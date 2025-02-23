import sys;

from hf_distillation_trainer import DistillationTrainer
from modules.hf_mamba2 import MambaHFWrapper
from modules.lm_head import LMHeadModel
from modules.modeling_phi_adjusted2 import PhiForCausalLM


import argparse
import os
import math
import torch
from accelerate import Accelerator
from transformers import (
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    TrainerCallback
)

# PEFT for LoRA
try:
    from peft import LoraConfig, get_peft_model, TaskType, PeftModel
except ImportError:
    raise ImportError(
        "Please install PEFT to use LoRA. For example: `pip install peft`"
    )

from MambaInLlama.mamba2.hybrid_wrapper import MambaTransformerHybridModelWrapper
from modules.hf_lm_head import CustomMambaLMHeadModel
from original_mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from utils.ppl import get_wikitext_dataloader, evaluate_wikitext


class PerplexityCallback(TrainerCallback):
    """
    A simple callback that computes and logs perplexity based on eval_loss.
    """

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics is not None and "eval_loss" in metrics:
            eval_loss = metrics["eval_loss"]
            # Avoid math overflow if eval_loss is extremely large
            eval_ppl = math.exp(eval_loss) if eval_loss < 100 else float('inf')
            metrics["eval_perplexity"] = eval_ppl
            # The Trainer will log everything in `metrics` to wandb automatically.


def get_llm(model_name, is_mamba=False, is_lm_head=False, is_mamba_in_llama=False, is_phi=False, is_pure_lm_head=False):
    """
    Returns the model loaded from a Hugging Face Hub or local path.
    Handles Mamba vs standard HF models.
    """
    if is_mamba:
        if is_lm_head:
            # Mamba LM Head model with a custom wrapper
            model = CustomMambaLMHeadModel.from_pretrained(
                model_name,
                is_pure_lm_head=True
            ).to(torch.bfloat16)
        elif is_pure_lm_head:
            # Standard HF model with custom LM head
            model = LMHeadModel.from_pretrained(
                model_name,
            ).to(torch.bfloat16)
        elif is_mamba_in_llama:
            # Mamba + LLaMA hybrid
            model = MambaTransformerHybridModelWrapper.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16
            ).model
        else:
            # Classic MambaLMHeadModel
            model = MambaHFWrapper.from_pretrained(
                model_name,
                device='cuda',
                dtype=torch.bfloat16
            )
    elif is_phi:
        # Phi model
        model = PhiForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    else:
        # Standard HF model
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
        )
    return model


def freeze_model(model, is_mamba):
    """
    Example freeze logic:
    - If Mamba, unfreeze 'mamba' or 'mixer' layers (except for 'attn' in mixer).
    - If not Mamba, only unfreeze 'attn' layers.
    Adjust to your needs.
    """
    for name, param in model.named_parameters():
        if 'mamba' in name or ('mixer' in name and 'attn' not in name) or (not is_mamba and 'attn' in name):
            param.requires_grad = True
        else:
            param.requires_grad = False


def apply_lora(args, model, lora_rank=8, lora_alpha=16, lora_dropout=0.05):
    """
    Wrap the given model with LoRA modules for the target modules.
    Then freeze all non-LoRA parameters.
    """
    if args.is_mamba:
        target_modules = ['in_proj', 'out_proj',]
        if args.is_lm_head:
            target_modules.extend(['fc2', 'fc1'])
        else:
            target_modules.extend( ["gate_proj", "down_proj", "up_proj"])
    else:
        target_modules = ["q_proj", "k_proj", "v_proj"]
        if args.is_phi:
            target_modules.extend(["dense",  "fc1", "fc2"])
        else:
            target_modules.extend([ "o_proj", "gate_proj", "down_proj", "up_proj",])


    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )

    # Convert model to a PEFT model
    peft_model = get_peft_model(model, lora_config)

    # Freeze base model parameters; only LoRA-added parameters will remain trainable
    for name, param in peft_model.named_parameters():
        if "lora" not in name.lower():
            param.requires_grad = False
        else:
            param.requires_grad = True
    return peft_model


def main(args):
    accelerator = Accelerator()
    world_size = accelerator.num_processes
    rank = accelerator.process_index

    # Setup wandb environment
    os.environ["WANDB_PROJECT"] = "Pruned_then_Finetuned"

    # Prepare data
    train_dataloader = get_wikitext_dataloader(
        percentage=100,
        seq_len=args.seq_llen,
        tokenizer_path=args.tokenizer,
        split='train',
        batch=args.batch_size,
        return_iter_dataset=True,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        world_size=world_size,
        rank=rank,
        streaming='c4' in args.dataset_path
    )

    eval_dataloader = get_wikitext_dataloader(
        percentage=100,
        seq_len=args.seq_llen,
        tokenizer_path=args.tokenizer,
        split='test',
        batch=args.batch_size,
        return_iter_dataset=False,
        dataset_path="cimec/lambada",
        dataset_name=None,
        world_size=world_size,
        rank=rank,
        return_dataset=True
    )

    slurm_job_id = os.environ.get('SLURM_JOB_ID', '')
    run_name = f"FTP{'_LORA' if args.use_lora else ''}{f'_DISTIL_{args.distil_alpha}' if args.distil else ''}_{args.model.replace('/', '_')}_{slurm_job_id}"

    # Load model
    model = get_llm(args.model, args.is_mamba, args.is_lm_head, args.is_mamba_in_llama)

    if args.use_lora:
        # Apply LoRA
        model = apply_lora(
            args,
            model,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout
        )
    else:
        # Normal freeze logic if not using LoRA
        freeze_model(model, args.is_mamba)

    model.train()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=run_name,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=2,
        save_total_limit=3,
        save_strategy="steps",
        save_steps=5000,
        logging_steps=500,
        evaluation_strategy="steps",
        eval_steps=50,  # Evaluate every 500 steps
        eval_on_start=True,
        optim="adamw_torch",
        learning_rate=args.lr,
        bf16=True,
        warmup_steps=500,
        run_name=run_name,
        report_to=["wandb"],  # Log to wandb
        prediction_loss_only=True,  # We only need loss for perplexity
        weight_decay=0.01,
    )

    if args.distil:
        teacher_model = get_llm(args.teacher, args.is_mamba_teacher, args.is_lm_head_teacher, args.is_mamba_in_llama_teacher)
        teacher_model = teacher_model.eval()
        teacher_model = teacher_model.to(model.device)
        trainer = DistillationTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataloader,
            eval_dataset=eval_dataloader,
            teacher_model=teacher_model,  # Set teacher model here
            alpha=args.distil_alpha,
            temperature=1.0
        )
    else:
        # Initialize Trainer with our callback for perplexity
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataloader,
            eval_dataset=eval_dataloader,
            callbacks=[PerplexityCallback()]
        )


    # Start training
    trainer.train()

    trainer.save_model(output_dir=os.path.join(run_name, "final_model"))

    #evaluate
    eval_metrics = trainer.evaluate()
    print("Final eval metrics:", eval_metrics)


def evaluate(args):
    model = get_llm(args.model, is_mamba=args.is_mamba, is_lm_head=args.is_lm_head, is_mamba_in_llama=args.is_mamba_in_llama, is_pure_lm_head=args.is_pure_lm_head)


    if args.use_lora:
        assert args.adapter is not None, "LoRA requires an adapter model path"
        model = PeftModel.from_pretrained(model, args.adapter)

    model.eval()

    ppl = evaluate_wikitext(model, use_cache=False, pass_attention_mask=True, tokenizer_path=args.tokenizer, dataset_name='wikitext-2-raw-v1')
    print(f"Perplexity: {ppl}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='HuggingFace model name or path')
    parser.add_argument('--teacher', type=str, required=False, help='HuggingFace teacher model name or path', default=None)
    parser.add_argument('--adapter', type=str, required=False, help='HuggingFace model name or path', default=None)
    parser.add_argument("--is_mamba", action="store_true", default=False)
    parser.add_argument("--is_mamba_teacher", action="store_true", default=False)
    parser.add_argument("--is_lm_head", action="store_true", default=False)
    parser.add_argument("--is_pure_lm_head", action="store_true", default=False)
    parser.add_argument("--is_lm_head_teacher", action="store_true", default=False)
    parser.add_argument("--is_mamba_in_llama", action="store_true", default=False)
    parser.add_argument("--is_mamba_in_llama_teacher", action="store_true", default=False)
    parser.add_argument("--is_phi", action="store_true", default=False)
    parser.add_argument("--dataset_name", type=str, default='wikitext-103-v1')
    parser.add_argument("--dataset_path", type=str, default='Salesforce/wikitext')
    parser.add_argument('--batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--lr', type=float,  default=5e-5, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1, help='epochs')
    parser.add_argument('--seq_llen', type=int, default=2048, help='sequence length')
    parser.add_argument('--tokenizer', type=str, required=True, help='Tokenizer path or name')
    parser.add_argument('--only_eval', action="store_true", default=False)
    parser.add_argument('--distil', action="store_true", default=False)
    parser.add_argument('--distil_alpha', type=float, default=0.5)



    # LoRA-specific arguments
    parser.add_argument("--use_lora", action="store_true", default=False,
                        help="If set, apply LoRA fine-tuning on attention layers.")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="Rank for the LoRA layers.")
    parser.add_argument("--lora_alpha", type=int, default=16,
                        help="Lora alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="Dropout probability for LoRA.")

    args = parser.parse_args()
    if args.dataset_name == 'None':
        args.dataset_name = None
    if args.only_eval:
        evaluate(args)
    else:
        main(args)