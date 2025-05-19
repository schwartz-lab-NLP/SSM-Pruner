import argparse
import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from importlib.metadata import version

from MambaInLlama.mamba2.hybrid_wrapper import MambaTransformerHybridModelWrapper
from original_mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from phi_mamba.modules.lm_head import LMHeadModel
from cartesia_pytorch.Llamba.llamba import LlambaLMHeadModel


from lib.prune import prune_wanda, prune_magnitude, prune_sparsegpt, prune_ablate, check_sparsity, find_layers
from lib.eval import eval_ppl, eval_zero_shot
from utils.ppl import evaluate_wikitext

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())


def get_llm(model_name, cache_dir="llm_weights", is_mamba=False, is_lm_head=False, is_mamba_in_llama=False,
            split_mamba=False, is_llamba=False):
    if is_mamba:
        if is_lm_head:
            model = LMHeadModel.from_pretrained(
                model_name,
                attn_type="flash_attention_2" if torch.is_autocast_enabled() else "eager",
                strict=True,
            ).to(torch.bfloat16)
        elif is_mamba_in_llama:
            model = MambaTransformerHybridModelWrapper.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        elif is_llamba:
            model = LlambaLMHeadModel.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        else:
            model = MambaLMHeadModel.from_pretrained(model_name, device='cuda', dtype=torch.bfloat16)
            if split_mamba:
                model.split_in_proj()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            cache_dir=cache_dir,
            low_cpu_mem_usage=True,
            device_map="auto"
        )

    model.seqlen = 128  #model.config.max_position_embeddings if hasattr(model.config, "max_position_embeddings") else (model.config.MixerModel.input.d_model if is_lm_head else model.config.d_model)
    model.config.hidden_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else (
        model.config.MixerModel.input.d_model if is_lm_head else model.config.d_model)

    if is_mamba:
        model.config.use_cache = False
        if hasattr(model.config, "hidden_size")  and model.config.hidden_size is None:
            model.config.hidden_size = model.config.d_model
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help='LLaMA model')
    parser.add_argument('--lm_eval_name', type=str, help='LLaMA model', required=False, default=None)
    parser.add_argument("--is_mamba", action="store_true", default=False)
    parser.add_argument("--is_lm_head", action="store_true", default=False)
    parser.add_argument("--is_mamba_in_llama", action="store_true", default=False)
    parser.add_argument("--is_llamba", action="store_true", default=False)
    parser.add_argument('--s_prune', action="store_true", default=False, help='structural pruning.')
    parser.add_argument("--split_mamba", action="store_true", default=False)
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=128, help='Number of calibration samples.')
    parser.add_argument('--sparsity_ratio', type=float, default=0, help='Sparsity level')
    parser.add_argument("--sparsity_type", type=str, choices=["unstructured", "4:8", "2:4"])
    parser.add_argument("--prune_method", type=str, choices=["magnitude", "wanda", "sparsegpt",
                                                             "ablate_mag_seq", "ablate_wanda_seq", "ablate_mag_iter",
                                                             "ablate_wanda_iter", "search"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--use_variant', action="store_true",
                        help="whether to use the wanda variant described in the appendix")
    parser.add_argument('--save', type=str, default=None, help='Path to save results.')
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--pre_eval', action="store_true", default=False, help='Evaluate the model before pruning.')
    parser.add_argument('--plot', action="store_true", default=False, help='plot')

    parser.add_argument("--eval_zero_shot", action="store_true")
    args = parser.parse_args()

    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Handling n:m sparsity
    prune_n, prune_m = 0, 0
    if args.sparsity_type != "unstructured":
        assert args.sparsity_ratio == 0.5, "sparsity ratio must be 0.5 for structured N:M sparsity"
        prune_n, prune_m = map(int, args.sparsity_type.split(":"))

    print(f"loading llm model {args.model}")
    model = get_llm(args.model, args.cache_dir, is_mamba=args.is_mamba, is_lm_head=args.is_lm_head,
                    is_mamba_in_llama=args.is_mamba_in_llama, split_mamba=args.split_mamba, is_llamba=args.is_llamba)
    model.eval()
    if args.is_mamba:
        if args.is_lm_head:
            tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-1.7B', use_fast=False)
            tokenizer_path = 'HuggingFaceTB/SmolLM2-1.7B'
            # tokenizer_path = 'microsoft/phi-1_5'
        elif args.is_mamba_in_llama:
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
            tokenizer_path = args.model
        elif args.is_llamba:
            tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
            tokenizer_path = "meta-llama/Llama-3.2-1B"
        else:
            tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
            tokenizer_path = 'EleutherAI/gpt-neox-20b'
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

        tokenizer_path = args.model
    import pdb; pdb.set_trace()
    device = torch.device("cuda:0")
    if "30b" in args.model or "65b" in args.model:  # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    if args.pre_eval:
        print("*" * 30)
        print('pre pruning checks and evaluation')
        sparsity_ratio = check_sparsity(model)
        print(f"sparsity sanity check {sparsity_ratio:.4f}")
        # ppl_test = eval_ppl(args, model, tokenizer, device)
        ppl_test = evaluate_wikitext(model, tokenizer_path=tokenizer_path)
        print(f"wikitext perplexity {ppl_test}")
        print("*" * 30)

    if args.sparsity_ratio != 0:
        print("pruning starts")
        if args.prune_method == "wanda":
            prune_wanda(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        # if args.prune_method == "taylor":
        #     prune_taylor(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "magnitude":
            prune_magnitude(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif args.prune_method == "sparsegpt":
            prune_sparsegpt(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)
        elif "ablate" in args.prune_method:
            prune_ablate(args, model, tokenizer, device, prune_n=prune_n, prune_m=prune_m)

    ################################################################
    print("*" * 30)
    sparsity_ratio = check_sparsity(model)
    print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print("*" * 30)
    ################################################################
    # ppl_test = eval_ppl(args, model, tokenizer, device)

    ppl_test = evaluate_wikitext(model, tokenizer_path=tokenizer_path)
    # print(f"wikitext perplexity {ppl_test}")
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    save_filepath = os.path.join(args.save, f"log_{args.prune_method}.txt")
    with open(save_filepath, "w") as f:
        print("method\tactual_sparsity\tppl_test", file=f, flush=True)
        print(f"{args.prune_method}\t{sparsity_ratio:.4f}\t{ppl_test:.4f}", file=f, flush=True)

    if args.eval_zero_shot:
        assert args.lm_eval_name is not None, "Please provide the name of the model for zero-shot evaluation"
        accelerate = False
        if "30b" in args.model or "65b" in args.model or "70b" in args.model:
            accelerate = True

        # task_list = ["boolq", "rte","hellaswag","winogrande", "arc_easy","arc_challenge", "openbookqa"]
        task_list = ["piqa", "lambada_openai", "hellaswag", "winogrande", "arc_easy", "arc_challenge"]
        num_shot = 0
        results = eval_zero_shot(args.lm_eval_name, args.model, tokenizer_path, task_list, num_shot, accelerate)
        print("********************************")
        print("zero_shot evaluation results")
        print(results)

    if args.save_model:
        if args.is_lm_head:
            model.save_pretrained_distributed(args.save_model,
                                              is_main_process=True,
                                              update_config=True)
        elif args.is_mamba_in_llama and args.is_mamba:
            model.save_config(args.save_model)
            model.model.save_pretrained(args.save_model, safe_serialization=False, max_shard_size="50GB")
        else:
            model.save_pretrained(args.save_model)
        # tokenizer.save_pretrained(args.save_model)
        model = get_llm(args.save_model, args.cache_dir, is_mamba=args.is_mamba, is_lm_head=args.is_lm_head,
                        is_mamba_in_llama=args.is_mamba_in_llama, split_mamba=args.split_mamba, is_llamba=args.is_llamba)
    ppl_test = evaluate_wikitext(model, tokenizer_path=tokenizer_path)
    print(f"wikitext perplexity loaded {ppl_test}")


if __name__ == '__main__':
    main()
