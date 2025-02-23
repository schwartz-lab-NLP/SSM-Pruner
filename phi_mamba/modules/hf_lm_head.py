import sys;
from typing import Dict, Any

from datasets import load_dataset
from transformers.modeling_outputs import CausalLMOutputWithPast

sys.path.extend(['.', './phi_mamba',
                 './MambaInLlama', './original_mamba'])

import os
import json
import torch

from transformers import (
    PreTrainedModel,
    PretrainedConfig,
    GenerationMixin,
)

from phi_mamba.utils.config import Config  # example
from phi_mamba.modules.lm_head import LMHeadModel


##############################################################################
# Step 1: Define a Hugging Face–style Config that knows how to extract
# fields like vocab_size, pad_token_id, etc. from your model’s custom config.
##############################################################################

class CustomMambaConfig(PretrainedConfig):
    """
    A small wrapper around HF's `PretrainedConfig`.
    We store your custom Config inside `self.mamba_config` and also
    expose standard Hugging Face fields (`vocab_size`, etc.).
    """

    model_type = "custom-mamba"  # Just a descriptive string.

    def __init__(self, mamba_dict_or_config=None, **kwargs):
        """
        mamba_dict_or_config can be:
          - a python dict that matches your custom Config.from_dict()
          - or an instance of your `Config`
        We'll store the raw config in `self.mamba_config`.
        Then we also set HF-specific fields: vocab_size, etc.
        """
        super().__init__(**kwargs)

        # If it's already your Config object, use it directly;
        # else convert dict -> Config.
        if isinstance(mamba_dict_or_config, dict):
            mamba_dict_or_config_ = mamba_dict_or_config.get('mamba_config', mamba_dict_or_config)
            self.mamba_config = Config.from_dict(mamba_dict_or_config_)
        else:
            self.mamba_config = mamba_dict_or_config

        # Example: read the vocab size from your config
        if self.mamba_config is not None:
            vocab_size = self.mamba_config.LanguageModel.input.vocab_size
        else:
            vocab_size = 0
        self.vocab_size = vocab_size

        self.eos_token_id = None
        self.bos_token_id = None

        # For a causal language model:
        self.is_decoder = True

    def to_dict(self) -> Dict[str, Any]:
        res = super().to_dict()
        if self.mamba_config is not None:
            res.update({"mamba_config": self.mamba_config.to_dict()})
        return res

    @classmethod
    def from_json_file(cls, json_file: str):
        """Helper to load config from a local JSON file."""
        with open(json_file, "r") as f:
            data = json.load(f)
        return cls(data)


##############################################################################
# Step 2: Define a HF-compatible model that wraps your LMHeadModel.
##############################################################################
class CustomMambaLMHeadModel(PreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    """
    A Hugging Face–compatible model that internally holds your `LMHeadModel`.

    - Inherits from `PreTrainedModel` so that .save_pretrained() / .from_pretrained() work.
    - Implements a `forward(...)` that matches the standard HF CausalLM signature.
    - Calls through to your existing LMHeadModel for the actual forward pass.
    """

    # Tie this class to our custom config.
    config_class = CustomMambaConfig

    def __init__(self, config: CustomMambaConfig):
        """
        config is an instance of CustomMambaConfig, which itself wraps your custom config.
        """
        super().__init__(config)

        # Store the "real" custom config inside self.mamba_config
        self.mamba_config = config.mamba_config

        self.model = LMHeadModel(self.mamba_config.to_dict())

        # If you need to tie embeddings, you can do so here or inside LMHeadModel.
        if  (
            hasattr(self.mamba_config.LanguageModel.input, 'tie_word_embeddings')
            and self.mamba_config.LanguageModel.input.tie_word_embeddings
        ):
            self.tie_weights()

    def get_input_embeddings(self):
        return self.model.backbone.embedding

    def set_input_embeddings(self, value):
        self.model.backbone.embedding = value

    def get_output_embeddings(self):
        return self.model.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model.backbone = decoder

    def get_decoder(self):
        return self.model.backbone

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            labels=None,
            return_dict=None,
            **kwargs,
    ):
        """
        Standard HF forward signature for causal language models.

        You can pass any extra arguments from the trainer in **kwargs and

        If `return_dict=True`, return a `CausalLMOutputWithPast`.
        Otherwise, return tuple as in other HF CLM classes
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )

        loss = outputs.loss
        logits = outputs.logits

        if not return_dict:
            output = (logits,) + (None, outputs.all_hidden_states, outputs.all_transfer_matrices)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.all_hidden_states,
            attentions=outputs.all_transfer_matrices,
        )

    # @dataclass
    # class CustomMambaCausalLMOutput(ModelOutput):
    #     loss: Optional[torch.FloatTensor] = None
    #     logits: Optional[torch.FloatTensor] = None
    #     all_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    #     all_transfer_matrices: Optional[Tuple[torch.FloatTensor, ...]] = None
    #     all_mamba_outputs: Optional[Tuple[torch.FloatTensor, ...]] = None
    #     last_hidden_state: Optional[torch.FloatTensor] = None

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        """
        If you need your custom logic for generation (e.g., handling caches, positions),
        delegate to LMHeadModel's custom methods. Otherwise you can rely on the base
        `GenerationMixin.prepare_inputs_for_generation`.
        """
        return self.model.prepare_inputs_for_generation(input_ids, **kwargs)

    def _reorder_cache(self, past_key_values, beam_idx):
        """
        If you have caching logic for beam search or similar, implement it here.
        Otherwise you can remove this method or pass through.
        """
        if hasattr(self.model, "_reorder_cache"):
            return self.model._reorder_cache(past_key_values, beam_idx)
        # or just do nothing
        return past_key_values

    def tie_weights(self):
        """
        call lm_head's tie_weights
        """
        self.model.tie_weights()

    ############################################################################
    # Step 3: Overriding .from_pretrained() / .save_pretrained() so you can
    # reuse your existing logic, or rely on PreTrainedModel defaults.
    ############################################################################

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        You can intercept the load process if you need custom logic for your config
        or state dict. Otherwise, calling super() will do the usual HF flow:
          1) Load config from JSON
          2) Create model instance
          3) Load model weights
        """
        is_pure_lm_head = kwargs.pop("is_pure_lm_head", False)
        if is_pure_lm_head:
            lm_head_model = LMHeadModel.from_pretrained(pretrained_model_name_or_path, strict=True)
            hf_config = CustomMambaConfig(lm_head_model.config)
        else:
            # 1) Load the HF config from <dir>/config.json
            #    which we expect to contain your custom fields.
            config_path = kwargs.pop("config_path", None)
            if config_path is None:
                # By default, HF looks for 'config.json' in pretrained_model_name_or_path
                config_path = os.path.join(pretrained_model_name_or_path, "config.json")
            # Build our CustomMambaConfig
            hf_config = CustomMambaConfig.from_json_file(config_path)

        # 2) Create the model using that config
        model = cls(hf_config, *model_args, **kwargs)
        if is_pure_lm_head:
            model.model.load_state_dict(lm_head_model.state_dict())
            model.original_model = lm_head_model
            # del lm_head_model
        else:
            # 3) Let HF load the weights from <dir>/pytorch_model.bin
            state_dict = None
            if os.path.isdir(pretrained_model_name_or_path):
                weights_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
                if os.path.isfile(weights_path):
                    state_dict = torch.load(weights_path, map_location="cpu")

            if state_dict is not None:
                # If you need a custom key conversion, do it here, e.g.:
                # state_dict = model.mamba_model._convert_state_dict(state_dict)
                model.load_state_dict(state_dict, strict=False)

        return model

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the config + weights in a typical Hugging Face format:
          - <save_directory>/config.json
          - <save_directory>/pytorch_model.bin
        """
        os.makedirs(save_directory, exist_ok=True)

        # 1) Save config
        self.config.save_pretrained(save_directory)

        # 2) Save model weights
        weights_path = os.path.join(save_directory, "pytorch_model.bin")
        state_dict = self.state_dict()

        # If you need to invert the key conversion, do it here.
        # state_dict = invert_key_conversion(state_dict)

        torch.save(state_dict, weights_path)


if __name__ == '__main__':
    # TEST

    from trl import DPOTrainer, DPOConfig
    from transformers import TrainingArguments, AutoTokenizer, AutoModelForCausalLM  #, Trainer

    tokenizer_path = "microsoft/phi-1_5"
    model_path = "goombalab/Phi-Mamba"
    model = CustomMambaLMHeadModel.from_pretrained(model_path, is_pure_lm_head=True).to(torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    training_args = TrainingArguments(
        output_dir="./hf_results",
        num_train_epochs=1,
        optim="sgd",
        report_to="none",
        max_steps=1000,
    )

    # Load dataset
    train_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', streaming=False)
    # train_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:1%]', streaming=True)
    eval_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation', streaming=False)


    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Prepare labels
    tokenized_train = tokenized_train.map(lambda x: {"labels": x["input_ids"]}, batched=True)
    tokenized_eval = tokenized_eval.map(lambda x: {"labels": x["input_ids"]}, batched=True)

    model.train()
    model.requires_grad_(True)
    # for name, param in model.named_parameters():
    #     if 'mixer' in name or 'embedding' in name:
    #         param.requires_grad_(True)
    #     else:
    #         param.requires_grad_(False)

    dpo_config = DPOConfig(
        bf16=True,
        beta=0.01,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=2000,
        gradient_accumulation_steps=1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        learning_rate=5.0e-7,
        log_level="info",
        logging_steps=10,
        lr_scheduler_type="cosine",
        max_length=512,
        max_prompt_length=512,
        num_train_epochs=1,
        optim="sgd",
        output_dir='./hf_test',
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        push_to_hub=False,
        save_strategy="steps",
        save_steps=10000,
        save_total_limit=1,
        seed=42,
        warmup_ratio=0.1,
        report_to=["wandb"],
        max_steps=10_000
    )
    teacher = AutoModelForCausalLM.from_pretrained('microsoft/phi-1_5', torch_dtype=torch.bfloat16)
    teacher.eval()

    trainer = DPOTrainer(
        model,
        teacher,
        args=dpo_config,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        processing_class=tokenizer
    )

    trainer.train()
