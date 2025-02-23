import json
import sys
from dataclasses import asdict
from typing import Dict, Any, Optional


sys.path.extend(['.', './phi_mamba',
                 './MambaInLlama', './original_mamba'])
import torch
import torch.nn as nn

from original_mamba.mamba_ssm.models.config_mamba import MambaConfig
from original_mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from transformers import PretrainedConfig, PreTrainedModel


# Assume the original MambaLMHeadModel and its dependencies are already imported
# from your provided code.


class CustomMambaOrigConfig(PretrainedConfig):
    """
    A small wrapper around HF's `PretrainedConfig`.
    We store your custom Config inside `self.mamba_config` and also
    expose standard Hugging Face fields (`vocab_size`, etc.).
    """
    model_type = "custom-mamba"  # Just a descriptive string.

    def __init__(self, mamba_config:Optional[MambaConfig] = None, **kwargs):
        """
        mamba_dict_or_config can be:
        """
        super().__init__(**kwargs)

        if mamba_config is not None:
            self.mamba_config = mamba_config
        else:
            self.mamba_config = MambaConfig()
        self.vocab_size =  self.mamba_config.vocab_size
        self.eos_token_id = None
        self.bos_token_id = None
        self.is_decoder = True

    def to_dict(self) -> Dict[str, Any]:
        res = super().to_dict()
        if self.mamba_config is not None:
            res.update({"mamba_config": asdict(self.mamba_config)})
        return res

    @classmethod
    def from_json_file(cls, json_file: str):
        """Helper to load config from a local JSON file."""
        with open(json_file, "r") as f:
            data = json.load(f)
        return cls(data)


class MambaHFWrapper(PreTrainedModel):
    """
    A wrapper for MambaLMHeadModel to make it compatible with the Hugging Face Trainer.
    This wrapper does not modify the underlying operations of MambaLMHeadModel.
    """

    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}
    """
    A Hugging Face–compatible model that internally holds your `LMHeadModel`.

    - Inherits from `PreTrainedModel` so that .save_pretrained() / .from_pretrained() work.
    - Implements a `forward(...)` that matches the standard HF CausalLM signature.
    - Calls through to your existing LMHeadModel for the actual forward pass.
    """

    # Tie this class to our custom config.
    config_class = CustomMambaOrigConfig

    def __init__(self, mamba_model: MambaLMHeadModel, **kwargs):
        """
        Initializes the wrapper with an instance of MambaLMHeadModel.

        Args:
            mamba_model (nn.Module): An instance of MambaLMHeadModel.
        """
        config  = CustomMambaOrigConfig(mamba_config=mamba_model.config)
        super().__init__(config)
        self.model: MambaLMHeadModel = mamba_model
        # Expose the model config for compatibility.
        self.config = config

        if self.config.mamba_config.tie_embeddings:
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

    def tie_weights(self):
        if self.config.mamba_config.tie_embeddings:
            self.model.tie_weights()

    def forward(self, input_ids, labels=None, **kwargs):
        """
        Forward pass for the model.

        Args:
            input_ids (torch.Tensor): Input token IDs.
            labels (torch.Tensor, optional): Labels (unused, as the underlying model computes loss).
            **kwargs: Additional keyword arguments forwarded to MambaLMHeadModel.

        Returns:
            dict: A dictionary with keys 'loss' and 'logits' for Trainer compatibility.
        """
        # Call the underlying model. Note: position_ids and other generation-specific kwargs can be passed via **kwargs.
        outputs = self.model(input_ids, **kwargs)
        # The original model returns a namedtuple with `loss` and `logits`.
        return outputs

    def generate(self, *args, **kwargs):
        """
        Delegate generation to the underlying model.
        """
        return self.model.generate(*args, **kwargs)

    def save_pretrained(self, save_directory: str, **kwarg):
        """
        Delegate saving to the underlying model.
        """
        return self.model.save_pretrained(save_directory)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, device=None, dtype=None, **kwargs):
        """
        Create a wrapper model from a pretrained checkpoint.

        Args:
            pretrained_model_name_or_path (str): The directory or model identifier.
            device (optional): Device to load the model on.
            dtype (optional): Desired torch dtype.
            **kwargs: Additional keyword arguments passed to MambaLMHeadModel.from_pretrained.

        Returns:
            MambaHFWrapper: An instance of the wrapper with the loaded model.
        """
        # Use the class method from MambaLMHeadModel to load the underlying model.
        from_pretrained_model = MambaLMHeadModel.from_pretrained(
            pretrained_model_name_or_path, device=device, dtype=dtype, **kwargs
        )
        return cls(from_pretrained_model)

    def prepare_inputs_for_generation(self, *args, **kwargs):
        """
        Delegate input preparation for generation to the underlying model.
        """
        return self.model.prepare_inputs_for_generation(*args, **kwargs)



if __name__ == "__main__":
    from transformers import Trainer, TrainingArguments, AutoTokenizer, PretrainedConfig
    from datasets import load_dataset

    # Assume you have a valid MambaConfig instance:
    # from your original code
    config = MambaConfig(
        d_model=2560,
        n_layer=2,
        d_intermediate=0,
        vocab_size=50277,
        ssm_cfg={"layer": "Mamba2"}
        # ... (other parameters as needed)
    )
    # Create the underlying model
    mamba_model = MambaLMHeadModel(config, dtype=torch.bfloat16)

    # Wrap the model for Trainer compatibility.
    model = MambaHFWrapper(mamba_model)

    # Prepare training arguments (customize as needed)
    training_args = TrainingArguments(
        output_dir="./mamba_model",
        num_train_epochs=1,
        per_device_train_batch_size=8,
        logging_steps=2,
        save_steps=10,
        max_steps=10,
        eval_steps=2
    )

    # # Assume you have a Hugging Face Dataset or DataLoader that provides input_ids
    # # For example, a dummy batch:
    # dummy_batch = {"input_ids": torch.randint(0, config.vocab_size, (4, 128))}

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
    tokenizer.pad_token = tokenizer.eos_token


    # Load dataset
    train_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', streaming=False)
    # train_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train[:1%]', streaming=True)
    eval_dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='validation', streaming=False)


    # Tokenize
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)


    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Prepare labels
    tokenized_train = tokenized_train.map(lambda x: {"labels": x["input_ids"]}, batched=True)


    # Create a Trainer instance.
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,  # Replace with your actual dataset
        eval_dataset=tokenized_train,  # Replace with your actual dataset
    )

    # Start training
    trainer.train()


    model = MambaHFWrapper.from_pretrained("./mamba_model/checkpoint-10")

