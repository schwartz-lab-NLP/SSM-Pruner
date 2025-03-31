import os

import torch
import transformers
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from lm_eval.models.utils import stop_sequences_criteria
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from MambaInLlama.mamba2.hybrid_wrapper import MambaTransformerHybridModelWrapper
from modules.hf_lm_head import CustomMambaLMHeadModel
from modules.hf_mamba2 import MambaHFWrapper
from modules.lm_head import LMHeadModel
from modules.modeling_phi_adjusted import PhiForCausalLM as PhiForCausalLMAdjusted
from original_mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

os.environ["TQDM_DISABLE"] = "1"


# Code from https://github.com/state-spaces/mamba/tree/main


class BaseEvalWrapper(HFLM):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(
            self,
            pretrained=None,
            max_length=2048,
            batch_size=None,
            device="cuda",
            dtype=torch.float32,
            tokenizer="microsoft/phi-1_5"
    ):  # training is everything 32
        pretrained = pretrained.to(device).to(dtype).eval()
        print('-' * 100, pretrained.device)
        super().__init__(pretrained=pretrained, max_length=max_length, batch_size=batch_size, device=device,
                         dtype=dtype, tokenizer=tokenizer, backend="causal")

        # Parameters
        # self._batch_size = int(batch_size) if batch_size is not None else 64
        # self._max_length = max_length
        # self._device = torch.device(device)
        self._dtype = dtype
        # Tokenizer
        # self.tokenizer = transformers.AutoTokenizer.from_pretrained("microsoft/phi-1_5")
        assert self.tokenizer.pad_token_id == self.tokenizer.eos_token_id
        assert self.vocab_size == self.tokenizer.vocab_size

    # @property
    # def batch_size(self):
    #     return self._batch_size

    def _model_generate(self, **kwargs):
        raise NotImplementedError()


@register_model("phi-mamba")
class PhiMambaEvalWrapper(BaseEvalWrapper):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, **kwargs):
        _model = LMHeadModel.from_pretrained("goombalab/Phi-Mamba", strict=True)
        super().__init__(pretrained=_model, **kwargs)
        self._model.to(self._device).to(self._dtype).eval()


@register_model("hybrid-phi-mamba")
class HybPhiMambaEvalWrapper(BaseEvalWrapper):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, **kwargs):
        _model = LMHeadModel.from_pretrained(
            "goombalab/Hybrid-Phi-Mamba",
            attn_type="flash_attention_2" if torch.is_autocast_enabled() else "eager",
            strict=True,
        )
        super().__init__(_model, **kwargs)
        self._model.to(self._device).to(self._dtype).eval()


@register_model("phi")
class PhiEvalWrapper(BaseEvalWrapper):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, **kwargs):
        _model = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5")
        super().__init__(_model, **kwargs)
        self._model.to(self._device).to(self._dtype).eval()


@register_model("phi-adj")
class AdjPhiEvalWrapper(BaseEvalWrapper):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, **kwargs):
        path = os.environ.get("EVAL_PATH", "microsoft/phi-1_5")
        print('PATH IN USE ------------------->', path)
        _model = PhiForCausalLMAdjusted.from_pretrained(path, attn_implementation='eager')  #, strict=True)
        super().__init__(_model, **kwargs)
        self._model.to(self._device).to(self._dtype).eval()


################### SMOL

@register_model("smol360")
class SmolEvalWrapper(BaseEvalWrapper):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, **kwargs):
        _model = AutoModelForCausalLM.from_pretrained("HuggingFaceTB/SmolLM2-360M")
        super().__init__(_model, tokenizer="HuggingFaceTB/SmolLM2-360M", **kwargs)
        self._model.to(self._device).to(self._dtype).eval()


@register_model("smol17")
class Smol17EvalWrapper(BaseEvalWrapper):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained="HuggingFaceTB/SmolLM2-1.7B", **kwargs):
        _model = AutoModelForCausalLM.from_pretrained(pretrained)
        kwargs.pop('pretrained', None)
        super().__init__(_model, tokenizer="HuggingFaceTB/SmolLM2-1.7B", **kwargs)
        self._model.to(self._device).to(self._dtype).eval()


@register_model("lm-head-smol17")
class Smol17LMheadEvalWrapper(BaseEvalWrapper):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained=None, peft=None, **kwargs):
        path = os.environ.get("EVAL_PATH", ".")
        _model = LMHeadModel.from_pretrained(path, strict=True)
        if peft is not None:
            _model = PeftModel.from_pretrained(_model, peft)
            print('PEFT PATH IN USE ------------------->', peft)
        super().__init__(_model, tokenizer="HuggingFaceTB/SmolLM2-1.7B", **kwargs)
        self._model.to(self._device).to(self._dtype).eval()


@register_model("lm-head-smol17-hf")
class Smol17LMheadHFEvalWrapper(BaseEvalWrapper):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained=None, peft=None,**kwargs):
        path = os.environ.get("EVAL_PATH", ".")
        _model = CustomMambaLMHeadModel.from_pretrained(path, is_pure_lm_head=True)
        if peft is not None:
            _model = PeftModel.from_pretrained(_model, peft)
            print('PEFT PATH IN USE ------------------->', peft)
        super().__init__(_model, tokenizer="HuggingFaceTB/SmolLM2-1.7B", **kwargs)
        self._model.to(self._device).to(self._dtype).eval()


@register_model("lm-head-phi")
class PhiLMHeadEvalWrapper(BaseEvalWrapper):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, **kwargs):
        path = os.environ.get("EVAL_PATH", ".")
        _model = LMHeadModel.from_pretrained(path, strict=True)
        super().__init__(_model, tokenizer="microsoft/phi-1_5", **kwargs)
        self._model.to(self._device).to(self._dtype).eval()

@register_model("auto-lm-head")
class AutoLMHeadEvalWrapper(BaseEvalWrapper):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, **kwargs):
        path = kwargs.get('pretrained', None)
        tokenizer = kwargs.get('tokenizer', None)
        assert path is not None, "Pretrained model path is required"
        assert tokenizer is not None, "Tokenizer is required"
        del kwargs['pretrained']
        del kwargs['tokenizer']
        _model = LMHeadModel.from_pretrained(path, strict=True)
        super().__init__(_model, tokenizer=tokenizer, **kwargs)
        self._model.to(self._device).to(self._dtype).eval()

@register_model("auto-mamba-llama")
class AutoLMHeadEvalWrapper(BaseEvalWrapper):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, **kwargs):
        path = kwargs.get('pretrained', None)
        tokenizer = kwargs.get('tokenizer', None)
        assert path is not None, "Pretrained model path is required"
        assert tokenizer is not None, "Tokenizer is required"
        del kwargs['pretrained']
        del kwargs['tokenizer']
        _model = MambaTransformerHybridModelWrapper.from_pretrained(path).bfloat16()
        super().__init__(_model, tokenizer=tokenizer, **kwargs)
        self._model.to(self._device).to(self._dtype).eval()


@register_model("lm-head-phi-hf")
class PhiLMHeadHfEvalWrapper(BaseEvalWrapper):
    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, **kwargs):
        peft_path = kwargs.get('peft', None)
        path = os.environ.get("EVAL_PATH", ".")
        _model = CustomMambaLMHeadModel.from_pretrained(path, is_pure_lm_head=True)
        if peft_path is not None:
            _model = PeftModel.from_pretrained(_model, peft_path)
            del kwargs['peft']
            print('PEFT PATH IN USE ------------------->', peft_path)
        super().__init__(_model, tokenizer="microsoft/phi-1_5", **kwargs)
        self._model.to(self._device).to(self._dtype).eval()



@register_model("mamba")
class MambaEvalWrapper(BaseEvalWrapper):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained="state-spaces/mamba2-2.7b",  **kwargs):
        path = os.environ.get("EVAL_PATH", None)
        path = pretrained if path is None else path
        _model = MambaLMHeadModel.from_pretrained(path, device='cuda')
        super().__init__(_model, tokenizer="EleutherAI/gpt-neox-20b", **kwargs)
        self._model.to(self._device).to(self._dtype).eval()


@register_model("hf-mamba")
class HFMambaEvalWrapper(BaseEvalWrapper):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained="state-spaces/mamba2-2.7b",  **kwargs):
        peft_path = kwargs.get('peft', None)
        path = os.environ.get("EVAL_PATH", None)
        path = pretrained if path is None else path
        _model = MambaHFWrapper.from_pretrained(path)
        if peft_path is not None:
            _model = PeftModel.from_pretrained(_model, peft_path)
            del kwargs['peft']
            print('PEFT PATH IN USE ------------------->', peft_path)
        super().__init__(_model, tokenizer="EleutherAI/gpt-neox-20b", **kwargs)
        self._model.to(self._device).to(self._dtype).eval()

@register_model("mamba2_hybrid")
class Mamba2EvalWrapper(HFLM):

    AUTO_MODEL_CLASS = transformers.AutoModelForCausalLM

    def __init__(self, pretrained, max_length=2048, batch_size=None, device="cuda",
                 dtype=torch.bfloat16,tokenizer=None, peft=None, ** kwargs):
        _model = MambaTransformerHybridModelWrapper.from_pretrained(pretrained, torch_dtype=dtype).model
        if peft is not None:
            _model = PeftModel.from_pretrained(_model, peft)
            print('PEFT PATH IN USE ------------------->', peft)
        tokenizer = tokenizer if tokenizer is not None else pretrained
        print(_model)
        super().__init__(_model, tokenizer=tokenizer, dtype=dtype, **kwargs)
        # self.tokenizer = AutoTokenizer.from_pretrained(pretrained)
        self._model.to(self._device).to(dtype).eval()
        self.vocab_size = self.tokenizer.vocab_size
        self._batch_size = int(batch_size) if batch_size is not None else 64
        self._max_length = max_length
        self.truncation = False
        self._device = torch.device(device)

    @property
    def batch_size(self):
        return self._batch_size

    # this is copied from https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/models/huggingface.py#L824-L849
    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        # temperature = 0.0 if not set
        # if do_sample is false and temp==0.0:
        # remove temperature, as do_sample=False takes care of this
        # and we don't want a warning from HF
        generation_kwargs["temperature"] = generation_kwargs.get("temperature", 0.0)
        do_sample = generation_kwargs.get("do_sample", None)

        # The temperature has to be a strictly positive float -- if it is 0.0, use greedy decoding strategies
        if generation_kwargs.get("temperature") == 0.0 and do_sample is None:
            generation_kwargs["do_sample"] = do_sample = False

        if do_sample is False and generation_kwargs.get("temperature") == 0.0:
            generation_kwargs.pop("temperature")
        # build stopping criteria
        stopping_criteria = stop_sequences_criteria(
            self.tokenizer, stop, context.shape[1], context.shape[0]
        )
        return self.model.generate(
            input_ids=context,
            max_length=max_length,
            stopping_criteria=stopping_criteria,
            pad_token_id=self.tokenizer.pad_token_id,
            use_cache=False,
            **generation_kwargs,
        )


if __name__ == "__main__":
    cli_evaluate()
