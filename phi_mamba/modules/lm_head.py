# Copyright (c) 2024, Kevin Li, Aviv Bick.
import copy
import json
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Callable, Union, Dict

import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from huggingface_hub.hub_mixin import T
from transformers import StaticCache

from modules.modeling_llama import _prepare_4d_causal_attention_mask_with_cache_position, LlamaForCausalLM
from modules.modeling_phi_adjusted import PhiForCausalLM

try:
    from mamba_ssm.utils.generation import GenerationMixin
except ImportError:
    class GenerationMixin: pass

from transformers.utils import ModelOutput

from modules.backbone import MixerModel
try:
    from modules.mixers.discrete_mamba2 import Mixer as MambaMixer
except ImportError:
    class MambaMixer: pass
from modules.mixers.phi_attention import Mixer as PhiMixer
from utils.config import Config




@dataclass
class CustomMambaCausalLMOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    all_hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_transfer_matrices: Optional[Tuple[torch.FloatTensor, ...]] = None
    all_mamba_outputs: Optional[Tuple[torch.FloatTensor, ...]] = None
    last_hidden_state: Optional[torch.FloatTensor] = None


class LMHeadModel(nn.Module, GenerationMixin, PyTorchModelHubMixin):
    def __init__(
            self, config: dict, initializer_cfg=None, device='cuda', dtype=None, mask_head_indexes=None, prune_dstates_ratio=None, **kwargs
    ):

        super().__init__()
        self.model_base_type = 'phi' # can be changed prepare_inputs_for_generation choice
        # Load config
        if not isinstance(config, Config):
            config = Config.from_dict(config)
        self.config = config

        # Factory kwargs
        factory_kwargs = {"device": device, "dtype": dtype}
        self.device = device
        self.dtype = dtype
        # Pad vocab size to be a multiple of pad_vocab_size_multiple
        vocab_size = config.LanguageModel.input.vocab_size
        lm_head_bias = config.LanguageModel.input.lm_head_bias if hasattr(config.LanguageModel.input, "lm_head_bias") else True
        pad_vocab_size_multiple = config.LanguageModel.input.pad_vocab_size_multiple
        if vocab_size % pad_vocab_size_multiple != 0:
            vocab_size += pad_vocab_size_multiple - (
                    vocab_size % pad_vocab_size_multiple
            )
        self.config.LanguageModel.input.vocab_size = vocab_size
        self.mask_head_indexes = mask_head_indexes
        # Mixer model
        self.backbone = MixerModel(
            input_size=vocab_size,
            config=self.config,
            initializer_cfg=initializer_cfg,
            mask_head_indexes=mask_head_indexes,
            prune_dstates_ratio=prune_dstates_ratio,
            final_layernorm_bias=lm_head_bias,
            **factory_kwargs,
            **kwargs
        )

        # LM head
        d_model = self.config.MixerModel.input.d_model
        vocab_size = self.config.LanguageModel.input.vocab_size
        self.lm_head = nn.Linear(
            d_model, vocab_size, bias=lm_head_bias, **factory_kwargs
        )  # changed for Phi
        if hasattr(config.LanguageModel.input, 'tie_word_embeddings') and config.LanguageModel.input.tie_word_embeddings:
            self.tie_weights()




    def allocate_inference_cache(self, *args, **kwargs):
        return self.backbone.allocate_inference_cache(*args, **kwargs)


    def forward(
            self,
            input_ids,
            position_ids=None,
            return_mixer_matrix=False,
            return_mamba_outputs=False,
            return_hidden_states=False,
            return_logits=True,
            inference_params=None,
            num_last_tokens=0,
            attention_mask=None,
            labels=None,
            use_cache=False,
            mask_head_indexes=None,
            return_attn_presoftmax=False,
            **kwargs
    ) -> CustomMambaCausalLMOutput:
        """
        "position_ids" is just to be compatible with Transformer generation. We don't use it.
        num_last_tokens: if > 0, only return the logits for the last n tokens
        """
        return_hidden_states = return_hidden_states or kwargs.get("output_hidden_states", False)
        return_mamba_outputs = return_mamba_outputs or kwargs.get("output_attention_results", False)
        return_mixer_matrix = return_mixer_matrix or kwargs.get("output_attentions", False)
        outputs = self.backbone(
            input_ids,
            return_mixer_matrix=return_mixer_matrix,
            return_mamba_outputs=return_mamba_outputs,
            return_hidden_states=return_hidden_states,
            inference_params=inference_params,
            position_ids=position_ids,
            attention_mask=attention_mask,
            use_cache=use_cache,
            mask_head_index=mask_head_indexes if mask_head_indexes is not None else self.mask_head_indexes,
            return_attn_presoftmax=return_attn_presoftmax,
            **kwargs
        )

        if outputs["last_hidden_state"] is not None and return_logits:
            logits = self.lm_head(outputs["last_hidden_state"]).float()
            outputs["logits"] = (
                logits if num_last_tokens == 0 else logits[:, -num_last_tokens:]
            )
            if labels is not None:
                outputs["loss"] = nn.CrossEntropyLoss()(
                    logits.view(-1, logits.size(-1)), labels.view(-1)
                )
        else:
            outputs["logits"] = None

        loss = None
        if "loss" in outputs:
            loss = outputs["loss"]
        else:
            # calculate loss ?
            if labels is None:
                shift_labels = input_ids[:, 1:].contiguous()
            else:  # shift labels
                shift_labels = labels[:, 1:].contiguous()
            shift_logits = logits[:, :-1, :].contiguous()

            shift_logits_flat = shift_logits.view(-1, shift_logits.size(-1))
            shift_labels_flat = shift_labels.view(-1)
            # Compute cross-entropy loss:
            loss = nn.CrossEntropyLoss()(shift_logits_flat, shift_labels_flat)

        return CustomMambaCausalLMOutput(
            loss=loss,
            logits=outputs["logits"],
            all_hidden_states=outputs["all_hidden_states"],
            all_transfer_matrices=outputs["all_transfer_matrices"],
            all_mamba_outputs=outputs["all_mamba_outputs"],
            last_hidden_state=outputs["last_hidden_state"],
        )

    def _save_pretrained(self, save_directory: str, save_function: Callable = torch.save, is_main_process: bool = True, update_config: bool = False):
        """
        Minimal implementation of save_pretrained for MambaLMHeadModel.
        Save the model and its configuration file to a directory.
        """
        if is_main_process:
            # Ensure save_directory exists
            os.makedirs(save_directory, exist_ok=True)

            # Save the model's state_dict
            model_path = os.path.join(save_directory, "pytorch_model.bin")
            save_function(self.state_dict(), model_path)
            if update_config:
                # Save the configuration of the model as old_confgi.json
                config_path = os.path.join(save_directory, "old_config.json")
                with open(config_path, "w") as f:
                    json.dump(self.config.to_dict(), f)

                # Save the updated configuration of the model as config.json
                updated_config = self.create_updated_config()
                config_path = os.path.join(save_directory, "config.json")
                with open(config_path, "w") as f:
                    json.dump(updated_config, f)
            else:
                # Save the configuration of the model as config.json
                config_path = os.path.join(save_directory, "config.json")
                with open(config_path, "w") as f:
                    json.dump(self.config.to_dict(), f)


    def create_updated_config(self):
        """
        Create a configuration file for the model.
        """
        module_dir = os.path.dirname(__file__)
        is_phi = 'llama' not in str(type(self.backbone.layers[0]))
        template_path = os.path.join(module_dir, '..', 'assets', f'{"PHI" if is_phi else "LLAMA"}_TEMPLATE_DONT_CHANGE.json')
        with open(template_path, 'r') as json_file:
            mohawk_config_template = json.load(json_file)
        current_config = self.config.to_dict()
        new_config = {
            'LanguageModel': current_config['LanguageModel'],
            'MixerModel': current_config['MixerModel']
        }

        type_map = {MambaMixer: 'Block_MAMBA', PhiMixer: 'Block_PHI'}
        blocks = []
        for idx, layer in enumerate(self.backbone.layers):
            # import pdb; pdb.set_trace()
            new_block_config = copy.deepcopy(mohawk_config_template[type_map[type(layer.mixer)]])
            new_block_config['n_layers'] = 1
            if type(layer.mixer) == MambaMixer:
                new_block_config['core_input']['in_proj'] = {'in_proj_in_features': layer.mixer.in_proj.in_features, 'in_proj_out_features': layer.mixer.in_proj.out_features}
                new_block_config['core_input']['out_proj'] = {'out_proj_in_features': layer.mixer.out_proj.in_features, 'out_proj_out_features': layer.mixer.out_proj.out_features}
                new_block_config['core_input']['in_proj_bias'] = layer.mixer.in_proj.bias is not None
                new_block_config['core_input']['out_proj_bias'] = layer.mixer.out_proj.bias is not None
                new_block_config['core_input']['conv_dim'] = layer.mixer.conv1d.in_channels
                new_block_config['core_input']['d_inner'] = layer.mixer.d_inner
                new_block_config['core_input']['d_state'] = layer.mixer.d_state
                new_block_config['core_input']['n_v_heads'] = layer.mixer.n_v_heads
                new_block_config['core_input']['n_qk_heads'] = layer.mixer.n_qk_heads
                new_block_config['core_input']['d_conv'] = layer.mixer.d_conv
                new_block_config['core_input']['conv_bias'] = layer.mixer.conv_bias
                new_block_config['core_input']['expand'] = layer.mixer.expand
                new_block_config['core_input']['chunk_size'] = layer.mixer.chunk_size
                new_block_config['core_input']['activation'] = layer.mixer.activation
                new_block_config['core_input']['bias'] = layer.mixer.bias
                new_block_config['mlp']['intermediate_size'] = layer.mlp.fc1.weight.data.shape[0] if hasattr(layer.mlp, 'fc1') else layer.mlp.gate_proj.weight.data.shape[0]
                new_block_config['mlp']['last_mlp_bias'] = (layer.mlp.fc1.bias if hasattr(layer.mlp, 'fc2') else layer.mlp.down_proj.bias) is not None
            else:
                new_block_config['core_input']['inner_hidden_size'] = layer.mixer.self_attn.inner_hidden_size
                new_block_config['core_input']['num_key_value_heads'] = layer.mixer.self_attn.num_key_value_heads
                new_block_config['core_input']['num_attention_heads'] = layer.mixer.self_attn.num_heads
                new_block_config['mlp']['intermediate_size'] = layer.mlp.fc1.weight.data.shape[0]
            if len(blocks) == 0:
                blocks.append(new_block_config)
                continue
            # check equality on all fields but the n_layers
            are_equal = True
            for k in new_block_config:
                if k == 'n_layers':
                    continue
                if new_block_config[k] != blocks[-1][k]:
                    are_equal = False
                    break
            if are_equal:
                blocks[-1]['n_layers'] += 1
            else:
                blocks.append(new_block_config)

        for idx, block in enumerate(blocks):
            new_config[f"Block_{idx}"] = block
        return new_config

    def save_pretrained_distributed(self, save_directory: str, save_function: Callable = torch.save, is_main_process: bool = False, safe_serialization: bool = False, update_config: bool = False):
        """
        Save the model and its configuration file to a directory, distributed setup.
        """
        self._save_pretrained(save_directory, save_function, is_main_process, update_config)

    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output layer.
        """
        if hasattr(self.config.LanguageModel.input, 'tie_word_embeddings') and self.config.LanguageModel.input.tie_word_embeddings:
            self.lm_head.weight = self.backbone.embedding.weight
        else:
            print("WARNING: tie_word_embeddings is not set to True in the config. Skipping weight tying.")

    def prune_dstates(self, ratio, method='taylor', exclude_layers=None):
        if ratio == 0:
            return
        layers = self.backbone.layers if exclude_layers is None else [layer for i, layer in enumerate(self.backbone.layers) if i not in exclude_layers]
        for layer in layers:
            layer.mixer.prune_dstates(ratio, method=method)

    def prune_dstate_and_dinner(self, ratio, exclude_layers=None):
        layers = self.backbone.layers if exclude_layers is None else [layer for i, layer in enumerate(self.backbone.layers) if i not in exclude_layers]
        for layer in layers:
            layer.mixer.prune_dstate_and_dinner(ratio)

    def prune_by_mask_head_index(self):
        for layer in self.backbone.layers:
            layer.mixer.prune_by_mask_head_index()
        self.mask_head_indexes = None

    def prune_v_head_internal(self, ratio):
        layers = self.backbone.layers
        for layer in layers:
            layer.mixer.prune_v_head_internal(ratio)

    def prune_kq_heads(self, new_qk_heads: int, clustering_method='mean_pooling', exclude_layers=None):
        layers = self.backbone.layers if exclude_layers is None else [layer for i, layer in enumerate(self.backbone.layers) if i not in exclude_layers]
        for layer in layers:
            layer.mixer.prune_kq_heads(new_qk_heads, clustering_method)

    def get_memory_footprint(self):
        """
        Get the memory footprint of the model.
        """
        return sum(p.numel() * p.element_size() for p in self.parameters())

    def prepare_inputs_for_generation(
        self,
        input_ids,
        **kwargs,
    ):
        if self.model_base_type == 'phi':
            return self.phi_prepare_inputs_for_generation(input_ids, **kwargs)
        else:
            return self.llama_prepare_inputs_for_generation(input_ids, **kwargs)

    def llama_prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
            if model_inputs["inputs_embeds"] is not None:
                batch_size, sequence_length, _ = model_inputs["inputs_embeds"].shape
                device = model_inputs["inputs_embeds"].device
            else:
                batch_size, sequence_length = model_inputs["input_ids"].shape
                device = model_inputs["input_ids"].device

            dtype = self.lm_head.weight.dtype
            min_dtype = torch.finfo(dtype).min

            attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
                attention_mask,
                sequence_length=sequence_length,
                target_length=past_key_values.get_max_length(),
                dtype=dtype,
                device=device,
                min_dtype=min_dtype,
                cache_position=cache_position,
                batch_size=batch_size,
            )

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    def phi_prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        **kwargs,
    ):
        # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
        # Exception 1: when passing input_embeds, input_ids may be missing entries
        # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
        if past_key_values is not None:
            if inputs_embeds is not None:  # Exception 1
                input_ids = input_ids[:, -cache_position.shape[0] :]
            elif input_ids.shape[1] != cache_position.shape[0]:  # Default case (the "else", a no op, is Exception 2)
                input_ids = input_ids[:, cache_position]

        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

                # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
                position_ids = position_ids.clone(memory_format=torch.contiguous_format)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and cache_position[0] == 0:
            model_inputs = {"inputs_embeds": inputs_embeds, "input_ids": None}
        else:
            # The clone here is for the same reason as for `position_ids`.
            model_inputs = {"input_ids": input_ids.clone(memory_format=torch.contiguous_format), "inputs_embeds": None}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


    @property
    def model(self):
        """ for hugging face api compatibility """
        return self.backbone

    @property
    def layers(self):
        return self.backbone.layers

    @staticmethod
    def _convert_state_dict(state_dict_: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Convert an HF state dict to the format expected lm_head
        model. -> backbone.
        embed_tokens -> embedding
        self_attn -> mixer.self_attn
        """
        new_state_dict = {}
        for k, v in state_dict_.items():
            if k == 'model.embed_tokens.weight':
                k = 'backbone.embedding.weight'
            elif k == 'model.norm.weight':
                k = 'backbone.final_layernorm.weight'
            elif k == 'model.norm.bias':
                k = 'backbone.final_layernorm.bias'
            else:
                k = k.replace("model.", "backbone.")
                k = k.replace(".self_attn", ".mixer.self_attn") if "mixer" not in k else k

            new_state_dict[k] = v
        return new_state_dict

    @staticmethod
    def from_hf_model(hf_model: Union[PhiForCausalLM, LlamaForCausalLM], config_path: str = "config.json", *model_args, **kwargs) -> T:
        """ create config from phi model """
        lm_head_config = Config.from_file(config_path)
        model = LMHeadModel(lm_head_config, *model_args, **kwargs)
        state_dict = hf_model.state_dict()
        state_dict = LMHeadModel._convert_state_dict(state_dict)
        model.load_state_dict(state_dict)
        return model

    @staticmethod
    def from_pretrained(
            pretrained_model_name_or_path: Union[str, os.PathLike],
            *model_args,
            **kwargs
    ) -> T:
        """
        Instantiate a model from a pretrained model.
        """
        local = kwargs.pop("local", False) or os.path.exists(pretrained_model_name_or_path)

        if local:
            config_path = kwargs.get("config_path", os.path.join(pretrained_model_name_or_path, "config.json"))
            config = Config.from_file(config_path)
            model = LMHeadModel(config, *model_args, **kwargs)
            # import pdb; pdb.set_trace()
            model_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
            state_dict = torch.load(model_path)
            state_dict = LMHeadModel._convert_state_dict(state_dict)
            model.load_state_dict(state_dict)
            return model
        else:
            if "config_path" in kwargs:
                config_path = kwargs.pop("config_path")
                config = Config.from_file(config_path)
                kwargs["config"] = config
            return super(LMHeadModel, LMHeadModel).from_pretrained(
                pretrained_model_name_or_path, *model_args, **kwargs
            )