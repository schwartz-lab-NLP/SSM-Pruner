import types

from phi_mamba.modules.modeling_llama import LlamaForCausalLM, LlamaAttention
from llama_custom import tova_llama_prepare_inputs_for_generation_generation, tova_llama_attention_forward
from phi_mamba.modules.modeling_phi_adjusted import PhiAttention, PhiForCausalLM
from transformers.models.phi.modeling_phi import PhiAttention as PhiAttentionOrig
from phi_custom import forward, prepare_inputs_for_generation


def enable_tova_caching(model):
    if hasattr(model, "model") and hasattr(model.model, "_use_sdpa"):
        model.model._use_sdpa = False

    if isinstance(model, LlamaForCausalLM):
        model.prepare_inputs_for_generation = types.MethodType(
            tova_llama_prepare_inputs_for_generation_generation, model
        )
        print('llama prepare_inputs_for_generation replaced')
    elif isinstance(model, PhiForCausalLM):
        model.prepare_inputs_for_generation = types.MethodType(
            prepare_inputs_for_generation, model
        )
        print('phi prepare_inputs_for_generation replaced')

    for name, module in reversed(model._modules.items()):
        if len(list(module.children())) > 0:
            enable_tova_caching(
                module,
            )
        if isinstance(module, (PhiAttention, PhiAttentionOrig)):
            model._modules[name].forward = types.MethodType(
                forward, model._modules[name]
            )

        elif isinstance(module, LlamaAttention):
            model._modules[name].forward = types.MethodType(
                tova_llama_attention_forward, model._modules[name]
            )
