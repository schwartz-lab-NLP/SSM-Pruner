import sys; sys.path.extend(['.', './phi_mamba'])
import torch
from transformers import AutoTokenizer
from phi_mamba.modules.modeling_phi_adjusted import PhiForCausalLM
from phi_mamba.modules.modeling_llama import LlamaForCausalLM
from phi_mamba.utils.ppl import evaluate_wikitext
from transformers.models.phi.modeling_phi import PhiForCausalLM as PhiForCausalLMOriginal
from tova_cache import TOVACache
from convert import enable_tova_caching



if __name__ == "__main__":
    # Load model
    # tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-1_5')
    tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM-360M')
    tokenizer.pad_token = tokenizer.eos_token
    device = 'cuda' if torch.cuda.is_available() else 'mps'
    # model = PhiForCausalLM.from_pretrained('microsoft/phi-1_5', attn_implementation='eager').to(device).eval()
    model = LlamaForCausalLM.from_pretrained('HuggingFaceTB/SmolLM-360M', attn_implementation='eager').to(device).eval()

    prompt = "you are a math teacher. output only relevant information to the answer. what is the answer to the equation 1 + 1 / 2 ? a) 1.5 b) 2 c) 2.5 d) 3"
    input_ids = tokenizer(prompt, return_tensors="pt", padding="max_length", max_length=512, truncation=True
                          ).input_ids

    # print("pre TOVA")
    # evaluate_wikitext(model)

    # use TOVA
    enable_tova_caching(model)
    multi_state_size = 1024
    cache = TOVACache(multi_state_size)
    # #
    output = model.generate(input_ids.to(device), past_key_values=cache, do_sample=True, max_length=1024)
    print(len(output))
    print(tokenizer.decode(output[0], skip_special_tokens=True))

    # print("post TOVA")
    # evaluate_wikitext(model, past_key_values=cache, use_cache=True, pass_attention_mask=False)



    # outputs = model.forward(input_ids, past_key_values=cache, use_cache=True)
