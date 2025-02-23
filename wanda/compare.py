import argparse
import time
from importlib.metadata import version

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from phi_mamba.modules.lm_head import LMHeadModel
from wanda.lib.data import get_c4

print('torch', version('torch'))
print('transformers', version('transformers'))
print('accelerate', version('accelerate'))
print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, is_mamba=False):
    if is_mamba:
        model = LMHeadModel.from_pretrained(
            model_name,
            attn_type="flash_attention_2" if torch.is_autocast_enabled() else "eager",
            local=True,
        ).to(torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    return model

def profile_inference_speed(model, data_loader, tokenizer, runs=10):
    model.eval()
    model = model.cuda()
    model = model.to(torch.float32)
    avgs = []
    with torch.no_grad():
        # Warm-up
        first = True
        for inputs in data_loader:
            # with torch.autocast(device_type='cuda', dtype=torch.float16):
            torch.cuda.synchronize()  # Ensure GPU ops are finished
            start_time = time.time()
            for _ in range(runs):
                model(inputs[0].to('cuda'))
                torch.cuda.synchronize()  # Ensure GPU ops are finished
            end_time = time.time()
            avg_time = (end_time - start_time) / runs
            if first:
                first = False
                continue
            avgs.append(avg_time)
    avg_time = sum(avgs) / len(avgs)
    torch.cuda.empty_cache()
    return avg_time * 1000  # Return time in milliseconds


def compare_models(model_1, model_2):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5")
    #Load subset of C4
    tokenizer.pad_token_id = tokenizer.eos_token_id
    dataloader, _ = get_c4(128, 0, 2048, tokenizer)
    m_time = profile_inference_speed(model_1, dataloader, tokenizer)
    print(f"Model model_1 inference time: {m_time:.2f} ms")
    del model_1
    torch.cuda.empty_cache()
    dataloader, _ = get_c4(128, 0, 2048, tokenizer)
    pm_time = profile_inference_speed(model_2, dataloader, tokenizer)

    print(f"Model model_2 inference time: {pm_time:.2f} ms")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_1', type=str)
    parser.add_argument('--model_2', type=str)
    parser.add_argument("--is_mamba", action="store_true", default=False)
    args = parser.parse_args()

    model_1 = get_llm(args.model_1, args.is_mamba).cuda()
    model_2 = get_llm(args.model_2, args.is_mamba).cuda()

    compare_models(model_1, model_2)






if __name__ == '__main__':
    main()