import argparse
import os 
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from MambaInLlama.mamba2.hybrid_wrapper import MambaTransformerHybridModelWrapper
from modules.modeling_phi_adjusted2 import PhiForCausalLM
from original_mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from phi_mamba.modules.lm_head import LMHeadModel


from lib.prune import prune_wanda_sp, prune_flap, prune_magnitude_sp, check_sparsity
from utils.ppl import evaluate_wikitext, evaluate_with_lm_eval_harness


# from lib.eval import eval_ppl

# print('torch', version('torch'))
# print('transformers', version('transformers'))
# print('accelerate', version('accelerate'))
# print('# of gpus: ', torch.cuda.device_count())

def get_llm(model_name, is_mamba=False, is_lm_head=False, split_mamba=False, is_phi=False, is_mamba_in_llama=False, skip_mlp=False):
    if is_mamba:
        if is_lm_head:
            model = LMHeadModel.from_pretrained(
                model_name,
                attn_type="flash_attention_2" if torch.is_autocast_enabled() else "eager",
                strict=True,
            ).to(torch.bfloat16)
        elif is_mamba_in_llama:
            model = MambaTransformerHybridModelWrapper.from_pretrained(model_name, torch_dtype=torch.bfloat16)
        else:
            model = MambaLMHeadModel.from_pretrained(model_name,device='cuda',dtype=torch.bfloat16)
            if split_mamba:
                model.split_in_proj()
    else:
        if is_phi:
            model = PhiForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto"
            )

    backbone = model.model if hasattr(model, "model") else model.backbone
    layers = backbone.layers if hasattr(backbone, "layers") else backbone.model.layers
    for i in range(len(layers)):
        if not is_mamba:
            out_projection  = model.model.layers[i].self_attn.o_proj if hasattr(model.model.layers[i].self_attn, 'o_proj') else model.model.layers[i].self_attn.dense
            out_projection.bias = torch.nn.Parameter(torch.zeros(out_projection.weight.shape[0], device='cpu', dtype=torch.bfloat16))  # 或 'cuda'
            last_mlp = model.model.layers[i].mlp.down_proj if hasattr(model.model.layers[i].mlp, 'down_proj') else model.model.layers[i].mlp.fc2
            #last_mlp.bias
            last_mlp.bias = torch.nn.Parameter(torch.zeros(last_mlp.weight.shape[0], device='cpu', dtype=torch.bfloat16))  # 或 'cuda'
            torch.nn.init.zeros_(out_projection.bias)
            torch.nn.init.zeros_(last_mlp.bias)
        else:
            if hasattr(layers[i], 'mixer') and layers[i].mixer.out_proj.bias is  None:
                bias_param = torch.zeros(layers[i].mixer.out_proj.out_features, device='cpu', dtype=torch.bfloat16)
                layers[i].mixer.out_proj.bias = torch.nn.Parameter(bias_param)  # 或 'cuda'
                torch.nn.init.zeros_(layers[i].mixer.out_proj.bias)
            if (is_mamba_in_llama or is_lm_head) and not skip_mlp:
                if hasattr(layers[i].mlp, 'fc2') and layers[i].mlp.fc2.bias is None:
                    torch.zeros(layers[i].mlp.fc2.out_features, device='cpu', dtype=torch.bfloat16)
                    layers[i].mlp.fc2.bias = torch.nn.Parameter(bias_param)  # 或 'cuda'
                    torch.nn.init.zeros_(layers[i].mlp.fc2.bias)
                elif hasattr(layers[i].mlp, 'down_proj') and layers[i].mlp.down_proj.bias is None:
                    torch.zeros(layers[i].mlp.down_proj.out_features, device='cpu', dtype=torch.bfloat16)
                    layers[i].mlp.down_proj.bias = torch.nn.Parameter(bias_param)
                    torch.nn.init.zeros_(layers[i].mlp.down_proj.bias)

    if is_mamba_in_llama:
        model.model.seqlen = 128
        model.seqlen = 128
    else:
        model.seqlen = 128
    # model.seqlen = model.config.max_position_embeddings if hasattr(model.config, "max_position_embeddings") else (
    #     model.config.MixerModel.input.d_model if is_lm_head else model.config.d_model)
    model.config.hidden_size = model.config.hidden_size if hasattr(model.config, "hidden_size") else (
        model.config.MixerModel.input.d_model if is_lm_head else model.config.d_model)
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, help=' model')    # Huggingface model name
    parser.add_argument("--is_mamba", action="store_true", default=False)
    parser.add_argument("--is_lm_head", action="store_true", default=False)
    parser.add_argument("--is_mamba_in_llama", action="store_true", default=False)
    parser.add_argument("--is_phi", action="store_true", default=False)
    parser.add_argument('--is_orig_smol', action="store_true", default=False)
    parser.add_argument("--split_mamba", action="store_true", default=False)
    parser.add_argument('--seed', type=int, default=0, help='Seed for sampling the calibration data.')
    parser.add_argument('--nsamples', type=int, default=2048, help='Number of calibration samples.')
    parser.add_argument('--pruning_ratio', type=float, default=0, help='Pruning ratio.')
    parser.add_argument('--remove_heads', type=int, default=8, help='Remove num_heads')
    parser.add_argument("--metrics", type=str, default="WIFV", choices=["IFV", "WIFV", "WIFN", 'N/A'])
    parser.add_argument("--structure", type=str, default="AL-AM", choices=["UL-UM", "UL-MM", "AL-MM", "AL-AM", 'N/A'])
    parser.add_argument("--prune_method", type=str, default="flap", choices=["flap", "wanda_sp", "mag_sp"])
    parser.add_argument("--cache_dir", default="llm_weights", type=str)
    parser.add_argument('--unstr', action="store_true")
    parser.add_argument('--eval', action="store_true")
    parser.add_argument('--save_model', type=str, default=None, help='Path to save the pruned model.')
    parser.add_argument('--skip_mlp', action="store_true", default=False)
    parser.add_argument('--skip_attn', action="store_true", default=False)
    
    args = parser.parse_args()
    
    # Setting seeds for reproducibility
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    # Build the model and tokenizer
    print(f"loading llm model {args.model}")
    model = get_llm(args.model,  args.is_mamba, args.is_lm_head, args.split_mamba, args.is_phi, args.is_mamba_in_llama, args.skip_mlp)
    device = torch.device("cuda:0")
    model.to(device)
    model.eval()
    if args.is_mamba:
        if args.is_lm_head:
            # tokenizer = AutoTokenizer.from_pretrained('microsoft/phi-1_5', use_fast=False)
            # tokenizer_path = 'microsoft/phi-1_5'
            tokenizer = AutoTokenizer.from_pretrained('HuggingFaceTB/SmolLM2-1.7B', use_fast=False)
            tokenizer_path = 'HuggingFaceTB/SmolLM2-1.7B'
        elif args.is_mamba_in_llama:
            tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
            tokenizer_path = args.model
        else:
            tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
            tokenizer_path = 'EleutherAI/gpt-neox-20b'
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)
        tokenizer_path = args.model

    if "30b" in args.model or "65b" in args.model: # for 30b and 65b we use device_map to load onto multiple A6000 GPUs, thus the processing here.
        device = model.hf_device_map["lm_head"]
    print("use device ", device)

    # Prune the model
    before_mixer = sum(p.numel() for n, p in model.named_parameters() if 'mixer' in n) if not args.is_orig_smol else sum(p.numel() for n, p in model.named_parameters() if 'attn' in n)
    print("pruning starts")
    if args.prune_method == "flap":
        if args.metrics == 'N/A':
            raise ValueError("For FLAP pruning, the metrics parameter must be chosen from ['IFV', 'WIFV', 'WIFN']. 'N/A' is not a valid choice.")  
        if args.structure == 'N/A':
            raise ValueError("For FLAP pruning, the compressed model structure parameter must be chosen from ['UL-UM', 'UL-MM', 'AL-MM', 'AL-AM']. 'N/A' is not a valid choice.")  
        prune_flap(args, model, tokenizer, device, retain_heads_min=16 if args.is_mamba_in_llama else 1)
    elif args.prune_method == "wanda_sp":
        prune_wanda_sp(args, model, tokenizer, device)
    elif args.prune_method == "mag_sp":
        prune_magnitude_sp(args, model, tokenizer, device)

    # Check the sparsity of the model

    print("*"*30)
    if args.unstr:
        sparsity_ratio = check_sparsity(model)
        print(f"sparsity sanity check {sparsity_ratio:.4f}")
    print(f"model parameter {sum(p.numel() for p in model.parameters()) / 1000 ** 3:.4f}B")
    print("*"*30)
    # Evaluate the model

    if args.eval:
        ppl = evaluate_wikitext(model, tokenizer_path=tokenizer_path)
        print(f"ppl on wikitext {ppl}")
        ppl = evaluate_with_lm_eval_harness(model, tokenizer_path=tokenizer_path, batch_size=64)
        print(f"ppl on lm_eval_harness {ppl}")
        
    # Save the model
    if args.save_model:
        if not os.path.exists(args.save_model):
            os.makedirs(args.save_model)
        # torch.save(model, f'{args.save_model}/pruned_model.pt')
        if args.is_mamba_in_llama:
            model.save_pretrained(args.save_model)
        elif args.is_lm_head:
            model.save_pretrained_distributed(args.save_model, is_main_process=True, update_config=True)
        else:
            model.save_pretrained(args.save_model)
        model = get_llm(args.save_model, args.is_mamba, args.is_lm_head, args.split_mamba, args.is_phi, args.is_mamba_in_llama, args.skip_mlp)

    print("*" * 30)
    print(f"model parameter post pruning {sum(p.numel() for p in model.parameters()) / 1000 ** 3:.4f}B")
    print("*" * 30)
    after_mixer = sum(p.numel() for n, p in model.named_parameters() if 'mixer' in n) if not args.is_orig_smol else sum(p.numel() for n, p in model.named_parameters() if 'attn' in n)
    print(f"number of mixer parameters after pruning: {after_mixer}, before {before_mixer}")
    print(f"compression ratio mixer: {after_mixer / before_mixer}")
    if args.eval:
        ppl = evaluate_wikitext(model, tokenizer_path=tokenizer_path)
        print(f"ppl on wikitext {ppl}")
        ppl = evaluate_with_lm_eval_harness(model, tokenizer_path=tokenizer_path, batch_size=64)
        print(f"ppl on lm_eval_harness {ppl}")
        # in_proj_sizes =[l.mixer.in_proj.shape for l in model.backbone.layers]
        # out_proj_sizes =[l.mixer.out_proj.shape for l in model.backbone.layers]
        # print(f"input projection sizes {in_proj_sizes}")
        # print(f"output projection sizes {out_proj_sizes}")

if __name__ == '__main__':
    main()