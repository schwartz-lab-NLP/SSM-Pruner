#!/bin/bash

#model="state-spaces/mamba2-2.7b"
# model="schwartz-lab/Smol2-Mamba-1.9B"
# model="schwartz-lab/SmolLM2-1.7B_lm_head"
#model="goombalab/Phi-Mamba"
#model="JunxiongWang/Llama3.2-Mamba2-3B-dpo"
# model="HuggingFaceTB/SmolLM2-1.7B"
model="LLAMBA_1B"

cuda_device=0

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device
export PYTHONPATH=".:$PYTHONPATH"
export PYTHONPATH="./phi_mamba:$PYTHONPATH"
export PYTHONPATH="./MambaInLlama:$PYTHONPATH"
export PYTHONPATH="./edge/cartesia-pytorch:$PYTHONPATH"

# Define function to run python command
run_python_command () {
    python FLAP/main.py \
    --model $model \
    --prune_method $1 \
    --pruning_ratio $2 \
    --remove_heads $3 \
    --metrics $4 \
    --structure $5 \
    --nsamples 1024 \
    --skip_mlp \
    --eval \
    --save_model "llm_weights/${1}_p${2}_${4}_${5}_LLAMBA1B" \
    --is_lm_head \
    --is_mamba \
    # --is_orig_smol
    
    
    # --is_orig_smol 
    # --is_mamba_in_llama  # uncomment for JunxiongWang/Llama3.2-Mamba2-3B-dpo
#    --skip_attn \
#    --unstr
#    --is_phi
}

echo "Running with flap pruning method"
run_python_command "flap" 0.25 -1 "WIFV" "AL-AM"
echo "-------------------------------------------------------------------------------------------------"
 run_python_command "flap" 0.5 -1 "WIFV" "AL-AM"

