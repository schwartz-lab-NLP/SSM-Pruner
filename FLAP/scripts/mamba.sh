#!/bin/bash

#model="state-spaces/mamba2-2.7b"
model="tGhattas/Smol2-Mamba-1.9B"
#model="goombalab/Phi-Mamba"
#model="JunxiongWang/Llama3.2-Mamba2-3B-dpo"

cuda_device=0

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device
export PYTHONPATH=".:$PYTHONPATH"
export PYTHONPATH="./phi_mamba:$PYTHONPATH"
export PYTHONPATH="./MambaInLlama:$PYTHONPATH"
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
    --is_mamba \
    --skip_mlp \
    --save_model "llm_weights/${1}_p${2}_${4}_${5}_MAMBA2" \
    --eval \
#    --is_lm_head \ # uncomment for tGhattas/Smol2-Mamba-1.9B & goombalab/Phi-Mamba
#    --is_mamba_in_llama \ # uncomment for JunxiongWang/Llama3.2-Mamba2-3B-dpo
#    --skip_attn \
#    --unstr
#    --is_phi
}

echo "Running with flap pruning method"
run_python_command "flap" 0.25 -1 "WIFV" "AL-AM"
echo "-------------------------------------------------------------------------------------------------"
 run_python_command "flap" 0.5 -1 "WIFV" "AL-AM"

