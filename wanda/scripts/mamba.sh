#!/bin/bash

# Set common variables
#model="goombalab/Phi-Mamba"
#model="state-spaces/mamba2-2.7b"
# model="schwartz-lab/Smol2-Mamba-1.9B"
model="HuggingFaceTB/SmolLM2-1.7B"
#model="JunxiongWang/Llama3.2-Mamba2-3B-dpo"

cuda_device=0

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device
export PYTHONPATH=".:$PYTHONPATH"
export PYTHONPATH="./phi_mamba:$PYTHONPATH"
export PYTHONPATH="./MambaInLlama:$PYTHONPATH"
export PYTHONPATH="./wanda:$PYTHONPATH"

# Define function to run python command
run_python_command () {
    python wanda/main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $5 \
    --sparsity_type $2 \
    --save $3 \
    --save_model $4 \
    --is_mamba_in_llama 
    # --is_mamba \
    # --is_lm_head \
#    --eval_zero_shot \
#    --lm_eval_name auto-lm-head

#    --s_prune
#    --split_mamba \
#    --pre_eval
}

# phi with wanda pruning method
#echo "Running with wanda pruning method"
#run_python_command "wanda" "unstructured" "out/mamba/unstructured/wanda/phi_0" "out/mamba/unstructured/wanda/phi_0/model/" 0
#echo "--------------------------------------------------------------------------------------"
#echo "--------------------------------------------------------------------------------------"
#echo "--------------------------------------------------------------------------------------"
#run_python_command "wanda" "unstructured" "out/mamba/unstructured/wanda/phi_02" "out/mamba/unstructured/wanda/phi_02/model/" 0.2
#echo "--------------------------------------------------------------------------------------"
#echo "--------------------------------------------------------------------------------------"
#echo "--------------------------------------------------------------------------------------"
#run_python_command "wanda" "unstructured" "out/mamba/unstructured/wanda/phi_03" "out/mamba/unstructured/wanda/phi_03/model/" 0.3
#echo "--------------------------------------------------------------------------------------"
#echo "--------------------------------------------------------------------------------------"
#echo "--------------------------------------------------------------------------------------"
#run_python_command "wanda" "unstructured" "out/mamba/unstructured/wanda/phi_05" "out/mamba/unstructured/wanda/phi_05/model/" 0.5

# run_python_command "wanda" "unstructured" "out/mamba/unstructured/wanda/SMOL19_025" "out/mamba/unstructured/wanda/SMOL19_05/model/" 0.5

# run_python_command "wanda" "unstructured" "out/mamba/unstructured/wanda/SMOL17_025" "out/mamba/unstructured/wanda/SMOL17_025/model/" 0.25
run_python_command "wanda" "unstructured" "out/mamba/unstructured/wanda/SMOL17_05" "out/mamba/unstructured/wanda/SMOL17_05/model/" 0.5



#run_python_command "wanda" "unstructured" "out/mamba/unstructured/wanda/mamba2_0" "out/mamba/unstructured/wanda/mamba2_0/model/" 0
#echo "--------------------------------------------------------------------------------------"
#echo "--------------------------------------------------------------------------------------"
#echo "--------------------------------------------------------------------------------------"
#run_python_command "wanda" "unstructured" "out/mamba/unstructured/wanda/mamba2_02" "out/mamba/unstructured/wanda/mamba2_02/model/" 0.2
#echo "--------------------------------------------------------------------------------------"
#echo "--------------------------------------------------------------------------------------"
#echo "--------------------------------------------------------------------------------------"
#run_python_command "wanda" "unstructured" "out/mamba/unstructured/wanda/mamba2_03" "out/mamba/unstructured/wanda/mamba2_03/model/" 0.3
#echo "--------------------------------------------------------------------------------------"
#echo "--------------------------------------------------------------------------------------"
#echo "--------------------------------------------------------------------------------------"
#run_python_command "wanda" "unstructured" "out/mamba/unstructured/wanda/mamba2_05" "out/mamba/unstructured/wanda/mamba2_05/model/" 0.5


#run_python_command "wanda" "unstructured" "out/mamba/unstructured/wanda/mLlama_0" "out/mamba/unstructured/wanda/mLlama_0/model/" 0
#echo "--------------------------------------------------------------------------------------"
#echo "--------------------------------------------------------------------------------------"
#echo "--------------------------------------------------------------------------------------"
#run_python_command "wanda" "unstructured" "out/mamba/unstructured/wanda/mLlama_02" "out/mamba/unstructured/wanda/mLlama_02/model/" 0.25
#echo "--------------------------------------------------------------------------------------"
#echo "--------------------------------------------------------------------------------------"
#echo "--------------------------------------------------------------------------------------"
#run_python_command "wanda" "unstructured" "out/mamba/unstructured/wanda/mLlama_03" "out/mamba/unstructured/wanda/mLlama_03/model/" 0.3
#echo "--------------------------------------------------------------------------------------"
#echo "--------------------------------------------------------------------------------------"
#echo "--------------------------------------------------------------------------------------"
#run_python_command "wanda" "unstructured" "out/mamba/unstructured/wanda/mLlama_05" "out/mamba/unstructured/wanda/mLlama_05/model/" 0.7




echo "Finished wanda pruning method"