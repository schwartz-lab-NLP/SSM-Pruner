#!/bin/bash

# Set common variables
#model="goombalab/Phi-Mamba"
model="state-spaces/mamba2-2.7b"
#model="./phi_mamba/tGhattas/Smol2-Mamba-1.9B" #19

#model="JunxiongWang/Llama3.2-Mamba2-3B-dpo"
#model="JunxiongWang/Mamba2InLlama_1"
#sparsity_ratio=0.5 #0.3, 0.5
cuda_device=0

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device
export PYTHONPATH=".:$PYTHONPATH"
export PYTHONPATH="./phi_mamba:$PYTHONPATH"
export PYTHONPATH="./MambaInLlama:$PYTHONPATH"

# Define function to run python command
run_python_command () {
    python main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $5 \
    --sparsity_type $2 \
    --save $3 \
    --is_mamba \
#    --is_lm_head \
#    --save_model $4 \
#    --is_mamba_in_llama \
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
#run_python_command "wanda" "unstructured" "out/mamba/unstructured/wanda/SMOL19_025" "out/mamba/unstructured/wanda/SMOL19_025/model/" 0.25
#run_python_command "wanda" "unstructured" "out/mamba/unstructured/wanda/SMOL19_08" "out/mamba/unstructured/wanda/SMOL19_08/model/" 0.8



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
run_python_command "wanda" "unstructured" "out/mamba/unstructured/wanda/mamba2_05" "out/mamba/unstructured/wanda/mamba2_05/model/" 0.5


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




#run_python_command "wanda" "2:4" "out/mamba/2-4/wanda/" "out/mamba/2-4/wanda/model/"
#run_python_command "wanda" "4:8" "out/mamba/4-8/wanda/" "out/mamba/4-8/wanda/model/"
echo "Finished wanda pruning method"
# wikitext perplexity 29.44671630859375
# wikitext perplexity 45.57360076904297
# wikitext perplexity 35.61686325073242


#echo "Running with sparsegpt pruning method"
#run_python_command "sparsegpt" "unstructured" "out/mamba/unstructured/sparsegpt/" "out/mamba/unstructured/sparsegpt/model/"
#run_python_command "sparsegpt" "2:4" "out/mamba/2-4/sparsegpt/" "out/mamba/2-4/sparsegpt/model/"
#run_python_command "sparsegpt" "4:8" "out/mamba/4-8/sparsegpt/" "out/mamba/4-8/sparsegpt/model/"
#echo "Finished sparsegpt pruning method"


# 2:4 -  wikitext perplexity 34.88787078857422