#!/bin/bash

# Set common variables
model="microsoft/phi-1_5"
sparsity_ratio=0.5
cuda_device=0

# Set CUDA device visibility
export CUDA_VISIBLE_DEVICES=$cuda_device

# Define function to run python command
run_python_command () {
    python main.py \
    --model $model \
    --prune_method $1 \
    --sparsity_ratio $sparsity_ratio \
    --sparsity_type $2 \
    --save $3 \
    --save_model $4
#    --pre_eval
}

# phi with wanda pruning method
#echo "Running with wanda pruning method"
#run_python_command "wanda" "unstructured" "out/phi/unstructured/wanda/" "out/phi/unstructured/wanda/model/"
#run_python_command "wanda" "2:4" "out/phi/2-4/wanda/" "out/phi/2-4/wanda/model/"
#run_python_command "wanda" "4:8" "out/phi/4-8/wanda/" "out/phi/4-8/wanda/model/"
#echo "Finished wanda pruning method"


echo "Running with sparsegpt pruning method"
#run_python_command "sparsegpt" "unstructured" "out/llama_7b/unstructured/sparsegpt/"
run_python_command "sparsegpt" "unstructured" "out/phi/unstructured/sparsegpt/" "out/phi/unstructured/sparsegpt/model/"
#run_python_command "sparsegpt" "2:4" "out/phi/2-4/sparsegpt/" "out/phi/2-4/sparsegpt/model/"
#run_python_command "sparsegpt" "4:8" "out/llama_7b/4-8/sparsegpt/"
echo "Finished sparsegpt pruning method"


# 2:4 -  wikitext perplexity 33.31394958496094
