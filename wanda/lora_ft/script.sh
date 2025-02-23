# set wandb project name to WANDA
export WANDB_PROJECT="WANDA"
#CUDA_VISIBLE_DEVICES=0
python lora_ft/finetune_lm.py \
    --model_name_or_path "$1" \
    --config_name "microsoft/phi-1_5" \
    --dataset_name c4 \
    --num_train_epochs 1 \
    --block_size 2048 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --max_train_samples 120000 \
    --max_eval_samples 128 \
    --learning_rate 1e-4 \
    --overwrite_output_dir \
    --output_dir "$2" \
    --push_to_hub

#CUDA_VISIBLE_DEVICES=0 python evaluate_ppl.py \
#    --model "$1" \
#    --lora_weights "$2"