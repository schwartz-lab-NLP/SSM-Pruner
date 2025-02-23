# Instructions to Replicate Mamba-Zephyr

As described in our technical report, training this model proceeds in three steps:

1. Generate pseudo labels from a teacher model. We provide the generated pseudo labels using the seed dataset of the UltraChat and UltraFeedback dataset [here](https://drive.google.com/drive/folders/1KzmFOJ6_pBZOuSYQKDj5jSD5rsvzFy-U?usp=sharing). Please download it and change the `train_datasets_path` in `zephyr_0.25_mamba.yaml` and `zephyr_0.50_mamba.yaml` to the path of your downloaded `zephyr_ultrafeedback` and `zephyr_ultrachat`.

2. Apply SFT to distilled model. We collected the SFT dataset from multiple sources and preprocessed those datasets using our style. The SFT dataset can be found [here](https://huggingface.co/datasets/JunxiongWang/sftdataset). The result is an SFT model like [`JunxiongWang/mamba_0_5_sft`](https://huggingface.co/JunxiongWang/mamba_0_5_sft).

3. Align the SFT model to AI feedback via DPO on a preprocessed version of the UltraFeedback dataset ([link](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized)). 

Following the Zephyr paper, we tested two hyperparameters:

- 1 epoch with `beta=0.01`, resulting in a DPO model [here](https://huggingface.co/JunxiongWang/mamba_0_5_dpo_ep1).
- 3 epochs with `beta=0.1`, resulting in a DPO model [here](https://huggingface.co/JunxiongWang/mamba_0_5_dpo_ep3).

Here are detailed commands to reproduce those models. Make sure you are in the root folder of the project.

## Hybrid Mamba (50% attention / 16 attention layer)

### Layerwise Distillation phrase:

We start with [HuggingFaceH4/zephyr-7b-beta](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta). First, we replace 25% of the attention layers with Mamba, and then replace another 25% of the attention layers with Mamba by running the following command. 

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file multi_gpu.yaml train_mamba/train_hybrid.py mamba_zephyr/zephyr_0.25_mamba.yaml

ACCELERATE_LOG_LEVEL=info accelerate launch --config_file multi_gpu.yaml train_mamba/train_hybrid.py mamba_zephyr/zephyr_0.50_mamba.yaml
```

This should rougly takes 10 hours in 8x80G A100.

Now, we have a distilled hybrid mamba model with 50% attention and 50% mamba. We will then want to align it with human feedback.

This model is available [here](https://huggingface.co/JunxiongWang/zephyr_0.50_mamba_progressive).

### End-to-end Training phrase

We explore two ways for this,

Approach 1: SFT using CE loss of GPT-4 synthetic data

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mamba/train_sft.py mamba_zephyr/zephyr_0.50_mamba_sft.yaml
```

This should rougly takes 3 days in 8x80G A100. This model is available [here](https://huggingface.co/JunxiongWang/mamba_0_5_sft).

Approach 2: SFT using KL loss of a larger teacher model, for example `Llama-70B-instruct`.

Please check `train_mamba/train_distill.py` and the Mamba-Llama-3.1 for details. It should have better results comapred with SFT using CE loss of GPT-4 synthetic data. This model is available [here](https://huggingface.co/JunxiongWang/mamba_0_5_distill).

If you don't do Layerwise Distillation phrase, you should set `with_distill` to False and it will initialize using attention linear layers. If you already do Layerwise Distillation phrase, you should set `with_distill` to True, and it loads the model trained after the first phrase.

### DPO

Zephyr provides two hyperparameters. You can choose one config from those two.

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mamba/train_dpo.py mamba_zephyr/zephyr_0.50_mamba_dpo_ep1.yaml
```

This model is available [here](https://huggingface.co/JunxiongWang/mamba_0_5_dpo_ep1).

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mamba/train_dpo.py mamba_zephyr/zephyr_0.50_mamba_dpo_ep3.yaml
```

This model is available [here](https://huggingface.co/JunxiongWang/mamba_0_5_dpo_ep3).

This should rougly takes few hours in 8x80G A100.

## Hybrid Mamba (25% attention / 8 attention layer)

We use the distilled SFT model from 50% attention to initialize this model.

### End-to-end Training phrase

We explore two ways for this,

Approach 1: SFT using CE loss of GPT-4 synthetic data

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mamba/train_sft.py mamba_zephyr/zephyr_0.75_mamba_sft.yaml
```

This model is available [here](https://huggingface.co/JunxiongWang/mamba_0_75_sft).

Approach 2: SFT using KL loss of a larger teacher model, for example `Llama-70B-instruct`.

Please check `train_mamba/train_distill.py` and the Mamba-Llama-3.1 for details. It should have better results comapred with SFT using CE loss of GPT-4 synthetic data.

If you don't do Layerwise Distillation phrase, you should set `with_distill` to False and it will initialize using attention linear layers. If you already do Layerwise Distillation phrase, you should set `with_distill` to True, and it loads the model trained after the first phrase.

### DPO

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mamba/train_dpo.py mamba_zephyr/zephyr_0.75_mamba_dpo_ep1.yaml
```

This model is available [here](https://huggingface.co/JunxiongWang/mamba_0_75_dpo_ep1).

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mamba/train_dpo.py mamba_zephyr/zephyr_0.75_mamba_dpo_ep3.yaml
```

This model is available [here](https://huggingface.co/JunxiongWang/mamba_0_75_dpo_ep3).

## Hybrid Mamba (12.5% attention / 4 attention layer)

We use the distilled SFT model from 25% attention to initialize this model.

### End-to-end Training phrase

We explore two ways for this,

Approach 1: SFT using CE loss of GPT-4 synthetic data

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mamba/train_sft.py mamba_zephyr/zephyr_0.875_mamba_sft.yaml
```

This model is available [here](https://huggingface.co/JunxiongWang/mamba_0_875_sft).

Approach 2: SFT using KL loss of a larger teacher model, for example `Llama-70B-instruct`.

Please check `train_mamba/train_distill.py` and the Mamba-Llama-3.1 for details. It should have better results comapred with SFT using CE loss of GPT-4 synthetic data.

If you don't do Layerwise Distillation phrase, you should set `with_distill` to False and it will initialize using attention linear layers. If you already do Layerwise Distillation phrase, you should set `with_distill` to True, and it loads the model trained after the first phrase.

### DPO

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mamba/train_dpo.py mamba_zephyr/zephyr_0.875_mamba_dpo_ep1.yaml
```

This model is available [here](https://huggingface.co/JunxiongWang/mamba_0_875_dpo_ep1).

```
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file deepspeed_zero3.yaml train_mamba/train_dpo.py mamba_zephyr/zephyr_0.875_mamba_dpo_ep3.yaml
```

This model is available [here](https://huggingface.co/JunxiongWang/mamba_0_875_dpo_ep3).

## Evaluation

Please follow the instructions [here](https://github.com/jxiw/MambaInLlama/tree/main/benchmark)
