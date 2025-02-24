


## Introduction
  
> **[On Pruning State-Space LLMs]()** [[arXiv]]()   
> *Tamer Ghattas, Michael Hassid and Roy Schwartz*   
> *Hebrew University of Jerusalem*  

This repo include the adaptation of WANDA and FLAP pruning methods to Mamba2 models along with the headdim and dstate pruning methods explained in the paper.
The code is based on the original repos, you'll find pruning methods implementations in the Mamba layer in each one of [discrete_mamba2.py](phi_mamba/modules/mixers/discrete_mamba2.py) , [mixer_seq_simple.py](original_mamba/mamba_ssm/models/mixer_seq_simple.py) and [hybrid_mamba_layer.py](MambaInLlama/mamba2/hybrid_mamba_layer.py) and modified versions of WANDA and FLAP.

## Installation
- `conda create -n ssm-pruner python=3.10`
- `conda activate ssm-pruner`
- `pip install torch==2.4.0+cu121  --extra-index-url https://download.pytorch.org/whl/cu121 --no-cache-dir`
- `pip install datasets==3.0.0`
- `pip install transformers==4.48.1`
- `pip install triton mamba-ssm==2.2.2 flash-attn==2.6.3`: the core Mamba package.

## Quickstart

### WANDA

```bash
wanda/scripts/mamba.sh
```

### FLAP

```bash
FLAP/scripts/mamba.sh
```

### Headdim & State Pruning

```bash
python prune_mha.py
```



## Smol2-Mamba-1.9B Making
This model was distilled from [SmolLM2-1.7B](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B) using our implementation of [MOHAWK](https://goombalab.github.io/blog/2024/distillation-part1-mohawk/) in [train.py](phi_mamba/train.py).

## Fine-tuning
For fine-tuning our pruned models with distillation loss we used [finetune.py](phi_mamba/finetune.py).

## Citation

```bibtex

```