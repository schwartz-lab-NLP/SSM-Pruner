

import argparse
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from original_mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

from phi_mamba.modules.lm_head import LMHeadModel

import weightwatcher as ww


def run():
    model = MambaLMHeadModel.from_pretrained('state-spaces/mamba2-2.7b',device='cuda',dtype=torch.bfloat16)
    watcher = ww.WeightWatcher(model=model)
    details = watcher.analyze()
    summary = watcher.get_summary(details)

