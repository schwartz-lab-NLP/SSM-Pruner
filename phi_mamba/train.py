# Copyright (c) Tamer Ghattas 2024
import argparse
import logging
import os
import random
import shutil
import time
from argparse import Namespace
from math import isnan
from typing import Optional

from trl import DPOTrainer, DPOConfig
import torch
from accelerate import Accelerator, ProfileKwargs
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from torch import nn
from torch.nn.functional import softmax, log_softmax
from tqdm import tqdm
from transformers import get_wsd_schedule, get_constant_schedule, AutoTokenizer

from modules.lm_head import LMHeadModel
from modules.hf_lm_head import CustomMambaLMHeadModel
from modules.mixers.discrete_mamba2 import Mixer
# import deepspeed
from modules.modeling_phi_adjusted import PhiForCausalLM
from modules.modeling_llama import LlamaForCausalLM
from utils.config import Config
from utils.convert_to_sparse import check_sparsity, print_mem_footprint
from utils.criterions import matrix_orientation_criterion_light, hidden_states_alignment_criterion, \
    hidden_states_alignment_criterion_light, pad_weight, matrix_orientation_criterion_MHA_importance
from utils.prepare_c4 import get_data_loaders
from utils.visualize import run_stage1_app
from prune_mha import prune_mha_heads


logger = get_logger(__name__)

debug_count = 0


class Trainer:
    '''
    Trainer class for training the student model using the teacher model
    '''

    def __init__(self, parsed_args: Namespace, accelerator_: Accelerator):
        self.output_dir = parsed_args.output_dir
        self.parsed_args = parsed_args
        self.random_suffix = f"{str(random.randint(0, 100))}_{time.strftime('%Y%m%d-%H%M%S')}"
        # sanitize the string to be compatible to save
        self.random_suffix = self.random_suffix.replace(":", "-")
        self.accelerator = accelerator_
        self.rank = self.accelerator.process_index
        self.world_size = self.accelerator.num_processes  # Total number of processes

        self.num_epochs = parsed_args.num_epochs
        self.dtype = torch.bfloat16

        if self.accelerator.is_main_process:
            experiment_config = {
                "args": vars(parsed_args),
                "num_epochs": self.num_epochs,
                "run_name": f"mamba_phi",
                "grad_limit": parsed_args.grad_limit
            }
            self.accelerator.init_trackers("mamba_phi", experiment_config)
        self.accelerator.wait_for_everyone()
        if not parsed_args.test_mode:
            # prepare c4 dataloaders
            self.get_train_dataloaders = lambda epoch: get_data_loaders(parsed_args.batch, parsed_args.length,
                                           (parsed_args.dataset_path, parsed_args.dataset_name),
                                           self.world_size, self.rank, split="train", tokenizer_path=parsed_args.tokenizer_path, seed=parsed_args.seed, epoch=epoch, return_iter_datasets=parsed_args.train_dpo)
            get_eval_dataloaders = lambda: get_data_loaders(parsed_args.batch, parsed_args.length,
                                                            (parsed_args.dataset_path, parsed_args.dataset_name),
                                                            self.world_size, self.rank, split=parsed_args.validation, tokenizer_path=parsed_args.tokenizer_path,
                                                            seed=parsed_args.seed, return_iter_datasets=parsed_args.train_dpo)
            if not parsed_args.train_dpo:
                dataloaders = self.get_train_dataloaders(0)
            else:
                dataloaders, data_collator = self.get_train_dataloaders(0)

        else:
            data_collator = None
            dataloaders = tuple([None, None, None])
            get_eval_dataloaders = lambda: None

        # init teacher
        if parsed_args.teacher_model_path.endswith(".bin"):
            self.teacher = torch.load(args.teacher_model_path, map_location='cpu')
            if parsed_args.teacher_pruned_arch_path is not None:
                pruned_arch = torch.load(parsed_args.teacher_pruned_arch_path, map_location='cpu')['model'].bfloat16()
                pruned_arch.load_state_dict(self.teacher)
                self.teacher = pruned_arch
            else:
                self.teacher = self.teacher['model'].bfloat16() if isinstance(self.teacher, dict) and 'model' in self.teacher else self.teacher.bfloat16()
        else:
            if parsed_args.teacher_config_path is not None or parsed_args.lm_head_teacher:
                kwargs = {} if parsed_args.teacher_config_path is None else {"config_path": parsed_args.teacher_config_path}
                self.teacher = LMHeadModel.from_pretrained(parsed_args.teacher_model_path, **kwargs).bfloat16()
            else:
                if parsed_args.teacher_type == "phi":
                    self.teacher = (
                        PhiForCausalLM.from_pretrained(parsed_args.teacher_model_path,
                                                       attn_implementation="eager")
                        .bfloat16()
                    )
                elif parsed_args.teacher_type == "llama":
                    self.teacher = (
                        LlamaForCausalLM.from_pretrained(parsed_args.teacher_model_path,
                                                         attn_implementation="eager")
                        .bfloat16()
                    )
        # Freeze all parameters in teacher model by setting requires_grad to False
        self.teacher.requires_grad_(False)
        self.teacher.eval()

        self.accelerator.print("Teacher model loaded")
        self.accelerator.print(self.teacher)

        if parsed_args.test_mode:

            dl = get_data_loaders(parsed_args.batch, parsed_args.length,
                                  (parsed_args.dataset_path, parsed_args.dataset_name),
                                  self.world_size, self.rank, split="validation")[-1]
            self.student = LMHeadModel.from_pretrained(pretrained_model_name_or_path=parsed_args.model_path,)
            optim = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.student.parameters()),
                betas=(0.9, 0.95),
                lr=parsed_args.stage_1_lr,
                weight_decay=parsed_args.weight_decay,
                eps=parsed_args.eps
            )
            self.student, dl, optim = self.accelerator.prepare(self.student, dl, optim)
            self.student.eval()
            self.teacher.to(self.accelerator.device)
            data = next(iter(dl))
            with torch.no_grad():
                teacher_result = self.teacher(input_ids=data['input_ids'],
                                              attention_mask=data['attention_mask'],
                                              output_hidden_states=True,
                                              output_attention_results=True,
                                              output_attentions=True,
                                              use_cache=False,
                                              )
                attns1, attns2 = [], []
                for layer_idx, student_layer in enumerate(self.student.backbone.layers):
                    student_input = teacher_result.all_hidden_states[layer_idx]
                    student_result = student_layer(hidden_states=student_input,
                                                   run_mlp_component=False,
                                                   return_mixer_matrix=True,
                                                   attention_mask=data['attention_mask'])
                    student_mixer_matrices = student_result['transfer_matrix']
                    attn_matrices = teacher_result.all_attn_matrices[layer_idx] if hasattr(teacher_result,
                                                                                           "all_attn_matrices") else teacher_result.all_transfer_matrices[layer_idx]
                    attns1.append(student_mixer_matrices)
                    attns2.append(attn_matrices)
                    print(f"Layer {layer_idx}")
                    print(f"student layer {student_layer}")
                    print(f"teacher layer {self.teacher.model.layers[layer_idx]}")
                    print("Student", student_mixer_matrices.size())
                    print("Teacher", attn_matrices.size())
                run_stage1_app(attns1, attns2)

            exit(0)

        self.num_epochs = None
        if parsed_args.grad_limit is None:
            self.num_epochs = parsed_args.num_epochs if parsed_args.num_epochs is not None else 1

        # stage 1 *****************************************************************************************************
        if parsed_args.train_stage_1:
            # init student
            cfg = Config.from_json(parsed_args.config_path)
            if parsed_args.model_path is not None:
                self.student = LMHeadModel.from_pretrained(pretrained_model_name_or_path=parsed_args.model_path,
                                                           local=True).bfloat16()
            else:
                self.student = LMHeadModel(cfg).bfloat16()
                # fill weights from teacher
                self._init_from_teacher(use_clean_teacher=False, padding=parsed_args.init_padding)
                if parsed_args.kqv:
                    self._init_with_kqvo(parsed_args.init_padding)

            self.student_backbone_layers_len = len(self.student.backbone.layers)


            self._freeze_stg_1()

            if parsed_args.grad_limit is None:
                self.matrix_orientation_total_steps = 80_000_000 // parsed_args.batch // parsed_args.length // self.world_size
                if parsed_args.num_epochs is not None:
                    self.matrix_orientation_total_steps *= parsed_args.num_epochs
                self.accelerator.print("Matrix orientation total steps per process: ", self.matrix_orientation_total_steps)
            else:
                self.matrix_orientation_total_steps = parsed_args.grad_limit * parsed_args.stage_1_grad_accumulation_steps * self.world_size

            if parsed_args.prune_mha:
                self.matrix_orientation_criterion = matrix_orientation_criterion_MHA_importance
            elif parsed_args.init_padding:
                self.matrix_orientation_criterion = matrix_orientation_criterion_light
            else:
                self.matrix_orientation_criterion = lambda transfer_matrix, attn_matrix: torch.linalg.matrix_norm(
                    transfer_matrix - attn_matrix, ord="fro").mean()

            self.matrix_orientation_train_loader = dataloaders[0]
            self.matrix_orientation_eval_loader = lambda: self.accelerator.prepare(get_eval_dataloaders()[0])
            self.matrix_orientation_grad_accumulation_steps = parsed_args.stage_1_grad_accumulation_steps

            self.matrix_orientation_optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.student.parameters()),
                betas=(0.9, 0.95),
                lr=parsed_args.stage_1_lr,
                weight_decay=parsed_args.weight_decay,
                eps=parsed_args.eps
            )
            self.stage1_actual_total_steps = self.matrix_orientation_total_steps // self.matrix_orientation_grad_accumulation_steps
            self.accelerator.print(f"stage1_actual_total_steps: {self.stage1_actual_total_steps}")
            self.matrix_orientation_scheduler = get_wsd_schedule(
                self.matrix_orientation_optimizer,
                num_warmup_steps=self.stage1_actual_total_steps // 10,
                num_stable_steps=8*(self.stage1_actual_total_steps // 10),
                num_decay_steps=self.stage1_actual_total_steps // 10,
                min_lr_ratio=0.1
            ) if not parsed_args.no_scheduler else get_constant_schedule(self.matrix_orientation_optimizer)


            if parsed_args.prune_mha:
                assert isinstance(self.teacher, LMHeadModel), "Pruning only supported for LMHeadModel"
                self.mask_head_indexes = prune_mha_heads(0, 24, self.teacher, self.teacher, ratio=0.25)

            # accelerate prepare
            (self.student,
             self.matrix_orientation_optimizer,
             self.matrix_orientation_train_loader,
             self.matrix_orientation_scheduler
             ) = self.accelerator.prepare(
                self.student,
                self.matrix_orientation_optimizer,
                self.matrix_orientation_train_loader,
                self.matrix_orientation_scheduler
            )
            # place the teacher on the current device
            self.teacher.to(self.accelerator.device)

        # stage 2 *****************************************************************************************************
        if parsed_args.train_stage_2:
            # init student
            if parsed_args.model_path is not None:
                self.student = LMHeadModel.from_pretrained(pretrained_model_name_or_path=parsed_args.model_path,
                                                           local=True).bfloat16()
                if parsed_args.reinit_from_teacher:
                    self._init_from_teacher()
            else:
                cfg = Config.from_json(parsed_args.config_path)
                self.student = LMHeadModel(cfg).bfloat16()
                self._init_from_teacher()
                if parsed_args.kqv:
                    self._init_with_kqvo()
            self.student_backbone_layers_len = len(self.student.backbone.layers)
            if parsed_args.grad_limit is None:
                self.hidden_states_alignment_total_steps = 160_000_000 // parsed_args.batch // parsed_args.length // self.world_size
                if parsed_args.num_epochs is not None:
                    self.hidden_states_alignment_total_steps *= parsed_args.num_epochs
            else:
                self.hidden_states_alignment_total_steps = parsed_args.grad_limit * parsed_args.stage_2_grad_accumulation_steps * self.world_size
            self.accelerator.print("Hidden states alignment total steps per process: ",
                                   self.hidden_states_alignment_total_steps)
            self.hidden_states_alignment_train_loader = dataloaders[1]
            self.hidden_states_alignment_eval_loader = lambda: self.accelerator.prepare(get_eval_dataloaders()[1])
            self._freeze_stg_2()
            self.hidden_states_alignment_optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.student.parameters()),
                betas=(0.9, 0.95),
                lr=parsed_args.stage_2_lr,
                weight_decay=parsed_args.weight_decay,
                eps=parsed_args.eps
            )
            self.hidden_states_alignment_grad_accumulation_steps = parsed_args.stage_2_grad_accumulation_steps
            self.stage2_actual_total_steps = self.hidden_states_alignment_total_steps // self.hidden_states_alignment_grad_accumulation_steps
            self.hidden_states_alignment_scheduler_with_warmup = get_wsd_schedule(
                self.hidden_states_alignment_optimizer,
                num_warmup_steps=self.stage2_actual_total_steps // 10,
                num_stable_steps=self.stage2_actual_total_steps // 10 * 8,
                num_decay_steps=self.stage2_actual_total_steps // 10,
            ) if not parsed_args.no_scheduler else get_constant_schedule(self.hidden_states_alignment_optimizer)
            self.hidden_states_alignment_criterion = hidden_states_alignment_criterion_light if parsed_args.init_padding else hidden_states_alignment_criterion

            (self.student,
             self.hidden_states_alignment_optimizer,
             self.hidden_states_alignment_train_loader,
             self.hidden_states_alignment_scheduler_with_warmup) = self.accelerator.prepare(
                self.student,
                self.hidden_states_alignment_optimizer,
                self.hidden_states_alignment_train_loader,
                self.hidden_states_alignment_scheduler_with_warmup,
            )

            if parsed_args.resume:
                self.accelerator.load_state(parsed_args.model_path)
                self.hidden_states_alignment_train_loader = self.accelerator.skip_first_batches(
                    self.hidden_states_alignment_train_loader,
                    parsed_args.skip)
                if parsed_args.new_lr:
                    self.distillation_scheduler = get_constant_schedule(self.hidden_states_alignment_optimizer)
            # place the tracher on the current device
            self.teacher.to(self.accelerator.device)

        # stage 3 *****************************************************************************************************
        if parsed_args.train_stage_3:
            if parsed_args.sanity_check:
                _path = parsed_args.model_path if parsed_args.model_path is not None else "microsoft/phi-1_5"
                if _path.endswith(".bin"):
                    self.student = torch.load(_path, map_location='cpu')
                    self.student = self.student['model'].bfloat16() if isinstance(self.student, dict) else self.student.bfloat16()
                else:
                    self.student = PhiForCausalLM.from_pretrained(_path,
                                                                  attn_implementation="eager").bfloat16()  # sanity check
            else:
                if parsed_args.model_path is not None:
                    if parsed_args.model_path.endswith(".bin"):
                        self.student = torch.load(parsed_args.model_path, map_location='cpu')
                        self.student = self.student['model'].bfloat16() if isinstance(self.student, dict) else self.student.bfloat16()
                    else:
                        self.student = LMHeadModel.from_pretrained(pretrained_model_name_or_path=parsed_args.model_path,
                                                                   local=True).bfloat16()
                else:
                    cfg = Config.from_json(parsed_args.config_path)
                    self.student = LMHeadModel(cfg).bfloat16()
                    self._init_from_teacher()
                    self._init_with_kqvo()
                if not parsed_args.no_transfer_stg3:
                    self._transfer_weights_stg3()
                self._set_mixer_conv(True)

            self._freeze_stg_3()
            if hasattr(self.student, "backbone"):
                self.student_backbone_layers_len = len(self.student.backbone.layers)
            else:
                self.student_backbone_layers_len = len(self.student.model.layers)
            self.distillation_grad_accumulation_steps = parsed_args.stage_3_grad_accumulation_steps
            if parsed_args.grad_limit is None:
                self.distillation_total_steps = 2_700_000_000 // parsed_args.batch // parsed_args.length // self.world_size
                self.distillation_total_steps *= self.num_epochs
            else:
                self.distillation_total_steps = parsed_args.grad_limit * self.distillation_grad_accumulation_steps * self.world_size
            self.distillation_train_loader = dataloaders[2]
            self.distillation_get_eval_loader = lambda: self.accelerator.prepare(get_eval_dataloaders()[2])

            self.distillation_optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, self.student.parameters()),
                betas=(0.9, 0.95),
                lr=parsed_args.stage_3_lr,
                weight_decay=parsed_args.weight_decay,
                eps=parsed_args.eps
            )
            self.stg3_actual_total_steps = self.distillation_total_steps // self.distillation_grad_accumulation_steps
            self.distillation_scheduler = get_wsd_schedule(
                self.distillation_optimizer,
                num_warmup_steps= self.stg3_actual_total_steps // 10,
                num_stable_steps= int(self.stg3_actual_total_steps // 10 * 8),
                num_decay_steps= self.stg3_actual_total_steps // 10,
            ) if not parsed_args.no_scheduler else get_constant_schedule(self.distillation_optimizer)
            # use CE loss for distillation
            temperature = parsed_args.temp
            self.CE_criterion = lambda logits, labels: nn.CrossEntropyLoss(ignore_index=-100)(logits, labels)

            def soft_ce(student_logits, teacher_logits):
                teacher_probs = softmax(teacher_logits / temperature, dim=-1)
                log_student_probs = log_softmax(student_logits / temperature, dim=-1)
                loss = -(teacher_probs * log_student_probs).sum(dim=-1).mean()
                return loss * (temperature ** 2)

            self.KD_criterion = soft_ce if not parsed_args.kl_div else \
                lambda student_logits, teacher_logits: nn.KLDivLoss(reduction='batchmean')(
                log_softmax(student_logits / temperature, dim=-1),
                softmax(teacher_logits / temperature, dim=-1)
            ) * (temperature ** 2)
            self.KD_loss_weight = parsed_args.kd_loss_weight
            self.kd_smoother = parsed_args.kd_smoother

            if self.accelerator.is_main_process:
                print("Teacher Memory footprint")
                print_mem_footprint(self.teacher)
                print("Student Memory footprint")
                print_mem_footprint(self.student)

            (self.student,
             self.distillation_optimizer,
             self.distillation_train_loader,
             self.distillation_scheduler) = self.accelerator.prepare(
                self.student,
                self.distillation_optimizer,
                self.distillation_train_loader,
                self.distillation_scheduler,
            )
            if parsed_args.resume:
                if parsed_args.new_opt:
                    self.distillation_optimizer = self.accelerator.prepare(
                        torch.optim.AdamW(
                            filter(lambda p: p.requires_grad, self.student.parameters()),
                            betas=(0.9, 0.95),
                            lr=parsed_args.stage_3_lr,
                            weight_decay=parsed_args.weight_decay
                        )
                    )

                self.accelerator.load_state(parsed_args.model_path)

                self.distillation_train_loader = accelerator.skip_first_batches(self.distillation_train_loader,
                                                                                parsed_args.skip)
                if parsed_args.new_lr:
                    self.distillation_scheduler = get_constant_schedule(
                        self.distillation_optimizer,
                    )
                    self.distillation_scheduler = self.accelerator.prepare(self.distillation_scheduler)

            # place the teacher on the current device
            self.teacher.to(self.accelerator.device)

        # DPO *****************************************************************************************************
        if parsed_args.train_dpo:
            self.student = CustomMambaLMHeadModel.from_pretrained(pretrained_model_name_or_path=parsed_args.model_path,
                                                                 is_pure_lm_head=not parsed_args.hf_student).bfloat16()

            self.student_backbone_layers_len = len(self.student.model.layers)
            self._freeze_stg_3()
            self.distillation_grad_accumulation_steps = parsed_args.stage_3_grad_accumulation_steps

            self.tokenizer = AutoTokenizer.from_pretrained(parsed_args.tokenizer_path)
            self.DPO_train_iter_dataset = dataloaders[0]
            self.DPO_eval_ter_dataset = get_eval_dataloaders()[0][0]
            self.DPO_total_steps = 80_000_000 // parsed_args.batch // parsed_args.length
            self.DPO_total_steps *= self.num_epochs




    def save_description(self):
        """ save the parsed arguments and wandb run details to a file """
        if self.accelerator.is_main_process:
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
            if not os.path.exists(os.path.join(self.output_dir, "description.txt")):
                with open(os.path.join(self.output_dir, "description.txt"), "w") as f:
                    # write the parsed arges in a command line fashion --argname value for reproduction
                    for argname, value in vars(self.parsed_args).items():
                        if value is not None and value is not False:
                            if type(value) == bool:
                                f.write(f"--{argname} \ \n")
                            else:
                                f.write(f"--{argname} {value} \ \n")
                    f.write("\n")
                    slurm_job_id = os.getenv('SLURM_JOB_ID')
                    slurm_job_name = os.getenv('SLURM_JOB_NAME')
                    f.write(f"SLURM_JOB_ID: {slurm_job_id}\n")
                    f.write(f"SLURM_JOB_NAME: {slurm_job_name}\n")
                    f.write("\n")
                    f.write(f"world_size: {self.world_size}\n")
                    f.write(f"rank: {self.rank}\n")
                    f.write(f"accelerator: {self.accelerator}\n")
                    f.write("\n")
            else:
                self.accelerator.print("description.txt already exists")


    def _init_with_kqvo(self, padding=False):
        student_module: LMHeadModel = self.student
        teacher_module = self.teacher

        for layer_idx, mamba_phi_block in enumerate(student_module.backbone.layers):
            mixer: Mixer = mamba_phi_block.mixer
            teacher_backbone = teacher_module.model if hasattr(teacher_module, "model") else teacher_module.backbone
            teacher_layer = teacher_backbone.layers[layer_idx]
            teacher_attn = teacher_layer.mixer.self_attn if hasattr(teacher_layer, "mixer") else teacher_layer.self_attn
            # Initialize weights using attention weights
            with torch.no_grad():
                # Load the output projection weights
                if not padding:
                    if self.parsed_args.teacher_type == "llama":
                        mixer.out_proj.load_state_dict(teacher_attn.o_proj.state_dict(), strict=False)
                    else:
                        mixer.out_proj.load_state_dict(teacher_attn.dense.state_dict(), strict=False)
                else:
                    if self.parsed_args.teacher_type == "llama":
                        padded_dense_weight = pad_weight(mixer.out_proj.weight, teacher_attn.o_proj.weight, axis=1)
                        mixer.out_proj.weight.data.copy_(padded_dense_weight)
                    else:
                        padded_dense_weight = pad_weight(mixer.out_proj.weight, teacher_attn.dense.weight, axis=1)
                        mixer.out_proj.weight.data.copy_(padded_dense_weight)

                # xBCzA_log = self.in_proj(u)

                # Define indices for slicing
                xBCzA_log_indices = mixer.get_in_proj_indices()
                start_v, end_v = xBCzA_log_indices['X']
                start_k, end_k = xBCzA_log_indices['B']
                start_q, end_q = xBCzA_log_indices['C']

                # Copy V_proj weights
                padded_teacher_layer_v_proj_weight = pad_weight(mixer.in_proj.weight[start_v:end_v, :],
                                                                teacher_attn.v_proj.weight,
                                                                axis=0) if padding else teacher_attn.v_proj.weight
                mixer.in_proj.weight[start_v:end_v, :].copy_(padded_teacher_layer_v_proj_weight)

                # Copy K_proj weights
                padded_teacher_layer_k_proj_weight = pad_weight(mixer.in_proj.weight[start_k:end_k, :],
                                                                teacher_attn.k_proj.weight,
                                                                axis=0) if padding else teacher_attn.k_proj.weight
                mixer.in_proj.weight[start_k:end_k, :].copy_(padded_teacher_layer_k_proj_weight)

                # Copy Q_proj weights
                padded_teacher_layer_q_proj_weight = pad_weight(mixer.in_proj.weight[start_q:end_q, :],
                                                                teacher_attn.q_proj.weight,
                                                                axis=0) if padding else teacher_attn.q_proj.weight
                mixer.in_proj.weight[start_q:end_q, :].copy_(padded_teacher_layer_q_proj_weight)

                # init A_log
                # A_log_start, A_log_end = xBCzA_log_indices['A_log']
                # A_in_in_proj_cols = mixer.in_proj.weight[A_log_start:A_log_end, :].shape[1]
                # A = torch.empty(mixer.n_v_heads, dtype=torch.float32, device=mixer.in_proj.weight.device).uniform_(1, 16)
                # A = A.unsqueeze(1).expand(-1, A_in_in_proj_cols)
                # A_log = torch.log(A).to(dtype=mixer.in_proj.weight.dtype)
                # mixer.in_proj.weight[A_log_start:A_log_end, :].copy_(A_log)

                # init conv weights
                if self.parsed_args.conv_init:
                    nn.init.uniform_(mixer.conv1d.weight, -self.parsed_args.conv_init, self.parsed_args.conv_init)

    def _init_from_teacher(self, use_clean_teacher=False, padding=False):
        student_module = self.student
        teacher_module = self.teacher if not use_clean_teacher else PhiForCausalLM.from_pretrained(
            "microsoft/phi-1_5",
            attn_implementation="eager"
        ).cpu().bfloat16()
        teacher_backbone = teacher_module.model if hasattr(teacher_module, "model") else teacher_module.backbone
        with torch.no_grad():
            # Copy embedding weights
            student_module.backbone.embedding.load_state_dict(
                teacher_backbone.embed_tokens.state_dict() if hasattr(teacher_backbone, "embed_tokens") else teacher_backbone.embedding.state_dict()
            )

            # Iterate over each layer
            for layer_idx, mamba_phi_block in enumerate(student_module.backbone.layers):
                teacher_layer = teacher_backbone.layers[layer_idx]
                teacher_attn = teacher_layer.mixer.self_attn if hasattr(teacher_layer, "mixer") else teacher_layer.self_attn
                # Load state dicts
                mamba_phi_block.mlp.load_state_dict(teacher_layer.mlp.state_dict())
                mamba_phi_block.input_layernorm.load_state_dict(teacher_layer.input_layernorm.state_dict())
                if hasattr(mamba_phi_block, "post_attention_layernorm"):
                    mamba_phi_block.post_attention_layernorm.load_state_dict(teacher_layer.post_attention_layernorm.state_dict())
                if not padding:
                    if self.parsed_args.teacher_type == "llama":
                        mamba_phi_block.mixer.out_proj.load_state_dict(teacher_attn.o_proj.state_dict(), strict=False)
                    else:
                        mamba_phi_block.mixer.out_proj.load_state_dict(teacher_attn.dense.state_dict(),
                                                                       strict=False)
                else:
                    # Load the state dict from the teacher
                    if self.parsed_args.teacher_type == "llama":
                        teacher_proj_state_dict = teacher_attn.o_proj.state_dict()
                    else:
                        teacher_proj_state_dict = teacher_attn.dense.state_dict()

                    # Get the current shape of the weights and bias in the student model
                    student_proj_weights = mamba_phi_block.mixer.out_proj.weight

                    # Resize the weights from teacher to fit the student's shape
                    # Assuming the teacher's weight shape is [2048, 1472] and the student's is [2048, 1536]
                    teacher_proj_weight = teacher_proj_state_dict['weight']

                    padded_teacher_weight = pad_weight(student_proj_weights, teacher_proj_weight, axis=1)

                    # Load the padded weights and bias into the student model
                    mamba_phi_block.mixer.out_proj.weight.data.copy_(padded_teacher_weight)


                # Convert to desired dtype in-place
                mamba_phi_block.mlp.to(dtype=self.dtype)
                mamba_phi_block.input_layernorm.to(dtype=self.dtype)
                mamba_phi_block.mixer.out_proj.to(dtype=self.dtype)
                if hasattr(mamba_phi_block, "post_attention_layernorm"):
                    mamba_phi_block.post_attention_layernorm.to(dtype=self.dtype)

            # Copy final layernorm and lm_head weights
            if self.parsed_args.teacher_type == "llama":
                student_module.backbone.final_layernorm.load_state_dict(teacher_backbone.norm.state_dict())
            else:
                student_module.backbone.final_layernorm.load_state_dict(teacher_backbone.final_layernorm.state_dict())
            student_module.lm_head.load_state_dict(teacher_module.lm_head.state_dict())

        if use_clean_teacher:
            del teacher_module

    def _transfer_weights_stg3(self):
        """Transfer token embedding, final layer normalization, and the Language Model head."""
        student_module = self.student
        teacher_module = self.teacher

        with torch.no_grad():
            # Copy lm_head weights and convert dtype
            student_module.lm_head.load_state_dict(teacher_module.lm_head.state_dict())
            student_module.lm_head.to(dtype=self.dtype)

            # Copy embedding weights and convert dtype
            student_module.backbone.embedding.load_state_dict(teacher_module.model.embed_tokens.state_dict())
            student_module.backbone.embedding.to(dtype=self.dtype)

            # Copy final layernorm weights and convert dtype
            if self.parsed_args.teacher_type == "llama":
                student_module.backbone.final_layernorm.load_state_dict(teacher_module.model.norm.state_dict())
            else:
                student_module.backbone.final_layernorm.load_state_dict(teacher_module.model.final_layernorm.state_dict())
            student_module.backbone.final_layernorm.to(dtype=self.dtype)

            for layer_idx, mamba_phi_block in enumerate(student_module.backbone.layers):
                teacher_layer = teacher_module.model.layers[layer_idx]

                if not self.parsed_args.no_transfer_mlp:
                    # Copy MLP weights and convert dtype
                    mamba_phi_block.mlp.load_state_dict(teacher_layer.mlp.state_dict())
                    mamba_phi_block.mlp.to(dtype=self.dtype)

                # Copy input layernorm weights and convert dtype
                mamba_phi_block.input_layernorm.load_state_dict(teacher_layer.input_layernorm.state_dict())
                mamba_phi_block.input_layernorm.to(dtype=self.dtype)

    def _set_mixer_conv(self, value: bool):
        for layer_idx, mamba_phi_block in enumerate(self.student.backbone.layers):
            if hasattr(mamba_phi_block.mixer, "conv1d"):
                mamba_phi_block.mixer.conv1d.weight.requires_grad_(value)
                mamba_phi_block.mixer.conv1d.bias.requires_grad_(value)

    def _freeze_stg_1(self):
        model = self.student
        model.requires_grad_(True)
        for name, param in model.named_parameters():
            if any(
                    [
                        n in name
                        for n in [
                        "mlp",
                        "input_layernorm",
                        "embedding",
                        "final_layernorm",
                        "norm",
                        "lm_head",
                        ] + (["self_attn"] if self.parsed_args.hybrid else [])
                    ]
            ):
                param.requires_grad_(False)
            else:
                self.accelerator.print(f"param: {name}")
                param.requires_grad_(True)
        for layer_idx, mamba_phi_block in enumerate(self.student.backbone.layers):
            if hasattr(mamba_phi_block.mixer, "in_proj"):
                mamba_phi_block.mixer.in_proj.requires_grad_(True)
                mamba_phi_block.mixer.out_proj.requires_grad_(False)
                keep_mamba_conv = self.parsed_args.keep_mamba_conv
                mamba_phi_block.mixer.disable_conv = not keep_mamba_conv
                mamba_phi_block.mixer.conv1d.weight.requires_grad_(keep_mamba_conv)
                mamba_phi_block.mixer.conv1d.bias.requires_grad_(keep_mamba_conv)
            else:
                mamba_phi_block.mixer.self_attn.requires_grad_(not self.parsed_args.hybrid)


    def simple_prune(self, percentage: float):
        # Zero out weights directly without using PyTorch's pruning
        for name, module in self.student.named_modules():
            if isinstance(module, torch.nn.Linear) and module.weight.requires_grad:
                with torch.no_grad():
                    # Get the absolute values of the weights
                    weight_abs = module.weight.abs()
                    # Calculate the threshold for pruning
                    threshold = torch.quantile(weight_abs, percentage)
                    # Zero out weights below the threshold
                    mask = weight_abs >= threshold
                    module.weight *= mask

    def _freeze_stg_2(self):
        self.student.requires_grad_(True)
        for name, param in self.student.named_parameters():
            if any(
                    [
                        n in name
                        for n in [
                                     "input_layernorm",
                                     "embedding",
                                     "post_attention_layernorm",
                                     "final_layernorm",
                                     "norm",
                                     "lm_head",
                                 ] + (['mlp'] if not self.parsed_args.train_mlp else [])
                                   + (['self_attn'] if self.parsed_args.hybrid else [])

                    ]
            ):
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

        #enable conv
        for layer_idx, mamba_phi_block in enumerate(self.student.backbone.layers):
            if hasattr(mamba_phi_block.mixer, "in_proj"):
                mamba_phi_block.mixer.in_proj.requires_grad_(True)
                mamba_phi_block.mixer.conv1d.weight.requires_grad_(True)
                mamba_phi_block.mixer.conv1d.bias.requires_grad_(True)
                mamba_phi_block.mixer.out_proj.requires_grad_(True)
                mamba_phi_block.mixer.disable_conv = False
            else:
                mamba_phi_block.mixer.self_attn.requires_grad_(not self.parsed_args.hybrid)

    def _freeze_stg_3(self):
        """ freeze the MLPs only """
        self.student.requires_grad_(True)
        for name, param in self.student.named_parameters():
            if (("mlp" in name and not self.parsed_args.train_mlp) or
                (not self.parsed_args.unfreeze_embeds and ("embedding" in name or "lm_head" in name)) or
                (self.parsed_args.freeze_norms and ("input_layernorm" in name or "final_layernorm" in name or "norm" in name)) or
                (self.parsed_args.hybrid and "self_attn" in name)):
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)

    def train_stg1(self, limit: Optional[int] = None, grad_limit: Optional[int] = None):
        """ Train the student mixer layers to align with the teacher mixer layers """
        self.accelerator.print("Training matrix orientation")
        self.student.train()

        self.teacher.eval()
        self.print_trainable_params_num()
        if self.accelerator.is_main_process and self.parsed_args.sparse:
            self.accelerator.print("Teacher Sparsity")
            self._check_sparsity(self.teacher, include_no_grad=True)
            self.accelerator.print("Student Sparsity")
            sparsity = self._check_sparsity(self.student)
            self.accelerator.log({"sparsity_stg1": sparsity})
        self.accelerator.wait_for_everyone()


        # train using matrix_orientation_optimizer to minimize the loss between the student mixer and teacher
        # attentions layers
        grad_steps = 0
        progress_bar = tqdm("Training matrix orientation",
                            total=self.stage1_actual_total_steps,
                            disable=not self.accelerator.is_main_process)
        top_model_checkpoints_score = {}
        curr_epoch = 0
        while (grad_limit is not None and grad_steps < grad_limit) or (
                self.num_epochs is not None and curr_epoch < self.num_epochs):
            for step, data in enumerate(self.matrix_orientation_train_loader):
                with (self.accelerator.accumulate(self.student)):
                    if step > self.matrix_orientation_total_steps:
                        self.accelerator.print("Step exceeds the total steps !!!")
                        break
                    if limit is not None and step >= limit:
                        break

                    if data['input_ids'].numel() == 0:
                        # Skip empty batch
                        self.accelerator.print("Empty training batch")
                        continue
                    total_loss = self._stg1_inner_train_iter(data)
                    self.accelerator.backward(total_loss)

                    if step % 100 == 0:
                        self.accelerator.print(f"Step: {step}, Loss: {total_loss.item()}")
                        self.accelerator.log({
                            "stg1_loss": total_loss.item(),
                            "stg1_grad_steps": grad_steps,
                            "stg1_lr": self.matrix_orientation_scheduler.get_last_lr()[0],
                            "stg1_total_steps": step
                        })

                    # clip norm with accelerator and assume it handles the gradient accumulation
                    if self.accelerator.sync_gradients:
                        self.accelerator.clip_grad_norm_(self.student.parameters(), 1.0)
                    self.matrix_orientation_optimizer.step()
                    self.matrix_orientation_scheduler.step()
                    self.matrix_orientation_optimizer.zero_grad()

                    if ((step + 1) % self.matrix_orientation_grad_accumulation_steps == 0
                        or (step + 1) == self.matrix_orientation_total_steps) or (limit is not None and step == limit // 4):
                        grad_steps += 1
                        progress_bar.set_description(f"Loss: {total_loss.item()}")
                        progress_bar.update(self.matrix_orientation_grad_accumulation_steps)

                    if (step + 1) % (self.matrix_orientation_grad_accumulation_steps * 25) == 0 or (
                            limit is not None and step == limit // 2):
                        self.accelerator.wait_for_everyone()

                        #evaluate the model
                        self.student.eval()
                        for data in self.matrix_orientation_eval_loader():
                            with torch.no_grad():
                                total_eval_loss = self._stg1_inner_train_iter(data)
                                self.accelerator.log({"stg1_eval_loss": total_eval_loss.item()})
                        self.student.train()

                        # log sparsity
                        if self.accelerator.is_main_process and self.parsed_args.sparse:
                            sparsity = self._check_sparsity(self.student)
                            self.accelerator.log({"sparsity_stg1": sparsity})
                        # save checkpoint
                        path = f"checkpoint-{grad_steps}-{self.random_suffix}"
                        path = os.path.join(self.output_dir, path)
                        if step > 0:
                            top_model_checkpoints_score[path] = total_eval_loss.item()
                        # save model weight
                        unwrapped_model = self.accelerator.unwrap_model(self.student)
                        unwrapped_model.save_pretrained_distributed(path,
                                                                    save_function=self.accelerator.save,
                                                                    is_main_process=self.accelerator.is_main_process)
                        accelerator.save_state(path)
                        self.clean_checkpoints(top_model_checkpoints_score)
                        if self.parsed_args.sparse:
                            # if reached 50% of training steps
                            if step >= self.matrix_orientation_total_steps // 2:
                                # prune the model with up to 50% sparsity starting from 10% sparsity

                                self.simple_prune(0.1 + (step - self.matrix_orientation_total_steps // 2) / (
                                        self.matrix_orientation_total_steps // 2) * 0.4)
            curr_epoch += 1
            self.reset_stg1_dataloader(curr_epoch)


        self.accelerator.wait_for_everyone()
        self.accelerator.print(top_model_checkpoints_score)
        path = f"student_model_stage1_end-{self.random_suffix}"
        path = os.path.join(self.output_dir, path)
        unwrapped_model = self.accelerator.unwrap_model(self.student)
        unwrapped_model.save_pretrained_distributed(path, is_main_process=self.accelerator.is_main_process,
                                                    save_function=self.accelerator.save)
        accelerator.save_state(path)
        progress_bar.close()
        self.accelerator.wait_for_everyone()

    def _stg1_inner_train_iter(self, data):
        total_loss = 0
        with torch.no_grad():
            teacher_result = self.teacher(input_ids=data['input_ids'],
                                          attention_mask=data['attention_mask'],
                                          output_hidden_states=True,
                                          output_attention_results=True,
                                          output_attentions=True,
                                          use_cache=False,
                                          )
        # move the teacher and results to cpu
        if self.parsed_args.cpu_teacher:
            self.teacher.to('cpu')
        [_.to('cpu') for _ in teacher_result.all_hidden_states]

        for layer_idx, student_layer in enumerate(self.student.backbone.layers):
            if student_layer.config.CoreType == "modules.mixers.llama_attention" or student_layer.config.CoreType == "modules.mixers.phi_attention":
                continue
            student_input = teacher_result.all_hidden_states[layer_idx]
            student_input.to(self.accelerator.device)
            student_result = student_layer(hidden_states=student_input,
                                           run_mlp_component=False,
                                           return_mixer_matrix=True,
                                           attention_mask=data['attention_mask'])
            student_mixer_matrices = student_result['transfer_matrix']

            attn_matrices = teacher_result.all_attn_matrices[layer_idx] if hasattr(teacher_result,
                                                                                   "all_attn_matrices") else \
            teacher_result.all_transfer_matrices[layer_idx]
            assert (self.parsed_args.init_padding or
                    student_mixer_matrices.size() == attn_matrices.size()), f"Student: {student_mixer_matrices.size()}, Teacher: {attn_matrices.size()}"
            if self.parsed_args.prune_mha:
                loss = self.matrix_orientation_criterion(student_mixer_matrices, attn_matrices,
                                                         self.mask_head_indexes[layer_idx])
            else:
                loss = self.matrix_orientation_criterion(student_mixer_matrices, attn_matrices)
            total_loss += loss  # Accumulate the loss

            # offload the student input to cpu
            student_input.to('cpu', non_blocking=True)

        if self.parsed_args.cpu_teacher:
            self.teacher.to(self.accelerator.device)

        total_loss /= self.student_backbone_layers_len
        return total_loss

    def train_stg2(self, limit: Optional[int] = None):
        """ Train the student hidden states to align with the teacher hidden states """
        self.accelerator.print("Training hidden states alignment")
        self.teacher.requires_grad_(False)
        self.print_trainable_params_num()
        top_model_checkpoints_score = {}
        freeze_mlp = not self.parsed_args.train_mlp
        grad_steps = 0
        progress_bar = tqdm("Training hidden states alignment",
                            total=self.stage2_actual_total_steps,
                            disable=not self.accelerator.is_main_process)
        if self.accelerator.is_main_process and self.parsed_args.sparse:
            sparsity = self._check_sparsity()
            self.accelerator.log({"sparsity_stg2": sparsity})
        curr_epoch = 0
        grad_limit = self.parsed_args.grad_limit
        while (grad_limit is not None and grad_steps < grad_limit) or (
                self.num_epochs is not None and curr_epoch < self.num_epochs):
            for step, data in enumerate(self.hidden_states_alignment_train_loader):
                with self.accelerator.accumulate(self.student):
                    if step > self.hidden_states_alignment_total_steps:
                        self.accelerator.print("Step exceeds the total steps !!!")
                        break
                    if data['input_ids'].numel() == 0:
                        # Skip empty batch
                        self.accelerator.print("Empty training batch")
                        continue
                    total_loss = self._stg2_inner_train_iter(data, freeze_mlp, self.parsed_args.run_mlp)
                    self.accelerator.backward(total_loss)

                    if step % 100 == 0:
                        self.accelerator.print(f"Step: {step}, Loss: {total_loss.item()}")
                        self.accelerator.log({
                            "stg2_loss": total_loss.item(),
                            "stg2_grad_steps": grad_steps,
                            "stg2_lr": self.hidden_states_alignment_scheduler_with_warmup.get_last_lr()[0],
                            "stg2_total_steps": step
                        })
                    if self.accelerator.sync_gradients:
                        if self.parsed_args.sparse:
                            self.mask_grad()
                        self.accelerator.clip_grad_norm_(self.student.parameters(), 1.0)
                    self.hidden_states_alignment_optimizer.step()
                    self.hidden_states_alignment_scheduler_with_warmup.step()
                    self.hidden_states_alignment_optimizer.zero_grad()

                    if ((step + 1) % self.hidden_states_alignment_grad_accumulation_steps == 0 or
                        (step + 1) == self.hidden_states_alignment_total_steps) or (
                            limit is not None and step == limit // 4):
                        grad_steps += 1
                        progress_bar.set_description(f"Loss: {total_loss.item()}")
                        progress_bar.update(self.hidden_states_alignment_grad_accumulation_steps)

                    if (step + 1) % (self.hidden_states_alignment_grad_accumulation_steps * 25) == 0:
                        self.accelerator.wait_for_everyone()
                        # evaluate the model
                        self.student.eval()
                        for data in self.hidden_states_alignment_eval_loader():
                            with torch.no_grad():
                                total_eval_loss = self._stg2_inner_train_iter(data, freeze_mlp, self.parsed_args.run_mlp)
                                self.accelerator.log({"stg2_eval_loss": total_eval_loss.item()})
                        self.student.train()

                        # log sparsity
                        if self.accelerator.is_main_process and self.parsed_args.sparse:
                            sparsity = self._check_sparsity()
                            self.accelerator.log({"sparsity_stg2": sparsity})
                        # save checkpoint
                        path = f"checkpoint-stg2-{grad_steps}-{self.random_suffix}"
                        path = os.path.join(self.output_dir, path)
                        if step > 0:
                            top_model_checkpoints_score[path] = total_eval_loss.item()
                        unwrapped_model = accelerator.unwrap_model(self.student)
                        unwrapped_model.save_pretrained_distributed(
                            path, save_function=self.accelerator.save, is_main_process=self.accelerator.is_main_process
                        )
                        accelerator.save_state(path)

                        self.clean_checkpoints(top_model_checkpoints_score)



            curr_epoch += 1
            self.reset_stg2_dataloader(curr_epoch)


        self.accelerator.wait_for_everyone()
        self.accelerator.print(top_model_checkpoints_score)
        path = f"student_model_stage2_end"
        path = os.path.join(self.output_dir, path)
        unwrapped_model = self.accelerator.unwrap_model(self.student)
        unwrapped_model.save_pretrained_distributed(path, save_function=self.accelerator.save,
                                                    is_main_process=self.accelerator.is_main_process)
        accelerator.save_state(path)
        progress_bar.close()

    def _stg2_inner_train_iter(self, data, freeze_mlp, run_mlp):
        output_attn_result = freeze_mlp and not run_mlp
        with torch.no_grad():
            teacher_outputs = self.teacher(
                input_ids=data['input_ids'],
                output_hidden_states=True,
                use_cache=False,
                output_attention_results=output_attn_result,
                attention_mask=data['attention_mask'],
                run_mlp=(not freeze_mlp or run_mlp)
            )
        total_loss = 0
        if self.parsed_args.cpu_teacher:
            self.teacher.to('cpu')
        # [teacher_outputs.all_hidden_states[0].to(self.accelerator.device)]+[_.to('cpu') for _ in teacher_outputs.all_hidden_states[1:]]

        for layer_idx, student_layer in enumerate(self.student.backbone.layers):
            if student_layer.config.CoreType == "modules.mixers.llama_attention" or student_layer.config.CoreType == "modules.mixers.phi_attention":
                continue
            student_input = teacher_outputs.all_hidden_states[layer_idx]
            student_input.to(self.accelerator.device)
            # Forward pass
            student_output = student_layer(
                hidden_states=student_input,
                run_mlp_component=(not freeze_mlp or run_mlp),
                return_hidden_states=(not freeze_mlp or run_mlp),
                attention_mask=data['attention_mask']
            )
            if hasattr(teacher_outputs, "all_attn_outputs"):
                attn_results = teacher_outputs.all_attn_outputs
            else:
                attn_results = teacher_outputs.all_mamba_outputs
            if output_attn_result:
                assert layer_idx < len(attn_results), f"Layer index: {layer_idx}, attn_results: {len(attn_results)}"
            teacher_hidden_state = (
                attn_results[layer_idx]
                if (freeze_mlp and not run_mlp)
                else teacher_outputs.all_hidden_states[layer_idx + 1]
            )
            teacher_hidden_state.to(self.accelerator.device)

            assert student_output[
                       "hidden_states"].size() == teacher_hidden_state.size(), f"Student: {student_output['hidden_states'].size()}, Teacher: {teacher_hidden_state.size()}"

            total_loss += self.hidden_states_alignment_criterion(
                student_output["hidden_states"], teacher_hidden_state
            )
            # student_input.to('cpu', non_blocking=True)
            # teacher_hidden_state.to('cpu', non_blocking=True)

        if self.parsed_args.cpu_teacher:
            self.teacher.to(self.accelerator.device)

        total_loss /= self.student_backbone_layers_len
        return total_loss

    def clean_checkpoints(self, top_model_checkpoints_score):
        # remove the lowest loss checkpoint
        if self.accelerator.is_main_process:
            if len(top_model_checkpoints_score) > self.parsed_args.max_saves:
                sorted_top_models = sorted(top_model_checkpoints_score.items(), key=lambda x: x[1])
                # remove the highest loss checkpoint
                try:
                    # remove whole dir
                    if os.path.exists(sorted_top_models[-1][0]):
                        shutil.rmtree(sorted_top_models[-1][0])
                    else:
                        self.accelerator.print(f"Path {sorted_top_models[-1][0]} does not exist, current dir: {os.getcwd()}")
                    del top_model_checkpoints_score[sorted_top_models[-1][0]]
                except Exception as e:
                    self.accelerator.print("Error removing checkpoint", e)
        self.accelerator.print(f"top models: {top_model_checkpoints_score}")

    def train_stg3(self, limit: Optional[int] = None, grad_limit: Optional[int] = None):
        self.accelerator.print("Weight transfer and KD fine-tuning")
        teacher_model = self.teacher
        teacher_model.eval()
        teacher_model.requires_grad_(False)
        if self.accelerator.is_main_process and self.parsed_args.sparse:
            sparsity = self._check_sparsity()
            self.accelerator.log({"sparsity_stg3": sparsity})

        self.student.train()

        self.print_trainable_params_num()
        top_model_checkpoints_score = {}
        progress_bar = tqdm("Training KD fine-tuning",
                            total=self.stg3_actual_total_steps,
                            disable=not self.accelerator.is_main_process)
        grad_steps = 0
        curr_epoch = 0
        while (grad_limit is not None and grad_steps < grad_limit) or (self.num_epochs is not None and curr_epoch < self.num_epochs):
            for step, data in enumerate(self.distillation_train_loader):
                with self.accelerator.accumulate(self.student):
                    if limit is not None and step >= limit:
                        break
                    if data['input_ids'].numel() == 0:
                        # warn if the input_ids is empty
                        self.accelerator.print("Empty input_ids in training")
                        continue

                    grad_steps = self._stg3_inner_train_iter(data, grad_steps, step, teacher_model, progress_bar=progress_bar)

                    # eval and save best
                    if step == 0 or (step + 1) % (self.distillation_grad_accumulation_steps * 100) == 0:
                        self.student.eval()
                        total_eval_loss = 0
                        total_eval_kl_loss = 0
                        total_eval_student_ce_loss = 0
                        total_eval_teacher_ce_loss = 0
                        cnt = 0
                        if self.accelerator.is_main_process and self.parsed_args.sparse:
                            sparsity = self._check_sparsity()
                            self.accelerator.log({"sparsity": sparsity})
                        for batch in self.distillation_get_eval_loader():
                            cnt += 1
                            input_ids = batch["input_ids"]
                            labels = batch["labels"]
                            attention_mask = batch["attention_mask"]
                            if input_ids.numel() == 0:
                                # warn if the input_ids is empty
                                self.accelerator.print("Empty input_ids in evaluation")
                                continue
                            kl_loss, loss, student_ce_loss, teacher_ce_loss = self._stg3_inner_eval_iter(attention_mask,
                                                                                                         input_ids,
                                                                                                         labels,
                                                                                                         teacher_model)
                            total_eval_loss += loss.item()
                            total_eval_kl_loss += kl_loss.item()
                            total_eval_student_ce_loss += student_ce_loss.item()
                            total_eval_teacher_ce_loss += teacher_ce_loss.item()

                        cnt = max(cnt, 1) # in case the eval loader is empty
                        avg_eval_loss = total_eval_loss / cnt
                        avg_eval_kl_loss = total_eval_kl_loss / cnt
                        avg_eval_student_ce_loss = total_eval_student_ce_loss / cnt
                        avg_eval_teacher_ce_loss = total_eval_teacher_ce_loss / cnt

                        avg_eval_loss = self.accelerator.gather(
                            torch.tensor(avg_eval_loss).to(accelerator.device)).mean().item()
                        avg_eval_student_ce_loss = self.accelerator.gather(
                            torch.tensor(avg_eval_student_ce_loss).to(accelerator.device)).mean().item()
                        avg_eval_kl_loss = self.accelerator.gather(
                            torch.tensor(avg_eval_kl_loss).to(accelerator.device)).mean().item()
                        avg_eval_teacher_ce_loss = self.accelerator.gather(
                            torch.tensor(avg_eval_teacher_ce_loss).to(accelerator.device)).mean().item()

                        self.accelerator.print(
                            f"Eval Loss: {avg_eval_loss}, CE Loss: {avg_eval_student_ce_loss}, KD Loss: {avg_eval_kl_loss}")
                        self.accelerator.log({
                            "eval_loss": avg_eval_loss,
                            "eval_kl_loss": avg_eval_kl_loss,
                            "eval_student_ce_loss": avg_eval_student_ce_loss,
                            "eval_teacher_ce_loss": avg_eval_teacher_ce_loss,
                        })
                        if self.accelerator.is_main_process and self.parsed_args.sparse:
                            sparsity = self._check_sparsity()
                            self.accelerator.log({"sparsity_stg3": sparsity})

                        # save checkpoint
                        self.accelerator.wait_for_everyone()
                        path = f"checkpoint-stg3-{grad_steps}-{self.random_suffix}"
                        path = os.path.join(self.output_dir, path)
                        if step > 0 and not isnan(avg_eval_loss):
                            # to exclude the starting checkpoint
                            top_model_checkpoints_score[path] = avg_eval_loss
                        unwrapped_model = accelerator.unwrap_model(self.student)
                        save_func = unwrapped_model.save_pretrained_distributed if not self.parsed_args.sanity_check else unwrapped_model.save_pretrained
                        save_func(
                            path, save_function=self.accelerator.save, is_main_process=self.accelerator.is_main_process
                        )
                        self.accelerator.save_state(path)
                        self.clean_checkpoints(top_model_checkpoints_score)

                        self.student.train()
            curr_epoch += 1
            self.reset_stg3_dataloader(curr_epoch)




        self.accelerator.wait_for_everyone()
        self.accelerator.print(top_model_checkpoints_score)
        path = f"student_model_stage3_end"
        path = os.path.join(self.output_dir, path)
        unwrapped_model = self.accelerator.unwrap_model(self.student)
        save_func = unwrapped_model.save_pretrained_distributed if not self.parsed_args.sanity_check else unwrapped_model.save_pretrained
        save_func(path, save_function=self.accelerator.save,
                  is_main_process=self.accelerator.is_main_process,
                  safe_serialization=False)
        accelerator.save_state(path)
        progress_bar.close()


    def _stg3_inner_eval_iter(self, attention_mask, input_ids, labels, teacher_model):
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=input_ids,
                output_hidden_states=False,
                output_logits=True,
                use_cache=False,
                output_attentions=False,
                attention_mask=attention_mask,
            )
            student_outputs = self.student(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        student_logits = student_outputs.logits
        student_ce_loss = self.CE_criterion(student_logits.view(-1, student_logits.size(-1)),
                                            labels.view(-1))
        teacher_ce_loss = self.CE_criterion(
            teacher_outputs.logits.view(-1, teacher_outputs.logits.size(-1)),
            labels.view(-1))
        kl_loss = self.KD_criterion(student_outputs.logits, teacher_outputs.logits)
        loss = (1.0 - self.KD_loss_weight) * student_ce_loss + self.KD_loss_weight * (
                kl_loss / self.kd_smoother)
        return kl_loss, loss, student_ce_loss, teacher_ce_loss

    def _stg3_inner_train_iter(self, data, grad_steps, step, teacher_model, progress_bar=None):
        with torch.no_grad():
            teacher_outputs = teacher_model(
                input_ids=data['input_ids'],
                output_hidden_states=False,
                output_logits=True,
                use_cache=False,
                output_attentions=False,
                attention_mask=data['attention_mask'],

            )
        if self.parsed_args.sanity_check:
            student_outputs = self.student(
                input_ids=data['input_ids'],
                attention_mask=data['attention_mask'],
                use_cache=False
            )
        else:
            student_outputs = self.student(
                input_ids=data['input_ids'],
                attention_mask=data['attention_mask'],
            )
        labels = data['labels']
        student_logits = student_outputs.logits
        ce_loss = self.CE_criterion(student_logits.view(-1, student_logits.size(-1)), labels.view(-1))
        kd_loss = self.KD_criterion(student_outputs.logits, teacher_outputs.logits)
        loss = (1.0 - self.KD_loss_weight) * ce_loss + self.KD_loss_weight * (kd_loss / self.kd_smoother)
        self.accelerator.backward(loss)
        if (step + 1) % 100 == 0:
            self.accelerator.print(
                f"Step: {step}, Loss: {loss.item()}, CE Loss: {ce_loss.item()}, KD Loss: {kd_loss.item()}")
            self.accelerator.log({
                "stg3_loss": loss.item(),
                "stg3_grad_steps": grad_steps,
                "stg3_lr": self.distillation_scheduler.get_last_lr()[0],
                "stg3_total_steps": step,
                "stg3_ce_loss": ce_loss.item(),
                "stg3_kd_loss": kd_loss.item()
            })
        if self.accelerator.sync_gradients:
            if self.parsed_args.sparse:
                self.mask_grad()
            self.accelerator.clip_grad_norm_(self.student.parameters(), 1.0)
        self.distillation_optimizer.step()
        self.distillation_scheduler.step()
        self.distillation_optimizer.zero_grad()
        if (step + 1) % self.distillation_grad_accumulation_steps == 0 or (
                step + 1) == self.distillation_total_steps:
            grad_steps += 1
            if progress_bar is not None:
                progress_bar.set_description(f"Loss: {loss.item()}")
                progress_bar.update(self.distillation_grad_accumulation_steps)
        return grad_steps

    def train_dpo(self, limit: Optional[int] = None):
        self.accelerator.print("Training DPO")
        self.student.train()
        self.teacher.eval()
        self.print_trainable_params_num()

        training_args = DPOConfig(
            bf16=True,
            beta=0.01,
            do_eval=True,
            evaluation_strategy="steps",
            eval_steps=2000,
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            learning_rate=5.0e-7,
            log_level="info",
            logging_steps=10,
            lr_scheduler_type="cosine",
            max_length=self.parsed_args.length,
            max_prompt_length=512,
            num_train_epochs=self.parsed_args.num_epochs,
            optim="adamw_torch",
            output_dir=self.parsed_args.output_dir,
            per_device_train_batch_size=self.parsed_args.batch,
            per_device_eval_batch_size=self.parsed_args.batch,
            push_to_hub=False,
            save_strategy="steps",
            save_steps=10000,
            save_total_limit=self.parsed_args.max_saves,
            seed=42,
            warmup_ratio=0.1,
            report_to=["wandb"],
            max_steps=self.DPO_total_steps if limit is None else limit,
        )
        self.accelerator.print('Training DPO with the following args:', training_args)
        trainer = DPOTrainer(
            self.student,
            self.teacher,
            args=training_args,
            train_dataset=self.DPO_train_iter_dataset,
            eval_dataset=self.DPO_eval_ter_dataset,
            processing_class=self.tokenizer
        )

        trainer.train()



    def reset_stg1_dataloader(self, epoch):
        self.matrix_orientation_train_loader = self.get_train_dataloaders(epoch)[0]
        self.matrix_orientation_train_loader = self.accelerator.prepare(self.matrix_orientation_train_loader)

    def reset_stg2_dataloader(self, epoch):
        self.hidden_states_alignment_train_loader = self.get_train_dataloaders(epoch)[1]
        self.hidden_states_alignment_train_loader = self.accelerator.prepare(self.hidden_states_alignment_train_loader)

    def reset_stg3_dataloader(self, epoch):
        self.distillation_train_loader = self.get_train_dataloaders(epoch)[2]
        self.distillation_train_loader = self.accelerator.prepare(self.distillation_train_loader)


    def lora_ft(self, limit:int=None):
        raise NotImplementedError

    def print_trainable_params_num(self):
        if self.accelerator.is_main_process:
            self.accelerator.print("student_model:", self.student)
            total_params = sum(p.numel() for p in self.student.parameters())
            total_trainable_params = sum(
                p.numel() for p in self.student.parameters() if p.requires_grad)
            self.accelerator.print(f"number of total params:{total_params}")
            self.accelerator.print(f"number of total trainable params:{total_trainable_params}")

    def _check_sparsity(self, model=None, include_no_grad=False):
        """checks number of sparse weights in the model"""
        if self.accelerator.is_main_process and self.accelerator.sync_gradients:
            sparsity = check_sparsity(self.student if model is None else model, include_no_grad)
            return sparsity
        else:
            return -1

    def mask_grad(self):
        global debug_count
        count = 0
        for name, param in self.student.named_parameters():
            if param.requires_grad and 'weight' in name and param.grad is not None:
                count += 1
                if debug_count < 50:
                    self.accelerator.print(f"[mask_grad] Masking gradients for {name}")
                    self.accelerator.print(f"[mask_grad] Param: {param}")
                    self.accelerator.print(f"[mask_grad] Grad: {param.grad}")
                with torch.no_grad():
                    param.grad *= (param != 0).bfloat16()  # Mask gradients where the parameter is zero

                if debug_count < 50:
                    self.accelerator.print(f"[mask_grad] Masked Grad: {param.grad}")
                    debug_count += 1
        if count == 0:
            self.accelerator.print("[mask_grad] No trainable parameters found")


if __name__ == "__main__":
    # prep dataloader of wikitext 2 and call trainer

    # args parse for running the above:
    # python train.py --batch 2 --length 64 --config_path assets/sample_config.json --dataset_path wikitext-2-v1 --minimize_dataset False --num_epochs 1

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--length", type=int, default=2048)
    parser.add_argument("--config_path", type=str, default="assets/sample_config.json")
    parser.add_argument("--dataset_path", type=str, default='allenai/c4')  # wikitext
    parser.add_argument("--dataset_name", type=str, default='en')  # wikitext-2-v1
    parser.add_argument("--validation", type=str, default='validation')  # wikitext-2-v1
    parser.add_argument("--minimize_dataset", type=bool, default=False)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--stage_1_grad_accumulation_steps", type=int, default=4) # 4 gpus expected, batch 1
    parser.add_argument("--stage_2_grad_accumulation_steps", type=int, default=4) # 4 gpus expected, batch 1
    parser.add_argument("--stage_3_grad_accumulation_steps", type=int, default=16) # 8 gpus expected, batch 2
    parser.add_argument("--stage_1_lr", type=float, default=5e-4)
    parser.add_argument("--stage_2_lr", type=float, default=2e-3)
    parser.add_argument("--stage_3_lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_stage_1", action='store_true', default=False)
    parser.add_argument("--train_stage_2", action='store_true', default=False)
    parser.add_argument("--train_stage_3", action='store_true', default=False)
    parser.add_argument("--train_dpo", action='store_true', default=False)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--grad_limit", type=int, default=None)
    parser.add_argument("--profile", type=bool, default=False)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--teacher_model_path", type=str, default="microsoft/phi-1_5")
    parser.add_argument("--tokenizer_path", type=str, default="microsoft/phi-1_5")
    parser.add_argument("--test_mode", action='store_true', default=False)
    parser.add_argument("--kqv", action='store_true', default=False)
    parser.add_argument("--no_wandb", action='store_true', default=False)
    parser.add_argument("--temp", type=float, default=1.0)
    parser.add_argument("--kd_smoother", type=float, default=1.0)
    parser.add_argument("--kd_loss_weight", type=float, default=1.0)
    parser.add_argument("--max_saves", type=int, default=2)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--sanity_check", action='store_true', default=False)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--project_dir", type=str, default=None)
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--no_transfer_stg3", action='store_true', default=False)
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--sparse", action='store_true', default=False)
    parser.add_argument("--train_mlp", action='store_true', default=False)
    parser.add_argument("--run_mlp", action='store_true', default=False)
    parser.add_argument("--no_transfer_mlp", action='store_true', default=False)
    parser.add_argument("--resume", action='store_true', default=False)
    parser.add_argument("--new_opt", action='store_true', default=False)
    parser.add_argument("--init_padding", action='store_true', default=False)
    parser.add_argument("--pruned_arch_path", type=str, default=None)
    parser.add_argument("--teacher_pruned_arch_path", type=str, default=None)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--teacher_config_path", type=str, default=None)
    parser.add_argument("--lm_head_teacher", action='store_true', default=False)
    parser.add_argument("--new_lr", action='store_true', default=False)
    parser.add_argument("--lora_ft", action='store_true', default=False)
    parser.add_argument("--prune_mha", action='store_true', default=False)
    parser.add_argument("--conv_init", type=int, default=None)
    parser.add_argument("--teacher_type", type=str, default="phi")
    parser.add_argument("--cpu_teacher", action='store_true', default=False)
    parser.add_argument("--reinit_from_teacher", action='store_true', default=False)
    parser.add_argument("--kl_div", action='store_true', default=False)
    parser.add_argument("--unfreeze_embeds", action='store_true', default=False)
    parser.add_argument("--freeze_norms", action='store_true', default=False)
    parser.add_argument("--keep_mamba_conv", action='store_true', default=False)
    parser.add_argument("--no_scheduler", action='store_true', default=False)
    parser.add_argument("--hybrid", action='store_true', default=False)
    parser.add_argument("--hf_student", action='store_true', default=False)




    args = parser.parse_args()

    assert args.skip == 0 or args.resume, "skip is only valid when resuming"
    # profiler
    profile_kwargs = ProfileKwargs(
        activities=["cpu", "cuda"],
        output_trace_dir="trace"
    )
    # set HF timeout
    os.environ["HF_HUB_HTTP_TIMEOUT"] = "60"
    os.environ["HF_HUB_ETAG_TIMEOUT"] = "500"

    # initialize the accelerator
    if args.no_wandb or args.test_mode:
        os.environ["WANDB_MODE"] = "dryrun"

    if args.wandb_run_id is not None:
        os.environ["WANDB_RUN_ID"] = args.wandb_run_id
        os.environ["WANDB_RESUME"] = "must"
    slurm_job_id = os.environ.get('SLURM_JOB_ID', '')
    if args.wandb_name is not None:
        os.environ["WANDB_NAME"] = f"{args.wandb_name}_{slurm_job_id}"
    else:
        os.environ["WANDB_NAME"] = f"{args.output_dir}_{slurm_job_id}"

    if args.train_stage_1:
        grad_accum = args.stage_1_grad_accumulation_steps
    elif args.train_stage_2:
        grad_accum = args.stage_2_grad_accumulation_steps
    else:
        grad_accum = args.stage_3_grad_accumulation_steps
    if args.profile:
        accelerator = Accelerator(log_with="wandb",
                                  kwargs_handlers=[profile_kwargs],
                                  gradient_accumulation_steps=grad_accum)
    else:
        accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=grad_accum)

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        # ser verbosity to info for main process
        logger.setLevel(logging.INFO)
    else:
        # set verbosity to error for all other processes
        logger.setLevel(logging.ERROR)

    accelerator.print("Args: ", args)
    set_seed(args.seed)

    args.output_dir = f"{args.output_dir}_{slurm_job_id}"
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    accelerator.wait_for_everyone()

    trainer = Trainer(args, accelerator)
    trainer.save_description()
    if args.train_stage_1:
        trainer.train_stg1(limit=args.limit, grad_limit=args.grad_limit)
    if args.train_stage_2:
        trainer.train_stg2(limit=args.limit)
    if args.train_stage_3:
        assert args.num_epochs is not None or args.grad_limit is not None, "Either epoch or grad_limit should be provided"
        assert args.grad_limit is None or args.num_epochs is None, "grad_limit and num_epochs cannot be set at the same time"
        trainer.train_stg3(limit=args.limit, grad_limit=args.grad_limit)
    if args.train_dpo:
        trainer.train_dpo(limit=args.limit)

"""
PhiForCausalLM(
  (model): PhiModel(
    (embed_tokens): Embedding(51200, 2048)
    (embed_dropout): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0-23): 24 x PhiDecoderLayer(
        (self_attn): PhiSdpaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (k_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (v_proj): Linear(in_features=2048, out_features=2048, bias=True)
          (dense): Linear(in_features=2048, out_features=2048, bias=True)
          (rotary_emb): PhiRotaryEmbedding()
        )
        (mlp): PhiMLP(
          (activation_fn): NewGELUActivation()
          (fc1): Linear(in_features=2048, out_features=8192, bias=True)
          (fc2): Linear(in_features=8192, out_features=2048, bias=True)
        )
        (input_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
        (resid_dropout): Dropout(p=0.0, inplace=False)
      )
    )
    (final_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=2048, out_features=51200, bias=True)
)

LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(49152, 960)
    (layers): ModuleList(
      (0-31): 32 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=960, out_features=960, bias=False)
          (k_proj): Linear(in_features=960, out_features=320, bias=False)
          (v_proj): Linear(in_features=960, out_features=320, bias=False)
          (o_proj): Linear(in_features=960, out_features=960, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=960, out_features=2560, bias=False)
          (up_proj): Linear(in_features=960, out_features=2560, bias=False)
          (down_proj): Linear(in_features=2560, out_features=960, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((960,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((960,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((960,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=960, out_features=49152, bias=False)
)

>>> student
LMHeadModel(
  (backbone): MixerModel(
        (embedding): Embedding(51200, 2048)
        (layers): ModuleList(
              (0-23): 24 x Block(
                        (mixer): Mixer(
                          (in_proj): Linear(in_features=2048, out_features=8224, bias=False)
                          (conv1d): Conv1d(6144, 6144, kernel_size=(4,), stride=(1,), padding=(3,), groups=6144)
                          (act): Identity()
                          (norm): Identity()
                          (out_proj): Linear(in_features=2048, out_features=2048, bias=False)
                        )
                        (mlp): PhiMLP(
                          (activation_fn): NewGELUActivation()
                          (fc1): Linear(in_features=2048, out_features=8192, bias=True)
                          (fc2): Linear(in_features=8192, out_features=2048, bias=True)
                        )
                        (input_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
                        (resid_dropout): Dropout(p=0.0, inplace=False)
              )
        )
        (final_layernorm): LayerNorm((2048,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=2048, out_features=51200, bias=True)
)



Stage 1
bs = 2^15, lr = 5 × 10−4
Stage 2
bs = 2^15, lr = 2 × 10−3
In Stage 3, 
bs=2^19 ≈ 0.5M and focused solely on varying the learning
rate, resulting in 5 × 10−4
"""
