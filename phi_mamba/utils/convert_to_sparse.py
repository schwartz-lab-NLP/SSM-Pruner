import argparse
import os
from functools import lru_cache

import torch
from torch import count_nonzero, nn
from phi_mamba.modules.lm_head import LMHeadModel
from phi_mamba.modules.modeling_phi_adjusted import PhiForCausalLM
from wanda.compare import compare_models

# @lru_cache(maxsize=2)
def find_layers(module, core_layers=[nn.Linear], name=''):
    if type(module) in core_layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, core_layers=core_layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def layer_sparsity(layer):
    tensor = layer.weight.data
    total_elements = tensor.numel()
    non_zero_elements = count_nonzero(tensor)
    zero_elements = total_elements - non_zero_elements
    sparsity = zero_elements / total_elements
    return sparsity


def check_sparsity(model, include_no_grad=False):
    # use_cache = model.config.use_cache
    backbone = model.backbone if hasattr(model, "backbone") else model.model
    layers = backbone.layers
    count = 0
    total_params = 0

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:

            W_grad = subset[name].weight.requires_grad
            W_data = subset[name].weight.data
            if W_grad or include_no_grad:
                count += (W_data == 0).sum().item()
                total_params += W_data.numel()

                sub_count += (W_data == 0).sum().item()
                sub_params += W_data.numel()
            # else:
            #     print(f"layer {i} {name} requires grad: {subset[name].weight.requires_grad}")
            if sub_params > 0:
                print(f"layer {i} {name} sparsity {float(sub_count) / sub_params:.4f}")

    assert total_params > 0, "No parameters found for sparsity calculation"
    print(f"Total sparsity: {float(count) / total_params:.4f}")

    return float(count) / total_params


def convert_to_sparse(model_path: str, output_dir: str, is_mamba: bool, compare: bool):
    """converts the actual sparse weights to pytorch sparse tensors using to_sparse()"""
    print("Converting model to sparse")

    if is_mamba:
        student = LMHeadModel.from_pretrained(pretrained_model_name_or_path=model_path,
                                              local=True).to(torch.float32)

    else:
        student = PhiForCausalLM.from_pretrained(pretrained_model_name_or_path=model_path,
                                                    attn_implementation='eager').to(torch.float32)

    check_sparsity(student)

    student.config.sparse = True
    print("memory footprint before conversion")
    print_mem_footprint(student)


    if output_dir is not None:
        # save
        path = f"sparse_student_model"
        path = os.path.join(output_dir, path)
        if hasattr(student, "save_pretrained_distributed"):
            student.save_pretrained_distributed(path, is_main_process=True)
        else:
            student.save_pretrained(path)
        print(f"Model saved to {path}")

    print("memory footprint after conversion")
    print_mem_footprint(student)
    if compare:
        student = student.cpu()
        baseline = LMHeadModel.from_pretrained(pretrained_model_name_or_path="goombalab/Phi-Mamba",
                                               local=False).to(torch.float32).cuda()
        print("Baseline model")
        check_sparsity(baseline)
        compare_models(baseline.cpu(), student)



def print_mem_footprint(model):
    total_size = 0
    for p in model.parameters():
        if p.layout == torch.sparse_coo:
            values_size = p._values().numel() * p.element_size()
            indices_size = p._indices().numel() * p._indices().element_size()
            total_size += values_size + indices_size
        else:
            total_size += p.numel() * p.element_size()
    print(f"Memory footprint: {total_size / 1024 / 1024:.2f} MB")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, help='Path to the model')
    parser.add_argument('--output_dir', type=str, help='Output directory', default=None)
    parser.add_argument('--is_mamba', action="store_true", default=False)
    parser.add_argument('--compare', action="store_true", default=False)

    args = parser.parse_args()
    if args.output_dir is not None:
        os.makedirs(args.output_dir, exist_ok=True)
    convert_to_sparse(args.model_path, args.output_dir, args.is_mamba, args.compare)
