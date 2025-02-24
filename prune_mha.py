
import sys
sys.path.extend(['.', './phi_mamba', './MambaInLlama', './original_mamba'])
from pprint import pprint
from typing import Union

from torch.cuda.amp import autocast
from original_mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from MambaInLlama.mamba2.hybrid_wrapper import MambaTransformerHybridModelWrapper
from phi_mamba.modules.lm_head import LMHeadModel
from phi_mamba.modules.modeling_phi_adjusted import PhiForCausalLM
from phi_mamba.modules.modeling_phi_adjusted2 import PhiForCausalLM as PhiForCausalLM2, convert_kv_heads
import torch
from tqdm import tqdm

from logging import getLogger

from phi_mamba.utils.convert_to_sparse import print_mem_footprint, check_sparsity
from phi_mamba.utils.ppl import get_wikitext_dataloader, evaluate_wikitext

logger = getLogger(__name__)


def prune_mha_heads(start_layer: int, end_layer: int, baseline_: Union[PhiForCausalLM, LMHeadModel],
                    to_prune_model: LMHeadModel, ratio: float, return_pruned_set=False) -> torch.Tensor:
    """
    Prune MHA heads from a model by comparing the output of the model with and without the head zeroed out.
    :param baseline_: The baseline model to compare against
    :param to_prune_model: The model to prune
    :return: A tensor of shape (num_layers, num_heads) containing the indices of the heads sorted by impact ascending
    """
    dataloader = get_wikitext_dataloader(percentage=1)

    # Ensure models are on the same device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    baseline_model = baseline_.to(device).eval()
    to_prune_model = to_prune_model.to(device).eval()

    # Number of layers and heads
    num_layers = len(to_prune_model.model.layers)

    def get_num_heads(layer_ind):
        layer_ = to_prune_model.model.layers[layer_ind]
        return layer_.mixer.self_attn.num_heads if hasattr(layer_.mixer, 'self_attn') else layer_.mixer.n_qk_heads

    head_impacts = {layer_idx: {head_idx: [] for head_idx in range(get_num_heads(layer_idx))} for layer_idx in
                    range(num_layers)}
    with torch.no_grad():
        for data in tqdm(dataloader, desc='Pruning heads'):
            attention_mask = data['attention_mask'].to(device)
            baseline_result = baseline_model(input_ids=data['input_ids'].to(device),
                                             attention_mask=attention_mask,
                                             output_hidden_states=True,
                                             output_attention_results=False,
                                             output_attentions=False,
                                             use_cache=False,
                                             )
            for layer_idx, to_prune_layer in enumerate(to_prune_model.backbone.layers):
                if layer_idx < start_layer or layer_idx >= end_layer:
                    continue
                to_prune_input = baseline_result.all_hidden_states[layer_idx]
                for head_idx in range(get_num_heads(layer_idx)):
                    layer_result = to_prune_layer(hidden_states=to_prune_input,
                                                  run_mlp_component=False,
                                                  return_mixer_matrix=True,
                                                  attention_mask=attention_mask)
                    pruned_layer_result = to_prune_layer(hidden_states=to_prune_input,
                                                         run_mlp_component=False,
                                                         return_mixer_matrix=True,
                                                         attention_mask=attention_mask,
                                                         layer_mask_head_index=head_idx)

                    # Calculate loss with the head zeroed out
                    head_impacts[layer_idx][head_idx].append(
                        torch.norm(layer_result['hidden_states'] - pruned_layer_result['hidden_states']))

    # print(to_prune_input, layer_result['hidden_states'], attention_mask)
    head_impacts_avg = {
        layer_idx: {head_idx: torch.mean(torch.tensor(head_impacts[layer_idx][head_idx])) for head_idx in
                    range(get_num_heads(layer_idx))} for layer_idx in range(num_layers)}
    # pprint(head_impacts_avg)

    head_impacts_tensor = torch.tensor(
        [[head_impacts_avg[layer_idx][head_idx] for head_idx in range(get_num_heads(layer_idx))] for layer_idx in
         range(num_layers)])
    sorted_head_impacts = torch.argsort(head_impacts_tensor, dim=1, descending=return_pruned_set)
    # prune the heads with the lowest impact (ratio * num_heads)
    res_sorted_head_impacts = []
    for layer_idx in range(num_layers):
        if layer_idx < start_layer or layer_idx >= end_layer:
            continue
        if return_pruned_set:
            num_heads_to_keep = int(ratio * get_num_heads(layer_idx))
        else:
            num_heads_to_keep = int((1 - ratio) * get_num_heads(layer_idx))
        res_sorted_head_impacts.append(sorted_head_impacts[layer_idx].resize_(num_heads_to_keep))
    logger.debug(f'Layer {layer_idx}: Pruned {get_num_heads(layer_idx) - num_heads_to_keep} heads')
    return torch.stack(res_sorted_head_impacts)


import torch
import copy


def calc_taylor_acc_grad_second(model: LMHeadModel, tokenizer_path='microsoft/phi-1_5'):
    """
    For each parameter p in 'model', this function creates two attributes:
      p.acc_grad : The accumulated (average) first derivative w.r.t. p
      p.acc_hess : The accumulated (average) approximate diagonal Hessian for p

    Steps:
      1) For each batch, compute 'first_grads' = grad(loss, params, create_graph=True).
      2) Sum of squares 'grad_sq = sum(each_grad**2)'.
      3) 'hessian_tuple' = grad(grad_sq, params) => approximate diag Hessian.
      4) Accumulate into p.acc_grad, p.acc_hess.
      5) Normalize by len(dataloader) if desired.

    Arguments:
      model:           A PyTorch model (e.g. GPT2LMHeadModel) with .parameters().
      dataloader:      A DataLoader over the training data subset for computing this approximation.
      device:          A torch.device. If None, defaults to CUDA if available, else CPU.

    """

    dataloader = get_wikitext_dataloader(percentage=1, tokenizer_path=tokenizer_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(torch.bfloat16)
    model.to(device)
    model.train()
    with autocast(dtype=torch.bfloat16):
        # 1) Initialize/clear any old accumulators
        for p in model.parameters():
            if hasattr(p, 'acc_grad'):
                del p.acc_grad
            if hasattr(p, 'acc_hess'):
                del p.acc_hess

        # 2) Iterate over the dataloader
        num_batches = len(dataloader)
        for batch_data in dataloader:
            # Some dataset dicts might have 'input_ids', 'attention_mask', etc.
            # Adjust as needed to match your dataset structure
            batch_input = batch_data['input_ids'].to(device)

            # a) Forward pass: get the loss
            #    Do *not* do loss.backward() -- we'll use torch.autograd.grad instead
            outputs = model(batch_input, labels=batch_input, use_cache=False)
            loss = outputs.loss

            # b) First gradient pass (create_graph=True so we can do grad-of-grad)
            #    This returns a tuple of gradients, each matching a parameter's shape
            first_grads = torch.autograd.grad(
                loss,
                list(model.parameters()),
                create_graph=True,  # keep the graph so we can do a second backward
                retain_graph=True  # keep the graph in memory for the next step
            )

            # c) Build grad_sq = sum of squares of the first grads
            grad_sq_sum = 0.0
            for g in first_grads:
                grad_sq_sum = grad_sq_sum + g.pow(2).sum()

            # d) Second gradient pass => approximate diagonal Hessian
            hessian_tuple = torch.autograd.grad(
                grad_sq_sum,
                list(model.parameters()),
                create_graph=False,  # typically you don't need further grad-of-grad-of-grad
                retain_graph=False
            )

            # e) Accumulate into p.acc_grad and p.acc_hess
            for (p, g, h) in zip(model.parameters(), first_grads, hessian_tuple):
                # Initialize if needed
                if not hasattr(p, 'acc_grad'):
                    p.acc_grad = torch.zeros_like(p.data, device=p.data.device)
                if not hasattr(p, 'acc_hess'):
                    p.acc_hess = torch.zeros_like(p.data, device=p.data.device)
                # Accumulate
                p.acc_grad += g.detach()  # store sum of first-order grads
                p.acc_hess += h.detach()  # store sum of approximate Hessian diag

            # f) Cleanup
            model.zero_grad()
            del loss, outputs  # not strictly necessary, but helps free memory

        # 3) Normalize by the number of batches (optional, but common)
        for p in model.parameters():
            if hasattr(p, 'acc_grad'):
                p.acc_grad /= float(num_batches)
            if hasattr(p, 'acc_hess'):
                p.acc_hess /= float(num_batches)

    # Now each parameter p has:
    #   p.acc_grad ~ average gradient
    #   p.acc_hess ~ average diagonal Hessian approximation
    #
    # Then you can do second-order Taylor pruning or analysis:
    #   p_salience = (p.data * p.acc_grad) + 0.5 * p.acc_hess * (p.data**2)
    #   p_salience = p_salience.abs()
    # etc.


def calc_taylor_acc_grad(model: LMHeadModel, limit=None, seq_len=2048, tokenizer_path='microsoft/phi-1_5', acc=None):
    dataloader = get_wikitext_dataloader(percentage=5, seq_len=seq_len, tokenizer_path=tokenizer_path)
    if acc is not None:
        opt = torch.optim.AdamW(model.parameters(), lr=0)
        model, dataloader, opt = acc.prepare(model, dataloader, opt)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.requires_grad_(True)
    count = 0
    with autocast(dtype=torch.bfloat16):
        for data in tqdm(dataloader):
            if limit is not None and count > limit:
                break
            batch_input = data['input_ids']
            if acc is None:
                batch_input = batch_input.to(device)
            loss = model(batch_input, labels=batch_input, use_cache=False).loss
            loss.backward() if acc is None else acc.backward(loss)

            for name, module_param in model.named_parameters():

                module_param.grad = module_param.grad * module_param.grad / len(dataloader)
                if hasattr(module_param, 'acc_grad'):
                    module_param.acc_grad += module_param.grad
                else:
                    module_param.acc_grad = copy.deepcopy(module_param.grad)

            if acc is not None:
                opt.step()
                opt.zero_grad()

            model.zero_grad()
            del loss.grad


def test_pruning_phi(ratio=0.25):
    baseline = PhiForCausalLM.from_pretrained('microsoft/phi-1_5', attn_implementation='eager')
    to_prune = LMHeadModel.from_pretrained('../prune_log/phi_lm_head')
    sorted_head_impacts = prune_mha_heads(0, 24, baseline, to_prune, ratio, return_pruned_set=True)

    pprint(sorted_head_impacts)
    print("evaluate baseline")
    evaluate_wikitext(baseline)
    print("evaluate original")
    evaluate_wikitext(to_prune)
    del to_prune
    pruned = LMHeadModel.from_pretrained('microsoft/phi-1_5', attn_implementation='eager',
                                         config_path='assets/phi_config.json', mask_head_index=sorted_head_impacts)
    print("evaluate pruned")
    evaluate_wikitext(pruned)


def test_pruning_phi_adjusted(new_num_kv_heads=16, method="mean", exclude_layers=None):
    model = PhiForCausalLM2.from_pretrained('microsoft/phi-1_5', attn_implementation='eager').bfloat16()
    convert_kv_heads(model, new_num_kv_heads=new_num_kv_heads, method=method, exclude_layers=exclude_layers)
    evaluate_wikitext(model)


def test_pruning_goombalab_headdim(ratio, method, exclude_layers=None, acc_grad_light=False, save=False):
    torch.cuda.empty_cache()

    # model = LMHeadModel.from_pretrained('goombalab/Phi-Mamba').bfloat16()
    model = LMHeadModel.from_pretrained('schwartz-lab/Smol2-Mamba-1.9B').bfloat16().to('cuda')
    if 'taylor' in method:
        calc_taylor_acc_grad(model, tokenizer_path='HuggingFaceTB/SmolLM2-1.7B')
    print_mem_footprint(model)
    before = sum(p.numel() for p in model.parameters())
    before_mixer = sum(p.numel() for n, p in model.named_parameters() if 'mixer' in n)

    model.prune_v_head_internal(ratio)
    # model.prune_dstate_and_dinner(ratio, exclude_layers=exclude_layers)
    if ratio:
        print(f"evaluate prune_headdim pruned {ratio}")
    evaluate_wikitext(model, tokenizer_path='HuggingFaceTB/SmolLM2-1.7B')
    print_mem_footprint(model)
    after = sum(p.numel() for p in model.parameters())
    after_mixer = sum(p.numel() for n, p in model.named_parameters() if 'mixer' in n)
    print(f"number of parameters after pruning: {after}")
    print(f"number of mixer parameters after pruning: {after_mixer}")
    print(f"compression ratio: {after / before}")
    print(f"compression ratio mixer: {after_mixer / before_mixer}")

    if save:
        model.save_pretrained_distributed(f'mha_pruned_bc_headdim/smol19_{str(ratio).replace(".", "")}_{method}',
                                          is_main_process=True,
                                          update_config=True)
        model = LMHeadModel.from_pretrained(f'mha_pruned_bc_headdim/smol19_{str(ratio).replace(".", "")}_{method}')
        print(" evaluate after save:")
        evaluate_wikitext(model, tokenizer_path='HuggingFaceTB/SmolLM2-1.7B')
    print_mem_footprint(model)

    torch.cuda.empty_cache()


def test_pruning_phi_adjusted(new_num_kv_heads=16, method="mean", exclude_layers=None):
    model = PhiForCausalLM2.from_pretrained('microsoft/phi-1_5', attn_implementation='eager').bfloat16()
    convert_kv_heads(model, new_num_kv_heads=new_num_kv_heads, method=method, exclude_layers=exclude_layers)
    evaluate_wikitext(model)


def test_pruning_goombalab_dstate(ratio, method, exclude_layers=None, acc_grad_light=True, save=False):
    torch.cuda.empty_cache()

    model = LMHeadModel.from_pretrained('goombalab/Phi-Mamba').bfloat16()
    # model = LMHeadModel.from_pretrained('schwartz-lab/Smol2-Mamba-1.9B').bfloat16().to('cuda')
    if 'taylor' in method:
        if acc_grad_light:
            calc_taylor_acc_grad(model)
            # calc_taylor_acc_grad(model, tokenizer_path='HuggingFaceTB/SmolLM2-1.7B')
        else:
            calc_taylor_acc_grad_second(model)
    print_mem_footprint(model)
    before = sum(p.numel() for p in model.parameters())
    before_mixer = sum(p.numel() for n, p in model.named_parameters() if 'mixer' in n)

    model.prune_dstates(ratio, method, exclude_layers=exclude_layers)
    if ratio:
        print(f"evaluate prune_dstate pruned {ratio}")
    # evaluate_wikitext(model)
    print_mem_footprint(model)
    after = sum(p.numel() for p in model.parameters())
    after_mixer = sum(p.numel() for n, p in model.named_parameters() if 'mixer' in n)
    print(f"number of parameters after pruning: {after}")
    print(f"number of mixer parameters after pruning: {after_mixer}")
    print(f"compression ratio: {after / before}")
    print(f"compression ratio mixer: {after_mixer / before_mixer}")

    # print(pruned)
    if save:
        # save_path = f'mha_pruned_bc_dstates/smol19_{str(ratio).replace(".", "")}_{method}'
        model.save_pretrained_distributed(save_path,
                                          is_main_process=True,
                                          update_config=True)
        model = LMHeadModel.from_pretrained(save_path)
        print("&&&&&&&&&&&&&      evaluate after save:")
        evaluate_wikitext(model, tokenizer_path='HuggingFaceTB/SmolLM2-1.7B')
    print_mem_footprint(model)
    torch.cuda.empty_cache()


def test_pruning_mamba2_dstate(ratio, method, exclude_layers=None, acc_grad_light=True, save=False):
    torch.cuda.empty_cache()

    model = MambaLMHeadModel.from_pretrained('state-spaces/mamba2-2.7b', device='cuda', dtype=torch.bfloat16)
    print(model)
    # model = MambaLMHeadModel.from_pretrained('state-spaces/mamba2-130m', device='cuda', dtype=torch.bfloat16)
    if 'taylor' in method:
        if acc_grad_light:
            calc_taylor_acc_grad(model, tokenizer_path="EleutherAI/gpt-neox-20b")
        else:
            calc_taylor_acc_grad_second(model, tokenizer_path="EleutherAI/gpt-neox-20b")
    print_mem_footprint(model)
    before = sum(p.numel() for p in model.parameters())
    before_mixer = sum(p.numel() for n, p in model.named_parameters() if 'mixer' in n)

    model.prune_dstates(ratio, method, exclude_layers=exclude_layers)
    if ratio:
        print(f"evaluate prune_dstate pruned {ratio}")

    print_mem_footprint(model)
    after = sum(p.numel() for p in model.parameters())
    after_mixer = sum(p.numel() for n, p in model.named_parameters() if 'mixer' in n)
    print(f"number of parameters after pruning: {after}")
    print(f"number of mixer parameters after pruning: {after_mixer}")
    print(f"compression ratio: {after / before}")
    print(f"compression ratio mixer: {after_mixer / before_mixer}")
    print(model)

    if save:
        model.save_pretrained(f'mha_pruned_bc_dstates/mamba2_{str(ratio).replace(".", "")}')
        model = MambaLMHeadModel.from_pretrained(f'mha_pruned_bc_dstates/mamba2_{str(ratio).replace(".", "")}',
                                                 device='cuda', dtype=torch.bfloat16)
    evaluate_wikitext(model, tokenizer_path="EleutherAI/gpt-neox-20b")
    print_mem_footprint(model)
    
    torch.cuda.empty_cache()


def test_pruning_mamba2_headdim(ratio, method, exclude_layers=None, acc_grad_light=True, save=False):
    print("FUNCTION test_pruning_mamba2_headdim")
    print(f"ARGS ratio: {ratio}, method: {method}, exclude_layers: {exclude_layers}, acc_grad_light: {acc_grad_light}")
    torch.cuda.empty_cache()

    model = MambaLMHeadModel.from_pretrained('state-spaces/mamba2-2.7b', device='cuda', dtype=torch.bfloat16)
    if 'taylor' in method:
        if acc_grad_light:
            calc_taylor_acc_grad(model, tokenizer_path="EleutherAI/gpt-neox-20b")
        else:
            calc_taylor_acc_grad_second(model, tokenizer_path="EleutherAI/gpt-neox-20b")
    print_mem_footprint(model)
    before = sum(p.numel() for p in model.parameters())
    before_mixer = sum(p.numel() for n, p in model.named_parameters() if 'mixer' in n)

    model.prune_headdim(ratio, method, exclude_layers=exclude_layers, only_nullify=False, exclude_out_proj=False)
    if ratio:
        print(f"evaluate prune_headdim pruned {ratio}")
    print(model)

    if save:
        model.save_pretrained(f'mha_pruned_headdim_mamba2/mamba2_{method}_{str(ratio).replace(".", "")}')
        model = MambaLMHeadModel.from_pretrained(f'mha_pruned_headdim_mamba2/mamba2_{method}_{str(ratio).replace(".", "")}',
                                                    device='cuda', dtype=torch.bfloat16)
    evaluate_wikitext(model, tokenizer_path="EleutherAI/gpt-neox-20b")
    after = sum(p.numel() for p in model.parameters())
    after_mixer = sum(p.numel() for n, p in model.named_parameters() if 'mixer' in n)
    print(f"number of parameters after pruning: {after}")
    print(f"number of mixer parameters after pruning: {after_mixer}")
    print(f"compression ratio: {after / before}")
    print(f"compression ratio mixer: {after_mixer / before_mixer}")
    torch.cuda.empty_cache()


def test_pruning_mambaLlama_dstate(ratio, method, exclude_layers=None, acc_grad_light=True, save=False):
    test_pruning_mambaLlama_aux('prune_dstates', ratio, method, exclude_layers, acc_grad_light, save=save)


def test_pruning_mambaLlama_headdim(ratio, method, exclude_layers=None, acc_grad_light=True, save=False):
    test_pruning_mambaLlama_aux('prune_xb', ratio, method, exclude_layers, acc_grad_light, save=save)


def test_pruning_mambaLlama_aux(func, ratio, method, exclude_layers=None, acc_grad_light=True, save=False):
    print(f"FUNCTION test_pruning_mambaLlama_{func}")
    print(f"ARGS ratio: {ratio}, method: {method}, exclude_layers: {exclude_layers}, acc_grad_light: {acc_grad_light}")
    torch.cuda.empty_cache()
    model_path = 'JunxiongWang/Llama3.2-Mamba2-3B-dpo'
    model = MambaTransformerHybridModelWrapper.from_pretrained(model_path, torch_dtype=torch.bfloat16)
    for layer in model.model.model.layers:
        if hasattr(layer, 'mamba'):
            mamba_first = layer
            break

    print(
        f' dstate={layer.mamba.d_state}, dmodel={layer.mamba.d_model}, dconv={layer.mamba.d_conv}, ngroups={layer.mamba.ngroups}, expand={layer.mamba.expand}, d_inner={layer.mamba.d_inner}, nheads={layer.mamba.nheads}, headdim={layer.mamba.headdim}')
    print(f'in_proj indices {layer.mamba.get_in_proj_indices()}')
    if 'taylor' in method:
        calc_taylor_acc_grad(model, limit=10, seq_len=128, tokenizer_path=model_path)

    print_mem_footprint(model)
    before = sum(p.numel() for p in model.parameters())
    before_mixer = sum(p.numel() for n, p in model.named_parameters() if ('mamba' in n or 'mixer' in n))

    func_ = getattr(model.unwrapped_model, func) if hasattr(model, 'unwrapped_model') else getattr(model, func)
    func_(ratio, method, exclude_layers=exclude_layers)

    if ratio:
        print(f"evaluate {func} pruned {ratio}")
    print_mem_footprint(model)
    after = sum(p.numel() for p in model.parameters())
    after_mixer = sum(p.numel() for n, p in model.named_parameters() if ('mamba' in n or 'mixer' in n))
    name = f'mha_pruned_mambaLlama_{func}_{method}_{str(ratio).replace(".", "")}'
    if save:
        model.save_pretrained(name)
        model = MambaTransformerHybridModelWrapper.from_pretrained(name, torch_dtype=torch.bfloat16)
    evaluate_wikitext(model, tokenizer_path=model_path)
    print(f"number of parameters after pruning: {after}")
    print(f"number of mixer parameters after pruning: {after_mixer}")
    print(f"compression ratio: {after / before}")
    print(f"compression ratio mixer: {after_mixer / before_mixer}")

    print_mem_footprint(model)
    torch.cuda.empty_cache()


def test_pruning_goombalab_gqa(pruning_methods, head_num, exclude_layers=None):
    torch.cuda.empty_cache()

    # model = LMHeadModel.from_pretrained('goombalab/Phi-Mamba', attn_implementation='eager')
    model = LMHeadModel.from_pretrained('schwartz-lab/Smol2-Mamba-1.9B').bfloat16().to('cuda')
    calc_taylor_acc_grad(model, tokenizer_path='HuggingFaceTB/SmolLM2-1.7B')
    print_mem_footprint(model)
    before = sum(p.numel() for p in model.parameters())
    before_mixer = sum(p.numel() for n, p in model.named_parameters() if 'mixer' in n)

    model.prune_kq_heads(head_num, pruning_methods, exclude_layers=exclude_layers)
    print_mem_footprint(model)
    after = sum(p.numel() for p in model.parameters())
    after_mixer = sum(p.numel() for n, p in model.named_parameters() if 'mixer' in n)
    print(f"number of parameters after pruning: {after}")
    print(f"number of mixer parameters after pruning: {after_mixer}")
    print(f"compression ratio: {after / before}")
    print(f"compression ratio mixer: {after_mixer / before_mixer}")

    #save the model
    model.save_pretrained_distributed(
        f'mha_pruned_bc_heads_{head_num}_{pruning_methods}_{str(exclude_layers).replace(",", "_")}/smol',
        is_main_process=True, update_config=True)
    #
    model = LMHeadModel.from_pretrained(
        f'mha_pruned_bc_heads_{head_num}_{pruning_methods}_{str(exclude_layers).replace(",", "_")}/smol')

    evaluate_wikitext(model, tokenizer_path='HuggingFaceTB/SmolLM2-1.7B')
    print_mem_footprint(model)


def test_pruning_mamba_gqa(num_heads, pruning_methods, exclude_layers=None, save=False):
    torch.cuda.empty_cache()
    model = MambaLMHeadModel.from_pretrained('state-spaces/mamba2-2.7b', device='cuda', dtype=torch.bfloat16)
    print(model)
    first_mixer = model.backbone.layers[0].mixer
    print("d_model:{}, d_state:{}, d_conv:{}, ngroups:{}, expand:{}, d_inner:{}, nheads:{}, headdim:{}".format(
        first_mixer.d_model, first_mixer.d_state, first_mixer.d_conv, first_mixer.ngroups,
        first_mixer.expand, first_mixer.d_inner, first_mixer.nheads, first_mixer.headdim))
    if 'taylor' in pruning_methods:
        calc_taylor_acc_grad(model)
    print_mem_footprint(model)
    before = sum(p.numel() for p in model.parameters())
    before_mixer = sum(p.numel() for n, p in model.named_parameters() if 'mixer' in n)
    model.prune_kq_heads(num_heads, pruning_methods, exclude_layers=exclude_layers)
    evaluate_wikitext(model, tokenizer_path="EleutherAI/gpt-neox-20b")
    print_mem_footprint(model)
    after = sum(p.numel() for p in model.parameters())
    after_mixer = sum(p.numel() for n, p in model.named_parameters() if 'mixer' in n)
    print(f"number of parameters after pruning: {after}")
    print(f"number of mixer parameters after pruning: {after_mixer}")
    print(f"compression ratio: {after / before}")
    print(f"compression ratio mixer: {after_mixer / before_mixer}")

    # print(pruned)
    if save:
        model.save_pretrained_distributed(
            f'mha_pruned_mean_pooling_{num_heads}_{pruning_methods}_{str(exclude_layers).replace(",", "_")}/mamba2',
            is_main_process=True, update_config=True)
    print_mem_footprint(model)
    
    torch.cuda.empty_cache()


def pruning_goombalab_dstate_experiment():
    baseline = LMHeadModel.from_pretrained('goombalab/Phi-Mamba', attn_implementation='eager')
    print("evaluate baseline")
    evaluate_wikitext(baseline)
    del baseline
    #'taylor_first', 'taylor_second', 'magnitude', 'random', 'mean_pooling',
    #   'taylor_first_B', 'taylor_first_C', 'taylor_second_B', 'taylor_second_C'
    for method in ['taylor_second']:
        for ration in [0, 0.2, 0.3, 0.5, 0.8]:
            # for exclude in [None, [0,1,2]]
            print('===' * 100)
            print(f"Pruning method: {method}, ratio: {ration}")
            try:
                test_pruning_goombalab_dstate(ration, method, exclude_layers=None, acc_grad_light=True,
                                              save=(method == 'taylor_second' or method == 'taylor_first'))
            except Exception as e:
                print(e)


def run():
    SAVE_PRUNED = False
    test_pruning_goombalab_dstate(0.25, 'taylor_second', exclude_layers=None, acc_grad_light=True, save=SAVE_PRUNED)
    test_pruning_goombalab_dstate(0.5, 'taylor_second', exclude_layers=None, acc_grad_light=True, save=SAVE_PRUNED)
    test_pruning_goombalab_headdim(0.25, 'taylor_second', exclude_layers=None, save=SAVE_PRUNED)
    test_pruning_goombalab_headdim(0.5, 'taylor_second', exclude_layers=None, save=SAVE_PRUNED)
    test_pruning_goombalab_gqa('mean_pooling', 8, exclude_layers=None)

    test_pruning_mambaLlama_dstate(0.25, 'taylor_second', exclude_layers=None, acc_grad_light=True, save=SAVE_PRUNED)
    test_pruning_mambaLlama_dstate(0.5, 'taylor_second', exclude_layers=None, acc_grad_light=True, save=SAVE_PRUNED)
    test_pruning_mambaLlama_headdim(0.25, 'taylor_second', exclude_layers=None, acc_grad_light=True, save=SAVE_PRUNED)
    test_pruning_mambaLlama_headdim(0.5, 'taylor_second', exclude_layers=None, acc_grad_light=True,  save=SAVE_PRUNED)


    test_pruning_mamba2_dstate(0.25, 'taylor_second', acc_grad_light=True, save=SAVE_PRUNED)
    test_pruning_mamba2_dstate(0.5, 'taylor_second', acc_grad_light=True, save=SAVE_PRUNED)
    test_pruning_mamba2_headdim(0.25, 'taylor_second', acc_grad_light=True,  save=SAVE_PRUNED)
    test_pruning_mamba2_headdim(0.5, 'taylor_second', acc_grad_light=True, save=SAVE_PRUNED)
    test_pruning_mamba_gqa(8, 'mean_pooling', exclude_layers=None, save=SAVE_PRUNED)



if __name__ == '__main__':
    run()