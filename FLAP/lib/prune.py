import math

import torch
import torch.nn as nn
from .layerwrapper import WrappedGPT, BiasGPT
from .data import get_loaders
from tqdm import tqdm
from original_mamba.mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated

# create a dictionary to map the method name to the function
"""
    'IFV': Input Feature Variance
    'WIFV': Weighted Input Feature Variance
    'WIFN': Weighted Input Feature Norm
"""
metrics = {
    'IFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp,
    'WIFV': lambda wrapped_layers, subset, name: wrapped_layers[name].fluc_inp * torch.sum(
        subset[name].weight.data.pow(2), dim=0),
    'WIFN': lambda wrapped_layers, subset, name: (torch.abs(subset[name].weight.data) * torch.sqrt(
        wrapped_layers[name].scaler_inp.reshape((1, -1)))).mean(axis=0),
}


def find_layers(module, layers=[nn.Linear], name=''):
    """
    Recursively find the layers of a certain type in a module.

    Args:
        module (nn.Module): PyTorch module.
        layers (list): List of layer types to find.
        name (str): Name of the module.

    Returns:
        dict: Dictionary of layers of the given type(s) within the module.
    """
    if type(module) in layers:
        return {name: module}
    res = {}
    for name1, child in module.named_children():
        res.update(find_layers(
            child, layers=layers, name=name + '.' + name1 if name != '' else name1
        ))
    return res


def flap_check_sparsity(model):
    """
    Check the sparsity of the weights in different layers of the model.
    
    Args:
        model (nn.Module): The model to check.
        
    Returns:
        float: Ratio of the count of non-zero weights to total parameters in the model.
    """
    if hasattr(model.config, "use_cache"):
        use_cache = model.config.use_cache
        model.config.use_cache = False

    backbone = model.model if hasattr(model, "model") else model.backbone
    layers = backbone.layers if hasattr(model, "layers") else backbone.model.layers

    intermediate_size = model.config.intermediate_size if hasattr(model.config, "intermediate_size") else layers[
        0].mlp.fc1.out_features if hasattr(layers[0].mlp, 'fc1') else layers[0].mlp.down_proj.in_features
    hidden_size = model.config.hidden_size if hasattr(model.config,
                                                      "hidden_size") else model.config.MixerModel.input.d_model

    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            sub_count += W.numel()
            count += W.numel()
            if 'self_attn' in name or 'mixer' in name:
                total_params += hidden_size * hidden_size
                sub_params += hidden_size * hidden_size
            else:
                total_params += hidden_size * intermediate_size
                sub_params += hidden_size * intermediate_size
            if subset[name].bias is not None:
                count += subset[name].bias.data.numel()
                sub_count += subset[name].bias.data.numel()

        print(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = use_cache
    return float(count) / total_params


def check_sparsity(model):
    if hasattr(model.config, "use_cache"):
        use_cache = model.config.use_cache
        model.config.use_cache = False

    layers = model.model.layers if hasattr(model, "model") else model.backbone.layers
    count = 0
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W == 0).sum().item()
            total_params += W.numel()

            sub_count += (W == 0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count) / sub_params:.6f}")

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = use_cache
    return float(count) / total_params


def prepare_calibration_input(model, dataloader, device):
    """
    Prepare inputs for model calibration. 
    
    Args:
        model (nn.Module): The model to prepare inputs for.
        dataloader (DataLoader): DataLoader object to fetch input data.
        device (torch.device): Device on which the model is loaded. 
        
    Returns:
        inps (torch.Tensor): Input tensor for calibration.
        outs (torch.Tensor): Output tensor for calibration.
        attention_mask (torch.Tensor): Attention mask tensor.
        position_ids (torch.Tensor): Position IDs tensor.
    """
    if hasattr(model.config, "use_cache"):
        use_cache = model.config.use_cache
        model.config.use_cache = False

    backbone = model.model if hasattr(model, "model") else model.backbone
    layers = backbone.layers if hasattr(backbone, "layers") else backbone.model.layers
    # layers = model.model.layers

    if "model.embed_tokens" in getattr(model, 'hf_device_map', {}):
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    if hasattr(model.config, "hidden_size") and model.config.hidden_size is None:
        model.config.hidden_size = model.config.d_model
    inps = torch.zeros((2048,
                        model.seqlen,
                        model.config.hidden_size if hasattr(model.config, "hidden_size") else (model.config.MixerModel.input.d_model if hasattr(model.config, 'MixerModel') else model.config.d_model)), dtype=dtype, device=device)

    inps.requires_grad = False
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, *args, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs.get('attention_mask', None)
            cache['position_ids'] = kwargs.get('position_ids', None)
            cache['residual'] = args[0] if len(args) > 0 else None
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    residual = cache['residual']
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = use_cache

    return inps, outs, attention_mask, position_ids if residual is None else (
    inps, outs, attention_mask, position_ids, residual)


def compress(layer, attn_mask, mlp_mask, attn_mean_inp, mlp_mean_inp, device, bias=True, unstr=False, args=None):
    """
    Compress a model layer by masking or pruning based on the given masks.
    
    Args:
        layer (nn.Module): The model layer to compress.
        attn_mask (torch.Tensor): The mask to apply to the attention weights.
        mlp_mask (torch.Tensor): The mask to apply to the MLP weights.
        attn_mean_inp (torch.Tensor): The mean attention input.
        mlp_mean_inp (torch.Tensor): The mean MLP input.
        device (torch.device): Device on which the model is loaded.
        bias (bool, optional): Whether to consider bias while compressing. Defaults to True.
        unstr (bool, optional): If True, only mask without real pruning. Defaults to False.
        
    Returns:
        None: This function modifies the layer in-place and doesn't return anything.
    """
    repeated_group_size = None
    if hasattr(layer, 'mixer') and hasattr(layer.mixer, 'in_proj'):
        in_proj_indices = layer.mixer.get_in_proj_indices()
        z_start, z_end = in_proj_indices['z']
        x_start, x_end = in_proj_indices['x']
        B_start, B_end = in_proj_indices['B']
        C_start, C_end = in_proj_indices['C']
        A_log_or_Dt_start, A_log_or_Dt_end = in_proj_indices['A_log'] if 'A_log' in in_proj_indices else \
        in_proj_indices['dt']
        x_start_conv, x_end_conv = in_proj_indices['x_conv']
        b_start_conv, b_end_conv = in_proj_indices['B_conv']
        c_start_conv, c_end_conv = in_proj_indices['C_conv']

        headdim = layer.mixer.headdim
        dstate = layer.mixer.d_state
        if hasattr(layer.mixer, 'repeat_group'):
            repeated_group_size = layer.mixer.nheads  //  layer.mixer.repeat_group
        elif args.is_mamba and not args.is_lm_head and not args.is_mamba_in_llama:
            repeated_group_size = 1
        else:
            repeated_group_size = layer.mixer.nheads if hasattr(layer.mixer, 'nheads') else layer.mixer.n_v_heads

    else:
        headdim = layer.self_attn.head_dim

    if unstr:  # Only mask, do not really prune
        # Attention Weight Masking
        if attn_mask is not None:
            retain_heads = torch.count_nonzero(attn_mask)
            attn_mask = attn_mask.repeat_interleave(headdim)
            if headdim != dstate:
                attn_mask_dstate = attn_mask.repeat_interleave(dstate)
            if hasattr(layer, 'self_attn'):
                # Apply the mask to the query, key and value projection weights
                layer.self_attn.q_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)
                layer.self_attn.k_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)
                layer.self_attn.v_proj.weight.data *= attn_mask.unsqueeze(-1).to(device)
            else:
                layer.mixer.in_proj.weight.data[z_start:z_end] *= attn_mask.unsqueeze(-1).to(device)
                layer.mixer.in_proj.weight.data[x_start:x_end] *= attn_mask.unsqueeze(-1).to(device)
            if hasattr(layer, 'self_attn'):
                o_proj = layer.self_attn.o_proj if hasattr(layer.self_attn, 'o_proj') else layer.self_attn.dense
            else:
                o_proj = layer.mixer.out_proj
            output_weight = o_proj.weight.data
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((attn_mean_inp * ~attn_mask.to(device)) @ output_weight.T)

            # Note: the weight data is masked, but the weight tensor shape remains unchanged
            if bias:
                o_proj.bias.data = output_bias
            o_proj.weight.data = output_weight

        # MLP Weight Masking
        if mlp_mask is not None and hasattr(layer, 'mlp'):
            # Apply the mask to the up and gate projection weights
            if hasattr(layer.mlp, 'up_proj'):
                layer.mlp.up_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)

            gate_proj = layer.mlp.gate_proj if hasattr(layer.mlp, 'gate_proj') else layer.mlp.fc1
            gate_proj.weight.data *= mlp_mask.unsqueeze(-1).to(device)

            down_proj = layer.mlp.down_proj if hasattr(layer.mlp, 'down_proj') else layer.mlp.fc2
            output_weight = down_proj.weight.data
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ output_weight.T)

            # Note: the weight data is masked, but the weight tensor shape remains unchanged
            if bias:
                down_proj.bias.data = output_bias
            down_proj.weight.data = output_weight

    else:
        # Real Pruning
        # Attention Weight Pruning
        if attn_mask is not None:
            retain_heads = torch.count_nonzero(attn_mask)
            attn_mask_pre_repeat = attn_mask
            attn_mask = attn_mask.repeat_interleave(headdim)  # 128
            if headdim != dstate:
                attn_mask_dstate = attn_mask_pre_repeat.repeat_interleave(dstate)
            else:
                attn_mask_dstate = attn_mask
            if repeated_group_size is not None:
                grouped_heads_mask = attn_mask_pre_repeat[:repeated_group_size].repeat_interleave(headdim)
            else:
                grouped_heads_mask = attn_mask

            if hasattr(layer, 'self_attn'):
                # Prune the query, key and value projection weights
                # We reduce the size of the weights based on the attention mask
                layer.self_attn.q_proj.weight.data = layer.self_attn.q_proj.weight.data[torch.where(attn_mask)[0]]
                layer.self_attn.k_proj.weight.data = layer.self_attn.k_proj.weight.data[torch.where(attn_mask)[0]]
                layer.self_attn.v_proj.weight.data = layer.self_attn.v_proj.weight.data[torch.where(attn_mask)[0]]

                # Update output dimensions of q, k, v projections based on remaining heads
                layer.self_attn.q_proj.out_features = attn_mask.sum().item()
                layer.self_attn.k_proj.out_features = attn_mask.sum().item()
                layer.self_attn.v_proj.out_features = attn_mask.sum().item()

                o_proj = layer.self_attn.o_proj if hasattr(layer.self_attn, 'o_proj') else layer.self_attn.dense
                output_weight = o_proj.weight.data
            else:
                ################################### pruning IN_PROJ weights ###################################
                in_proj_mask = torch.ones(layer.mixer.in_proj.weight.data.shape[0], device=device, dtype=torch.bool)
                xBC_pruned_num= 0
                in_proj_mask[z_start:z_end] = False
                if  args.is_lm_head:
                    in_proj_mask[x_start:x_end] = False
                    in_proj_mask[B_start:B_end] = False
                    in_proj_mask[C_start:C_end] = False
                elif args.is_mamba_in_llama:
                    in_proj_mask[C_start:C_end] = False
                else:
                    in_proj_mask[x_start:x_end] = False
                in_proj_mask[A_log_or_Dt_start:A_log_or_Dt_end] = False

                in_proj_mask[torch.where(attn_mask)[0] + z_start] = True
                if args.is_lm_head:
                    in_proj_mask[torch.where(attn_mask)[0] + x_start] = True
                    in_proj_mask[torch.where(attn_mask_dstate)[0] + B_start] = True
                    in_proj_mask[torch.where(attn_mask_dstate)[0] + C_start] = True
                elif args.is_mamba_in_llama:
                    in_proj_mask[torch.where(attn_mask)[0] + C_start] = True
                else:
                    in_proj_mask[torch.where(attn_mask)[0] + x_start] = True
                xBC_pruned_num += in_proj_mask[x_start:x_end].sum().item()
                xBC_pruned_num += in_proj_mask[B_start:B_end].sum().item()
                xBC_pruned_num += in_proj_mask[C_start:C_end].sum().item()
                in_proj_mask[torch.where(attn_mask_pre_repeat)[0] + A_log_or_Dt_start] = True
                layer.mixer.in_proj.weight = nn.Parameter(layer.mixer.in_proj.weight[in_proj_mask, :])
                layer.mixer.in_proj.out_features = in_proj_mask.sum().item()
                layer.mixer.just_pruned = True

                ################################### pruning CONV1D weights ###################################
                # Build mask for conv1d channels
                conv_dim_old = layer.mixer.conv1d.weight.shape[0]
                conv_mask = torch.ones(conv_dim_old, dtype=torch.bool, device=layer.mixer.conv1d.weight.device)

                if args.is_lm_head:
                    conv_mask[x_start_conv:x_end_conv] = False
                    conv_mask[b_start_conv:b_end_conv] = False
                    conv_mask[c_start_conv:c_end_conv] = False
                    conv_mask[torch.where(attn_mask)[0] + x_start_conv] = True
                    conv_mask[torch.where(attn_mask_dstate)[0] + b_start_conv] = True
                    conv_mask[torch.where(attn_mask_dstate)[0] + c_start_conv] = True
                if args.is_mamba_in_llama:
                    conv_mask[c_start_conv:c_end_conv] = False
                    conv_mask[torch.where(attn_mask)[0] + c_start_conv] = True
                else:
                    conv_mask[x_start_conv:x_end_conv] = False
                    conv_mask[torch.where(attn_mask)[0] + x_start_conv] = True

                assert xBC_pruned_num == conv_mask.sum().item(), f"xBC_pruned_num: {xBC_pruned_num}, conv_mask.sum(): {conv_mask.sum()}, layer: {layer.layer_idx}"

                if hasattr(layer.mixer, 'z_bias') and layer.mixer.z_bias is not None:
                    layer.mixer.z_bias = nn.Parameter(layer.mixer.z_bias[torch.where(attn_mask)[0]])

                # Create new conv1d layer with adjusted dimensions
                conv_dim_new = conv_mask.sum().item()
                factory_kwargs = {"device": layer.mixer.conv1d.weight.device, "dtype": layer.mixer.conv1d.weight.dtype}
                new_conv = nn.Conv1d(
                    in_channels=conv_dim_new,
                    out_channels=conv_dim_new,
                    bias=layer.mixer.conv_bias,
                    kernel_size=layer.mixer.d_conv,
                    groups=conv_dim_new,
                    padding=layer.mixer.d_conv - 1,
                    **factory_kwargs,
                )

                pruned_conv_weights = layer.mixer.conv1d.weight[conv_mask, :, :]

                new_conv.weight.data.copy_(pruned_conv_weights)
                if layer.mixer.conv1d.bias is not None:
                    pruned_conv_bias = layer.mixer.conv1d.bias.data[conv_mask]
                    new_conv.bias.data.copy_(pruned_conv_bias)
                layer.mixer.conv1d = new_conv

                ################################### pruning D&A weights ###################################
                layer.mixer.D = nn.Parameter(layer.mixer.D[torch.where(attn_mask_pre_repeat)[0]])
                if hasattr(layer.mixer, 'A_log'):
                    layer.mixer.A_log = nn.Parameter(layer.mixer.A_log[torch.where(attn_mask_pre_repeat)[0]])
                if hasattr(layer.mixer, 'dt_bias'):
                    layer.mixer.dt_bias = nn.Parameter(layer.mixer.dt_bias[torch.where(attn_mask_pre_repeat)[0]])

                o_proj = layer.mixer.out_proj
                output_weight = o_proj.weight.data

            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((attn_mean_inp * ~attn_mask.to(device)) @ output_weight.T)
            ################################### pruning norm ###################################
            if hasattr(layer.mixer, 'norm'):

                new_norm_wights = nn.Parameter(layer.mixer.norm.weight[torch.where(attn_mask)[0]])
                layer.mixer.norm = RMSNormGated(new_norm_wights.shape[0], eps=1e-5,
                                                norm_before_gate=layer.mixer.norm_before_gate,
                                                group_size=new_norm_wights.shape[0] , ** {"device": device, "dtype": layer.mixer.in_proj.weight.dtype})

                layer.mixer.norm.weight = new_norm_wights
            ################################### pruning OUT_PROJ/O_PROJ weights ###################################
            # Prune the output projection weight
            output_weight = o_proj.weight.data[:, torch.where(attn_mask)[0]]

            # Update layer configurations for the new output shape after pruning
            if hasattr(layer, 'self_attn'):
                layer.self_attn.num_heads = retain_heads.item()
                layer.self_attn.hidden_size = retain_heads.item() * headdim
            else:
                if hasattr(layer.mixer, 'nheads'):
                    layer.mixer.nheads = retain_heads.item()
                if hasattr(layer.mixer, 'n_v_heads'):
                    layer.mixer.n_v_heads = retain_heads.item()
                    layer.mixer.n_qk_heads = retain_heads.item()
                if hasattr(layer.mixer, 'ngroups'):
                    layer.mixer.ngroups = retain_heads.item()

                layer.mixer.headdim = headdim
                layer.mixer.d_inner = retain_heads.item() * headdim


            if bias:
                # Re-initialize the Linear layer with new shape and bias
                o_proj.in_features = attn_mask.sum().item()
                # layer.self_attn.o_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
                o_proj.bias.data = output_bias

            # Assign the pruned weights
            o_proj.weight.data = output_weight

            if hasattr(layer, 'mixer'):
                layer.mixer.in_proj.out_features = layer.mixer.in_proj.weight.data.shape[0]
                layer.mixer.in_proj.in_features = layer.mixer.in_proj.weight.data.shape[1]
                layer.mixer.out_proj.in_features = layer.mixer.out_proj.weight.data.shape[1]
                layer.mixer.out_proj.out_features = layer.mixer.out_proj.weight.data.shape[0]
                layer.mixer.d_inner = layer.mixer.out_proj.in_features
                if hasattr(layer.mixer, 'd_ssm'):
                    layer.mixer.d_ssm = layer.mixer.d_inner
                if hasattr(layer.mixer, 'repeat_group'):
                    layer.mixer.repeat_group =  layer.mixer.d_inner //  layer.mixer.d_xb
                assert retain_heads == layer.mixer.d_inner // layer.mixer.headdim, f"retain_heads: {retain_heads}, d_inner: {layer.mixer.d_inner}, headdim: {layer.mixer.headdim}"
                if hasattr(layer.mixer, 'n_qk_heads'):
                    assert retain_heads % layer.mixer.n_qk_heads == 0, f"retain_heads: {retain_heads}, n_qk_heads: {layer.mixer.n_qk_heads}"

        # MLP Weight Pruning
        if mlp_mask is not None and hasattr(layer, 'mlp') and layer.mlp is not None:
            # Prune the up and gate projection weights
            if hasattr(layer.mlp, 'up_proj'):
                layer.mlp.up_proj.weight.data = layer.mlp.up_proj.weight.data[torch.where(mlp_mask)[0]]
            gate_proj = layer.mlp.gate_proj if hasattr(layer.mlp, 'gate_proj') else layer.mlp.fc1
            gate_proj.weight.data = gate_proj.weight.data[torch.where(mlp_mask)[0]]
            if gate_proj.bias is not None:
                gate_proj.bias.data = gate_proj.bias.data[torch.where(mlp_mask)[0]]

            # Update output dimensions of up and gate projections based on the mlp mask
            if hasattr(layer.mlp, 'up_proj'):
                layer.mlp.up_proj.out_features = mlp_mask.sum().item()
            gate_proj.out_features = mlp_mask.sum().item()

            down_proj = layer.mlp.down_proj if hasattr(layer.mlp, 'down_proj') else layer.mlp.fc2
            output_weight = down_proj.weight.data
            layer.mlp.intermediate_size = mlp_mask.sum().item()
            if bias:
                # Add the additional bias to compensate for the loss
                output_bias = ((mlp_mean_inp * ~mlp_mask.to(device)) @ output_weight.T)

            # Prune the down projection weight
            output_weight = down_proj.weight.data[:, torch.where(mlp_mask)[0]]

            if bias:
                # Re-initialize the Linear layer with new shape and bias
                down_proj.in_features = mlp_mask.sum().item()
                # layer.mlp.down_proj = torch.nn.Linear(in_features=output_weight.shape[1], out_features=output_weight.shape[0], bias=True).to(device)
                down_proj.bias.data += output_bias

            # Assign the pruned weights
            down_proj.weight.data = output_weight

    # Explicitly empty the CUDA cache to clean up some memory
    torch.cuda.empty_cache()


def cal_remove_neuron(args, model):
    intermediate_size = model.config.intermediate_size
    hidden_size = model.config.hidden_size
    num_layers = model.config.num_hidden_layers
    if args.structure == "UL-MM":
        remove_params = args.pruning_ratio * (intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
        remove_head_params = hidden_size * 4 * (args.remove_heads // num_layers) * 128
        return int((remove_params - remove_head_params) / (hidden_size * 3))
    else:
        remove_params = num_layers * args.pruning_ratio * (
                    intermediate_size * hidden_size * 3 + hidden_size * hidden_size * 4)
        remove_head_params = hidden_size * 4 * args.remove_heads * 128
        return int((remove_params - remove_head_params) / (hidden_size * 3))


def prune_flap(args, model, tokenizer, device=torch.device("cuda:0"), retain_heads_min=1, const_heads=None):
    """
    Our FLAP Pruning.
    
    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    if hasattr(model.config, "use_cache"):
        use_cache = model.config.use_cache
        model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen,
                                tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        calib_inputs = prepare_calibration_input(model, dataloader, device)
        residual = None
        if len(calib_inputs) == 4:
            inps, outs, attention_mask, position_ids = calib_inputs
        else:
            inps, outs, attention_mask, position_ids, residual = calib_inputs

    attn_metric_list, mlp_metric_list = [], []
    attn_baseline_inp_list, mlp_baseline_inp_list = [], []
    attn_mask, mlp_mask = [], []

    # Split into sub-problems, separate statistics for each module
    backbone = model.model if hasattr(model, "model") else model.backbone
    layers = backbone.layers if hasattr(backbone, "layers") else backbone.model.layers
    if args.skip_attn:
        layers = [l for l in layers if hasattr(l, 'mixer')]
    for i in tqdm(range(len(layers)), desc="Processing layers"):
        layer = layers[i]
        subset = {}
        if hasattr(layer, 'self_attn') and args.skip_attn:
            continue
        elif hasattr(layer, 'self_attn') or (hasattr(layer, 'mixer') and args.is_mamba_in_llama):
            o_proj_key = 'self_attn.o_proj' if hasattr(layer.self_attn, 'o_proj') else 'self_attn.dense'
            headdim = layer.self_attn.head_dim
        elif hasattr(layer, 'mamba'):
            o_proj_key = 'mamba.out_proj'
            headdim = layer.mamba.headdim
        else:
            o_proj_key = 'mixer.out_proj'
            import pdb; pdb.set_trace()
            headdim = layer.mixer.headdim
        subset.update({o_proj_key: find_layers(layer)[o_proj_key]})
        if (hasattr(layer, 'mlp') and layer.mlp is not None) and not args.skip_mlp:
            mlp_down_proj_key = 'mlp.down_proj' if hasattr(layer.mlp, 'down_proj') else 'mlp.fc2'
            subset.update({mlp_down_proj_key: find_layers(layer)[mlp_down_proj_key]})

        if f"model.layers.{i}" in getattr(model, 'hf_device_map',
                                          {}):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(
                dev), position_ids.to(dev)
            if residual is not None:
                residual = residual.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = BiasGPT(subset[name], args.metrics)

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                if residual is None:
                    if args.is_mamba_in_llama:
                        position_embeddings = backbone.model.rotary_emb(inps[j].unsqueeze(0), position_ids)
                        tmp_out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings )
                    else:
                        tmp_out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)
                else:

                    tmp_out, residual = layer(inps[j].unsqueeze(0), residual, attention_mask=attention_mask,
                                              position_ids=position_ids)
                outs[j] = tmp_out[0] if not args.is_lm_head else tmp_out['hidden_states']


        for h in handles:
            h.remove()

        for name in subset:
            # if name in ['mixer.out_proj', 'mamba.out_proj']:
            if name in ['self_attn.o_proj', 'self_attn.dense', 'mixer.out_proj', 'mamba.out_proj']:
                W_metric = metrics[args.metrics](wrapped_layers, subset, name) ** 2
                if args.structure == "UL-UM":
                    W_metric = W_metric.reshape(-1, headdim).sum(dim=1)
                    num_heads = layer.self_attn.num_heads if hasattr(layer, 'self_attn') else (
                        layer.mixer.nheads if hasattr(layer.mixer, 'nheads') else layer.mixer.n_v_heads
                    )
                    thresh = torch.sort(W_metric.cuda())[0][int(args.pruning_ratio * num_heads)].cpu()
                    W_mask = (W_metric >= thresh)
                    attn_mask.append(W_mask)
                elif args.structure == "UL-MM":
                    W_metric = W_metric.reshape(-1, headdim).sum(dim=1)
                    thresh = torch.sort(W_metric.cuda())[0][args.remove_heads // len(layers)].cpu()
                    W_mask = (W_metric >= thresh)
                    attn_mask.append(W_mask)
                else:
                    attn_metric_list.append(W_metric.cpu())
                attn_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.bfloat16))
            else:
                W_metric = metrics[args.metrics](wrapped_layers, subset, name)
                if args.structure == "UL-UM":
                    thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel() * args.pruning_ratio)].cpu()
                    W_mask = (W_metric >= thresh)
                    mlp_mask.append(W_mask)
                elif args.structure == "UL-MM":
                    thresh = torch.sort(W_metric.cuda())[0][cal_remove_neuron(args, model)].cpu()
                    W_mask = (W_metric >= thresh)
                    mlp_mask.append(W_mask)
                else:
                    mlp_metric_list.append(W_metric.cpu())
                mlp_baseline_inp_list.append(wrapped_layers[name].baseline_inp.type(torch.bfloat16))
            wrapped_layers[name].free()

        inps, outs = outs, inps  # Use the original output as input to the next layer
        torch.cuda.empty_cache()

    standarlization = lambda x: (x - torch.mean(x, axis=1, keepdim=True)) / torch.std(x, axis=1, keepdim=True)
    if args.structure in ["AL-MM", "AL-AM"]:
        attn_metric = torch.stack(attn_metric_list)
        attn_metric = standarlization(attn_metric)
        len_layers = sum(hasattr(l, 'mixer') for l in layers) if args.skip_attn else len(layers)
        attn_metric = attn_metric.reshape(len_layers, -1, headdim).mean(dim=2)
        if len(mlp_metric_list):
            mlp_metric = torch.stack(mlp_metric_list)
            mlp_metric = standarlization(mlp_metric)
        else:
            mlp_metric = None

        if args.structure == "AL-MM":
            sorted_attn = torch.sort(attn_metric.view(-1), descending=True)[0]
            attn_thres = sorted_attn[-int(args.remove_heads)]
            attn_mask = (attn_metric > attn_thres)  # 1 means retain
            if mlp_metric is not None:
                sorted_mlp = torch.sort(mlp_metric.view(-1), descending=True)[0]
                mlp_thres = sorted_mlp[-cal_remove_neuron(args, model)]
                mlp_mask = (mlp_metric > mlp_thres)
            else:
                mlp_mask = None
        else:
            prune_metric = torch.cat(
                [attn_metric.view(-1), mlp_metric.view(-1)]) if mlp_metric is not None else attn_metric.view(-1)
            sorted_prune, indices = torch.sort(prune_metric, descending=True)
            compression_weight = torch.ones_like(indices)
            compression_weight[indices < attn_metric.numel()] = 512.0 / 3
            threshold = sorted_prune[torch.argmin(torch.abs(
                torch.cumsum(compression_weight, 0) - torch.sum(compression_weight) * (1 - args.pruning_ratio)))]
            if const_heads is None:
                attn_mask = (attn_metric > threshold)
                num_layers, num_heads = attn_metric.shape
                for i in range(num_layers):
                    if hasattr(layers[i], 'self_attn') and args.skip_attn:
                        continue
                    current_count = int(attn_mask[i].sum().item())
                    # Ensure at least retain_heads_min are kept.
                    if current_count < retain_heads_min:
                        new_required = retain_heads_min
                    else:
                        new_required = retain_heads_min * math.ceil(current_count / retain_heads_min)

                    # Get the indices of heads sorted by metric in descending order.
                    sorted_indices = torch.argsort(attn_metric[i], descending=True)
                    # Force the top new_required heads to be retained.
                    attn_mask[i, sorted_indices[:new_required]] = True

            else:
                _, sorted_indices = torch.sort(attn_metric, descending=True, dim=1)
                top_indices = sorted_indices[:, :const_heads]
                attn_mask = torch.zeros_like(attn_metric, dtype=torch.bool)
                attn_mask[torch.arange(attn_mask.shape[0]).unsqueeze(1), top_indices] = True


            mlp_mask = (mlp_metric > threshold) if mlp_metric is not None else None
    else:
        attn_mask = torch.stack(attn_mask)
        mlp_mask = torch.stack(mlp_mask) if len(mlp_mask) else None

    for idx in range(len(layers)):
        if args.skip_attn and not hasattr(layers[idx], 'mixer'):
            continue
        if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):
            compress(layers[idx], attn_mask[idx], None, attn_baseline_inp_list[idx], None,
                     model.hf_device_map[f"model.layers.{idx}"], unstr=args.unstr, args=args)
        else:
            compress(layers[idx], attn_mask[idx], None, attn_baseline_inp_list[idx], None, device, unstr=args.unstr, args=args)
        if mlp_mask is not None:
            if f"model.layers.{i}" in getattr(model, 'hf_device_map', {}):
                compress(layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx],
                         model.hf_device_map[f"model.layers.{idx}"], unstr=args.unstr)
            else:
                compress(layers[idx], None, mlp_mask[idx], None, mlp_baseline_inp_list[idx], device, unstr=args.unstr, args=args)

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_wanda_sp(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    Wanda on structured pruning.

    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    use_cache = model.config.use_cache
    model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2", nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen,
                                tokenizer=tokenizer)
    print("dataset loading complete")

    with torch.no_grad():
        calib_inputs = prepare_calibration_input(model, dataloader, device)
        residual = None
        if len(calib_inputs) == 4:
            inps, outs, attention_mask, position_ids = calib_inputs
        else:
            inps, outs, attention_mask, position_ids, residual = calib_inputs

    backbone = model.model if hasattr(model, "model") else model.backbone
    layers = backbone.layers if hasattr(model, "layers") else backbone.model.layers
    for i in range(len(layers)):
        layer = layers[i]
        subset = {}
        if hasattr(layer, 'self_attn') and args.skip_attn:
            continue
        elif hasattr(layer, 'self_attn'):
            o_proj_key = 'self_attn.o_proj' if hasattr(layer.self_attn, 'o_proj') else 'self_attn.dense'
        elif hasattr(layer, 'mamba'):
            o_proj_key = 'mamba.out_proj'
            headdim = layer.mamba.headdim
        else:
            o_proj_key = 'mixer.out_proj'
            headdim = layer.mixer.headdim
        subset.update({o_proj_key: find_layers(layer)[o_proj_key]})
        if (hasattr(layer, 'mlp') and layer.mlp is not None) and not args.skip_mlp:
            mlp_down_proj_key = 'mlp.down_proj' if hasattr(layer.mlp, 'down_proj') else 'mlp.fc2'
            subset.update({mlp_down_proj_key: find_layers(layer)[mlp_down_proj_key]})

        if f"model.layers.{i}" in getattr(model, 'hf_device_map',
                                          {}):  ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(
                dev), position_ids.to(dev)

        wrapped_layers = {}
        for name in subset:
            wrapped_layers[name] = WrappedGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                wrapped_layers[name].add_batch(inp[0].data, out.data)

            return tmp

        handles = []
        for name in wrapped_layers:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(
                wrapped_layers[name].scaler_row.reshape((1, -1)))

            if name in ['self_attn.o_proj', 'self_attn.dense', 'mixer.out_proj', 'mamba.out_proj']:
                W_metric = W_metric.mean(axis=0).reshape(-1, headdim).sum(dim=1)  # importance score of each head
                nheads = layer.self_attn.num_heads if hasattr(layer, 'self_attn') else (
                    layer.mixer.nheads if hasattr(layer.mixer, 'nheads') else layer.mixer.n_v_heads
                )
                thresh = torch.sort(W_metric.cuda())[0][8].cpu()
                # thresh = torch.sort(W_metric.cuda())[0][int(args.pruning_ratio * nheads)].cpu()
                W_mask = (W_metric >= thresh)
                compress(layer, W_mask, None, None, None, device, bias=False, unstr=args.unstr, args=args)
            else:
                if args.skip_mlp:
                    continue
                W_metric = W_metric.mean(axis=0)
                thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel() * args.pruning_ratio)].cpu()
                W_mask = (W_metric >= thresh)
                compress(layer, None, W_mask, None, None, device, bias=False, unstr=args.unstr, args=args)

            wrapped_layers[name].free()

        for j in range(args.nsamples):
            with torch.no_grad():
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        inps, outs = outs, inps  # the pruned output as input to the next layer

        torch.cuda.empty_cache()

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()


def prune_magnitude_sp(args, model, tokenizer, device=torch.device("cuda:0")):
    """
    Magnitude Pruning on structured pruning.
    
    Args:
        args (object): Command line arguments parsed via argparse.
        model (nn.Module): PyTorch model to prune.
        tokenizer (Tokenizer): Tokenizer associated with the model.
        device (torch.device, optional): Device to move tensors to. Defaults to CUDA device 0.
    """
    layers = model.model.layers

    for i in range(len(layers)):
        layer = layers[i]
        subset = {}
        subset.update({'self_attn.o_proj': find_layers(layer)['self_attn.o_proj']})
        subset.update({'mlp.down_proj': find_layers(layer)['mlp.down_proj']})

        for name in subset:
            print(f"pruning layer {i} name {name}")
            W_metric = torch.norm(subset[name].weight.data, dim=0)

            if name == 'self_attn.o_proj':
                W_metric = W_metric.reshape(-1, 128).sum(dim=1)  # importance score of each head
                thresh = torch.sort(W_metric.cuda())[0][int(args.pruning_ratio * layer.self_attn.num_heads)].cpu()
                W_mask = (W_metric >= thresh)
                compress(layer, W_mask, None, None, None, device, bias=False, unstr=args.unstr)
            else:
                thresh = torch.sort(W_metric.cuda())[0][int(W_metric.numel() * args.pruning_ratio)].cpu()
                W_mask = (W_metric >= thresh)
                compress(layer, None, W_mask, None, None, device, bias=False, unstr=args.unstr)
