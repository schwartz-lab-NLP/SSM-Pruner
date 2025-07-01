import torch
import torch.nn as nn
from .sparsegpt import SparseGPT
from .layerwrapper import WrappedGPT
from .data import get_loaders 
from .ablate import AblateGPT


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

def check_sparsity(model):

    use_cache = model.config.use_cache 
    model.config.use_cache = False 
    backbone = model.model if hasattr(model, "model") else model.backbone
    layers = backbone.layers if hasattr(backbone, "layers") else backbone.model.layers
    count = 0 
    total_params = 0
    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        sub_count = 0
        sub_params = 0
        for name in subset:
            W = subset[name].weight.data
            count += (W==0).sum().item()
            total_params += W.numel()

            sub_count += (W==0).sum().item()
            sub_params += W.numel()

        print(f"layer {i} sparsity {float(sub_count)/sub_params:.6f}")

    model.config.use_cache = use_cache 
    return float(count)/total_params 

def prepare_calibration_input(model, dataloader, device):
    if hasattr(model.config, "use_cache"):
        use_cache = model.config.use_cache
        model.config.use_cache = False

    backbone = model.model if hasattr(model, "model") else model.backbone
    layers = backbone.layers if hasattr(backbone, "layers") else backbone.model.layers

    # dev = model.hf_device_map["model.embed_tokens"]
    if ( hasattr(model, "hf_device_map") and
            "model.embed_tokens" in model.hf_device_map):
        device = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128,
                        model.seqlen,
                        model.config.hidden_size if hasattr(model.config, "hidden_size") else (model.config.MixerModel.input.d_model if hasattr(model.config, "MixerModel") else model.config.d_model)
                        ), dtype=dtype, device=device)
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

    return inps, outs, attention_mask, position_ids if residual is None else (inps, outs, attention_mask, position_ids, residual)

def return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before):
    thres_cumsum = sum_before * alpha 
    sort_mask = tmp_metric <= thres_cumsum.reshape((-1,1))
    thres = torch.gather(sort_res[0], dim=1, index=sort_mask.sum(dim=1, keepdims=True)-1)
    W_mask = (W_metric <= thres)
    cur_sparsity = (W_mask==True).sum() / W_mask.numel()
    return W_mask, cur_sparsity

def prune_magnitude(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    layers = model.model.layers 

    for i in range(len(layers)):
        layer = layers[i]
        subset = find_layers(layer)

        for name in subset:
            W = subset[name].weight.data 
            W_metric = torch.abs(W)
            if prune_n != 0:
                W_mask = (torch.zeros_like(W)==1)
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                thresh = torch.sort(W_metric.flatten().cuda())[0][int(W.numel()*args.sparsity_ratio)].cpu()
                W_mask = (W_metric<=thresh)

            W[W_mask] = 0


def prune_wanda(args, model, tokenizer, device=torch.device("cuda:0"), prune_n=0, prune_m=0):
    if hasattr(model.config, "use_cache"):
        use_cache = model.config.use_cache
        model.config.use_cache = False

    print("loading calibdation data")
    dataloader, _ = get_loaders("wikitext2",
                                nsamples=args.nsamples,
                                seed=args.seed,
                                seqlen=model.seqlen,
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
    layers = backbone.layers if hasattr(backbone, "layers") else backbone.model.layers
    plots = {i:{
        'in_proj_magnitude':None,
        'in_proj_wanda_mask':None,
        'out_proj_magnitude':None,
        'out_proj_wanda_mask':None,
    } for i in range(len(layers))} if args.plot else None

    for i in range(len(layers)):

        layer = layers[i]
        subset = find_layers(layer)

        if hasattr(model, "hf_device_map") and f"model.layers.{i}" in model.hf_device_map:   ## handle the case for llama-30B and llama-65B, when the device map has multiple GPUs;
            dev = model.hf_device_map[f"model.layers.{i}"]
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)
            if residual is not None:
                residual = residual.to(dev)

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
                if residual is None:
                    if args.is_mamba_in_llama:
                        position_embeddings = (backbone.model if hasattr(backbone, 'model') else backbone).rotary_emb(inps[j].unsqueeze(0), position_ids)
                        tmp_out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings )
                    else:
                        tmp_out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)
                else:
                    tmp_out, residual = layer(inps[j].unsqueeze(0), residual, attention_mask=attention_mask, position_ids=position_ids)
                outs[j] = tmp_out[0] if (not args.is_lm_head and not args.is_llamba) else tmp_out['hidden_states']
        for h in handles:
            h.remove()

        mamba_indices = {}
        if args.is_mamba and not args.split_mamba and not args.is_llamba:
            mamba_indices = layer.mixer.get_in_proj_indices() if hasattr(layer, "mixer") else {}
        for name in subset:
            excluded = ['fc', 'up_proj', 'down_proj', 'gate_proj', 'mlp','out_proj']
            if any([ex in name for ex in excluded]):
                continue
            print(f"pruning layer {i} name {name}")
            W_metric = torch.abs(subset[name].weight.data) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1,-1)))

            W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False



            if prune_n != 0:
                # structured n:m sparsity
                for ii in range(W_metric.shape[1]):
                    if ii % prune_m == 0:
                        tmp = W_metric[:,ii:(ii+prune_m)].float()
                        W_mask.scatter_(1,ii+torch.topk(tmp, prune_n,dim=1, largest=False)[1], True)
            else:
                sort_res = torch.sort(W_metric, dim=-1, stable=True)

                if args.use_variant:
                    # wanda variant 
                    tmp_metric = torch.cumsum(sort_res[0], dim=1)
                    sum_before = W_metric.sum(dim=1)

                    alpha = 0.4
                    alpha_hist = [0., 0.8]
                    W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    while (torch.abs(cur_sparsity - args.sparsity_ratio)>0.001) and (alpha_hist[1]-alpha_hist[0]>=0.001):
                        if cur_sparsity > args.sparsity_ratio:
                            alpha_new = (alpha + alpha_hist[0]) / 2.0
                            alpha_hist[1] = alpha
                        else:
                            alpha_new = (alpha + alpha_hist[1]) / 2.0
                            alpha_hist[0] = alpha

                        alpha = alpha_new 
                        W_mask, cur_sparsity = return_given_alpha(alpha, sort_res, W_metric, tmp_metric, sum_before)
                    print(f"alpha found {alpha} sparsity {cur_sparsity:.6f}")
                else:
                    # unstructured pruning
                    indices = sort_res[1][:,:int(W_metric.shape[1]*args.sparsity_ratio)]
                    W_mask.scatter_(1, indices, True)

            if args.plot:
                kn = 'in_proj' if 'in_proj' in name else 'out_proj'
                plots[i][f"{kn}_magnitude"] = W_metric.cpu().numpy()
                plots[i][f"{kn}_wanda_mask"] = W_mask.cpu().numpy()

            if args.s_prune:
                if 'out_proj' in name:
                    continue
                layer.mixer.prune_headdim(args.sparsity_ratio, 'wanda', W_mask=W_mask, only_nullify=True, exclude_out_proj=True)
                # if 'in_proj' in name:
                #     # if i ==0:
                #     #     continue
                #     # subset[name].weight.data[W_mask] = 0
                #     # subset[name].weight.data = subset[name].weight.data[torch.where(W_mask)].reshape(-1, subset[name].weight.data.shape[1])
                #     import pdb; pdb.set_trace()
                #     subset[name].weight.data = subset[name].weight.data[torch.where(W_mask)[0]]
                #     subset[name].out_features = subset[name].weight.data.shape[0]
                #     subset[name].in_features = subset[name].weight.data.shape[1]
                #     # llayer.conv1d.weight.data = llayer.conv1d.weight.data

            else:
                # Order in in_proj: [z, x, B, C, dt]
                if 'in_proj' in name:
                    exluded_in_proj_parts = []
                    for exluded_in_proj_part in exluded_in_proj_parts:
                        start, end = mamba_indices[exluded_in_proj_part]
                        W_mask[start:end, :] = 0
                
                subset[name].weight.data[W_mask] = 0  ## set weights to zero
                




        for j in range(args.nsamples):
            with torch.no_grad():
                if residual is None:
                    if args.is_mamba_in_llama:
                        position_embeddings = (backbone.model if hasattr(backbone, 'model') else backbone).rotary_emb(inps[j].unsqueeze(0), position_ids)
                        tmp_out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids, position_embeddings=position_embeddings )
                    else:
                        tmp_out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)
                else:
                    tmp_out, residual = layer(inps[j].unsqueeze(0), residual, attention_mask=attention_mask, position_ids=position_ids)
                
                outs[j] = tmp_out[0] if (not args.is_lm_head and not args.is_llamba) else tmp_out['hidden_states']
        inps, outs = outs, inps
    if hasattr(model.config, "use_cache"):
        model.config.use_cache = use_cache
    torch.cuda.empty_cache()




@torch.no_grad()
def prune_sparsegpt(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False

    backbone = model.model if hasattr(model, "model") else model.backbone
    layers = backbone.layers if hasattr(backbone, "layers") else backbone.model.layers

    if hasattr(model, "hf_device_map") and "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
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
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']
    residual = cache['residual']
    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if hasattr(model, "hf_device_map") and f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = SparseGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            if residual is None:
                tmp_out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)
            else:
                tmp_out, residual = layer(inps[j].unsqueeze(0), residual, attention_mask=attention_mask,
                                          position_ids=position_ids)
            outs[j] = tmp_out[0] if (not args.is_lm_head and not args.is_llamba) else tmp_out['hidden_states']

        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            gpts[name].fasterprune(args.sparsity_ratio, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            if residual is None:
                tmp_out = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)
            else:
                tmp_out, residual = layer(inps[j].unsqueeze(0), residual, attention_mask=attention_mask,
                                          position_ids=position_ids)
            outs[j] = tmp_out[0] if (not args.is_lm_head and not args.is_llamba) else tmp_out['hidden_states']

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()



@torch.no_grad()
def prune_ablate(args, model, tokenizer, dev, prune_n=0, prune_m=0):
    ## SparseGPT code available at: https://github.com/IST-DASLab/sparsegpt/tree/f5c25005a61f96a0933ca2f95705a963585aafaa
    print('Starting ...')
    dataloader, _ = get_loaders("c4",nsamples=args.nsamples,seed=args.seed,seqlen=model.seqlen,tokenizer=tokenizer)

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.layers

    if "model.embed_tokens" in model.hf_device_map:
        dev = model.hf_device_map["model.embed_tokens"]

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            cache['attention_mask'] = kwargs['attention_mask']
            cache['position_ids'] = kwargs['position_ids']
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    layers[0] = layers[0].module
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache['attention_mask']
    position_ids = cache['position_ids']

    print('Ready.')

    for i in range(len(layers)):
        layer = layers[i]
        if f"model.layers.{i}" in model.hf_device_map:
            dev = model.hf_device_map[f"model.layers.{i}"]
            print(f"layer {i} device {dev}")
            inps, outs, attention_mask, position_ids = inps.to(dev), outs.to(dev), attention_mask.to(dev), position_ids.to(dev)

        subset = find_layers(layer)

        gpts = {}
        for name in subset:
            gpts[name] = AblateGPT(subset[name])

        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch(inp[0].data, out.data)
            return tmp

        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]
        for h in handles:
            h.remove()

        for name in gpts:
            print(i, name)
            print('Pruning ...')

            if args.prune_method == "ablate_wanda_seq":
                prune_mask = gpts[name].get_wanda_mask(args.sparsity_ratio, prune_n, prune_m)
            elif args.prune_method == "ablate_mag_seq":
                prune_mask = gpts[name].get_mag_mask(args.sparsity_ratio, prune_n, prune_m)
            elif "iter" in args.prune_method:
                prune_mask = None 

            gpts[name].fasterprune(args, args.sparsity_ratio, mask=prune_mask, prune_n=prune_n, prune_m=prune_m, percdamp=0.01, blocksize=128)
            gpts[name].free()

        for j in range(args.nsamples):
            outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[0]

        layers[i] = layer 
        torch.cuda.empty_cache()

        inps, outs = outs, inps

    model.config.use_cache = use_cache
    torch.cuda.empty_cache()