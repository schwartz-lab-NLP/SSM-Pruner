import os
import sys
import time
import json
import argparse
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add the necessary paths
sys.path.extend(['.', './phi_mamba', './MambaInLlama', './original_mamba'])

from phi_mamba.modules.lm_head import LMHeadModel
from phi_mamba.utils.ppl import get_wikitext_dataloader

def get_gpu_info() -> Dict:
    """Get GPU information for reproducibility."""
    if not torch.cuda.is_available():
        return {"available": False}
    
    return {
        "available": True,
        "name": torch.cuda.get_device_name(0),
        "device_count": torch.cuda.device_count(),
        "device_capability": torch.cuda.get_device_capability(0),
        "cuda_version": torch.version.cuda,
    }

def get_system_info() -> Dict:
    """Get system information for reproducibility."""
    import platform
    import psutil
    
    return {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "cpu_count": os.cpu_count(),
        "memory_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu_info": get_gpu_info(),
    }

def measure_throughput(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    n_runs: int = 10,
    warmup_runs: int = 5,
    use_amp: bool = True,
    batch_size: Optional[int] = None,
    sequence_length: Optional[int] = None,
) -> Tuple[float, float, List[float]]:
    """
    Measure the throughput of a model in tokens per second.
    
    Args:
        model: The model to benchmark
        input_ids: Input token IDs
        attention_mask: Attention mask
        n_runs: Number of runs for benchmarking
        warmup_runs: Number of warmup runs (not counted in benchmarking)
        use_amp: Whether to use automatic mixed precision
        batch_size: Override batch size for reporting (uses input shape if None)
        sequence_length: Override sequence length for reporting (uses input shape if None)
    
    Returns:
        Tuple containing (mean_throughput, std_throughput, all_throughputs)
    """
    device = next(model.parameters()).device
    
    # Ensure inputs are on the correct device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Get actual batch size and sequence length from inputs
    actual_batch_size = input_ids.shape[0]
    actual_seq_len = input_ids.shape[1]
    
    # Use provided batch_size/seq_len for reporting if given, otherwise use actual values
    report_batch_size = batch_size if batch_size is not None else actual_batch_size
    report_seq_len = sequence_length if sequence_length is not None else actual_seq_len
    
    # Calculate total tokens for reporting
    total_tokens = report_batch_size * report_seq_len
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            with autocast(enabled=use_amp):
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    # Benchmark runs
    throughputs = []
    with torch.no_grad():
        for _ in range(n_runs):
            start_time = time.time()
            with autocast(enabled=use_amp):
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            duration = end_time - start_time
            throughput = total_tokens / duration
            throughputs.append(throughput)
    
    return np.mean(throughputs), np.std(throughputs), throughputs

def measure_generation_throughput(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    generate_length: int,
    n_runs: int = 5,
    warmup_runs: int = 2,
    use_amp: bool = True,
) -> Tuple[float, float, List[float]]:
    """
    Measure the generation throughput of a model in tokens per second.
    
    Args:
        model: The model to benchmark
        input_ids: Input token IDs
        attention_mask: Attention mask
        generate_length: Number of tokens to generate
        n_runs: Number of runs for benchmarking
        warmup_runs: Number of warmup runs (not counted in benchmarking)
        use_amp: Whether to use automatic mixed precision
    
    Returns:
        Tuple containing (mean_throughput, std_throughput, all_throughputs)
    """
    device = next(model.parameters()).device
    
    # Ensure inputs are on the correct device
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    
    # Warmup runs
    with torch.no_grad():
        for _ in range(warmup_runs):
            with autocast(enabled=use_amp):
                _ = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=generate_length,
                    do_sample=False
                )
    
    # Synchronize before timing
    torch.cuda.synchronize()
    
    # Benchmark runs
    throughputs = []
    with torch.no_grad():
        for _ in range(n_runs):
            start_time = time.time()
            with autocast(enabled=use_amp):
                _ = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=generate_length,
                    do_sample=False
                )
            
            torch.cuda.synchronize()
            end_time = time.time()
            
            duration = end_time - start_time
            throughput = generate_length / duration
            throughputs.append(throughput)
    
    return np.mean(throughputs), np.std(throughputs), throughputs

def calculate_parameter_count(model: torch.nn.Module) -> Dict[str, int]:
    """
    Calculate the number of parameters in the model.
    
    Args:
        model: The model to count parameters for
        
    Returns:
        Dict containing total, trainable, and non-trainable parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        "total": total_params,
        "trainable": trainable_params,
        "non_trainable": total_params - trainable_params
    }

def measure_model_sparsity(model: torch.nn.Module) -> float:
    """
    Measure the sparsity of the model weights.
    
    Args:
        model: The model to measure sparsity for
        
    Returns:
        Percentage of zero-valued parameters
    """
    total_params = 0
    zero_params = 0
    
    for param in model.parameters():
        total_params += param.numel()
        zero_params += (param == 0).sum().item()
    
    return (zero_params / total_params) * 100 if total_params > 0 else 0

def run_benchmark(
    model_path: str,
    batch_sizes: List[int] = [1, 4, 16],
    sequence_lengths: List[int] = [128, 512, 1024],
    generation_lengths: List[int] = [128],
    use_amp: bool = True,
    precision: torch.dtype = torch.bfloat16,
    output_file: Optional[str] = None,
) -> Dict:
    """
    Run a comprehensive throughput benchmark on a model.
    
    Args:
        model_path: Path to the model to benchmark
        batch_sizes: List of batch sizes to benchmark
        sequence_lengths: List of sequence lengths to benchmark
        generation_lengths: List of generation lengths to benchmark
        use_amp: Whether to use automatic mixed precision
        precision: Precision to use for the model
        output_file: Path to save results (optional)
        
    Returns:
        Dict containing benchmark results
    """
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    print(f"Loading model from {model_path}")
    model = LMHeadModel.from_pretrained(model_path, device=device, dtype=precision)
    model.eval()
    model.to(device)
    
    # Record model metadata
    results = {
        "model_path": model_path,
        "system_info": get_system_info(),
        "parameters": calculate_parameter_count(model),
        "sparsity": measure_model_sparsity(model),
        "batch_inference": {},
        "generation": {},
        "test_params": {
            "use_amp": use_amp,
            "precision": str(precision),
        }
    }
    
    # Batch inference benchmarks
    print("Running batch inference benchmarks...")
    for batch_size in batch_sizes:
        for seq_len in sequence_lengths:
            print(f"  Benchmarking batch_size={batch_size}, seq_len={seq_len}")
            
            # Create random input data of the appropriate shape
            input_ids = torch.randint(0, model.config.input.vocab_size, (batch_size, seq_len), device=device)
            attention_mask = torch.ones_like(input_ids)
            
            # Benchmark
            mean_throughput, std_throughput, all_throughputs = measure_throughput(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_amp=use_amp,
                batch_size=batch_size,
                sequence_length=seq_len,
            )
            
            # Record results
            key = f"bs{batch_size}_seq{seq_len}"
            results["batch_inference"][key] = {
                "batch_size": batch_size,
                "sequence_length": seq_len,
                "throughput_mean": mean_throughput,
                "throughput_std": std_throughput,
                "throughput_list": all_throughputs,
                "throughput_unit": "tokens/second",
            }
    
    # Generation benchmarks
    print("Running generation benchmarks...")
    for gen_len in generation_lengths:
        print(f"  Benchmarking generation_length={gen_len}")
        
        # Generation typically uses small batch sizes
        batch_size = 1
        context_len = 32  # Short context for generation benchmarks
        
        # Create random input data
        input_ids = torch.randint(0, model.config.MixerModel.input.vocab_size, (batch_size, context_len), device=device)
        attention_mask = torch.ones_like(input_ids)
        
        # Benchmark
        mean_throughput, std_throughput, all_throughputs = measure_generation_throughput(
            model=model,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generate_length=gen_len,
            use_amp=use_amp,
        )
        
        # Record results
        key = f"gen{gen_len}"
        results["generation"][key] = {
            "batch_size": batch_size,
            "context_length": context_len,
            "generation_length": gen_len,
            "throughput_mean": mean_throughput,
            "throughput_std": std_throughput,
            "throughput_list": all_throughputs,
            "throughput_unit": "tokens/second",
        }
    
    # Save results if output file is specified
    if output_file:
        os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_file}")
    
    return results

def plot_comparison(
    results_files: List[str],
    labels: List[str],
    output_path: str = "throughput_comparison.png",
):
    """
    Plot a comparison of throughput results from multiple benchmark runs.
    
    Args:
        results_files: List of paths to result JSON files
        labels: List of labels for each result file
        output_path: Path to save the plot
    """
    if len(results_files) != len(labels):
        raise ValueError("Number of result files must match number of labels")
    
    results_list = []
    for file_path in results_files:
        with open(file_path, "r") as f:
            results_list.append(json.load(f))
    
    # Extract batch inference results
    batch_configs = set()
    for results in results_list:
        batch_configs.update(results["batch_inference"].keys())
    
    batch_configs = sorted(batch_configs)
    
    # Plot batch inference throughput
    fig, ax = plt.subplots(figsize=(12, 8))
    
    bar_width = 0.35
    positions = np.arange(len(batch_configs))
    
    for i, (results, label) in enumerate(zip(results_list, labels)):
        throughputs = []
        errors = []
        
        for config in batch_configs:
            if config in results["batch_inference"]:
                throughputs.append(results["batch_inference"][config]["throughput_mean"])
                errors.append(results["batch_inference"][config]["throughput_std"])
            else:
                throughputs.append(0)
                errors.append(0)
        
        ax.bar(
            positions + i * bar_width,
            throughputs,
            bar_width,
            yerr=errors,
            label=label,
            capsize=5,
        )
    
    # Update appearance
    ax.set_ylabel("Throughput (tokens/second)")
    ax.set_title("Batch Inference Throughput Comparison")
    ax.set_xticks(positions + bar_width / 2)
    ax.set_xticklabels(batch_configs)
    ax.legend()
    
    # Save plot
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Comparison plot saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Benchmark model throughput")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model to benchmark")
    parser.add_argument("--output_file", type=str, default=None, help="Path to save benchmark results")
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 4, 16], help="Batch sizes to benchmark")
    parser.add_argument("--sequence_lengths", type=int, nargs="+", default=[128, 512, 1024], help="Sequence lengths to benchmark")
    parser.add_argument("--generation_lengths", type=int, nargs="+", default=[128], help="Generation lengths to benchmark")
    parser.add_argument("--no_amp", action="store_true", help="Disable automatic mixed precision")
    parser.add_argument("--precision", type=str, choices=["float32", "float16", "bfloat16"], default="bfloat16", help="Model precision")
    
    # For comparison plotting
    parser.add_argument("--compare", action="store_true", help="Compare multiple benchmark results")
    parser.add_argument("--compare_files", type=str, nargs="+", default=[], help="Result files to compare")
    parser.add_argument("--compare_labels", type=str, nargs="+", default=[], help="Labels for comparison")
    parser.add_argument("--compare_output", type=str, default="throughput_comparison.png", help="Output path for comparison plot")
    
    args = parser.parse_args()
    
    # Handle precision
    precision_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    precision = precision_map[args.precision]
    
    # Compare mode
    if args.compare:
        if not args.compare_files:
            parser.error("--compare_files is required when --compare is set")
        if not args.compare_labels:
            parser.error("--compare_labels is required when --compare is set")
            
        plot_comparison(
            results_files=args.compare_files,
            labels=args.compare_labels,
            output_path=args.compare_output,
        )
    # Benchmark mode
    else:
        run_benchmark(
            model_path=args.model_path,
            batch_sizes=args.batch_sizes,
            sequence_lengths=args.sequence_lengths,
            generation_lengths=args.generation_lengths,
            use_amp=not args.no_amp,
            precision=precision,
            output_file=args.output_file,
        )

if __name__ == "__main__":
    main() 