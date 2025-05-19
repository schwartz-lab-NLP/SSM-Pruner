from typing import Any

import sys
sys.path.append("edge")
sys.path.append("phi_mamba")

import torch
from datasets import load_dataset
from lm_eval import simple_evaluate
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoTokenizer
from phi_mamba.modules.lm_head import LMHeadModel


def get_wikitext_dataloader(percentage: int, seq_len: int = 2048, tokenizer_path: str='microsoft/phi-1_5', split='test', batch:int = 1,
                            return_iter_dataset=False, dataset_path='wikitext', dataset_name='wikitext-2-raw-v1', world_size=1, rank=0, streaming=False, return_dataset=False) -> \
DataLoader[Any] | IterableDataset:
    # Load a subset of the Wikitext dataset
    dataset = load_dataset(dataset_path, dataset_name, split=f'{split}[:{percentage}%]', streaming=streaming)
    if world_size > 1:
        dataset = dataset.shard(num_shards=world_size, index=rank)
    # Tokenizer for 'microsoft/phi-1_5'
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token

    # Tokenize each text individually and collect input IDs
    texts = dataset['text']
    input_ids_list = []
    for text in texts:
        # Skip empty strings to avoid issues
        if text.strip() == '':
            continue
        tokenized_text = tokenizer.encode(text)
        input_ids_list.extend(tokenized_text)

    # Convert the list of token IDs to a tensor
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)

    # Chunk into sequences of 2048 tokens
    max_length = seq_len
    total_length = input_ids.size(0)
    num_chunks = (total_length + max_length - 1) // max_length  # Ceiling division
    chunks = []
    for i in range(num_chunks):
        start_idx = i * max_length
        end_idx = min(start_idx + max_length, total_length)
        chunk = input_ids[start_idx:end_idx]
        # Pad the last chunk if necessary
        if chunk.size(0) < max_length:
            padding_length = max_length - chunk.size(0)
            padding = torch.full((padding_length,), tokenizer.pad_token_id, dtype=torch.long)
            chunk = torch.cat([chunk, padding], dim=0)
        chunks.append(chunk)

    if not return_iter_dataset:
        # Create dataset and dataloader
        class ChunkedDataset(torch.utils.data.Dataset):
            def __init__(self, chunks):
                self.chunks = chunks

            def __len__(self):
                return len(self.chunks)

            def __getitem__(self, idx):
                return {'input_ids': self.chunks[idx],
                        'attention_mask': (self.chunks[idx] != tokenizer.pad_token_id).long()}

        dataset = ChunkedDataset(chunks)
        return DataLoader(dataset, batch_size=batch) if not return_dataset else dataset

    class IterChunkedDataset(torch.utils.data.IterableDataset):
        def __init__(self, chunks):
            self.chunks = chunks

        def __len__(self):
            return len(self.chunks)

        def __iter__(self):
            for chunk in self.chunks:
                yield {'input_ids': chunk,
                       'attention_mask': (chunk != tokenizer.pad_token_id).long()}

    dataset = IterChunkedDataset(chunks)


    return dataset


def evaluate_wikitext(model, use_cache=False, past_key_values=None, pass_attention_mask=True, tokenizer_path="microsoft/phi-1_5", dataset_name='wikitext-2-raw-v1'):

    dataloader = get_wikitext_dataloader(percentage=100, tokenizer_path=tokenizer_path, dataset_name=dataset_name)
    # evaluate the pruned model
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
    model = model.to(device).eval()
    losses = []
    cnt = 0
    with torch.no_grad():
        for data in dataloader:
            cnt += 1
            inp = data['input_ids'].to(device)
            result = model(input_ids=inp,
                           attention_mask=data['attention_mask'].to(device) if pass_attention_mask else None,
                           use_cache=use_cache,
                           past_key_values=past_key_values)
            if hasattr(result, 'logits'):
                logits = result.logits
                logits = logits.to(device)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = inp[:, 1:].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
                loss = loss_fct(shift_logits.reshape(-1, shift_logits.size(-1)), shift_labels.view(-1))
            else:
                loss = result[0] if isinstance(result, tuple) else result.loss
            losses.append(loss)
            # if cnt % 100 == 0:
            #     print(f'Processed {cnt} examples')
            #     print(f'past_key_values: {past_key_values}')

        loss = torch.cat(losses, dim=-1).mean()
        print(f'{dataset_name}. Loss: {loss.item()}')
        ppl = torch.exp(loss).item()
        print(f'{dataset_name} Perplexity: {ppl}')
    return ppl


def generate_text(prompt, device='cuda', model_path="goombalab/Phi-Mamba", tokenizer_path="microsoft/phi-1_5"):
    model = LMHeadModel.from_pretrained(model_path, strict=True) if type(model_path) is str else model_path

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # tokenizer.pad_token = tokenizer.eos_token
    model = model.to(device).eval()

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    print('start generating')
    generation_output = model.generate(
        input_ids=input_ids.to(device),
        top_k=50,
        max_length= 7,
    )
    print(generation_output)
    # generated_text = tokenizer.decode(generation_output[0], skip_special_tokens=True)
    return generation_output


def evaluate_with_lm_eval_harness(
    model_or_path,
    benchmarks=["lambada_openai","hellaswag","piqa","arc_easy","arc_challenge","winogrande"],
    limit=None,
    batch_size=64,
    device="cuda",
    tokenizer_path=None,
    no_cache=False,
    cache_dir=None,
    verbose=True,
    num_fewshot=0,
    model_type="hf",  # Default to HuggingFace models when using a string path
    use_accelerate=False  # Whether to use Accelerate for model loading
):
    """
    Evaluate a model using lm-eval-harness on specified benchmarks.
    
    Args:
        model_or_path: Either a model instance or a string path to a HuggingFace model
        benchmarks: List of benchmark names to evaluate on
        limit: Optional limit on number of examples to use
        batch_size: Batch size for evaluation
        device: Device to run evaluation on ('cuda', 'cpu', 'mps')
        tokenizer_path: Path to tokenizer (if None, uses model_or_path) or tokenizer object
        no_cache: Whether to disable caching
        cache_dir: Directory to use for caching (if None, uses default)
        verbose: Whether to print evaluation progress
        num_fewshot: Number of examples to use for few-shot learning (0 for zero-shot)
        model_type: Type of model to use when using a string path ('hf', 'vllm', 'mamba_ssm', etc.)
        use_accelerate: Whether to use Accelerate for model loading
        
    Returns:
        results: Dictionary containing evaluation results
    """
    try:
        from lm_eval import evaluator
        simple_evaluate = evaluator.simple_evaluate
    except ImportError:
        print("lm-eval-harness not found. Please install with: pip install lm-eval")
        return None
    
    # Handle use_cache parameter
    import inspect
    sig = inspect.signature(simple_evaluate)
    cache_param = None
    if 'use_cache' in sig.parameters:
        # use_cache expects a path string or None, not a boolean
        cache_param = None if no_cache else (cache_dir or "lm_cache")
        
    # Determine if we're using a string path or a model object
    using_model_object = not isinstance(model_or_path, str)
    
    # Basic evaluation parameters that are common across versions
    eval_params = {
        "tasks": benchmarks,
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "device": device,
        "limit": limit,
    }
    
    # Add use_cache parameter if available
    if cache_param is not None:
        eval_params["use_cache"] = cache_param
    
    # Get tokenizer path or object
    if not tokenizer_path and not using_model_object:
        tokenizer_path = model_or_path
    
    # Import necessary modules for handling models and tokenizers
    from transformers import PreTrainedModel, PreTrainedTokenizer, AutoTokenizer
    
    # Handle model configuration differently based on whether we have a string or object
    if using_model_object:
        # For lm-eval 0.4.8, we need to use a temporary file approach
        # First, let's import necessary modules
        import tempfile
        import os
        import shutil
        
        # Create a temporary directory to save the model
        temp_dir = tempfile.mkdtemp()
        temp_model_path = os.path.join(temp_dir, "model")
        
        try:
            if verbose:
                print("Using temporary file approach for model object with lm-eval 0.4.8")
                
            # Save the model to the temporary directory
            model = model_or_path
            model.save_pretrained(temp_model_path)
            
            # Handle tokenizer based on its type
            if tokenizer_path is not None:
                # Check if it's a string path or a tokenizer object
                if isinstance(tokenizer_path, str):
                    temp_tokenizer_path = tokenizer_path
                elif hasattr(tokenizer_path, "save_pretrained"):
                    # It's a tokenizer object, save it to temp dir
                    temp_tokenizer_dir = os.path.join(temp_dir, "tokenizer")
                    tokenizer_path.save_pretrained(temp_tokenizer_dir)
                    temp_tokenizer_path = temp_tokenizer_dir
                else:
                    raise ValueError("Tokenizer path must be a string or tokenizer object with save_pretrained method")
            else:
                # Try to infer from model config
                try:
                    if hasattr(model, "config") and hasattr(model.config, "_name_or_path"):
                        model_name = model.config._name_or_path
                        if model_name:
                            temp_tokenizer_path = model_name
                        else:
                            raise ValueError("No tokenizer path specified and couldn't infer from model")
                    else:
                        raise ValueError("Model object does not have required attributes to infer tokenizer")
                except Exception as e:
                    raise ValueError(f"No tokenizer path specified for model object: {e}")
            
            # Set up parameters for model path
            model_args_parts = [f"pretrained={temp_model_path}"]
            
            # Add tokenizer if specified
            model_args_parts.append(f"tokenizer={temp_tokenizer_path}")
            
            # Join all parts with commas
            model_args = ",".join(model_args_parts)
            
            # Set model type and args in eval_params
            eval_params["model"] = model_type
            eval_params["model_args"] = model_args
            
            if verbose:
                print(f"Using model object (saved temporarily at {temp_model_path})")
        except Exception as e:
            shutil.rmtree(temp_dir)
            raise ValueError(f"Failed to save model object for evaluation: {e}")
    else:
        # When a string path is passed, use model_type and model_args
        # Build model_args string without use_accelerate
        model_args_parts = [f"pretrained={model_or_path}"]
        
        # Only add use_accelerate if specified
        if use_accelerate:
            model_args_parts.append("use_accelerate=True")
            
        # Add tokenizer if specified
        if tokenizer_path and tokenizer_path != model_or_path:
            if isinstance(tokenizer_path, str):
                model_args_parts.append(f"tokenizer={tokenizer_path}")
            elif hasattr(tokenizer_path, "save_pretrained"):
                # It's a tokenizer object, save it to a temporary directory
                import tempfile
                import os
                import shutil
                
                temp_dir = tempfile.mkdtemp()
                temp_tokenizer_dir = os.path.join(temp_dir, "tokenizer")
                
                try:
                    tokenizer_path.save_pretrained(temp_tokenizer_dir)
                    model_args_parts.append(f"tokenizer={temp_tokenizer_dir}")
                except Exception as e:
                    shutil.rmtree(temp_dir)
                    raise ValueError(f"Failed to save tokenizer object: {e}")
            else:
                raise ValueError("Tokenizer must be a string path or tokenizer object with save_pretrained method")
            
        # Join all parts with commas
        model_args = ",".join(model_args_parts)
        
        # Set model type and args in eval_params
        eval_params["model"] = model_type
        eval_params["model_args"] = model_args
    
    # Run evaluation
    print(f"Starting evaluation on benchmarks: {', '.join(benchmarks)}")
    try:
        if verbose:
            print(f"Using parameters: {eval_params}")
        
        results = simple_evaluate(**eval_params)
        
        # Print summarized results
        if verbose and results and "results" in results:
            benchmarks = benchmarks.split(",") if isinstance(benchmarks, str) else benchmarks
            for benchmark in benchmarks:
                if benchmark in results["results"]:
                    print(f"\n{benchmark} results:")
                    for metric, value in results["results"][benchmark].items():
                        # Format value based on its type
                        if isinstance(value, (float, int)):
                            print(f"  {metric}: {value:.4f}")
                        else:
                            print(f"  {metric}: {value}")
        
        # Clean up temp directory if it exists
        if using_model_object and 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
        elif not using_model_object and isinstance(tokenizer_path, PreTrainedTokenizer) and 'temp_dir' in locals():
            shutil.rmtree(temp_dir)
            
        return results
    except Exception as e:
        # Handle any errors gracefully
        import traceback
        traceback.print_exc()
        
        print(f"Error running evaluation: {e}")
        print("This could be due to version mismatch in lm-eval-harness.")
        
        # Print actual supported parameters
        print("\nSupported parameters for simple_evaluate:")
        for param in sig.parameters:
            print(f"  - {param}")
            
        # Clean up temp directory if it exists
        if 'temp_dir' in locals():
            import shutil
            shutil.rmtree(temp_dir)
            
        return None


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate a model using lm-eval-harness")
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM2-1.7B", 
                        help="Path to model or model name on HuggingFace")
    parser.add_argument("--benchmarks", nargs="+", default=["lambada_openai","hellaswag","piqa","arc_easy","arc_challenge","winogrande"], 
                        help="List of benchmarks to evaluate on")
    parser.add_argument("--limit", type=int, default=100, 
                        help="Limit number of examples per benchmark")
    parser.add_argument("--batch-size", type=int, default=1, 
                        help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to run evaluation on (cuda, cpu, mps)")
    parser.add_argument("--num-fewshot", type=int, default=0, 
                        help="Number of examples to use for few-shot learning")
    parser.add_argument("--model-type", type=str, default="hf", 
                        help="Type of model to use when using a string path (hf, vllm, mamba_ssm, etc.)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer (if different from model)")
    parser.add_argument("--use-accelerate", action="store_true", 
                        help="Whether to use Accelerate for model loading")
    parser.add_argument("--no-cache", action="store_true",
                        help="Disable caching of model outputs")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="Directory to use for caching (if None, uses default)")
    parser.add_argument("--example", action="store_true",
                        help="Run an example with a model object instead of loading from path")
    
    args = parser.parse_args()
    
    # Detect available device if cuda not available
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        if torch.backends.mps.is_available():
            args.device = "mps"
            print("CUDA not available, using MPS device")
        else:
            args.device = "cpu"
            print("CUDA and MPS not available, using CPU")
    
    # Example of using with a model object
    if args.example:
        print("Running example with model object...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        # Load the model and tokenizer
        model = AutoModelForCausalLM.from_pretrained(args.model)
        tokenizer_path = args.tokenizer or args.model
        
        # Evaluate with the model object
        results = evaluate_with_lm_eval_harness(
            model_or_path=model,  # Pass model object directly
            benchmarks=args.benchmarks,
            limit=args.limit,
            batch_size=args.batch_size,
            device=args.device,
            tokenizer_path=tokenizer_path,
            num_fewshot=args.num_fewshot,
            no_cache=args.no_cache,
            cache_dir=args.cache_dir,
        )
    else:
        # Normal usage with model path
        print(f"Evaluating model {args.model} on benchmarks {args.benchmarks}")
        print(f"Using device: {args.device}")
        print(f"Model type: {args.model_type}")
        print(f"Using accelerate: {args.use_accelerate}")
        print(f"Caching: {'disabled' if args.no_cache else 'enabled'}")
        
        results = evaluate_with_lm_eval_harness(
            model_or_path=args.model,
            benchmarks=args.benchmarks,
            limit=args.limit,
            batch_size=args.batch_size,
            device=args.device,
            tokenizer_path=args.tokenizer,
            num_fewshot=args.num_fewshot,
            model_type=args.model_type,
            use_accelerate=args.use_accelerate,
            no_cache=args.no_cache,
            cache_dir=args.cache_dir,
        )
    
    if results is None:
        print("Evaluation failed. Please check errors above.")
    else:
        print("\nFull results:")
        if "results" in results:
            for benchmark, metrics in results["results"].items():
                print(f"\n{benchmark}:")
                for metric, value in metrics.items():
                    # Format value based on its type
                    if isinstance(value, (float, int)):
                        print(f"  {metric}: {value:.4f}")
                    else:
                        print(f"  {metric}: {value}")


# Example usage from Python code with a model object:
"""
from transformers import AutoModelForCausalLM, AutoTokenizer
from phi_mamba.utils.ppl import evaluate_with_lm_eval_harness

# Load your model
model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer_path = "gpt2"

# For custom models like LMHeadModel
# from phi_mamba.modules.lm_head import LMHeadModel
# model = LMHeadModel.from_pretrained("path/to/model")
# tokenizer_path = "microsoft/phi-1_5"

# Evaluate the model
results = evaluate_with_lm_eval_harness(
    model_or_path=model,  # Pass the model object directly
    benchmarks=["hellaswag", "winogrande"],
    limit=100,  # Optional limit on examples
    batch_size=8,
    device="cuda",  # or "cpu", "mps"
    tokenizer_path=tokenizer_path,
    num_fewshot=0,
    no_cache=False,
)

# Analyze results
for benchmark, metrics in results["results"].items():
    print(f"{benchmark} results:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value}")
"""



