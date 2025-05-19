from typing import Any

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
    benchmarks=["hellaswag", "winogrande", "arc_easy", "arc_challenge", "mmlu"],
    limit=None,
    batch_size=1,
    device="cuda",
    tokenizer_path=None,
    no_cache=False,
    verbose=True
):
    """
    Evaluate a model using lm-eval-harness on specified benchmarks.
    
    Args:
        model_or_path: Either a model instance or a string path to a HuggingFace model
        benchmarks: List of benchmark names to evaluate on
        limit: Optional limit on number of examples to use
        batch_size: Batch size for evaluation
        device: Device to run evaluation on ('cuda', 'cpu', 'mps')
        tokenizer_path: Path to tokenizer (if None, uses model_or_path)
        no_cache: Whether to use cached results
        verbose: Whether to print evaluation progress
        
    Returns:
        results: Dictionary containing evaluation results
    """
    # Prepare the model
    if isinstance(model_or_path, str):
        model_name = model_or_path
    else:
        model_name = "custom_model"
        model = model_or_path.to(device).eval()
    
    # If tokenizer path not specified, use model path
    if tokenizer_path is None and isinstance(model_or_path, str):
        tokenizer_path = model_or_path
    
    # Prepare task dict for lm-eval-harness
    task_dict = {}
    for benchmark in benchmarks:
        task_dict[benchmark] = {}
        if limit:
            task_dict[benchmark]["limit"] = limit
    
    # Set evaluation parameters
    eval_params = {
        "model": model_name if isinstance(model_or_path, str) else model,
        "model_args": None if not isinstance(model_or_path, str) else f"pretrained={model_or_path},use_accelerate=True",
        "tasks": benchmarks,
        "batch_size": batch_size,
        "device": device,
        "no_cache": no_cache,
        "verbosity": "INFO" if verbose else "ERROR",
    }
    
    # If custom tokenizer specified
    if tokenizer_path and isinstance(model_or_path, str):
        eval_params["model_args"] += f",tokenizer={tokenizer_path}"
    
    # Run evaluation
    print(f"Starting evaluation on benchmarks: {', '.join(benchmarks)}")
    results = simple_evaluate(**eval_params)
    
    # Print summarized results
    if verbose:
        for benchmark in benchmarks:
            if benchmark in results["results"]:
                print(f"\n{benchmark} results:")
                for metric, value in results["results"][benchmark].items():
                    print(f"  {metric}: {value:.4f}")
    
    return results


if __name__ == "__main__":
    evaluate_with_lm_eval_harness(
        model_or_path="HuggingFaceTB/SmolLM2-1.7B",
        benchmarks=["hellaswag"],
        limit=100,
        batch_size=1,
    )



