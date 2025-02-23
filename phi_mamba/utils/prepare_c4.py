import os
from hashlib import sha256
from time import sleep

import torch
from datasets import load_dataset, load_from_disk
from tenacity import retry
from torch.utils.data import DataLoader, IterableDataset
from tqdm import tqdm
from transformers import AutoTokenizer, DataCollatorForLanguageModeling


# Step 5: Define a custom iterable dataset with chunking
class TokenCountDataset(IterableDataset):
    def __init__(self, dataset_iter_, tokenizer_, max_tokens, max_seq_length, world_size=1, rank=0):
        super(TokenCountDataset).__init__()
        self.dataset_iter = dataset_iter_
        self.tokenizer = tokenizer_
        assert world_size > 0
        self.max_tokens = max_tokens // world_size
        self.total_tokens = 0
        self.max_seq_length = max_seq_length
        self.world_size = world_size
        self.rank = rank

        self.eos_token_id = tokenizer_.eos_token_id
        if self.eos_token_id is None:
            self.eos_token_id = tokenizer_.bos_token_id
            if self.eos_token_id is None:
                raise ValueError("Tokenizer does not have eos_token_id or bos_token_id")

    def __iter__(self):
        buffer = []
        for example in self.dataset_iter:
            text = example['text']
            example_hash = int(sha256(text.encode('utf-8')).hexdigest(), 16)
            if example_hash % self.world_size != self.rank:
                continue
            # Tokenize the text without padding or truncation
            tokens = self.tokenizer.encode(text)
            buffer.extend(tokens)
            buffer.append(self.eos_token_id)
            # Create chunks of max_seq_length
            while len(buffer) >= self.max_seq_length:
                chunk = buffer[:self.max_seq_length]
                buffer = buffer[self.max_seq_length:]
                if self.total_tokens + len(chunk) > self.max_tokens:
                    return
                self.total_tokens += len(chunk)
                yield {
                    'input_ids': chunk,
                    'attention_mask': [1] * self.max_seq_length
                }
            if self.total_tokens >= self.max_tokens:
                break
        # Handle remaining tokens in buffer
        if buffer and self.total_tokens < self.max_tokens:
            pad_length = self.max_seq_length - len(buffer)
            chunk = buffer + [self.eos_token_id] * pad_length
            attention_mask = [1] * len(buffer) + [0] * pad_length
            self.total_tokens += len(chunk)
            yield {
                'input_ids': chunk,
                'attention_mask': attention_mask
            }


def get_data_loaders(batch_size: int, max_seq_length: int, dataset_args: tuple[str, str],
                     world_size: int, rank: int, split: str, tokenizer_path='microsoft/phi-1_5', seed=42, epoch=0, return_iter_datasets=False) -> list[DataLoader]:
    # Step 6: Define cumulative token thresholds for splits
    split_tokens = [80_000_000, 240_000_000, 3_000_000_000]  # Cumulative tokens
    if split in ['validation', 'only_train']:
        split_tokens = [20_480, 40_960, 135_000]  # Cumulative tokens
        if split == 'only_train':
            split = 'train'
    # Step 1: Load the dataset in streaming mode
    print(f"Loading the {str(dataset_args)} dataset in streaming mode...")
    retry_limit = 100
    loaded_successfully = False
    dataset = None
    while not loaded_successfully and retry_limit > 0:
        try:
            dataset = load_dataset(*dataset_args, split=split, streaming=True, cache_dir="c4")
            loaded_successfully = True
        except Exception as e:
            print(f"Error loading the dataset: {e}")
            print(f"Retrying...{retry_limit} attempts left.")
            retry_limit -= 1
            sleep(1 + (100 - retry_limit))

    # shuffle
    dataset = dataset.shuffle(seed=seed, buffer_size=10_000)
    dataset.set_epoch(epoch)

    # Step 2: Initialize the tokenizer
    print("Initializing the tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token
    if not tokenizer.is_fast:
        raise ValueError("Please use a tokenizer that supports fast tokenization.")

    # Step 4: Create a shared iterator over the dataset
    dataset_iter = iter(dataset)

    # Step 7: Initialize the data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors='pt'
    )

    # Step 8: Training Loop with Splits
    start_token = 0
    dataloaders = []
    iter_datasets = []
    for idx, max_tokens in enumerate(split_tokens):
        max_split_tokens = max_tokens - start_token  # Tokens for this split
        split_dataset = TokenCountDataset(
            dataset_iter, tokenizer, max_split_tokens, max_seq_length, world_size, rank
        )
        start_token = max_tokens  # Update for next split
        if return_iter_datasets:
            iter_datasets.append(split_dataset)
        else:
            # Create DataLoader for the split
            dataloader = DataLoader(
                split_dataset, batch_size=batch_size, collate_fn=data_collator
            )

            dataloaders.append(dataloader)

    return dataloaders if not return_iter_datasets else (iter_datasets, data_collator)
