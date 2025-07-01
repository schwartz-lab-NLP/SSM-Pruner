import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from phi_mamba.utils.ppl import evaluate_with_lm_eval_harness

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained("gpt2", torch_dtype=torch.float16)

# Test both string path and tokenizer object
print("1. Testing with string tokenizer path...")
results_string = evaluate_with_lm_eval_harness(
    model_or_path=model,
    benchmarks=["hellaswag"],
    limit=10,  # Very small limit for quick testing
    batch_size=4,
    device="cpu",
    tokenizer_path="gpt2",  # String path
    num_fewshot=0,
    no_cache=True,
    verbose=True,
)

print("\n2. Testing with tokenizer object...")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
results_object = evaluate_with_lm_eval_harness(
    model_or_path=model,
    benchmarks=["hellaswag"],
    limit=10,  # Very small limit for quick testing
    batch_size=4,
    device="cpu",
    tokenizer_path=tokenizer,  # Tokenizer object
    num_fewshot=0,
    no_cache=True,
    verbose=True,
)

# Report on both approaches
print("\n=== SUMMARY ===")
print(f"String path approach: {'SUCCESS' if results_string is not None else 'FAILED'}")
print(f"Tokenizer object approach: {'SUCCESS' if results_object is not None else 'FAILED'}")

if results_string is not None and results_object is not None:
    print("\nBoth approaches worked correctly with lm-eval 0.4.8!") 