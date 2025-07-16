import sys

# Block TensorFlow and related problematic modules
sys.modules['tensorflow'] = None
sys.modules['tensorflow.python'] = None
sys.modules['ml_dtypes'] = None

import gc
import torch
import time
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from huggingface_hub import login
login("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")

# Models to compare
models = {
    "Qwen 2.5 7B Instruct": "Qwen/Qwen2-7B-Instruct",
    "Mistral 7B Instruct": "mistralai/Mistral-7B-Instruct-v0.2"
}


# Use CUDA if available
device = "cuda" if torch.cuda.is_available() else "cpu"
prompt = "Explain quantum entanglement in simple terms."

# Store results
results = {}

for name, model_id in models.items():
    print(f"\nLoading {name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16).to(device)
    streamer = TextStreamer(tokenizer)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print(f"Generating with {name}...")
    start_time = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            streamer=streamer,
            max_new_tokens=100,
            do_sample=False
        )
    end_time = time.time()

    total_tokens = output.shape[-1]
    elapsed_time = end_time - start_time
    tokens_per_sec = total_tokens / elapsed_time

    print(f"{name}: {total_tokens} tokens in {elapsed_time:.2f}s -> {tokens_per_sec:.2f} tokens/sec")
    results[name] = tokens_per_sec

    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()

# Plot results
plt.bar(results.keys(), results.values(), color=["skyblue", "salmon"])
plt.ylabel("Tokens per second")
plt.title("LLM Inference Speed (100 tokens)")
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig("llm_speed_comparison.png")
plt.show()
