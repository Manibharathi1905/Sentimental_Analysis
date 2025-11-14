import time
import torch
from transformers import AutoTokenizer, AutoModel
import numpy as np

def benchmark(model_name="distilbert-base-uncased", batch_size=8, seq_len=128, n_batches=50):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    model.to(device)

    # create synthetic inputs
    texts = ["this is a benchmark sentence"] * batch_size
    enc = tokenizer(texts, padding="max_length", max_length=seq_len, truncation=True, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    # warmup
    for _ in range(5):
        with torch.no_grad():
            _ = model(input_ids, attention_mask)

    start = time.time()
    for i in range(n_batches):
        with torch.no_grad():
            _ = model(input_ids, attention_mask)
    end = time.time()
    total = end - start
    batches_per_sec = n_batches / total
    samples_per_sec = batches_per_sec * batch_size
    print(f"Batches/sec: {batches_per_sec:.3f}, samples/sec: {samples_per_sec:.3f}")
    return samples_per_sec

if __name__ == "__main__":
    sps = benchmark()
    print("To estimate total training time (seconds):")
    print(" total_steps = (dataset_size / batch_size) * epochs")
    print(" seconds = total_steps / batches_per_sec")
