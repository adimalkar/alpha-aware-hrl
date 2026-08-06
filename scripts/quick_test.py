#!/usr/bin/env python3
"""
Quick test of feature extractor with synthetic data.
Avoids memory issues with large FI-2010 files.
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import numpy as np

print("="*60)
print("Quick Feature Extractor Test")
print("="*60)

# Device check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Import and test
print("\n" + "-"*60)
print("Testing MambaFeatureExtractor...")
print("-"*60)

from src.agents.mamba_extractor import MambaFeatureExtractor, LOBClassifier

# Create synthetic data matching FI-2010 format
batch_size = 32
seq_len = 50
input_dim = 144  # FI-2010 features

print(f"\nSynthetic data: batch={batch_size}, seq_len={seq_len}, features={input_dim}")

# Create model
model = MambaFeatureExtractor(
    input_dim=input_dim,
    d_model=64,
    n_layers=2,
    dropout=0.1,
    backend="auto",
)
model = model.to(device)
print(f"Model backend: {model.backend}")

# Count params
params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")

# Test forward pass
x = torch.randn(batch_size, seq_len, input_dim).to(device)
print(f"\nInput shape: {x.shape}")

with torch.no_grad():
    features, final_state = model(x)

print(f"Features shape: {features.shape}")
print(f"Final state shape: {final_state.shape}")

# Test classifier
print("\n" + "-"*60)
print("Testing LOBClassifier...")
print("-"*60)

classifier = LOBClassifier(
    input_dim=input_dim,
    d_model=64,
    n_layers=2,
    n_classes=3,
    backend="auto",
)
classifier = classifier.to(device)

with torch.no_grad():
    logits = classifier(x)
    predictions = classifier.predict(x)
    probs = classifier.predict_proba(x)

print(f"Logits shape: {logits.shape}")
print(f"Predictions shape: {predictions.shape}")
print(f"Sample predictions: {predictions[:5].cpu().numpy()}")
print(f"Sample probabilities:\n{probs[:3].cpu().numpy()}")

# Test training step
print("\n" + "-"*60)
print("Testing training step...")
print("-"*60)

classifier.train()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# Fake labels
y = torch.randint(0, 3, (batch_size,)).to(device)

optimizer.zero_grad()
logits = classifier(x)
loss = criterion(logits, y)
loss.backward()
optimizer.step()

print(f"Training loss: {loss.item():.4f}")
print("Backward pass: OK")
print("Optimizer step: OK")

# Quick benchmark
print("\n" + "-"*60)
print("Quick benchmark (100 forward passes)...")
print("-"*60)

import time
classifier.eval()
x_bench = torch.randn(16, 50, 144).to(device)

# Warmup
for _ in range(10):
    with torch.no_grad():
        _ = classifier(x_bench)

if device.type == "cuda":
    torch.cuda.synchronize()

start = time.time()
for _ in range(100):
    with torch.no_grad():
        _ = classifier(x_bench)
if device.type == "cuda":
    torch.cuda.synchronize()
elapsed = time.time() - start

print(f"100 forward passes: {elapsed:.3f}s")
print(f"Per batch: {elapsed/100*1000:.2f}ms")
print(f"Throughput: {100*16/elapsed:.0f} samples/sec")

print("\n" + "="*60)
print("All tests passed!")
print("="*60)
