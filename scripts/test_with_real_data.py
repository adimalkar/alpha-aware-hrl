#!/usr/bin/env python3
"""
Test feature extractor with a small portion of real FI-2010 data.
Uses chunked loading for memory efficiency.
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from pathlib import Path

print("="*60)
print("Feature Extractor Test with Real FI-2010 Data")
print("="*60)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDevice: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Configuration
SEQ_LEN = 50
N_SAMPLES = 5000  # Only load 5000 rows
N_FEATURES = 144
BATCH_SIZE = 32
EPOCHS = 3

# Load a small chunk of real data
print("\n" + "-"*60)
print(f"Loading first {N_SAMPLES} rows from FI-2010...")
print("-"*60)

data_path = Path("data/fi2010/FI2010/FI2010_train.csv")
df = pd.read_csv(data_path, index_col=0, nrows=N_SAMPLES + SEQ_LEN)

print(f"Loaded shape: {df.shape}")

# Extract features and labels
features = df.iloc[:, :N_FEATURES].values.astype(np.float32)
labels = df.iloc[:, 144].values.astype(np.int64) - 1  # Column 144 is first label, convert 1,2,3 -> 0,1,2
labels = np.clip(labels, 0, 2)

print(f"Features shape: {features.shape}")
print(f"Labels shape: {labels.shape}")
print(f"Label distribution: 0={np.sum(labels==0)}, 1={np.sum(labels==1)}, 2={np.sum(labels==2)}")

# Create sequences
print("\n" + "-"*60)
print("Creating sequences...")
print("-"*60)

n_sequences = len(features) - SEQ_LEN + 1
X = np.zeros((n_sequences, SEQ_LEN, N_FEATURES), dtype=np.float32)
y = np.zeros(n_sequences, dtype=np.int64)

for i in range(n_sequences):
    X[i] = features[i:i + SEQ_LEN]
    y[i] = labels[i + SEQ_LEN - 1]

print(f"Sequences shape: X={X.shape}, y={y.shape}")

# Train/test split
split_idx = int(len(X) * 0.8)
X_train, X_test = X[:split_idx], X[split_idx:]
y_train, y_test = y[:split_idx], y[split_idx:]

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# Create model
print("\n" + "-"*60)
print("Creating LOBClassifier...")
print("-"*60)

from src.agents.mamba_extractor import LOBClassifier

model = LOBClassifier(
    input_dim=N_FEATURES,
    d_model=64,
    n_layers=2,
    n_classes=3,
    backend="auto",
)
model = model.to(device)

params = sum(p.numel() for p in model.parameters())
print(f"Parameters: {params:,}")

# Convert to tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

# Training
print("\n" + "-"*60)
print("Training...")
print("-"*60)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

from torch.utils.data import TensorDataset, DataLoader

train_dataset = TensorDataset(X_train_t, y_train_t)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        _, pred = torch.max(logits, 1)
        total += batch_y.size(0)
        correct += (pred == batch_y).sum().item()
    
    avg_loss = total_loss / total
    accuracy = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS}: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")

# Test evaluation
print("\n" + "-"*60)
print("Test Evaluation...")
print("-"*60)

model.eval()
test_dataset = TensorDataset(X_test_t, y_test_t)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

correct = 0
total = 0
all_preds = []
all_labels = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        logits = model(batch_x)
        _, pred = torch.max(logits, 1)
        
        total += batch_y.size(0)
        correct += (pred == batch_y).sum().item()
        
        all_preds.extend(pred.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

test_accuracy = correct / total
print(f"Test Accuracy: {test_accuracy:.4f}")

# Per-class accuracy
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

for cls in range(3):
    mask = all_labels == cls
    if mask.sum() > 0:
        cls_acc = (all_preds[mask] == all_labels[mask]).mean()
        cls_names = ["Down", "Stable", "Up"]
        print(f"  {cls_names[cls]}: {cls_acc:.4f} (n={mask.sum()})")

print("\n" + "="*60)
print("Test with real FI-2010 data complete!")
print("="*60)
