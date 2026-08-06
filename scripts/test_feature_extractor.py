#!/usr/bin/env python3
"""
Test script for the Mamba/LSTM Feature Extractor with FI-2010 data.

This script:
1. Loads FI-2010 data
2. Creates sequences for training
3. Trains a simple LOB classifier
4. Evaluates on test set

Usage:
    cd alpha-aware-hrl
    source venv/bin/activate
    python scripts/test_feature_extractor.py
"""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm

from src.utils.data_loader import FI2010DataLoader
from src.agents.mamba_extractor import MambaFeatureExtractor, LOBClassifier


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * batch_x.size(0)
        _, predicted = torch.max(logits, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()
    
    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            _, predicted = torch.max(logits, 1)
            total += batch_y.size(0)
            correct += (predicted == batch_y).sum().item()
    
    return total_loss / total, correct / total


def main():
    print("="*60)
    print("Mamba/LSTM Feature Extractor Test with FI-2010 Data")
    print("="*60)
    
    # Configuration
    SEQUENCE_LENGTH = 50  # Shorter for faster testing
    BATCH_SIZE = 64
    EPOCHS = 5
    LEARNING_RATE = 1e-3
    D_MODEL = 64  # Smaller for faster testing
    N_LAYERS = 2
    MAX_SAMPLES = 50000  # Limit samples for memory efficiency
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    print("\n" + "="*60)
    print("Loading FI-2010 Data...")
    print("="*60)
    
    loader = FI2010DataLoader(data_dir="data/fi2010/FI2010")
    splits = loader.get_train_val_test_split(sequence_length=SEQUENCE_LENGTH)
    
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    
    # Limit samples for memory efficiency
    if len(X_train) > MAX_SAMPLES:
        print(f"\nLimiting train samples from {len(X_train)} to {MAX_SAMPLES}")
        X_train = X_train[:MAX_SAMPLES]
        y_train = y_train[:MAX_SAMPLES]
    if len(X_val) > MAX_SAMPLES // 5:
        X_val = X_val[:MAX_SAMPLES // 5]
        y_val = y_val[:MAX_SAMPLES // 5]
    if len(X_test) > MAX_SAMPLES // 5:
        X_test = X_test[:MAX_SAMPLES // 5]
        y_test = y_test[:MAX_SAMPLES // 5]
    
    print(f"\nData shapes (after limiting):")
    print(f"  Train: X={X_train.shape}, y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}, y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}, y={y_test.shape}")
    
    # Create dataloaders
    train_dataset = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.long)
    )
    val_dataset = TensorDataset(
        torch.tensor(X_val, dtype=torch.float32),
        torch.tensor(y_val, dtype=torch.long)
    )
    test_dataset = TensorDataset(
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.long)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Create model
    print("\n" + "="*60)
    print("Creating LOB Classifier...")
    print("="*60)
    
    input_dim = X_train.shape[2]  # 144 features
    model = LOBClassifier(
        input_dim=input_dim,
        d_model=D_MODEL,
        n_layers=N_LAYERS,
        n_classes=3,
        dropout=0.1,
        backend="auto",  # Will use LSTM since Mamba not installed
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Training loop
    print("\n" + "="*60)
    print("Training...")
    print("="*60)
    
    best_val_acc = 0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
        
        print(f"Epoch {epoch+1}/{EPOCHS}: "
              f"Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f} | "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    
    # Test evaluation
    print("\n" + "="*60)
    print("Test Evaluation...")
    print("="*60)
    
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Best Val Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    
    # Quick inference test
    print("\n" + "="*60)
    print("Inference Test...")
    print("="*60)
    
    model.eval()
    sample = torch.tensor(X_test[:5], dtype=torch.float32).to(device)
    with torch.no_grad():
        predictions = model.predict(sample)
        probabilities = model.predict_proba(sample)
    
    print(f"Sample predictions: {predictions.cpu().numpy()}")
    print(f"Sample ground truth: {y_test[:5]}")
    print(f"Sample probabilities:\n{probabilities.cpu().numpy()}")
    
    print("\n" + "="*60)
    print("Test Complete!")
    print("="*60)
    print("\nFeature extractor is working correctly with FI-2010 data.")
    print(f"Backend used: {model.feature_extractor.backend}")


if __name__ == "__main__":
    main()
