#!/usr/bin/env python3
"""
Full Training Script for LOB Mid-Price Prediction

This script trains and evaluates multiple model architectures on FI-2010 data.
Supports: LSTM, BiLSTM, TCN, Transformer

Usage:
    python scripts/train_lob_classifier.py --model lstm --epochs 50
    python scripts/train_lob_classifier.py --model transformer --epochs 30
"""

import sys
sys.path.insert(0, '.')

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm

# Model imports
from src.agents.mamba_extractor import LOBClassifier
from src.models.mamba_ssm import TCNEncoder


class TransformerLOBClassifier(nn.Module):
    """Transformer-based LOB classifier."""
    
    def __init__(
        self,
        input_dim: int = 144,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        n_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x[:, -1, :]  # Last timestep
        return self.classifier(x)


class BiLSTMLOBClassifier(nn.Module):
    """Bidirectional LSTM classifier."""
    
    def __init__(
        self,
        input_dim: int = 144,
        d_model: int = 128,
        n_layers: int = 4,
        n_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.input_proj = nn.Linear(input_dim, d_model)
        
        self.lstm = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model // 2,  # Half because bidirectional doubles it
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=True,
        )
        
        self.norm = nn.LayerNorm(d_model)
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )
        
    def forward(self, x):
        x = self.input_proj(x)
        x, _ = self.lstm(x)
        x = self.norm(x[:, -1, :])
        return self.classifier(x)


class TCNLOBClassifier(nn.Module):
    """TCN-based classifier."""
    
    def __init__(
        self,
        input_dim: int = 144,
        d_model: int = 128,
        n_layers: int = 4,
        n_classes: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.encoder = TCNEncoder(
            input_dim=input_dim,
            hidden_dim=d_model,
            n_layers=n_layers,
            dropout=dropout,
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes),
        )
        
    def forward(self, x):
        _, final_state = self.encoder(x)
        return self.classifier(final_state)


def load_data_chunked(data_path: Path, n_rows: int = None):
    """Load FI-2010 data with optional row limit."""
    print(f"Loading data from {data_path}...")
    
    if n_rows:
        df = pd.read_csv(data_path, index_col=0, nrows=n_rows)
    else:
        df = pd.read_csv(data_path, index_col=0)
    
    # Features: first 144 columns
    features = df.iloc[:, :144].values.astype(np.float32)
    # Labels: column 144 (first horizon), convert 1,2,3 -> 0,1,2
    labels = df.iloc[:, 144].values.astype(np.int64) - 1
    labels = np.clip(labels, 0, 2)
    
    print(f"Loaded {len(features)} samples")
    return features, labels


def create_sequences(features, labels, seq_len):
    """Create sequences from raw data."""
    n_seq = len(features) - seq_len + 1
    X = np.zeros((n_seq, seq_len, features.shape[1]), dtype=np.float32)
    y = np.zeros(n_seq, dtype=np.int64)
    
    for i in range(n_seq):
        X[i] = features[i:i + seq_len]
        y[i] = labels[i + seq_len - 1]
    
    return X, y


def train_epoch(model, loader, optimizer, criterion, device, scheduler=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for batch_x, batch_y in pbar:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        total_loss += loss.item() * batch_x.size(0)
        _, pred = torch.max(logits, 1)
        total += batch_y.size(0)
        correct += (pred == batch_y).sum().item()
        
        pbar.set_postfix({'loss': loss.item(), 'acc': correct/total})
    
    return total_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate on validation/test set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            _, pred = torch.max(logits, 1)
            total += batch_y.size(0)
            correct += (pred == batch_y).sum().item()
            
            all_preds.extend(pred.cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
    
    return total_loss / total, correct / total, np.array(all_preds), np.array(all_labels)


def get_model(model_name, input_dim, d_model, n_layers, n_classes, dropout):
    """Create model by name."""
    if model_name == "lstm" or model_name == "mamba":
        return LOBClassifier(
            input_dim=input_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_classes=n_classes,
            dropout=dropout,
            backend=model_name,
        )
    elif model_name == "bilstm":
        return BiLSTMLOBClassifier(
            input_dim=input_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_classes=n_classes,
            dropout=dropout,
        )
    elif model_name == "tcn":
        return TCNLOBClassifier(
            input_dim=input_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_classes=n_classes,
            dropout=dropout,
        )
    elif model_name == "transformer":
        return TransformerLOBClassifier(
            input_dim=input_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_classes=n_classes,
            dropout=dropout,
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Train LOB classifier")
    parser.add_argument("--model", type=str, default="lstm", 
                        choices=["lstm", "bilstm", "tcn", "transformer", "mamba"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=50)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max_samples", type=int, default=100000,
                        help="Max samples to use (for memory). Use 0 for all.")
    parser.add_argument("--save_path", type=str, default="checkpoints")
    args = parser.parse_args()
    
    print("="*70)
    print(f"Training {args.model.upper()} LOB Classifier")
    print("="*70)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Load data
    data_dir = Path("data/fi2010/FI2010")
    
    n_rows = args.max_samples + args.seq_len if args.max_samples > 0 else None
    train_features, train_labels = load_data_chunked(data_dir / "FI2010_train.csv", n_rows)
    test_features, test_labels = load_data_chunked(data_dir / "FI2010_test.csv", n_rows)
    
    # Create sequences
    print(f"\nCreating sequences (seq_len={args.seq_len})...")
    X_train, y_train = create_sequences(train_features, train_labels, args.seq_len)
    X_test, y_test = create_sequences(test_features, test_labels, args.seq_len)
    
    # Train/val split
    val_size = int(len(X_train) * 0.15)
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    X_train, y_train = X_train[:-val_size], y_train[:-val_size]
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Label distribution
    for name, labels in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
        dist = [np.sum(labels == i) for i in range(3)]
        print(f"{name} labels: Down={dist[0]}, Stable={dist[1]}, Up={dist[2]}")
    
    # Create dataloaders
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=args.batch_size, num_workers=2, pin_memory=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(X_test), torch.tensor(y_test)),
        batch_size=args.batch_size, num_workers=2, pin_memory=True
    )
    
    # Create model
    print(f"\nCreating {args.model.upper()} model...")
    model = get_model(
        args.model,
        input_dim=X_train.shape[2],
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_classes=3,
        dropout=args.dropout,
    )
    model = model.to(device)
    
    params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {params:,}")
    
    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=len(train_loader) * args.epochs
    )
    
    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("-"*70)
    
    best_val_acc = 0
    best_epoch = 0
    patience = 10
    patience_counter = 0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, scheduler)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        
        epoch_time = time.time() - start_time
        
        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Save checkpoint
            save_dir = Path(args.save_path)
            save_dir.mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'args': args,
            }, save_dir / f"{args.model}_best.pt")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch+1}")
                break
    
    # Load best model and evaluate on test
    print("\n" + "="*70)
    print("Final Evaluation on Test Set")
    print("="*70)
    
    checkpoint = torch.load(Path(args.save_path) / f"{args.model}_best.pt", weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, preds, labels = evaluate(model, test_loader, criterion, device)
    
    print(f"\nBest Validation Accuracy: {best_val_acc:.4f} (Epoch {best_epoch})")
    print(f"Test Accuracy: {test_acc:.4f}")
    
    # Per-class accuracy
    print("\nPer-class accuracy:")
    class_names = ["Down", "Stable", "Up"]
    for i, name in enumerate(class_names):
        mask = labels == i
        if mask.sum() > 0:
            acc = (preds[mask] == labels[mask]).mean()
            print(f"  {name}: {acc:.4f} (n={mask.sum()})")
    
    # Confusion matrix
    from collections import Counter
    print("\nPrediction distribution:")
    pred_dist = Counter(preds)
    for i, name in enumerate(class_names):
        print(f"  {name}: {pred_dist[i]} ({pred_dist[i]/len(preds)*100:.1f}%)")
    
    print("\n" + "="*70)
    print(f"Model saved to {args.save_path}/{args.model}_best.pt")
    print("="*70)


if __name__ == "__main__":
    main()
