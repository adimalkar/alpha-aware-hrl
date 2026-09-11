"""
Data Loading Utilities for FI-2010 and FNSPID Datasets
"""

import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict, List
import torch
from torch.utils.data import Dataset, DataLoader


class FI2010DataLoader:
    """
    Data loader for FI-2010 Limit Order Book dataset.
    
    The FI-2010 dataset contains normalized LOB features from 5 stocks
    traded on the Helsinki Stock Exchange.
    
    The CSV format has:
    - Columns 0-143: LOB features (normalized)
    - Columns 144-148: Labels for 5 prediction horizons (k=1,2,3,5,10)
    - Labels: 1=down, 2=stable, 3=up (we convert to 0,1,2)
    
    Args:
        data_dir: Path to FI-2010 data directory
        horizon_idx: Which prediction horizon to use (0-4, default=0 for k=10)
    """
    
    def __init__(
        self,
        data_dir: str = "data/fi2010/FI2010",
        horizon_idx: int = 0,  # 0=k10, 1=k20, 2=k30, 3=k50, 4=k100
    ):
        self.data_dir = Path(data_dir)
        self.horizon_idx = horizon_idx
        
        self.train_data = None
        self.train_labels = None
        self.test_data = None
        self.test_labels = None
        
        # Feature columns (first 144 columns are features)
        self.n_features = 144
        # Label columns are the last 5 columns
        self.label_col_offset = 144
        
    def load(self, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
        """
        Load and preprocess FI-2010 data.
        
        Args:
            split: 'train' or 'test'
            
        Returns:
            features: LOB features (n_samples, n_features)
            labels: Mid-price movement labels (n_samples,) - 0=down, 1=stable, 2=up
        """
        if split == "train":
            file_path = self.data_dir / "FI2010_train.csv"
        else:
            file_path = self.data_dir / "FI2010_test.csv"
            
        if not file_path.exists():
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        print(f"[FI2010] Loading {split} data from {file_path}...")
        
        # Load CSV
        df = pd.read_csv(file_path, index_col=0)
        
        # Extract features (first 144 columns)
        features = df.iloc[:, :self.n_features].values.astype(np.float32)
        
        # Extract labels for selected horizon (last 5 columns)
        # Labels are 1,2,3 -> convert to 0,1,2
        label_col = self.label_col_offset + self.horizon_idx
        labels = df.iloc[:, label_col].values.astype(np.int64) - 1  # Convert 1,2,3 -> 0,1,2
        
        # Clip labels to valid range (safety check)
        labels = np.clip(labels, 0, 2)

        # R2: data/live_market/FI2010_*.csv contained Coinbase crypto snapshots,
        # not FI-2010, and loaded through this class without complaint. A reader
        # of the call site could not tell which asset class produced a result.
        # FI-2010 is z-scored, so genuine FI-2010 features are O(1); real price
        # levels are not.
        max_abs = float(np.abs(features).max()) if features.size else 0.0
        if max_abs > 50.0:
            warnings.warn(
                f"{file_path} has |max| feature value {max_abs:.1f}. FI-2010 is "
                "z-score normalised and should be O(1); this looks like raw price "
                "levels from a different dataset. Loading it through "
                "FI2010DataLoader will silently mislabel the asset class -- use "
                "LiveMarketDataLoader for collected market data.",
                RuntimeWarning,
                stacklevel=2,
            )
        
        if split == "train":
            self.train_data = features
            self.train_labels = labels
        else:
            self.test_data = features
            self.test_labels = labels
        
        print(f"[FI2010] Loaded {len(features)} samples with {features.shape[1]} features")
        print(f"[FI2010] Label distribution: down={np.sum(labels==0)}, stable={np.sum(labels==1)}, up={np.sum(labels==2)}")
        
        return features, labels
    
    def load_all(self) -> None:
        """Load both train and test data."""
        self.load("train")
        self.load("test")
    
    def get_sequences(
        self,
        sequence_length: int = 100,
        split: str = "train",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for temporal modeling.
        
        Args:
            sequence_length: Length of input sequences
            split: 'train' or 'test'
            
        Returns:
            X: Input sequences (n_sequences, sequence_length, n_features)
            y: Labels (n_sequences,)
        """
        if split == "train":
            if self.train_data is None:
                self.load("train")
            data, labels = self.train_data, self.train_labels
        else:
            if self.test_data is None:
                self.load("test")
            data, labels = self.test_data, self.test_labels
        
        n_samples = len(data) - sequence_length + 1
        
        X = np.zeros((n_samples, sequence_length, data.shape[1]), dtype=np.float32)
        y = np.zeros(n_samples, dtype=np.int64)
        
        for i in range(n_samples):
            X[i] = data[i:i + sequence_length]
            # Use label at end of sequence
            y[i] = labels[i + sequence_length - 1]
        
        print(f"[FI2010] Created {n_samples} sequences of length {sequence_length}")
        
        return X, y
    
    def get_train_val_test_split(
        self,
        sequence_length: int = 100,
        val_ratio: float = 0.15,
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        Split data into train/val/test sets.
        
        Train and val come from train file (chronological split).
        Test comes from test file.
        """
        # Get train sequences
        X_train_full, y_train_full = self.get_sequences(sequence_length, "train")
        
        # Split train into train/val chronologically
        n_val = int(len(X_train_full) * val_ratio)
        n_train = len(X_train_full) - n_val
        
        X_train = X_train_full[:n_train]
        y_train = y_train_full[:n_train]
        X_val = X_train_full[n_train:]
        y_val = y_train_full[n_train:]
        
        # Get test sequences
        X_test, y_test = self.get_sequences(sequence_length, "test")
        
        return {
            "train": (X_train, y_train),
            "val": (X_val, y_val),
            "test": (X_test, y_test),
        }


class FI2010Dataset(Dataset):
    """PyTorch Dataset wrapper for FI-2010."""
    
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]





class LiveMarketDataLoader:
    """
    Loader for the per-symbol .npz series written by scripts/fetch_live_market_data.py.

    Exposes the same attribute surface as FI2010DataLoader so it can be dropped
    into HistoricalLOBEnv, plus the real mid-price series the env needs.

    Feature scaling is fit on the TRAINING SPLIT ONLY and then applied to both
    splits. Fitting on the concatenation -- or normalising each split by its own
    statistics -- leaks test distribution information into training and is the
    quiet version of the same mistake that made the old live test set a copy of
    the train set.

    The mid-price series is deliberately NOT scaled: it is the causal price path
    the environment trades on, and rescaling it (the old pipeline forced every
    mid to exactly 100.0) is what destroyed the only genuine price information
    in the previous dataset.
    """

    # Columns holding absolute prices: 10 levels x (ask_price, bid_price), plus
    # spread and mid. These are non-stationary in level, so they are converted
    # to fractions of the mid before scaling.
    _PRICE_COLS = [i * 4 + 0 for i in range(10)] + [i * 4 + 2 for i in range(10)]
    _MID_COL = 41
    _SPREAD_COL = 40

    def __init__(self, data_dir: str = "data/live_market", symbol: str = "BTC/USD"):
        self.data_dir = Path(data_dir)
        self.symbol = symbol
        self.safe = symbol.replace("/", "").lower()
        self.sym_dir = self.data_dir / self.safe

        self.train_data = None
        self.train_labels = None
        self.train_mid = None
        self.test_data = None
        self.test_labels = None
        self.test_mid = None

        self._scaler_mean = None
        self._scaler_std = None

    def _raw(self, split: str):
        path = self.sym_dir / f"{split}.npz"
        if not path.exists():
            raise FileNotFoundError(
                f"No {split} data at {path}. Run:\n"
                f"  python scripts/fetch_live_market_data.py --symbols {self.symbol}"
            )
        with np.load(path) as z:
            return (
                z["features"].astype(np.float64),
                z["mid"].astype(np.float64),
                z["label"].astype(np.int64),
            )

    def _to_stationary(self, feats: np.ndarray, mid: np.ndarray) -> np.ndarray:
        """Convert absolute price levels to mid-relative fractions."""
        out = feats.copy()
        m = mid[:, None]
        out[:, self._PRICE_COLS] = (out[:, self._PRICE_COLS] - m) / m
        out[:, self._SPREAD_COL] = out[:, self._SPREAD_COL] / mid
        # The mid column itself carries only level information, which the agent
        # must not key on; replace it with the one-step realised return.
        rets = np.zeros_like(mid)
        rets[1:] = np.diff(mid) / mid[:-1]
        out[:, self._MID_COL] = rets
        return out

    def load(self, split: str = "train"):
        """
        Load and scale one split. 'train' must be loaded before 'test' so the
        scaler exists; calling load('test') first does that automatically.
        """
        if split == "test" and self._scaler_mean is None:
            self.load("train")

        feats, mid, labels = self._raw(split)
        feats = self._to_stationary(feats, mid)

        if split == "train":
            self._scaler_mean = feats.mean(axis=0)
            self._scaler_std = feats.std(axis=0)
            # Constant columns (the reserved zero block) must not divide by ~0.
            self._scaler_std[self._scaler_std < 1e-12] = 1.0

        scaled = ((feats - self._scaler_mean) / self._scaler_std).astype(np.float32)

        if split == "train":
            self.train_data, self.train_mid, self.train_labels = scaled, mid, labels
        else:
            self.test_data, self.test_mid, self.test_labels = scaled, mid, labels

        return scaled, labels

    def prices(self, split: str = "train") -> np.ndarray:
        """The causal mid-price series for `split`, in real units."""
        if split == "train":
            if self.train_mid is None:
                self.load("train")
            return self.train_mid
        if self.test_mid is None:
            self.load("test")
        return self.test_mid
