#!/usr/bin/env python3
"""
Run Traditional Financial Baselines (Phase 3)

Evaluates traditional, non-RL algorithmic trading strategies on the historical
FI-2010 Limit Order Book test set to provide a baseline comparison for the 
Hierarchical RL agent.

Baselines evaluated:
1. MACD (Moving Average Convergence Divergence) - Momentum
2. Bollinger Bands - Mean Reversion
3. Supervised LSTM - Deterministic ML prediction
"""

import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
import json
import matplotlib.pyplot as plt

sys.path.insert(0, '.')
from src.utils.data_loader import FI2010DataLoader
from src.envs.historical_lob_env import HistoricalLOBEnv
from src.utils.metrics import (
    compute_sharpe, 
    compute_max_drawdown, 
    compute_var, 
    compute_cvar
)

# ---------------------------------------------------------------------------
# 1. Traditional Math Strategies
# ---------------------------------------------------------------------------

class MACDStrategy:
    """Momentum strategy based on EMA crossovers."""
    def __init__(self, short_window=12, long_window=26, signal_window=9):
        self.short_window = short_window
        self.long_window = long_window
        self.signal_window = signal_window
        self.prices = []
        
    def get_action(self, price: float) -> float:
        self.prices.append(price)
        if len(self.prices) < self.long_window + self.signal_window:
            return 0.0 # Hold while warming up
            
        prices_series = pd.Series(self.prices)
        ema_short = prices_series.ewm(span=self.short_window, adjust=False).mean()
        ema_long = prices_series.ewm(span=self.long_window, adjust=False).mean()
        macd_line = ema_short - ema_long
        signal_line = macd_line.ewm(span=self.signal_window, adjust=False).mean()
        
        # Action is based on the crossover
        current_macd = macd_line.iloc[-1]
        current_signal = signal_line.iloc[-1]
        
        if current_macd > current_signal:
            return 1.0  # Buy
        elif current_macd < current_signal:
            return -1.0 # Sell
        return 0.0

class BollingerStrategy:
    """Mean reversion strategy based on statistical bands."""
    def __init__(self, window=20, num_std=2.0):
        self.window = window
        self.num_std = num_std
        self.prices = []
        self.current_position = 0.0
        
    def get_action(self, price: float) -> float:
        self.prices.append(price)
        if len(self.prices) < self.window:
            return 0.0
            
        recent_prices = np.array(self.prices[-self.window:])
        sma = np.mean(recent_prices)
        std = np.std(recent_prices)
        
        upper_band = sma + (self.num_std * std)
        lower_band = sma - (self.num_std * std)
        
        if price > upper_band:
            self.current_position = -1.0  # Overbought, short it
        elif price < lower_band:
            self.current_position = 1.0   # Oversold, buy it
        elif sma - (0.5*std) <= price <= sma + (0.5*std):
            self.current_position = 0.0   # Reverted to mean, close positions
            
        return self.current_position

# ---------------------------------------------------------------------------
# 2. Supervised LSTM Strategy
# ---------------------------------------------------------------------------

class SimpleLSTMModel(nn.Module):
    def __init__(self, input_dim=144, hidden_dim=64, num_classes=3):
        super().__init__()
        # PyTorch LSTM expects input of shape (batch, seq_len, features)
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        # x is (batch, seq_len, features) or (batch, features)
        if len(x.shape) == 2:
            x = x.unsqueeze(1) # Add seq dimension if missing
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # Take last timestep
        return out

class SupervisedLSTMStrategy:
    """Trains an LSTM on historical data to predict next tick movement."""
    def __init__(self, data_loader, device="cpu"):
        self.model = SimpleLSTMModel().to(device)
        self.device = device
        self.data_loader = data_loader
        self._train_model()
        self.model.eval()
        
    def _train_model(self):
        print("\n--- Training Supervised LSTM Baseline ---")
        train_data = self.data_loader.train_data[:50000] # Use subset for speed
        train_labels = self.data_loader.train_labels[:50000]
        
        dataset = TensorDataset(torch.tensor(train_data), torch.tensor(train_labels))
        loader = DataLoader(dataset, batch_size=256, shuffle=True)
        
        optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        self.model.train()
        for epoch in range(1): # 1 Epoch is enough for baseline demonstration
            total_loss = 0
            for X_batch, y_batch in tqdm(loader, desc="Epoch 1/1"):
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                optimizer.zero_grad()
                preds = self.model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"LSTM Train Loss: {total_loss/len(loader):.4f}")
            
    def get_action(self, obs: np.ndarray) -> float:
        with torch.no_grad():
            x = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
            preds = self.model(x)
            predicted_class = torch.argmax(preds, dim=1).item()
            
        # Class 0: Down (Short), Class 1: Stable (Hold), Class 2: Up (Long)
        if predicted_class == 2:
            return 1.0
        elif predicted_class == 0:
            return -1.0
        return 0.0

# ---------------------------------------------------------------------------
# 3. Evaluation Loop
# ---------------------------------------------------------------------------

def evaluate_strategy(name, strategy_obj, env_kwargs, n_steps=31900):
    print(f"\nEvaluating: {name}")
    env = HistoricalLOBEnv(**env_kwargs)
    obs, info = env.reset()
    
    portfolio_history = [info["portfolio_value"]]
    returns = []
    
    for _ in tqdm(range(n_steps)):
        # Different strategies take different inputs
        if hasattr(strategy_obj, 'get_action'):
            if name == "Supervised LSTM":
                action = strategy_obj.get_action(obs)
            else:
                action = strategy_obj.get_action(info["current_price"])
        else:
            action = 0.0 # Fallback
            
        obs, rew, term, trunc, info = env.step(np.array([action]))
        
        port_val = info["portfolio_value"]
        returns.append( (port_val - portfolio_history[-1]) / portfolio_history[-1] )
        portfolio_history.append(port_val)
        
        if term or trunc:
            break
            
    returns = np.array(returns)
    
    # Calculate Metrics
    sharpe = compute_sharpe(returns)
    mdd, _, _ = compute_max_drawdown(portfolio_history)
    var = compute_var(returns)
    cvar = compute_cvar(returns)
    final_port = portfolio_history[-1]
    
    return {
        "Name": name,
        "Final Portfolio": final_port,
        "Return %": ((final_port - 100000) / 100000) * 100,
        "Sharpe Ratio": sharpe,
        "Max Drawdown %": mdd * 100,
        "VaR (95%)": var * 100,
        "CVaR (95%)": cvar * 100,
        "_history": portfolio_history
    }

def main():
    print("=" * 60)
    print("PHASE 3: Evaluating Traditional Financial Baselines")
    print("=" * 60)
    
    # 1. Load Data
    print("\nLoading FI-2010 Data...")
    loader = FI2010DataLoader(data_dir="data/fi2010/FI2010", horizon_idx=0)
    loader.load("train")
    loader.load("test")
    
    env_kwargs = {
        "data_loader": loader,
        "split": "test",
        "episode_length": 31900,  # Entire test set
        "starting_cash": 100000.0,
        "transaction_fee": 0.0001, # 1 bps
    }
    
    # 2. Initialize Strategies
    device = "cuda" if torch.cuda.is_available() else "cpu"
    strategies = [
        ("MACD (Momentum)", MACDStrategy()),
        ("Bollinger Bands (Mean Reversion)", BollingerStrategy()),
        ("Supervised LSTM", SupervisedLSTMStrategy(loader, device))
    ]
    
    # 3. Evaluate
    results = []
    histories = {}
    
    for name, strategy in strategies:
        res = evaluate_strategy(name, strategy, env_kwargs)
        histories[name] = res.pop("_history")
        results.append(res)
        
    # 4. Print & Save Results
    print("\n" + "=" * 80)
    print("BASELINE COMPARISON RESULTS")
    print("=" * 80)
    
    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    
    save_dir = Path("experiments/baselines")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_json(save_dir / "baseline_metrics.json", orient="records", indent=4)
    
    # 5. Plot Equity Curves
    plt.figure(figsize=(12, 6))
    for name, hist in histories.items():
        plt.plot(hist, label=name)
        
    plt.title("Baseline Equity Curves on FI-2010 Test Set (31,900 Ticks)")
    plt.xlabel("Ticks")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True)
    
    plot_dir = save_dir / "plots"
    plot_dir.mkdir(exist_ok=True)
    plt.savefig(plot_dir / "baseline_equity_curves.png")
    
    print(f"\n✅ Results and plots saved to {save_dir}")

if __name__ == "__main__":
    main()
