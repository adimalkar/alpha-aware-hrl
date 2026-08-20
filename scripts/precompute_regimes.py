#!/usr/bin/env python3
"""
Precompute LLM Regimes (Phase 2)

This script bridges the unstructured text data (FNSPID) with the structured
LOB data (FI-2010). It simulates a chronological alignment by passing batches
of news articles to the LLMAnalyst to generate a regime label. The labels are
then broadcasted to match the tick-by-tick length of the FI-2010 dataset.

Because running a 1.1B parameter LLM for 362,400 individual ticks is computationally
infeasible, we compute the regime for distinct "time blocks" (e.g., every 10,000 ticks
representing a few hours of market time) and expand the array.
"""

import sys
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, '.')
from src.agents.llm_analyst import LLMAnalyst, EventAwareRegimeAnalyst
from src.models.tpp_regime import NewsEvent

def main():
    parser = argparse.ArgumentParser(description="Precompute Macro Regimes from News")
    parser.add_argument("--mode", choices=["event", "classic"], default="event",
                        help="Regime detector mode: 'event' (TPP-LLM Hawkes) or 'classic' (TinyLlama)")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Ticks per regime chunk")
    args = parser.parse_args()

    print("=" * 60)
    print(f"PHASE 2/3: Precomputing Macro Regimes from FNSPID [Mode: {args.mode.upper()}]")
    print("=" * 60)
    
    # 1. Load Data
    try:
        train_df = pd.read_csv("data/fi2010/FI2010/FI2010_train.csv")
        test_df = pd.read_csv("data/fi2010/FI2010/FI2010_test.csv")
        news_df = pd.read_csv("data/news/Stock_News_Dataset.csv")
    except FileNotFoundError as e:
        print(f"Error loading datasets: {e}")
        print("Please ensure Phase 1 data is downloaded.")
        sys.exit(1)
        
    n_train = len(train_df)
    n_test = len(test_df)
    
    print(f"Loaded FI-2010 Train: {n_train} ticks")
    print(f"Loaded FI-2010 Test:  {n_test} ticks")
    print(f"Loaded FNSPID News:   {len(news_df)} articles")
    
    # 2. Initialize Analyst
    chunk_size = args.chunk_size

    if args.mode == "event":
        print("\nInitializing TPP-LLM Event Regime Analyst (Hawkes Intensity)...")
        analyst = EventAwareRegimeAnalyst()
    else:
        print("\nInitializing Classic LLM Analyst (TinyLlama-1.1B-Chat)...")
        analyst = LLMAnalyst(device="cuda")
    
    def compute_regimes_for_split(n_ticks, split_name, start_news_idx=0):
        n_chunks = int(np.ceil(n_ticks / chunk_size))
        regimes = np.zeros(n_ticks, dtype=np.int32)
        confidences = np.zeros(n_ticks, dtype=np.float32)
        
        print(f"\nProcessing {split_name} split ({n_chunks} chunks)...")
        news_idx = start_news_idx
        
        for i in tqdm(range(n_chunks)):
            # Grab 5 news articles to form the context window
            news_subset = news_df.iloc[news_idx:news_idx+5]['Article_title'].tolist()
            
            if args.mode == "event":
                # Construct NewsEvent objects with inter-arrival timestamps
                events = [
                    NewsEvent(
                        timestamp=float(i * 1.0 + j * 0.15),
                        headline=str(title),
                        sentiment_score=0.0,
                    )
                    for j, title in enumerate(news_subset)
                ]
                signal, _ = analyst.analyze_events(events, current_time=float(i * 1.0 + 1.0))
            else:
                signal = analyst.analyze(news_subset)
            
            # Broadcast to the tick array
            start_tick = i * chunk_size
            end_tick = min((i + 1) * chunk_size, n_ticks)
            
            regimes[start_tick:end_tick] = signal.regime
            confidences[start_tick:end_tick] = signal.confidence
            
            news_idx += 5
            # Wrap around if we run out of news
            if news_idx >= len(news_df) - 5:
                news_idx = 0
                
        return regimes, confidences, news_idx

    # 3. Compute
    train_regimes, train_conf, next_news_idx = compute_regimes_for_split(n_train, "Train")
    test_regimes, test_conf, _ = compute_regimes_for_split(n_test, "Test", next_news_idx)
    
    # 4. Save
    print("\nSaving precomputed regimes to disk...")
    save_dir = Path("data/precomputed_regimes")
    save_dir.mkdir(exist_ok=True)
    
    np.save(save_dir / "train_regimes.npy", train_regimes)
    np.save(save_dir / "train_confidences.npy", train_conf)
    np.save(save_dir / "test_regimes.npy", test_regimes)
    np.save(save_dir / "test_confidences.npy", test_conf)
    
    # Save a combined representation [regime_id, confidence]
    train_combined = np.column_stack([train_regimes, train_conf])
    test_combined = np.column_stack([test_regimes, test_conf])
    np.save(save_dir / "train_combined.npy", train_combined)
    np.save(save_dir / "test_combined.npy", test_combined)
    
    print("\n✅ Phase 2 precomputation complete!")
    print(f"Distribution of Train Regimes: {np.bincount(train_regimes)}")

if __name__ == "__main__":
    main()
