#!/usr/bin/env python3
"""
Download Real Datasets (FI-2010 & FNSPID)

This script uses the HuggingFace datasets library to download the real
FI-2010 benchmark dataset and converts it into the CSV format expected
by the data_loader.py pipeline.
"""

import sys
import pandas as pd
from pathlib import Path
from datasets import load_dataset
import os

def download_fi2010():
    print("=" * 50)
    print("Downloading Real FI-2010 Dataset from HuggingFace")
    print("=" * 50)
    
    save_dir = Path("data/fi2010/FI2010")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = save_dir / "FI2010_train.csv"
    test_path = save_dir / "FI2010_test.csv"
    
    if train_path.exists() and test_path.exists():
        # Check if they are dummy files (1000 rows vs 362400 rows)
        df_len = len(pd.read_csv(train_path, nrows=2000))
        if df_len > 2000:
            print("Real FI-2010 data already exists. Skipping download.")
            return
        else:
            print("Found dummy data. Overwriting with real data...")
    
    try:
        # Load datasets
        print("Fetching 'shanehans/FI2010' (Train Split)...")
        train_ds = load_dataset('shanehans/FI2010', split='train')
        print(f"Loaded train split: {len(train_ds)} rows")
        
        print("Fetching 'shanehans/FI2010' (Test Split)...")
        test_ds = load_dataset('shanehans/FI2010', split='test')
        print(f"Loaded test split: {len(test_ds)} rows")
        
        # Convert to pandas
        print("Converting to pandas DataFrames...")
        train_df = train_ds.to_pandas()
        test_df = test_ds.to_pandas()
        
        # The dataset has an 'Unnamed: 0' column which is the index, we can drop it
        if 'Unnamed: 0' in train_df.columns:
            train_df = train_df.drop(columns=['Unnamed: 0'])
        if 'Unnamed: 0' in test_df.columns:
            test_df = test_df.drop(columns=['Unnamed: 0'])
            
        # Save to CSV
        print(f"Saving to {train_path}...")
        train_df.to_csv(train_path, index=True)
        
        print(f"Saving to {test_path}...")
        test_df.to_csv(test_path, index=True)
        
        print("\n✅ Successfully downloaded and saved real FI-2010 dataset!")
        
    except ImportError:
        print("Error: The 'datasets' library is required.")
        print("Please run: pip install datasets")
        sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

def download_fnspid_subset():
    """Download a subset of FNSPID for testing Phase 2."""
    print("\n" + "=" * 50)
    print("Downloading FNSPID Subset from HuggingFace")
    print("=" * 50)
    
    save_dir = Path("data/news")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / "Stock_News_Dataset.csv"
    
    try:
        # The full FNSPID is 15.7M rows. We will just download the train split
        # but stream it and take the first 10,000 rows for development.
        print("Fetching 'Zihan1004/FNSPID' (Streaming subset)...")
        ds = load_dataset('Zihan1004/FNSPID', split='train', streaming=True)
        
        # Take 10000 records
        subset = []
        for i, row in enumerate(ds):
            subset.append(row)
            if i >= 9999:
                break
                
        df = pd.DataFrame(subset)
        print(f"Loaded subset of {len(df)} news records.")
        
        print(f"Saving to {save_path}...")
        df.to_csv(save_path, index=False)
        print("\n✅ Successfully downloaded FNSPID subset!")
        
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    download_fi2010()
    download_fnspid_subset()
