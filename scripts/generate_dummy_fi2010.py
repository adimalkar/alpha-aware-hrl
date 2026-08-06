import pandas as pd
import numpy as np
from pathlib import Path

def generate_dummy_data(n_samples, filename):
    # 144 features
    features = np.random.randn(n_samples, 144)
    # 5 horizons of labels (k=1,2,3,5,10) - labels are 1 (down), 2 (stable), 3 (up)
    labels = np.random.randint(1, 4, size=(n_samples, 5))
    
    data = np.hstack([features, labels])
    columns = [f'f_{i}' for i in range(144)] + ['k1', 'k2', 'k3', 'k5', 'k10']
    
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(filename)
    print(f"Saved {filename} with shape {df.shape}")

if __name__ == "__main__":
    Path("data/fi2010/FI2010").mkdir(parents=True, exist_ok=True)
    generate_dummy_data(5000, "data/fi2010/FI2010/FI2010_train.csv")
    generate_dummy_data(1000, "data/fi2010/FI2010/FI2010_test.csv")
