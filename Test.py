import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os

# Import model definition
from BiLSTM_MHA import BiLSTMAttention


# ==========================================
# Shared Utilities (Must match train.py)
# ==========================================
def calculate_xaj_features(rainfall_series, w_m=120.0):
    """Calculate Soil Saturation and Runoff using XAJ mechanism."""
    n = len(rainfall_series)
    sat, runoff = np.zeros(n), np.zeros(n)
    w_curr = w_m * 0.6
    for t in range(n):
        p = rainfall_series[t]
        pe = max(0, p - 0.2)
        if w_curr + pe > w_m:
            runoff[t] = (w_curr + pe) - w_m
            w_curr = w_m
        else:
            w_curr += pe
        sat[t] = w_curr / w_m
    return sat, runoff


class RealFloodDataset(Dataset):
    def __init__(self, X, y, rains, seq_len):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.rains = torch.FloatTensor(rains)
        self.seq_len = seq_len

    def __len__(self): return len(self.X) - self.seq_len

    def __getitem__(self, i): return (
    self.X[i:i + self.seq_len], self.y[i + self.seq_len], self.rains[i + self.seq_len])


# ==========================================
# Testing and Evaluation Core Logic
# ==========================================
def test_model():
    print("[TEST] 1. Processing Data...")

    # Check if the weight file exists
    if not os.path.exists('model_weights.pth'):
        raise FileNotFoundError("Error: 'model_weights.pth' not found. Please run train.py first.")

    # Read data
    df = pd.read_csv('Data.csv')
    df.columns = df.columns.str.strip()  # Remove spaces from column

    # Mapping
    rename_mapping = {'Rainfall(mm/h)': 'Rain', 'True Water Depth (m)': 'Depth_True'}
    df.rename(columns=rename_mapping, inplace=True)
    df = df.fillna(0)

    # Reconstruct feature engineering (must be identical to the training phase)
    raw_rain = df['Rain'].values
    raw_depth = df['Depth_True'].values
    sat, runoff = calculate_xaj_features(raw_rain)

    features = np.stack((raw_rain, sat, runoff, raw_depth), axis=1)
    targets = raw_depth.reshape(-1, 1)

    # Normalization
    scaler_x = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(features)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(targets)
    rains_scaled = X_scaled[:, 0]

    # Partitioning the test set (last 20%)
    seq_len = 6
    split_idx = int(len(X_scaled) * 0.8)

    test_ds = RealFloodDataset(X_scaled[split_idx:], y_scaled[split_idx:], rains_scaled[split_idx:], seq_len)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # Load model
    print("[TEST] 2. Loading Model Weights...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTMAttention(input_size=4, hidden_size=64).to(device)

    # Load weights (process map_location to account for device inconsistencies)
    model.load_state_dict(torch.load('model_weights.pth', map_location=device))
    model.eval()

    # Start prediction
    print("[TEST] 3. Running Inference...")
    preds, trues = [], []
    with torch.no_grad():
        for bx, by, _ in test_loader:
            bx = bx.to(device)
            p = model(bx)
            preds.append(p.cpu().numpy())
            trues.append(by.cpu().numpy())

    # De-normalization (conversion back to actual water level in meters)
    pred_real = scaler_y.inverse_transform(np.concatenate(preds))
    true_real = scaler_y.inverse_transform(np.concatenate(trues))

    # Calculate Core Metrics
    nse = 1 - np.sum((true_real - pred_real) ** 2) / np.sum((true_real - np.mean(true_real)) ** 2)
    rmse = np.sqrt(mean_squared_error(true_real, pred_real))

    print("-" * 30)
    print(f"Final Evaluation Results:")
    print(f"NSE  (Nash-Sutcliffe) : {nse:.4f}")
    print(f"RMSE (Root Mean Sq)   : {rmse:.4f}")
    print("-" * 30)


if __name__ == '__main__':
    test_model()