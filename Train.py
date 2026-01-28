import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

# Import model definition
from BiLSTM_MHA import BiLSTMAttention, PhysicsInformedLoss


# ==========================================
# Shared Utilities (Must match test.py)
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

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, i):
        return (self.X[i:i + self.seq_len], self.y[i + self.seq_len], self.rains[i + self.seq_len])


# ==========================================
# Training Pipeline
# ==========================================
def train_model():
    print("[TRAIN] 1. Loading Data...")
    df = pd.read_csv('Data.csv')
    df.columns = df.columns.str.strip()  # Clean whitespace

    # Map columns
    rename_mapping = {'Rainfall(mm/h)': 'Rain', 'True Water Depth (m)': 'Depth_True'}
    df.rename(columns=rename_mapping, inplace=True)
    df = df.fillna(0)

    # Feature Engineering
    raw_rain = df['Rain'].values
    raw_depth = df['Depth_True'].values
    sat, runoff = calculate_xaj_features(raw_rain)

    # Prepare Inputs
    features = np.stack((raw_rain, sat, runoff, raw_depth), axis=1)
    targets = raw_depth.reshape(-1, 1)

    # Normalization
    scaler_x = MinMaxScaler()
    X_scaled = scaler_x.fit_transform(features)
    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(targets)
    rains_scaled = X_scaled[:, 0]

    # Split Data (80% Train)
    seq_len = 6
    split_idx = int(len(X_scaled) * 0.8)
    train_ds = RealFloodDataset(X_scaled[:split_idx], y_scaled[:split_idx], rains_scaled[:split_idx], seq_len)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # Initialize Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiLSTMAttention(input_size=4, hidden_size=64).to(device)
    criterion = PhysicsInformedLoss(lambda_phy=0.2)
    optimizer = optim.Adam(model.parameters(), lr=0.005)

    # Training Loop
    print("[TRAIN] 2. Starting Training Loop...")
    model.train()
    best_loss = float('inf')

    for epoch in range(50):
        total_loss = 0
        for bx, by, brain in train_loader:
            bx, by, brain = bx.to(device), by.to(device), brain.to(device)
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by, brain)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 10 == 0:
            print(f"      Epoch {epoch + 1:02d} | Loss: {avg_loss:.6f}")

        # Save best model logic (simplified)
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'model_weights.pth')

    print(f"[TRAIN] Done. Best model saved to 'model_weights.pth' (Loss: {best_loss:.6f})")


if __name__ == '__main__':
    train_model()