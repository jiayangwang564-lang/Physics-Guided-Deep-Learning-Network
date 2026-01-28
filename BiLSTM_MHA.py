import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BiLSTMAttention(nn.Module):
    """
    Physics-Informed BiLSTM-Attention Model for Flood Forecasting.

    Architecture:
    1. Input Layer: Accepts multi-variable time series (Rainfall, Soil Saturation, Runoff, Historical Depth).
    2. Bi-LSTM Layer: Captures temporal dependencies in both forward and backward directions.
    3. Self-Attention Layer: Weights the importance of different time steps relative to the prediction target.
    4. Output Layer: Predicts the water depth for the next time step.
    """

    def __init__(self, input_size, hidden_size, num_layers=2, num_heads=4, output_size=1, dropout=0.2):
        """
        Args:
            input_size (int): Number of input features (e.g., Rain, Saturation, etc.).
            hidden_size (int): Number of features in the hidden state of LSTM.
            num_layers (int): Number of recurrent layers.
            num_heads (int): Number of heads in the multiheadattention models.
            output_size (int): Dimension of the output (usually 1 for water depth).
            dropout (float): Dropout probability.
        """
        super(BiLSTMAttention, self).__init__()

        # 1. Bidirectional LSTM
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 2. Multi-Head Attention
        # Note: embed_dim is hidden_size * 2 because LSTM is bidirectional
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=num_heads,
            batch_first=True,
            dropout=dropout
        )

        # 3. Layer Normalization & Fully Connected Layers
        self.layer_norm = nn.LayerNorm(hidden_size * 2)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_size)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Prediction of shape (batch_size, output_size).
            torch.Tensor: Attention weights (optional, for visualization).
        """
        # LSTM Forward
        # lstm_out shape: [batch_size, seq_len, hidden_size * 2]
        lstm_out, _ = self.lstm(x)

        # Attention Mechanism
        # query=key=value=lstm_out (Self-Attention)
        # attn_out shape: [batch_size, seq_len, hidden_size * 2]
        attn_out, attn_weights = self.attention(lstm_out, lstm_out, lstm_out)

        # Residual Connection + Norm
        out = self.layer_norm(attn_out + lstm_out)

        # Global Average Pooling (aggregating temporal information)
        context_vector = torch.mean(out, dim=1)

        # Final Prediction
        prediction = self.fc(context_vector)

        return prediction


class PhysicsInformedLoss(nn.Module):
    """
    Custom Loss Function incorporating Physical Constraints.

    Formula:
        Loss = MSE + lambda * L_phy

    Where L_phy penalizes predictions where water depth increases significantly
    without corresponding rainfall (violating mass conservation/causality).
    """

    def __init__(self, lambda_phy=0.1, threshold=0.1):
        """
        Args:
            lambda_phy (float): Weight for the physical constraint term.
            threshold (float): Tolerance threshold for noise in physical causality.
        """
        super(PhysicsInformedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.lambda_phy = lambda_phy
        self.threshold = threshold

    def forward(self, pred, target, rain_input):
        """
        Args:
            pred (torch.Tensor): Predicted water depth (Normalized).
            target (torch.Tensor): Actual water depth (Normalized).
            rain_input (torch.Tensor): Rainfall input corresponding to the prediction step (Normalized).
        """
        # 1. Data-driven Loss (MSE)
        loss_mse = self.mse(pred, target)

        # 2. Physics-based Constraint (Causality Check)
        # Logic: If Depth increases (pred > 0) but Rain is zero (rain_input ~ 0), it's a violation.
        # Simplified Implementation: Penalize if Pred > Rain + Threshold
        # (Assuming inputs are normalized, this checks relative magnitude violations)
        physical_violation = torch.relu(pred.squeeze() - (rain_input.squeeze() + self.threshold))

        loss_phy = torch.mean(physical_violation ** 2)

        # Total Loss
        total_loss = loss_mse + self.lambda_phy * loss_phy

        return total_loss


# ==========================================
# Utility: XAJ Physical Mechanism (Optional)
# ==========================================
def xaj_mechanism(rainfall, w_m=120.0):
    """
    A simplified Xinanjiang (XAJ) model component to generate physical features.
    Can be used as a preprocessing step before the Neural Network.

    Args:
        rainfall (np.array): Time series of rainfall.
        w_m (float): Tension water storage capacity (mm).

    Returns:
        tuple: (saturation_series, runoff_series)
    """
    n = len(rainfall)
    saturation = np.zeros(n)
    runoff = np.zeros(n)
    w_curr = w_m * 0.5  # Initial state assumption

    for t in range(n):
        p = rainfall[t]
        pe = max(0, p - 0.2)  # Net rain after simplified evaporation

        if w_curr + pe > w_m:
            runoff[t] = (w_curr + pe) - w_m
            w_curr = w_m
        else:
            runoff[t] = 0
            w_curr += pe

        saturation[t] = w_curr / w_m

    return saturation, runoff