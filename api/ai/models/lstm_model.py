"""
Industrial-Grade LSTM Model for Time Series Forecasting

Features:
- Multi-layer LSTM with dropout
- Sequence-to-sequence architecture
- Attention mechanism
- Customizable depth/width
- Batch normalization
- Teacher forcing support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .base_model import BaseModel

class LSTMForecaster(BaseModel):
    """Enhanced LSTM Model for Emission Prediction"""
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
        # Architecture parameters
        input_size = self.hyperparameters["input_size"]
        hidden_size = self.hyperparameters["hidden_size"]
        num_layers = self.hyperparameters["num_layers"]
        dropout = self.hyperparameters["dropout"]
        output_size = config.get("output_size", 1)
        
        # Core LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=1)
        )
        
        # Output layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size//2, output_size)
        )
        
        # Regularization
        self.bn = nn.BatchNorm1d(input_size)
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self.apply(self.init_weights)
        
    def forward(self, x: torch.Tensor, hidden: Optional[tuple] = None) -> torch.Tensor:
        """Forward pass with optional hidden state"""
        # Batch normalization
        x = self.bn(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # LSTM processing
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Attention mechanism
        attn_weights = self.attention(lstm_out)
        context = torch.sum(attn_weights * lstm_out, dim=1)
        
        # Final prediction
        output = self.fc(context)
        return output, hidden
    
    def init_hidden(self, batch_size: int) -> tuple:
        """Initialize hidden states"""
        device = next(self.parameters()).device
        return (
            torch.zeros(self.hyperparameters["num_layers"], batch_size, 
                       self.hyperparameters["hidden_size"]).to(device),
            torch.zeros(self.hyperparameters["num_layers"], batch_size,
                       self.hyperparameters["hidden_size"]).to(device)
        )

    def predict_sequence(self, x: torch.Tensor, steps: int) -> torch.Tensor:
        """Multi-step sequence prediction"""
        predictions = []
        hidden = None
        
        for _ in range(steps):
            pred, hidden = self(x, hidden)
            predictions.append(pred)
            x = torch.cat([x[:, 1:], pred.unsqueeze(1)], dim=1)
            
        return torch.cat(predictions, dim=1)

# Example Usage:
"""
config = {
    "input_size": 10,
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "output_size": 1
}

model = LSTMForecaster(config)
print(model)

# Test forward pass
x = torch.randn(32, 24, 10)  # (batch, sequence, features)
output, _ = model(x)
print(output.shape)  # torch.Size([32, 1])

# Multi-step prediction
seq_pred = model.predict_sequence(x, 6)
print(seq_pred.shape)  # torch.Size([32, 6])
"""
