"""
Industrial Time Series Dataset Handling

Features:
- Windowing for temporal models
- Multiple prediction horizons
- Dynamic feature selection
- On-the-fly normalization
- Data augmentation
- Memory-efficient loading
- Multi-worker compatibility
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Optional, Tuple

class TimeSeriesDataset(Dataset):
    """Enterprise-grade time series dataset"""
    
    def __init__(self, 
                 data: pd.DataFrame,
                 window_size: int = 24,
                 horizon: int = 1,
                 target_cols: list = ["co2e_kg"],
                 feature_cols: Optional[list] = None,
                 normalize: bool = True,
                 augmentation: bool = False):
        
        self.window_size = window_size
        self.horizon = horizon
        self.target_cols = target_cols
        self.feature_cols = feature_cols or [c for c in data.columns if c not in target_cols]
        self.augmentation = augmentation
        
        # Store raw data and parameters
        self.data = data[self.feature_cols + self.target_cols]
        self.n_samples = len(data) - window_size - horizon + 1
        
        # Initialize normalization
        self.normalize = normalize
        self.scalers = {}
        if self.normalize:
            self._init_normalization()
            
    def _init_normalization(self):
        """Fit scalers on initialization data"""
        for col in self.data.columns:
            scaler = StandardScaler()
            scaler.fit(self.data[[col]].values)
            self.scalers[col] = scaler
            
    def _normalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply stored normalization"""
        return pd.DataFrame(
            {col: self.scalers[col].transform(df[[col]].values.ravel())
             for col in df.columns}
        )
    
    def _augment(self, window: pd.DataFrame) -> pd.DataFrame:
        """Apply data augmentation"""
        # Add Gaussian noise
        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.01, size=window.shape)
            window += noise
            
        # Random scaling
        if np.random.rand() < 0.3:
            scale = np.random.uniform(0.9, 1.1)
            window *= scale
            
        return window
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if idx >= len(self):
            raise IndexError
            
        # Extract window
        window_start = idx
        window_end = idx + self.window_size
        target_end = window_end + self.horizon
        
        window_data = self.data.iloc[window_start:window_end]
        target_data = self.data[self.target_cols].iloc[window_end:target_end]
        
        # Apply normalization
        if self.normalize:
            window_data = self._normalize(window_data)
            target_data = self._normalize(target_data)
            
        # Convert to numpy arrays
        window = window_data.values.astype(np.float32)
        target = target_data.values.astype(np.float32)
        
        # Apply augmentation
        if self.augmentation and self.mode == "train":
            window = self._augment(window)
            
        return torch.from_numpy(window), torch.from_numpy(target)
    
    def split(self, test_size: float = 0.2) -> Tuple['TimeSeriesDataset', 'TimeSeriesDataset']:
        """Split dataset into train/test"""
        split_idx = int(len(self) * (1 - test_size))
        train_data = self.data.iloc[:split_idx + self.window_size]
        test_data = self.data.iloc[split_idx:]
        
        train_ds = TimeSeriesDataset(
            train_data,
            window_size=self.window_size,
            horizon=self.horizon,
            target_cols=self.target_cols,
            feature_cols=self.feature_cols,
            normalize=False  # Already normalized
        )
        
        test_ds = TimeSeriesDataset(
            test_data,
            window_size=self.window_size,
            horizon=self.horizon,
            target_cols=self.target_cols,
            feature_cols=self.feature_cols,
            normalize=False
        )
        
        # Copy scalers
        train_ds.scalers = self.scalers
        test_ds.scalers = self.scalers
        
        return train_ds, test_ds

class MultiHorizonDataset(TimeSeriesDataset):
    """Advanced dataset for multi-horizon forecasting"""
    
    def __init__(self, 
                 horizons: list = [1, 6, 24],
                 **kwargs):
        super().__init__(**kwargs)
        self.horizons = sorted(horizons)
        self.max_horizon = max(horizons)
        self.n_samples = len(self.data) - self.window_size - self.max_horizon + 1
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        window, _ = super().__getitem__(idx)
        targets = {}
        
        for horizon in self.horizons:
            target_end = idx + self.window_size + horizon
            target_data = self.data[self.target_cols].iloc[target_end - 1]
            if self.normalize:
                target_data = self._normalize(target_data)
            targets[f"h{horizon}"] = torch.from_numpy(target_data.values.astype(np.float32))
            
        return window, targets

# Example usage:
"""
processed_data = pd.read_parquet("data/processed.parquet")
dataset = TimeSeriesDataset(
    processed_data,
    window_size=24,
    horizon=6,
    target_cols=["co2e_kg"],
    normalize=True
)

train_ds, test_ds = dataset.split(test_size=0.2)

train_loader = DataLoader(
    train_ds,
    batch_size=32,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

# Multi-horizon example
multi_ds = MultiHorizonDataset(
    processed_data,
    window_size=24,
    horizons=[1, 6, 24],
    target_cols=["co2e_kg"]
)
"""
