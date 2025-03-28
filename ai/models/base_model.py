"""
Base Model Architecture with Core Functionality

Features:
- Weight initialization
- Model saving/loading
- Parameter counting
- Common layer definitions
- Hyperparameter tracking
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import torch
import torch.nn as nn
import torch.nn.init as init

class BaseModel(nn.Module):
    """Abstract base model with essential utilities"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self._init_weights = True
        self._record_hyperparameters()
        
    def _record_hyperparameters(self):
        """Store model configuration metadata"""
        self.hyperparameters = {
            "model_type": self.__class__.__name__,
            "input_size": self.config.get("input_size", 10),
            "hidden_size": self.config.get("hidden_size", 64),
            "num_layers": self.config.get("num_layers", 2),
            "dropout": self.config.get("dropout", 0.2),
            "created_at": torch.datetime.now().isoformat()
        }
        
    def init_weights(self, module: nn.Module):
        """Xavier/Glorot initialization for linear layers"""
        if isinstance(module, nn.Linear):
            init.xavier_normal_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if "weight_ih" in name:
                    init.xavier_normal_(param.data)
                elif "weight_hh" in name:
                    init.orthogonal_(param.data)
                elif "bias" in name:
                    param.data.fill_(0)
                    
    def save(self, path: Path, metadata: Optional[Dict] = None):
        """Save model with metadata"""
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "hyperparameters": self.hyperparameters,
            "metadata": metadata or {}
        }, path)
        print(f"Model saved to {path}")
        
    @classmethod
    def load(cls, path: Path, config: Dict, map_location=None):
        """Load model with configuration"""
        checkpoint = torch.load(path, map_location=map_location)
        model = cls(config)
        model.load_state_dict(checkpoint["state_dict"])
        model.hyperparameters.update(checkpoint["metadata"])
        return model
        
    def count_parameters(self) -> Dict[str, int]:
        """Return parameter counts (total/trainable)"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "total_parameters": f"{total:,}",
            "trainable_parameters": f"{trainable:,}"
        }
        
    def get_config(self) -> Dict:
        """Return model configuration"""
        return {
            "architecture": self.__class__.__name__,
            **self.hyperparameters,
            **self.config
        }
        
    def __str__(self) -> str:
        """Human-readable model summary"""
        config = json.dumps(self.get_config(), indent=2)
        params = self.count_parameters()
        return (
            f"{self.__class__.__name__} Summary:\n"
            f"Parameters: {params['trainable_parameters']} trainable/"
            f"{params['total_parameters']} total\n"
            f"Configuration:\n{config}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Base forward pass (to be implemented by subclasses)"""
        raise NotImplementedError("Forward method must be implemented by subclass")
