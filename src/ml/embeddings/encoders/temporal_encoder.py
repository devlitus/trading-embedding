"""Encoder temporal para secuencias OHLC."""

import numpy as np
import logging
from typing import Optional

# Optional imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

# Define base class conditionally
if TORCH_AVAILABLE:
    _TemporalEncoderBase = nn.Module
else:
    _TemporalEncoderBase = object


class TemporalEncoder(_TemporalEncoderBase):
    """
    Encoder para secuencias temporales de datos OHLC.
    Utiliza LSTM para capturar patrones temporales.
    """
    
    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int = 128, num_layers: int = 2):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TemporalEncoder but is not installed")
        
        super(TemporalEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # LSTM para procesar secuencias temporales
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2 if num_layers > 1 else 0
        )
        
        # Capa de proyección a embedding
        self.projection = nn.Sequential(
            nn.Linear(hidden_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x) -> np.ndarray:
        """
        Forward pass del encoder temporal.
        
        Args:
            x: Tensor de forma (batch_size, sequence_length, input_dim)
            
        Returns:
            Tensor de embeddings de forma (batch_size, embed_dim)
        """
        # Procesar con LSTM
        lstm_out, (hidden, _) = self.lstm(x)
        
        # Usar el último estado oculto
        last_hidden = hidden[-1]  # (batch_size, hidden_dim)
        
        # Proyectar a embedding
        embedding = self.projection(last_hidden)
        
        return embedding