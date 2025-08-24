"""Encoder para indicadores técnicos."""

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
    _TechnicalEncoderBase = nn.Module
else:
    _TechnicalEncoderBase = object


class TechnicalEncoder(_TechnicalEncoderBase):
    """
    Encoder para indicadores técnicos.
    Comprime múltiples indicadores en un embedding denso.
    """
    
    def __init__(self, input_dim: int, embed_dim: int):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for TechnicalEncoder but is not installed")
        
        super(TechnicalEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Red neuronal para procesar indicadores
        self.network = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x) -> np.ndarray:
        """
        Forward pass del encoder técnico.
        
        Args:
            x: Tensor de indicadores técnicos (batch_size, input_dim)
            
        Returns:
            Tensor de embeddings (batch_size, embed_dim)
        """
        # Procesar indicadores
        embedding = self.network(x)
        
        return embedding