"""Encoder para patrones de trading."""

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
    _PatternEncoderBase = nn.Module
else:
    _PatternEncoderBase = object


class PatternEncoder(_PatternEncoderBase):
    """
    Encoder para patrones de trading usando CNN.
    Detecta patrones locales en secuencias de precios.
    """
    
    def __init__(self, sequence_len: int, input_dim: int, embed_dim: int):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for PatternEncoder but is not installed")
        
        super(PatternEncoder, self).__init__()
        
        self.sequence_len = sequence_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Capas convolucionales para detectar patrones
        self.conv_layers = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        
        # Capa de proyección a embedding
        self.projection = nn.Sequential(
            nn.Linear(256, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x) -> np.ndarray:
        """
        Forward pass del encoder de patrones.
        
        Args:
            x: Tensor de forma (batch_size, sequence_len, input_dim)
            
        Returns:
            Tensor de embeddings (batch_size, embed_dim)
        """
        # Transponer para CNN: (batch_size, input_dim, sequence_len)
        x = x.transpose(1, 2)
        
        # Aplicar convoluciones
        conv_out = self.conv_layers(x)
        
        # Aplanar
        conv_out = conv_out.view(conv_out.size(0), -1)
        
        # Proyectar a embedding
        embedding = self.projection(conv_out)
        
        return embedding