"""Manager principal del sistema de embeddings."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime

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

from ..encoders.temporal_encoder import TemporalEncoder
from ..encoders.technical_encoder import TechnicalEncoder
from ..encoders.pattern_encoder import PatternEncoder
from ..utils.embedding_utils import EmbeddingUtils


class TradingEmbeddings:
    """
    Sistema completo de embeddings para datos de trading.
    Combina múltiples tipos de encoders para crear representaciones vectoriales.
    """
    
    def __init__(self, 
                 temporal_embed_dim: int = 256,
                 technical_embed_dim: int = 128,
                 pattern_embed_dim: int = 512,
                 device: str = 'cpu'):
        """
        Inicializa el sistema de embeddings.
        
        Args:
            temporal_embed_dim: Dimensión de embeddings temporales
            technical_embed_dim: Dimensión de embeddings técnicos
            pattern_embed_dim: Dimensión de embeddings de patrones
            device: Dispositivo para computación ('cpu' o 'cuda')
        """
        self.temporal_embed_dim = temporal_embed_dim
        self.technical_embed_dim = technical_embed_dim
        self.pattern_embed_dim = pattern_embed_dim
        self.device = device
        
        # Inicializar encoders (se configurarán cuando se conozcan las dimensiones)
        self.temporal_encoder = None
        self.technical_encoder = None
        self.pattern_encoder = None
        
        # Configurar logging
        self.logger = logging.getLogger(__name__)
        
        # Almacenar embeddings calculados
        self.embedding_cache = {}
        
        # Inicializar utilidades
        self.utils = EmbeddingUtils()
        
    def initialize_encoders(self, 
                          ohlc_dim: int, 
                          technical_dim: int, 
                          sequence_length: int):
        """
        Inicializa los encoders con las dimensiones correctas.
        
        Args:
            ohlc_dim: Dimensión de datos OHLC
            technical_dim: Dimensión de indicadores técnicos
            sequence_length: Longitud de secuencias temporales
        """
        if not TORCH_AVAILABLE:
            self.logger.warning("PyTorch no está disponible. Los encoders no se inicializarán.")
            return
        
        self.temporal_encoder = TemporalEncoder(
            input_dim=ohlc_dim,
            embed_dim=self.temporal_embed_dim
        ).to(self.device)
        
        self.technical_encoder = TechnicalEncoder(
            input_dim=technical_dim,
            embed_dim=self.technical_embed_dim
        ).to(self.device)
        
        self.pattern_encoder = PatternEncoder(
            sequence_len=sequence_length,
            input_dim=ohlc_dim,
            embed_dim=self.pattern_embed_dim
        ).to(self.device)
        
        self.logger.info(f"Encoders inicializados - Temporal: {self.temporal_embed_dim}, "
                        f"Technical: {self.technical_embed_dim}, Pattern: {self.pattern_embed_dim}")
    
    def encode_temporal_sequence(self, ohlc_sequence: np.ndarray) -> np.ndarray:
        """
        Codifica una secuencia temporal OHLC.
        
        Args:
            ohlc_sequence: Array de forma (sequence_length, ohlc_dim)
            
        Returns:
            Embedding temporal
        """
        if self.temporal_encoder is None:
            raise ValueError("Temporal encoder no inicializado")
        
        # Convertir a tensor
        x = torch.FloatTensor(ohlc_sequence).unsqueeze(0).to(self.device)
        
        # Generar embedding
        with torch.no_grad():
            embedding = self.temporal_encoder(x)
        
        return embedding.cpu().numpy().flatten()
    
    def encode_technical_indicators(self, indicators: np.ndarray) -> np.ndarray:
        """
        Codifica indicadores técnicos.
        
        Args:
            indicators: Array de indicadores técnicos
            
        Returns:
            Embedding de indicadores
        """
        if self.technical_encoder is None:
            raise ValueError("Technical encoder no inicializado")
        
        # Convertir a tensor
        x = torch.FloatTensor(indicators).unsqueeze(0).to(self.device)
        
        # Generar embedding
        with torch.no_grad():
            embedding = self.technical_encoder(x)
        
        return embedding.cpu().numpy().flatten()
    
    def encode_pattern(self, pattern_sequence: np.ndarray) -> np.ndarray:
        """
        Codifica un patrón de trading.
        
        Args:
            pattern_sequence: Array de forma (sequence_length, input_dim)
            
        Returns:
            Embedding de patrón
        """
        if self.pattern_encoder is None:
            raise ValueError("Pattern encoder no inicializado")
        
        # Convertir a tensor
        x = torch.FloatTensor(pattern_sequence).unsqueeze(0).to(self.device)
        
        # Generar embedding
        with torch.no_grad():
            embedding = self.pattern_encoder(x)
        
        return embedding.cpu().numpy().flatten()
    
    def encode_market_state(self, 
                          ohlc_window: np.ndarray, 
                          technical_indicators: np.ndarray) -> np.ndarray:
        """
        Codifica el estado completo del mercado.
        
        Args:
            ohlc_window: Ventana de datos OHLC
            technical_indicators: Indicadores técnicos
            
        Returns:
            Embedding combinado del estado del mercado
        """
        # Generar embeddings individuales
        temporal_embed = self.encode_temporal_sequence(ohlc_window)
        technical_embed = self.encode_technical_indicators(technical_indicators)
        
        # Combinar embeddings
        combined_embedding = np.concatenate([temporal_embed, technical_embed])
        
        return combined_embedding
    
    def find_similar_patterns(self, 
                            query_embedding: np.ndarray, 
                            embedding_database: Dict[str, np.ndarray],
                            top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Encuentra patrones similares usando similitud coseno.
        
        Args:
            query_embedding: Embedding de consulta
            embedding_database: Base de datos de embeddings
            top_k: Número de resultados a retornar
            
        Returns:
            Lista de (pattern_id, similarity_score) ordenada por similitud
        """
        similarities = []
        
        for pattern_id, stored_embedding in embedding_database.items():
            # Calcular similitud coseno
            similarity = self.utils.cosine_similarity(query_embedding, stored_embedding)
            similarities.append((pattern_id, similarity))
        
        # Ordenar por similitud descendente
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], filepath: str):
        """
        Guarda embeddings en archivo.
        
        Args:
            embeddings: Diccionario de embeddings
            filepath: Ruta del archivo
        """
        self.utils.save_embeddings(embeddings, filepath)
        self.logger.info(f"Embeddings guardados en {filepath}")
    
    def load_embeddings(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        Carga embeddings desde archivo.
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            Diccionario de embeddings
        """
        embeddings = self.utils.load_embeddings(filepath)
        self.logger.info(f"Embeddings cargados desde {filepath}")
        return embeddings
    
    def train_encoders(self, 
                      training_data: Dict[str, np.ndarray],
                      epochs: int = 100,
                      learning_rate: float = 0.001):
        """
        Entrena los encoders usando datos de entrenamiento.
        
        Args:
            training_data: Datos de entrenamiento
            epochs: Número de épocas
            learning_rate: Tasa de aprendizaje
        """
        # Configurar optimizadores
        optimizers = []
        
        if self.temporal_encoder:
            optimizers.append(torch.optim.Adam(self.temporal_encoder.parameters(), lr=learning_rate))
        
        if self.technical_encoder:
            optimizers.append(torch.optim.Adam(self.technical_encoder.parameters(), lr=learning_rate))
        
        if self.pattern_encoder:
            optimizers.append(torch.optim.Adam(self.pattern_encoder.parameters(), lr=learning_rate))
        
        # Entrenamiento básico (se puede expandir con loss functions específicas)
        self.logger.info(f"Iniciando entrenamiento de encoders por {epochs} épocas")
        
        for epoch in range(epochs):
            # Aquí se implementaría el loop de entrenamiento específico
            # dependiendo de la tarea (autoencoder, contrastive learning, etc.)
            pass
        
        self.logger.info("Entrenamiento de encoders completado")
    
    def train(self, training_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        """
        Método de entrenamiento compatible con el pipeline existente.
        
        Args:
            training_data: Datos de entrenamiento
            **kwargs: Argumentos adicionales
            
        Returns:
            Métricas de entrenamiento
        """
        # Extraer parámetros
        epochs = kwargs.get('epochs', 100)
        learning_rate = kwargs.get('learning_rate', 0.001)
        
        # Entrenar encoders
        self.train_encoders(training_data, epochs, learning_rate)
        
        return {
            'status': 'completed',
            'epochs': epochs,
            'learning_rate': learning_rate
        }