import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging
from datetime import datetime

class TemporalEncoder(nn.Module):
    """
    Encoder para secuencias temporales de datos OHLC.
    Utiliza LSTM para capturar patrones temporales.
    """
    
    def __init__(self, input_dim: int, embed_dim: int, hidden_dim: int = 128, num_layers: int = 2):
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

class TechnicalEncoder(nn.Module):
    """
    Encoder para indicadores técnicos.
    Comprime múltiples indicadores en un embedding denso.
    """
    
    def __init__(self, input_dim: int, embed_dim: int):
        super(TechnicalEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Red neuronal para comprimir indicadores
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(input_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim, embed_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del encoder técnico.
        
        Args:
            x: Tensor de indicadores técnicos (batch_size, input_dim)
            
        Returns:
            Tensor de embeddings (batch_size, embed_dim)
        """
        return self.encoder(x)

class PatternEncoder(nn.Module):
    """
    Encoder para patrones completos de Wyckoff.
    Combina información temporal y técnica.
    """
    
    def __init__(self, sequence_len: int, input_dim: int, embed_dim: int):
        super(PatternEncoder, self).__init__()
        
        self.sequence_len = sequence_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        
        # Encoder convolucional para patrones
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
        
        # Proyección final
        self.projection = nn.Linear(256, embed_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass del encoder de patrones.
        
        Args:
            x: Tensor de secuencias (batch_size, sequence_len, input_dim)
            
        Returns:
            Tensor de embeddings (batch_size, embed_dim)
        """
        # Transponer para conv1d: (batch_size, input_dim, sequence_len)
        x = x.transpose(1, 2)
        
        # Aplicar convoluciones
        conv_out = self.conv_layers(x)  # (batch_size, 256, 1)
        conv_out = conv_out.squeeze(-1)  # (batch_size, 256)
        
        # Proyección final
        embedding = self.projection(conv_out)
        
        return embedding

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
        Codifica un patrón completo.
        
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
        Codifica el estado completo del mercado combinando información temporal y técnica.
        
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
            Lista de (identificador, similitud) ordenada por similitud
        """
        similarities = []
        
        for pattern_id, stored_embedding in embedding_database.items():
            # Calcular similitud coseno
            similarity = self._cosine_similarity(query_embedding, stored_embedding)
            similarities.append((pattern_id, similarity))
        
        # Ordenar por similitud descendente
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calcula similitud coseno entre dos vectores.
        
        Args:
            a, b: Vectores a comparar
            
        Returns:
            Similitud coseno
        """
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def reduce_dimensionality(self, 
                            embeddings: np.ndarray, 
                            method: str = 'pca',
                            n_components: int = 2) -> np.ndarray:
        """
        Reduce la dimensionalidad de embeddings para visualización.
        
        Args:
            embeddings: Array de embeddings (n_samples, embed_dim)
            method: Método de reducción ('pca' o 'tsne')
            n_components: Número de componentes finales
            
        Returns:
            Embeddings reducidos
        """
        if method == 'pca':
            reducer = PCA(n_components=n_components)
        elif method == 'tsne':
            reducer = TSNE(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Método no soportado: {method}")
        
        reduced_embeddings = reducer.fit_transform(embeddings)
        
        return reduced_embeddings
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], filepath: str):
        """
        Guarda embeddings en archivo.
        
        Args:
            embeddings: Diccionario de embeddings
            filepath: Ruta del archivo
        """
        np.savez_compressed(filepath, **embeddings)
        self.logger.info(f"Embeddings guardados en {filepath}")
    
    def load_embeddings(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        Carga embeddings desde archivo.
        
        Args:
            filepath: Ruta del archivo
            
        Returns:
            Diccionario de embeddings
        """
        loaded = np.load(filepath)
        embeddings = {key: loaded[key] for key in loaded.files}
        
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