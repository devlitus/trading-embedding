"""Utilidades para el manejo de embeddings."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# Optional imports
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    PCA = None
    TSNE = None


class EmbeddingUtils:
    """
    Utilidades para el manejo y procesamiento de embeddings.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calcula la similitud coseno entre dos vectores.
        
        Args:
            a: Primer vector
            b: Segundo vector
            
        Returns:
            Similitud coseno (-1 a 1)
        """
        # Calcular producto punto
        dot_product = np.dot(a, b)
        
        # Calcular normas
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        # Evitar división por cero
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def euclidean_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calcula la distancia euclidiana entre dos vectores.
        
        Args:
            a: Primer vector
            b: Segundo vector
            
        Returns:
            Distancia euclidiana
        """
        return np.linalg.norm(a - b)
    
    def manhattan_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Calcula la distancia Manhattan entre dos vectores.
        
        Args:
            a: Primer vector
            b: Segundo vector
            
        Returns:
            Distancia Manhattan
        """
        return np.sum(np.abs(a - b))
    
    def normalize_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Normaliza un embedding a norma unitaria.
        
        Args:
            embedding: Vector de embedding
            
        Returns:
            Embedding normalizado
        """
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    
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
        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn no está disponible. Retornando embeddings originales.")
            return embeddings[:, :n_components] if embeddings.shape[1] >= n_components else embeddings
        
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
    
    def compute_embedding_statistics(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Calcula estadísticas de un conjunto de embeddings.
        
        Args:
            embeddings: Array de embeddings (n_samples, embed_dim)
            
        Returns:
            Diccionario con estadísticas
        """
        return {
            'mean': np.mean(embeddings, axis=0),
            'std': np.std(embeddings, axis=0),
            'min': np.min(embeddings, axis=0),
            'max': np.max(embeddings, axis=0),
            'shape': embeddings.shape,
            'norm_mean': np.mean([np.linalg.norm(emb) for emb in embeddings]),
            'norm_std': np.std([np.linalg.norm(emb) for emb in embeddings])
        }
    
    def find_outliers(self, embeddings: np.ndarray, threshold: float = 2.0) -> List[int]:
        """
        Encuentra embeddings atípicos basados en la distancia a la media.
        
        Args:
            embeddings: Array de embeddings (n_samples, embed_dim)
            threshold: Umbral en desviaciones estándar
            
        Returns:
            Lista de índices de embeddings atípicos
        """
        # Calcular centroide
        centroid = np.mean(embeddings, axis=0)
        
        # Calcular distancias al centroide
        distances = [self.euclidean_distance(emb, centroid) for emb in embeddings]
        
        # Calcular umbral
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        outlier_threshold = mean_dist + threshold * std_dist
        
        # Encontrar outliers
        outliers = [i for i, dist in enumerate(distances) if dist > outlier_threshold]
        
        return outliers
    
    def cluster_embeddings(self, embeddings: np.ndarray, n_clusters: int = 5) -> np.ndarray:
        """
        Agrupa embeddings usando K-means.
        
        Args:
            embeddings: Array de embeddings (n_samples, embed_dim)
            n_clusters: Número de clusters
            
        Returns:
            Array de etiquetas de cluster
        """
        try:
            from sklearn.cluster import KMeans
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embeddings)
            
            return labels
        except ImportError:
            self.logger.warning("scikit-learn no está disponible para clustering.")
            return np.zeros(len(embeddings))
    
    def compute_similarity_matrix(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Calcula la matriz de similitud coseno entre todos los embeddings.
        
        Args:
            embeddings: Array de embeddings (n_samples, embed_dim)
            
        Returns:
            Matriz de similitud (n_samples, n_samples)
        """
        n_samples = len(embeddings)
        similarity_matrix = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            for j in range(n_samples):
                similarity_matrix[i, j] = self.cosine_similarity(embeddings[i], embeddings[j])
        
        return similarity_matrix