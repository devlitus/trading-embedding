import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
import os
import uuid
from dataclasses import dataclass, asdict

@dataclass
class LabeledSample:
    """
    Representa una muestra etiquetada para entrenamiento.
    """
    timestamp: datetime
    symbol: str
    timeframe: str
    pattern: str
    confidence: float
    score: float
    data: Dict[str, Any]
    signals: List[str]
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte la muestra a diccionario."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'LabeledSample':
        """Crea una muestra desde un diccionario."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)

class DatasetManager:
    """
    Gestor de datasets para muestras etiquetadas.
    Maneja la creación, almacenamiento y recuperación de datasets de entrenamiento.
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "data/labeled_samples.json"
        self.datasets_path = "data/datasets.json"
        self.annotations_path = "data/annotations.json"
        self.samples: List[LabeledSample] = []
        self.datasets: List[Dict[str, Any]] = []
        self.annotations: List[Dict[str, Any]] = []
        self.load_samples()
        self.load_datasets()
        self.load_annotations()
    
    def add_sample(self, sample: LabeledSample) -> None:
        """
        Añade una nueva muestra al dataset.
        
        Args:
            sample: Muestra etiquetada a añadir
        """
        self.samples.append(sample)
    
    def create_sample(self, 
                     timestamp: datetime,
                     symbol: str,
                     timeframe: str,
                     pattern: str,
                     confidence: float,
                     score: float,
                     data: Dict[str, Any],
                     signals: List[str],
                     metadata: Optional[Dict[str, Any]] = None) -> LabeledSample:
        """
        Crea y añade una nueva muestra al dataset.
        
        Returns:
            La muestra creada
        """
        sample = LabeledSample(
            timestamp=timestamp,
            symbol=symbol,
            timeframe=timeframe,
            pattern=pattern,
            confidence=confidence,
            score=score,
            data=data,
            signals=signals,
            metadata=metadata or {}
        )
        
        self.add_sample(sample)
        return sample
    
    def get_samples(self, 
                   symbol: Optional[str] = None,
                   pattern: Optional[str] = None,
                   min_confidence: Optional[float] = None,
                   min_score: Optional[float] = None) -> List[LabeledSample]:
        """
        Recupera muestras filtradas según criterios.
        
        Args:
            symbol: Filtrar por símbolo
            pattern: Filtrar por patrón
            min_confidence: Confianza mínima
            min_score: Score mínimo
            
        Returns:
            Lista de muestras filtradas
        """
        filtered_samples = self.samples
        
        if symbol:
            filtered_samples = [s for s in filtered_samples if s.symbol == symbol]
        
        if pattern:
            filtered_samples = [s for s in filtered_samples if s.pattern == pattern]
        
        if min_confidence is not None:
            filtered_samples = [s for s in filtered_samples if s.confidence >= min_confidence]
        
        if min_score is not None:
            filtered_samples = [s for s in filtered_samples if s.score >= min_score]
        
        return filtered_samples
    
    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del dataset.
        
        Returns:
            Diccionario con estadísticas
        """
        if not self.samples:
            return {
                'total_samples': 0,
                'patterns': {},
                'symbols': {},
                'avg_confidence': 0.0,
                'avg_score': 0.0
            }
        
        patterns = {}
        symbols = {}
        
        for sample in self.samples:
            patterns[sample.pattern] = patterns.get(sample.pattern, 0) + 1
            symbols[sample.symbol] = symbols.get(sample.symbol, 0) + 1
        
        avg_confidence = np.mean([s.confidence for s in self.samples])
        avg_score = np.mean([s.score for s in self.samples])
        
        return {
            'total_samples': len(self.samples),
            'patterns': patterns,
            'symbols': symbols,
            'avg_confidence': float(avg_confidence),
            'avg_score': float(avg_score)
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """
        Convierte las muestras a DataFrame para análisis.
        
        Returns:
            DataFrame con las muestras
        """
        if not self.samples:
            return pd.DataFrame()
        
        data = []
        for sample in self.samples:
            row = {
                'timestamp': sample.timestamp,
                'symbol': sample.symbol,
                'timeframe': sample.timeframe,
                'pattern': sample.pattern,
                'confidence': sample.confidence,
                'score': sample.score,
                'num_signals': len(sample.signals)
            }
            
            # Añadir metadata si existe
            if sample.metadata:
                row.update(sample.metadata)
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def save_samples(self) -> None:
        """
        Guarda las muestras en el archivo de almacenamiento.
        """
        try:
            data = [sample.to_dict() for sample in self.samples]
            
            # Crear directorio si no existe
            import os
            os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
            
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error guardando muestras: {e}")
    
    def load_samples(self) -> None:
        """
        Carga las muestras desde el archivo de almacenamiento.
        """
        try:
            import os
            if os.path.exists(self.storage_path):
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                self.samples = [LabeledSample.from_dict(item) for item in data]
        except Exception as e:
            print(f"Error cargando muestras: {e}")
            self.samples = []
    
    def clear_samples(self) -> None:
        """
        Limpia todas las muestras del dataset.
        """
        self.samples = []
    
    def remove_sample(self, index: int) -> bool:
        """
        Elimina una muestra por índice.
        
        Args:
            index: Índice de la muestra a eliminar
            
        Returns:
            True si se eliminó correctamente
        """
        try:
            if 0 <= index < len(self.samples):
                del self.samples[index]
                return True
            return False
        except Exception:
            return False
    
    def create_dataset(self, 
                      name: str, 
                      samples: Optional[List[LabeledSample]] = None,
                      description: str = "",
                      annotation_sessions: Optional[List[str]] = None,
                      metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Crea un nuevo dataset.
        
        Args:
            name: Nombre del dataset
            samples: Lista de muestras a incluir
            description: Descripción del dataset
            annotation_sessions: IDs de sesiones de anotación
            metadata: Metadatos adicionales
            
        Returns:
            ID del dataset creado
        """
        dataset_id = str(uuid.uuid4())
        
        # Si se proporcionan muestras, agregarlas
        if samples:
            for sample in samples:
                self.add_sample(sample)
        
        dataset = {
            'dataset_id': dataset_id,
            'name': name,
            'description': description,
            'creation_date': datetime.now().isoformat(),
            'sample_count': len(samples) if samples else len(self.samples),
            'annotation_sessions': annotation_sessions or [],
            'metadata': metadata or {}
        }
        
        self.datasets.append(dataset)
        self.save_datasets()
        self.save_samples()
        
        return dataset_id
    
    def list_datasets(self) -> List[Dict[str, Any]]:
        """
        Lista todos los datasets disponibles.
        
        Returns:
            Lista de datasets
        """
        return self.datasets.copy()
    
    def get_dataset_summary(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        """
        Obtiene un resumen del dataset.
        
        Args:
            dataset_id: ID del dataset
            
        Returns:
            Resumen del dataset o None si no existe
        """
        dataset = next((d for d in self.datasets if d['dataset_id'] == dataset_id), None)
        if not dataset:
            return None
        
        # Obtener muestras relacionadas (simplificado - todas las muestras)
        related_samples = self.samples
        
        if not related_samples:
            return {
                'total_samples': 0,
                'patterns': {},
                'symbols': {},
                'avg_confidence': 0.0,
                'avg_score': 0.0
            }
        
        patterns = {}
        symbols = {}
        
        for sample in related_samples:
            patterns[sample.pattern] = patterns.get(sample.pattern, 0) + 1
            symbols[sample.symbol] = symbols.get(sample.symbol, 0) + 1
        
        return {
            'total_samples': len(related_samples),
            'patterns': patterns,
            'symbols': symbols,
            'avg_confidence': np.mean([s.confidence for s in related_samples]),
            'avg_score': np.mean([s.score for s in related_samples])
        }
    
    def delete_dataset(self, dataset_id: str) -> bool:
        """
        Elimina un dataset.
        
        Args:
            dataset_id: ID del dataset a eliminar
            
        Returns:
            True si se eliminó correctamente
        """
        try:
            self.datasets = [d for d in self.datasets if d['dataset_id'] != dataset_id]
            self.save_datasets()
            return True
        except Exception:
            return False
    
    def save_annotations(self, session_data: Dict[str, Any]) -> None:
        """
        Guarda una sesión de anotaciones.
        
        Args:
            session_data: Datos de la sesión de anotación
        """
        self.annotations.append(session_data)
        self.save_annotations_file()
    
    def get_training_data(self, dataset_id: str) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
        """
        Prepara datos para entrenamiento de ML.
        
        Args:
            dataset_id: ID del dataset
            
        Returns:
            ((X_train, y_train), (X_test, y_test))
        """
        # Implementación simplificada - retorna datos dummy
        # En una implementación real, esto procesaría las muestras del dataset
        n_samples = len(self.samples)
        n_features = 10  # Número de características dummy
        
        # Crear datos sintéticos para demostración
        X = np.random.random((n_samples, n_features))
        y = np.random.randint(0, 3, n_samples)  # 3 clases
        
        # División train/test simple
        split_idx = int(0.8 * n_samples)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        return (X_train, y_train), (X_test, y_test)
    
    def load_datasets(self) -> None:
        """
        Carga los datasets desde el archivo.
        """
        try:
            if os.path.exists(self.datasets_path):
                with open(self.datasets_path, 'r', encoding='utf-8') as f:
                    self.datasets = json.load(f)
        except Exception as e:
            print(f"Error cargando datasets: {e}")
            self.datasets = []
    
    def save_datasets(self) -> None:
        """
        Guarda los datasets en el archivo.
        """
        try:
            os.makedirs(os.path.dirname(self.datasets_path), exist_ok=True)
            with open(self.datasets_path, 'w', encoding='utf-8') as f:
                json.dump(self.datasets, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error guardando datasets: {e}")
    
    def load_annotations(self) -> None:
        """
        Carga las anotaciones desde el archivo.
        """
        try:
            if os.path.exists(self.annotations_path):
                with open(self.annotations_path, 'r', encoding='utf-8') as f:
                    self.annotations = json.load(f)
        except Exception as e:
            print(f"Error cargando anotaciones: {e}")
            self.annotations = []
    
    def save_annotations_file(self) -> None:
        """
        Guarda las anotaciones en el archivo.
        """
        try:
            os.makedirs(os.path.dirname(self.annotations_path), exist_ok=True)
            with open(self.annotations_path, 'w', encoding='utf-8') as f:
                json.dump(self.annotations, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"Error guardando anotaciones: {e}")