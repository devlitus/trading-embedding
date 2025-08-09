import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import json
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
        self.samples: List[LabeledSample] = []
        self.load_samples()
    
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