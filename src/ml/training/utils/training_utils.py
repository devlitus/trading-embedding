import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from pathlib import Path
import json
from datetime import datetime, timedelta
import pickle

# Importaciones opcionales
try:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    accuracy_score = None
    classification_report = None
    confusion_matrix = None
    SKLEARN_AVAILABLE = False

from ....config.config_manager import ConfigManager
from ...wyckoff.features.feature_extractor import WyckoffFeatureExtractor
from ...labeling.dataset_manager import LabeledSample

class TrainingUtils:
    """
    Utilidades para el proceso de entrenamiento de modelos.
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.feature_extractor = WyckoffFeatureExtractor()
    
    def create_sliding_windows(self, data: pd.DataFrame, window_size: int = 100, 
                             step_size: int = 20) -> List[pd.DataFrame]:
        """
        Crea ventanas deslizantes de los datos OHLC.
        
        Args:
            data: DataFrame con datos OHLC
            window_size: Tamaño de la ventana
            step_size: Paso entre ventanas
            
        Returns:
            Lista de DataFrames con ventanas
        """
        windows = []
        
        for i in range(0, len(data) - window_size + 1, step_size):
            window = data.iloc[i:i + window_size].copy()
            if len(window) == window_size:
                windows.append(window)
        
        self.logger.debug(f"Creadas {len(windows)} ventanas de tamaño {window_size}")
        return windows
    
    def generate_heuristic_label(self, window: pd.DataFrame) -> Tuple[str, float, float]:
        """
        Genera etiqueta heurística para una ventana de datos.
        
        Args:
            window: Ventana de datos OHLC
            
        Returns:
            Tupla con (patrón, confianza, score)
        """
        try:
            # Extraer características Wyckoff
            features = self.feature_extractor.extract_features(window)
            
            # Calcular score de calidad del patrón
            pattern_score = self._calculate_pattern_score(features)
            
            # Determinar patrón basado en características
            pattern = self._determine_pattern_from_features(features)
            
            # Calcular confianza basada en la consistencia de las señales
            confidence = self._calculate_confidence(features, pattern)
            
            return pattern, confidence, pattern_score
            
        except Exception as e:
            self.logger.warning(f"Error generando etiqueta heurística: {e}")
            return 'unknown', 0.0, 0.0
    
    def calculate_pattern_score(self, features: Dict[str, float]) -> float:
        """
        Calcula un score de calidad para un patrón basado en sus características.
        
        Args:
            features: Diccionario de características extraídas
            
        Returns:
            Score de calidad (0-1)
        """
        return self._calculate_pattern_score(features)
    
    def encode_pattern(self, pattern: str) -> np.ndarray:
        """
        Codifica un patrón como vector one-hot.
        
        Args:
            pattern: Nombre del patrón
            
        Returns:
            Vector one-hot codificado
        """
        pattern_mapping = {
            'accumulation': 0,
            'distribution': 1,
            'markup': 2,
            'markdown': 3,
            'spring': 4,
            'upthrust': 5,
            'unknown': 6
        }
        
        encoded = np.zeros(len(pattern_mapping))
        if pattern in pattern_mapping:
            encoded[pattern_mapping[pattern]] = 1.0
        else:
            encoded[pattern_mapping['unknown']] = 1.0
        
        return encoded
    
    def filter_training_samples(self, samples: List[LabeledSample], 
                              min_quality: float = 0.6,
                              balance_classes: bool = True,
                              max_samples_per_class: int = 1000) -> List[LabeledSample]:
        """
        Filtra y balancea muestras de entrenamiento.
        
        Args:
            samples: Lista de muestras etiquetadas
            min_quality: Score mínimo de calidad
            balance_classes: Si balancear las clases
            max_samples_per_class: Máximo de muestras por clase
            
        Returns:
            Lista filtrada de muestras
        """
        self.logger.info(f"Filtrando {len(samples)} muestras (min_quality={min_quality})")
        
        # Filtrar por calidad
        quality_filtered = [s for s in samples if s.score >= min_quality]
        self.logger.info(f"Después de filtro de calidad: {len(quality_filtered)} muestras")
        
        if not balance_classes:
            return quality_filtered[:max_samples_per_class * 10]  # Límite general
        
        # Agrupar por patrón
        pattern_groups = {}
        for sample in quality_filtered:
            pattern = sample.pattern
            if pattern not in pattern_groups:
                pattern_groups[pattern] = []
            pattern_groups[pattern].append(sample)
        
        # Balancear clases
        balanced_samples = []
        min_samples = min(len(group) for group in pattern_groups.values())
        target_samples = min(min_samples, max_samples_per_class)
        
        for pattern, group in pattern_groups.items():
            # Ordenar por score y tomar las mejores
            sorted_group = sorted(group, key=lambda x: x.score, reverse=True)
            selected = sorted_group[:target_samples]
            balanced_samples.extend(selected)
            
            self.logger.debug(f"Patrón {pattern}: {len(group)} -> {len(selected)} muestras")
        
        self.logger.info(f"Después de balanceo: {len(balanced_samples)} muestras")
        return balanced_samples
    
    def split_data(self, samples: List[LabeledSample], 
                   train_ratio: float = 0.8,
                   random_state: int = 42) -> Tuple[List[LabeledSample], List[LabeledSample]]:
        """
        Divide las muestras en entrenamiento y validación.
        
        Args:
            samples: Lista de muestras
            train_ratio: Proporción para entrenamiento
            random_state: Semilla aleatoria
            
        Returns:
            Tupla con (muestras_entrenamiento, muestras_validación)
        """
        np.random.seed(random_state)
        
        # Mezclar muestras
        shuffled_samples = samples.copy()
        np.random.shuffle(shuffled_samples)
        
        # Dividir
        split_idx = int(len(shuffled_samples) * train_ratio)
        train_samples = shuffled_samples[:split_idx]
        val_samples = shuffled_samples[split_idx:]
        
        self.logger.info(f"División de datos: {len(train_samples)} entrenamiento, {len(val_samples)} validación")
        
        return train_samples, val_samples
    
    def prepare_training_data(self, samples: List[LabeledSample]) -> Tuple[List[pd.DataFrame], List[str]]:
        """
        Prepara los datos para entrenamiento.
        
        Args:
            samples: Lista de muestras etiquetadas
            
        Returns:
            Tupla con (datos, etiquetas)
        """
        X = []
        y = []
        
        for sample in samples:
            try:
                # Convertir datos a DataFrame si es necesario
                if isinstance(sample.data, dict):
                    data_df = pd.DataFrame(sample.data)
                elif isinstance(sample.data, pd.DataFrame):
                    data_df = sample.data
                else:
                    self.logger.warning(f"Tipo de datos no soportado: {type(sample.data)}")
                    continue
                
                X.append(data_df)
                y.append(sample.pattern)
                
            except Exception as e:
                self.logger.warning(f"Error preparando muestra: {e}")
                continue
        
        self.logger.info(f"Datos preparados: {len(X)} muestras")
        return X, y
    
    def save_training_checkpoint(self, checkpoint_data: Dict[str, Any], 
                               checkpoint_path: Optional[str] = None) -> str:
        """
        Guarda un checkpoint del entrenamiento.
        
        Args:
            checkpoint_data: Datos del checkpoint
            checkpoint_path: Ruta donde guardar
            
        Returns:
            Ruta del checkpoint guardado
        """
        if checkpoint_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            checkpoint_path = f'training_checkpoint_{timestamp}.pkl'
        
        checkpoint_path = Path(checkpoint_path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint_data, f)
        
        self.logger.info(f"Checkpoint guardado en: {checkpoint_path}")
        return str(checkpoint_path)
    
    def load_training_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """
        Carga un checkpoint del entrenamiento.
        
        Args:
            checkpoint_path: Ruta del checkpoint
            
        Returns:
            Datos del checkpoint
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint_data = pickle.load(f)
        
        self.logger.info(f"Checkpoint cargado desde: {checkpoint_path}")
        return checkpoint_data
    
    def calculate_training_metrics(self, y_true: List[str], y_pred: List[str], 
                                 training_time: float) -> Dict[str, Any]:
        """
        Calcula métricas de entrenamiento.
        
        Args:
            y_true: Etiquetas verdaderas
            y_pred: Predicciones
            training_time: Tiempo de entrenamiento en segundos
            
        Returns:
            Diccionario con métricas
        """
        if not SKLEARN_AVAILABLE:
            logging.warning("scikit-learn no disponible. Usando métricas básicas.")
            # Calcular accuracy manualmente
            accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
            metrics = {
                'accuracy': float(accuracy),
                'training_time': float(training_time),
                'samples_processed': len(y_true),
                'samples_per_second': len(y_true) / training_time if training_time > 0 else 0,
                'timestamp': datetime.now().isoformat(),
                'classification_report': {'note': 'scikit-learn no disponible'}
            }
        else:
            metrics = {
                'accuracy': float(accuracy_score(y_true, y_pred)),
                'training_time': float(training_time),
                'samples_processed': len(y_true),
                'samples_per_second': len(y_true) / training_time if training_time > 0 else 0,
                'timestamp': datetime.now().isoformat()
            }
            
            # Reporte de clasificación
            try:
                report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
                metrics['classification_report'] = report
            except Exception as e:
                self.logger.warning(f"Error calculando reporte de clasificación: {e}")
                metrics['classification_report'] = {'note': 'Error en reporte de clasificación'}
            
            # Matriz de confusión
            try:
                cm = confusion_matrix(y_true, y_pred)
                metrics['confusion_matrix'] = cm.tolist()
            except Exception as e:
                self.logger.warning(f"Error calculando matriz de confusión: {e}")
                metrics['confusion_matrix'] = []
        
        return metrics
    
    def _calculate_pattern_score(self, features: Dict[str, float]) -> float:
        """
        Calcula score de calidad del patrón basado en características.
        
        Args:
            features: Características extraídas
            
        Returns:
            Score de calidad (0-1)
        """
        try:
            # Factores de calidad
            volume_factor = min(features.get('avg_volume', 0) / 1000000, 1.0)  # Normalizar volumen
            volatility_factor = min(features.get('volatility', 0) * 10, 1.0)  # Volatilidad moderada es mejor
            trend_strength = abs(features.get('trend_strength', 0))
            
            # Score de soporte/resistencia
            support_tests = features.get('support_resistance_tests', 0)
            sr_score = min(support_tests / 5.0, 1.0)  # Más tests = mejor
            
            # Score de acumulación/distribución
            acc_dist_score = abs(features.get('accumulation_distribution_score', 0))
            
            # Combinar factores
            pattern_score = (
                volume_factor * 0.3 +
                volatility_factor * 0.2 +
                trend_strength * 0.2 +
                sr_score * 0.2 +
                acc_dist_score * 0.1
            )
            
            return min(max(pattern_score, 0.0), 1.0)
            
        except Exception as e:
            self.logger.warning(f"Error calculando pattern score: {e}")
            return 0.5  # Score neutral por defecto
    
    def _determine_pattern_from_features(self, features: Dict[str, float]) -> str:
        """
        Determina el patrón basado en las características.
        
        Args:
            features: Características extraídas
            
        Returns:
            Nombre del patrón identificado
        """
        try:
            # Obtener características clave
            acc_dist = features.get('accumulation_distribution_score', 0)
            trend_strength = features.get('trend_strength', 0)
            spring_prob = features.get('spring_probability', 0)
            upthrust_prob = features.get('upthrust_probability', 0)
            phase = features.get('wyckoff_phase', 'unknown')
            
            # Lógica de determinación de patrón
            if spring_prob > 0.7:
                return 'spring'
            elif upthrust_prob > 0.7:
                return 'upthrust'
            elif acc_dist > 0.3 and trend_strength < 0:
                return 'accumulation'
            elif acc_dist < -0.3 and trend_strength > 0:
                return 'distribution'
            elif trend_strength > 0.5:
                return 'markup'
            elif trend_strength < -0.5:
                return 'markdown'
            elif phase in ['accumulation', 'distribution', 'markup', 'markdown']:
                return phase
            else:
                return 'unknown'
                
        except Exception as e:
            self.logger.warning(f"Error determinando patrón: {e}")
            return 'unknown'
    
    def _calculate_confidence(self, features: Dict[str, float], pattern: str) -> float:
        """
        Calcula la confianza en la identificación del patrón.
        
        Args:
            features: Características extraídas
            pattern: Patrón identificado
            
        Returns:
            Nivel de confianza (0-1)
        """
        try:
            # Factores de confianza específicos por patrón
            if pattern == 'spring':
                spring_prob = features.get('spring_probability', 0)
                volume_confirmation = min(features.get('avg_volume', 0) / 1000000, 1.0)
                return (spring_prob + volume_confirmation) / 2
            
            elif pattern == 'upthrust':
                upthrust_prob = features.get('upthrust_probability', 0)
                volume_confirmation = min(features.get('avg_volume', 0) / 1000000, 1.0)
                return (upthrust_prob + volume_confirmation) / 2
            
            elif pattern in ['accumulation', 'distribution']:
                acc_dist_strength = abs(features.get('accumulation_distribution_score', 0))
                sr_tests = min(features.get('support_resistance_tests', 0) / 5.0, 1.0)
                return (acc_dist_strength + sr_tests) / 2
            
            elif pattern in ['markup', 'markdown']:
                trend_strength = abs(features.get('trend_strength', 0))
                volume_trend = abs(features.get('volume_trend', 0))
                return (trend_strength + volume_trend) / 2
            
            else:
                # Confianza general basada en consistencia de señales
                signal_strength = (
                    abs(features.get('trend_strength', 0)) +
                    abs(features.get('accumulation_distribution_score', 0)) +
                    min(features.get('support_resistance_tests', 0) / 5.0, 1.0)
                ) / 3
                
                return signal_strength
                
        except Exception as e:
            self.logger.warning(f"Error calculando confianza: {e}")
            return 0.5  # Confianza neutral por defecto

class DataAugmentation:
    """
    Utilidades para aumentar datos de entrenamiento.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def add_noise(self, data: pd.DataFrame, noise_level: float = 0.01) -> pd.DataFrame:
        """
        Añade ruido gaussiano a los datos OHLC.
        
        Args:
            data: DataFrame con datos OHLC
            noise_level: Nivel de ruido (proporción del precio)
            
        Returns:
            DataFrame con ruido añadido
        """
        augmented_data = data.copy()
        
        for col in ['open', 'high', 'low', 'close']:
            if col in augmented_data.columns:
                noise = np.random.normal(0, noise_level, len(augmented_data))
                augmented_data[col] *= (1 + noise)
        
        # Asegurar consistencia OHLC
        augmented_data = self._fix_ohlc_consistency(augmented_data)
        
        return augmented_data
    
    def time_shift(self, data: pd.DataFrame, shift_periods: int = 1) -> pd.DataFrame:
        """
        Aplica desplazamiento temporal a los datos.
        
        Args:
            data: DataFrame con datos OHLC
            shift_periods: Número de períodos a desplazar
            
        Returns:
            DataFrame desplazado
        """
        shifted_data = data.shift(shift_periods).dropna()
        return shifted_data
    
    def scale_volume(self, data: pd.DataFrame, scale_factor: float = 1.2) -> pd.DataFrame:
        """
        Escala el volumen de los datos.
        
        Args:
            data: DataFrame con datos OHLC
            scale_factor: Factor de escalado
            
        Returns:
            DataFrame con volumen escalado
        """
        augmented_data = data.copy()
        
        if 'volume' in augmented_data.columns:
            augmented_data['volume'] *= scale_factor
        
        return augmented_data
    
    def _fix_ohlc_consistency(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Corrige inconsistencias en datos OHLC después de añadir ruido.
        
        Args:
            data: DataFrame con datos OHLC
            
        Returns:
            DataFrame con consistencia corregida
        """
        fixed_data = data.copy()
        
        for i in range(len(fixed_data)):
            row = fixed_data.iloc[i]
            
            # Asegurar que high >= max(open, close) y low <= min(open, close)
            if all(col in row for col in ['open', 'high', 'low', 'close']):
                open_price = row['open']
                close_price = row['close']
                high_price = row['high']
                low_price = row['low']
                
                # Corregir high
                min_high = max(open_price, close_price)
                if high_price < min_high:
                    fixed_data.iloc[i, fixed_data.columns.get_loc('high')] = min_high
                
                # Corregir low
                max_low = min(open_price, close_price)
                if low_price > max_low:
                    fixed_data.iloc[i, fixed_data.columns.get_loc('low')] = max_low
        
        return fixed_data