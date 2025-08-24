import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta

# Importaciones opcionales para scikit-learn
try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.model_selection import train_test_split
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    StandardScaler = None
    MinMaxScaler = None
    RobustScaler = None
    train_test_split = None

from ..data.data_manager import DataManager
from .labeling.dataset_manager import DatasetManager, LabeledSample

class DataPreprocessor:
    """
    Preprocesador de datos para modelos de ML.
    Maneja la preparación de datos OHLC y etiquetas para entrenamiento.
    """
    
    def __init__(self, 
                 scaler_type: str = 'standard',
                 sequence_length: int = 50,
                 prediction_horizon: int = 1):
        """
        Inicializa el preprocesador.
        
        Args:
            scaler_type: Tipo de escalador ('standard', 'minmax', 'robust')
            sequence_length: Longitud de secuencias para modelos temporales
            prediction_horizon: Horizonte de predicción
        """
        self.scaler_type = scaler_type
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        
        # Inicializar escaladores
        if SKLEARN_AVAILABLE:
            self.scalers = {
                'standard': StandardScaler(),
                'minmax': MinMaxScaler(),
                'robust': RobustScaler()
            }
        else:
            self.scalers = {}
            self.logger.warning("scikit-learn no está disponible. Funcionalidad de escalado deshabilitada.")
        
        self.fitted_scalers = {}
        self.feature_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Configurar logging
        self.logger = logging.getLogger(__name__)
    
    def prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara características adicionales a partir de datos OHLC.
        
        Args:
            data: DataFrame con datos OHLC
            
        Returns:
            DataFrame con características adicionales
        """
        df = data.copy()
        
        # Características básicas de precio
        df['price_change'] = df['close'].pct_change()
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']
        
        # Características de volumen
        df['volume_change'] = df['volume'].pct_change()
        df['volume_price_trend'] = df['volume'] * df['price_change']
        
        # Medias móviles
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
            df[f'volume_sma_{window}'] = df['volume'].rolling(window=window).mean()
        
        # Volatilidad
        df['volatility_5'] = df['close'].rolling(window=5).std()
        df['volatility_20'] = df['close'].rolling(window=20).std()
        
        # RSI simplificado
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bandas de Bollinger
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        return df
    
    def create_sequences(self, 
                        data: pd.DataFrame, 
                        labels: Optional[pd.Series] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Crea secuencias temporales para modelos LSTM/GRU.
        
        Args:
            data: DataFrame con características
            labels: Series con etiquetas (opcional)
            
        Returns:
            Tupla con secuencias X y etiquetas y (si se proporcionan)
        """
        # Seleccionar columnas numéricas
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        data_numeric = data[numeric_cols].fillna(method='ffill').fillna(0)
        
        sequences = []
        sequence_labels = []
        
        for i in range(self.sequence_length, len(data_numeric) - self.prediction_horizon + 1):
            # Secuencia de entrada
            seq = data_numeric.iloc[i-self.sequence_length:i].values
            sequences.append(seq)
            
            # Etiqueta correspondiente
            if labels is not None:
                label_idx = i + self.prediction_horizon - 1
                if label_idx < len(labels):
                    sequence_labels.append(labels.iloc[label_idx])
                else:
                    sequence_labels.append(0)  # Etiqueta por defecto
        
        X = np.array(sequences)
        y = np.array(sequence_labels) if labels is not None else None
        
        return X, y
    
    def scale_features(self, 
                      data: pd.DataFrame, 
                      fit: bool = True) -> pd.DataFrame:
        """
        Escala las características usando el escalador configurado.
        
        Args:
            data: DataFrame con características
            fit: Si ajustar el escalador (True para entrenamiento)
            
        Returns:
            DataFrame con características escaladas
        """
        if not SKLEARN_AVAILABLE:
            self.logger.warning("scikit-learn no está disponible. Devolviendo datos sin escalar.")
            return data.copy()
        
        df = data.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        scaler_name = f"{self.scaler_type}_scaler"
        
        if fit:
            # Ajustar y transformar
            if self.scaler_type in self.scalers:
                scaler = self.scalers[self.scaler_type]
                df[numeric_cols] = scaler.fit_transform(df[numeric_cols].fillna(0))
                self.fitted_scalers[scaler_name] = scaler
            else:
                self.logger.warning(f"Escalador {self.scaler_type} no disponible.")
        else:
            # Solo transformar
            if scaler_name in self.fitted_scalers:
                scaler = self.fitted_scalers[scaler_name]
                df[numeric_cols] = scaler.transform(df[numeric_cols].fillna(0))
            else:
                self.logger.warning(f"Escalador {scaler_name} no encontrado. Usando datos sin escalar.")
        
        return df
    
    def prepare_training_data(self, 
                            dataset_manager: DatasetManager,
                            data_manager: DataManager,
                            min_confidence: float = 0.7,
                            test_size: float = 0.2,
                            random_state: int = 42) -> Dict[str, Any]:
        """
        Prepara datos completos para entrenamiento.
        
        Args:
            dataset_manager: Gestor de datasets etiquetados
            data_manager: Gestor de datos OHLC
            min_confidence: Confianza mínima para incluir muestras
            test_size: Proporción de datos para test
            random_state: Semilla aleatoria
            
        Returns:
            Diccionario con datos preparados
        """
        # Obtener muestras etiquetadas
        samples = dataset_manager.get_samples(min_confidence=min_confidence)
        
        if not samples:
            raise ValueError("No hay muestras etiquetadas suficientes")
        
        self.logger.info(f"Preparando {len(samples)} muestras para entrenamiento")
        
        # Preparar datos por símbolo
        all_sequences = []
        all_labels = []
        
        # Mapeo de patrones a números
        pattern_mapping = self._create_pattern_mapping(samples)
        
        for sample in samples:
            try:
                # Obtener datos OHLC para el período
                start_time = sample.timestamp - timedelta(hours=self.sequence_length)
                end_time = sample.timestamp + timedelta(hours=self.prediction_horizon)
                
                ohlc_data = data_manager.get_data(
                    symbol=sample.symbol,
                    interval=sample.timeframe,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if len(ohlc_data) < self.sequence_length:
                    continue
                
                # Preparar características
                features = self.prepare_features(ohlc_data)
                
                # Escalar características
                scaled_features = self.scale_features(features, fit=True)
                
                # Crear secuencia
                X_seq, _ = self.create_sequences(scaled_features)
                
                if len(X_seq) > 0:
                    all_sequences.extend(X_seq)
                    # Usar el mapeo de patrones
                    pattern_label = pattern_mapping.get(sample.pattern, 0)
                    all_labels.extend([pattern_label] * len(X_seq))
                
            except Exception as e:
                self.logger.warning(f"Error procesando muestra {sample.timestamp}: {e}")
                continue
        
        if not all_sequences:
            raise ValueError("No se pudieron crear secuencias válidas")
        
        # Convertir a arrays numpy
        X = np.array(all_sequences)
        y = np.array(all_labels)
        
        # División train/test
        if SKLEARN_AVAILABLE and train_test_split is not None:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        else:
            # División manual simple si sklearn no está disponible
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            self.logger.warning("Usando división manual de datos (sklearn no disponible).")
        
        return {
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'pattern_mapping': pattern_mapping,
            'feature_shape': X.shape,
            'num_classes': len(pattern_mapping),
            'samples_processed': len(samples)
        }
    
    def _create_pattern_mapping(self, samples: List[LabeledSample]) -> Dict[str, int]:
        """
        Crea mapeo de patrones a índices numéricos.
        
        Args:
            samples: Lista de muestras etiquetadas
            
        Returns:
            Diccionario con mapeo patrón -> índice
        """
        unique_patterns = list(set(sample.pattern for sample in samples))
        unique_patterns.sort()  # Para consistencia
        
        pattern_mapping = {pattern: idx for idx, pattern in enumerate(unique_patterns)}
        
        self.logger.info(f"Patrones detectados: {pattern_mapping}")
        
        return pattern_mapping
    
    def inverse_transform_predictions(self, predictions: np.ndarray, pattern_mapping: Dict[str, int]) -> List[str]:
        """
        Convierte predicciones numéricas de vuelta a nombres de patrones.
        
        Args:
            predictions: Array con predicciones numéricas
            pattern_mapping: Mapeo de patrones
            
        Returns:
            Lista con nombres de patrones
        """
        inverse_mapping = {v: k for k, v in pattern_mapping.items()}
        return [inverse_mapping.get(pred, 'unknown') for pred in predictions]