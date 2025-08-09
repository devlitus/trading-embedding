#!/usr/bin/env python3
"""
Data Strategy - Estrategias de acceso a datos optimizadas
Implementa patrones de acceso específicos para diferentes casos de uso
"""

from enum import Enum
from typing import Dict, List, Optional, Union
import pandas as pd
from datetime import datetime, timedelta
import logging
from .data_access_layer import DataAccessLayer

logger = logging.getLogger(__name__)

class DataUsagePattern(Enum):
    """Patrones de uso de datos optimizados"""
    REALTIME_TRADING = "realtime_trading"  # Trading en tiempo real
    ML_TRAINING = "ml_training"            # Entrenamiento de modelos ML
    BACKTESTING = "backtesting"            # Backtesting de estrategias
    RESEARCH = "research"                  # Investigación y análisis
    API_SERVING = "api_serving"            # Servir datos vía API
    DASHBOARD = "dashboard"                # Dashboard y visualización

class DataStrategy:
    """
    Implementa estrategias de acceso a datos optimizadas según el patrón de uso.
    Cada patrón tiene configuraciones específicas para maximizar rendimiento.
    """
    
    def __init__(self, data_access: DataAccessLayer):
        self.data_access = data_access
        self.strategies = self._initialize_strategies()
    
    def get_data(self, pattern: DataUsagePattern, symbol: str, interval: str, 
                 **kwargs) -> pd.DataFrame:
        """
        Obtiene datos usando la estrategia optimizada para el patrón específico.
        
        Args:
            pattern: Patrón de uso de datos
            symbol: Símbolo de trading (ej: BTCUSDT)
            interval: Intervalo de tiempo (ej: 1h, 4h)
            **kwargs: Parámetros adicionales específicos del patrón
        
        Returns:
            DataFrame con datos optimizados para el patrón de uso
        """
        strategy_config = self.strategies[pattern]
        logger.info(f"Aplicando estrategia {pattern.value} para {symbol} {interval}")
        
        return strategy_config['handler'](symbol, interval, **kwargs)
    
    def _initialize_strategies(self) -> Dict[DataUsagePattern, Dict]:
        """Inicializa configuraciones de estrategias."""
        return {
            DataUsagePattern.REALTIME_TRADING: {
                'handler': self._realtime_trading_strategy,
                'description': 'Optimizado para trading en tiempo real con baja latencia',
                'data_source_priority': ['cache', 'database', 'csv'],
                'max_latency_ms': 100,
                'default_limit': 500
            },
            
            DataUsagePattern.ML_TRAINING: {
                'handler': self._ml_training_strategy,
                'description': 'Optimizado para entrenamiento de modelos ML con datos completos',
                'data_source_priority': ['csv', 'database'],
                'preprocessing': True,
                'feature_engineering': True
            },
            
            DataUsagePattern.BACKTESTING: {
                'handler': self._backtesting_strategy,
                'description': 'Optimizado para backtesting con datos históricos completos',
                'data_source_priority': ['csv', 'database'],
                'ensure_completeness': True,
                'sort_by_time': True
            },
            
            DataUsagePattern.RESEARCH: {
                'handler': self._research_strategy,
                'description': 'Optimizado para investigación con flexibilidad máxima',
                'data_source_priority': ['csv', 'database', 'cache'],
                'include_metadata': True,
                'flexible_timeframes': True
            },
            
            DataUsagePattern.API_SERVING: {
                'handler': self._api_serving_strategy,
                'description': 'Optimizado para servir datos vía API con cache inteligente',
                'data_source_priority': ['cache', 'database'],
                'cache_strategy': 'aggressive',
                'response_format': 'json_ready'
            },
            
            DataUsagePattern.DASHBOARD: {
                'handler': self._dashboard_strategy,
                'description': 'Optimizado para dashboards con agregaciones pre-calculadas',
                'data_source_priority': ['cache', 'database'],
                'aggregations': True,
                'visualization_ready': True
            }
        }
    
    # Estrategias específicas
    
    def _realtime_trading_strategy(self, symbol: str, interval: str, 
                                  limit: Optional[int] = None, **kwargs) -> pd.DataFrame:
        """
        Estrategia para trading en tiempo real.
        Prioriza velocidad y datos recientes.
        """
        limit = limit or self.strategies[DataUsagePattern.REALTIME_TRADING]['default_limit']
        
        # Obtener datos con prioridad en cache/DB
        data = self.data_access.get_data_for_realtime(
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        
        if data.empty:
            logger.warning(f"No hay datos disponibles para trading en tiempo real: {symbol} {interval}")
            return data
        
        # Optimizaciones para trading en tiempo real
        data = self._optimize_for_realtime(data)
        
        logger.info(f"Datos para trading en tiempo real: {len(data)} registros, último: {data['datetime'].max()}")
        return data
    
    def _ml_training_strategy(self, symbol: str, interval: str,
                             lookback_days: Optional[int] = None,
                             include_features: bool = True, **kwargs) -> pd.DataFrame:
        """
        Estrategia para entrenamiento de ML.
        Prioriza completitud de datos y features preparadas.
        """
        # Obtener datos optimizados para ML
        data = self.data_access.get_data_for_ml_analysis(
            symbol=symbol,
            interval=interval,
            lookback_days=lookback_days
        )
        
        if data.empty:
            logger.warning(f"No hay datos disponibles para ML: {symbol} {interval}")
            return data
        
        # Aplicar preprocesamiento específico para ML
        data = self._preprocess_for_ml(data, include_features)
        
        logger.info(f"Datos para ML: {len(data)} registros, features: {list(data.columns)}")
        return data
    
    def _backtesting_strategy(self, symbol: str, interval: str,
                             start_date: Optional[datetime] = None,
                             end_date: Optional[datetime] = None, **kwargs) -> pd.DataFrame:
        """
        Estrategia para backtesting.
        Asegura datos completos y ordenados cronológicamente.
        """
        # Obtener datos para ML (CSV preferido para backtesting)
        data = self.data_access.get_data_for_ml_analysis(symbol=symbol, interval=interval)
        
        if data.empty:
            logger.warning(f"No hay datos disponibles para backtesting: {symbol} {interval}")
            return data
        
        # Filtrar por fechas si se especifican
        if start_date or end_date:
            data = self._filter_by_date_range(data, start_date, end_date)
        
        # Optimizar para backtesting
        data = self._optimize_for_backtesting(data)
        
        logger.info(f"Datos para backtesting: {len(data)} registros, rango: {data['datetime'].min()} a {data['datetime'].max()}")
        return data
    
    def _research_strategy(self, symbol: str, interval: str,
                          include_metadata: bool = True, **kwargs) -> pd.DataFrame:
        """
        Estrategia para investigación.
        Máxima flexibilidad y información adicional.
        """
        # Intentar obtener de múltiples fuentes
        data = self.data_access.get_data_for_ml_analysis(symbol=symbol, interval=interval)
        
        if data.empty:
            # Fallback a datos en tiempo real
            data = self.data_access.get_data_for_realtime(symbol=symbol, interval=interval)
        
        if data.empty:
            logger.warning(f"No hay datos disponibles para investigación: {symbol} {interval}")
            return data
        
        # Agregar metadatos si se solicita
        if include_metadata:
            data = self._add_research_metadata(data, symbol, interval)
        
        logger.info(f"Datos para investigación: {len(data)} registros con metadatos")
        return data
    
    def _api_serving_strategy(self, symbol: str, interval: str,
                             limit: Optional[int] = None,
                             format_for_json: bool = True, **kwargs) -> pd.DataFrame:
        """
        Estrategia para servir datos vía API.
        Optimizado para respuestas rápidas y formato JSON.
        """
        # Priorizar cache y DB para APIs
        data = self.data_access.get_data_for_realtime(
            symbol=symbol,
            interval=interval,
            limit=limit or 100  # Límite por defecto para APIs
        )
        
        if data.empty:
            logger.warning(f"No hay datos disponibles para API: {symbol} {interval}")
            return data
        
        # Optimizar para respuesta API
        if format_for_json:
            data = self._format_for_api_response(data)
        
        logger.info(f"Datos para API: {len(data)} registros formateados")
        return data
    
    def _dashboard_strategy(self, symbol: str, interval: str,
                           include_aggregations: bool = True, **kwargs) -> pd.DataFrame:
        """
        Estrategia para dashboards.
        Incluye agregaciones y datos listos para visualización.
        """
        # Obtener datos recientes para dashboard
        data = self.data_access.get_data_for_realtime(
            symbol=symbol,
            interval=interval,
            limit=1000  # Suficiente para gráficos
        )
        
        if data.empty:
            logger.warning(f"No hay datos disponibles para dashboard: {symbol} {interval}")
            return data
        
        # Agregar cálculos para dashboard
        if include_aggregations:
            data = self._add_dashboard_aggregations(data)
        
        logger.info(f"Datos para dashboard: {len(data)} registros con agregaciones")
        return data
    
    # Métodos de optimización específicos
    
    def _optimize_for_realtime(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimizaciones específicas para trading en tiempo real."""
        if data.empty:
            return data
        
        # Normalizar nombres de columnas si vienen de la base de datos
        data = self._normalize_column_names(data)
        
        # Asegurar orden cronológico (más reciente al final)
        time_col = 'datetime' if 'datetime' in data.columns else 'timestamp'
        data = data.sort_values(time_col).reset_index(drop=True)
        
        # Agregar indicadores básicos para trading
        if 'close' in data.columns:
            data['price_change'] = data['close'].pct_change()
        if 'volume' in data.columns:
            data['volume_ma'] = data['volume'].rolling(window=20, min_periods=1).mean()
        
        # Eliminar valores nulos que puedan afectar trading
        required_cols = [col for col in ['close', 'volume'] if col in data.columns]
        if required_cols:
            data = data.dropna(subset=required_cols)
        
        return data
    
    def _preprocess_for_ml(self, data: pd.DataFrame, include_features: bool) -> pd.DataFrame:
        """Preprocesamiento específico para ML."""
        if data.empty:
            return data
        
        # Normalizar nombres de columnas
        data = self._normalize_column_names(data)
        
        # Asegurar tipos de datos correctos
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_cols:
            if col in data.columns:
                data[col] = pd.to_numeric(data[col], errors='coerce')
        
        if include_features and np is not None and 'close' in data.columns:
            # Agregar features básicas para ML
            data['returns'] = data['close'].pct_change()
            data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            data['volatility'] = data['returns'].rolling(window=20).std()
            data['rsi'] = self._calculate_rsi(data['close'])
            
            # Eliminar filas con NaN después de cálculos
            data = data.dropna()
        
        return data
    
    def _optimize_for_backtesting(self, data: pd.DataFrame) -> pd.DataFrame:
        """Optimizaciones para backtesting."""
        if data.empty:
            return data
        
        # Normalizar nombres de columnas
        data = self._normalize_column_names(data)
        
        # Orden cronológico estricto
        time_col = 'datetime' if 'datetime' in data.columns else 'timestamp'
        data = data.sort_values(time_col).reset_index(drop=True)
        
        # Verificar continuidad de datos
        data = self._ensure_data_continuity(data)
        
        # Agregar columnas útiles para backtesting
        if 'close' in data.columns:
            data['next_close'] = data['close'].shift(-1)  # Para calcular returns futuros
            data['forward_return'] = (data['next_close'] / data['close']) - 1
        
        return data
    
    def _add_research_metadata(self, data: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
        """Agrega metadatos útiles para investigación."""
        if data.empty:
            return data
        
        # Información del dataset
        data.attrs['symbol'] = symbol
        data.attrs['interval'] = interval
        data.attrs['total_records'] = len(data)
        data.attrs['date_range'] = (data['datetime'].min(), data['datetime'].max())
        data.attrs['data_quality'] = self._assess_data_quality(data)
        
        return data
    
    def _format_for_api_response(self, data: pd.DataFrame) -> pd.DataFrame:
        """Formatea datos para respuesta API."""
        if data.empty:
            return data
        
        # Convertir datetime a string para JSON
        if 'datetime' in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data['datetime']):
                data['datetime'] = data['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S')
        elif 'timestamp' in data.columns:
            if pd.api.types.is_datetime64_any_dtype(data['timestamp']):
                data['datetime'] = data['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Redondear números para reducir tamaño de respuesta
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_cols:
            if col in data.columns:
                if col in ['volume', 'quote_volume']:
                    data[col] = data[col].round(2)
                else:
                    data[col] = data[col].round(8)  # Precisión de precio
        
        return data
    
    def _add_dashboard_aggregations(self, data: pd.DataFrame) -> pd.DataFrame:
        """Agrega agregaciones útiles para dashboards."""
        if data.empty:
            return data
        
        # Medias móviles
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        # Bandas de Bollinger
        data['bb_upper'] = data['sma_20'] + (data['close'].rolling(window=20).std() * 2)
        data['bb_lower'] = data['sma_20'] - (data['close'].rolling(window=20).std() * 2)
        
        # Volumen promedio
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        
        return data
    
    # Métodos de utilidad
    
    def _filter_by_date_range(self, data: pd.DataFrame, 
                             start_date: Optional[datetime],
                             end_date: Optional[datetime]) -> pd.DataFrame:
        """Filtra datos por rango de fechas."""
        if data.empty:
            return data
        
        if start_date:
            data = data[data['datetime'] >= start_date]
        if end_date:
            data = data[data['datetime'] <= end_date]
        
        return data
    
    def _ensure_data_continuity(self, data: pd.DataFrame) -> pd.DataFrame:
        """Asegura continuidad en los datos para backtesting."""
        if data.empty or len(data) < 2:
            return data
        
        # Detectar gaps en los datos
        data['time_diff'] = data['datetime'].diff()
        
        # Log gaps significativos
        median_diff = data['time_diff'].median()
        large_gaps = data[data['time_diff'] > median_diff * 2]
        
        if not large_gaps.empty:
            logger.warning(f"Detectados {len(large_gaps)} gaps en los datos para backtesting")
        
        return data.drop('time_diff', axis=1)
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calcula RSI (Relative Strength Index)."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _normalize_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normaliza nombres de columnas entre base de datos y CSV."""
        if data.empty:
            return data
        
        # Resetear índice si hay conflictos
        if data.index.name == 'datetime' or 'datetime' in data.index.names:
            data = data.reset_index()
        
        # Mapeo de columnas de base de datos a formato estándar
        column_mapping = {
            'timestamp': 'datetime',
            'open_price': 'open',
            'high_price': 'high', 
            'low_price': 'low',
            'close_price': 'close',
            'trades_count': 'trades_count'
        }
        
        # Renombrar columnas si existen y no hay duplicados
        for old_col, new_col in column_mapping.items():
            if old_col in data.columns and new_col not in data.columns:
                data = data.rename(columns={old_col: new_col})
        
        # Convertir timestamp a datetime si es necesario
        if 'datetime' in data.columns:
            if data['datetime'].dtype in ['int64', 'float64']:
                data['datetime'] = pd.to_datetime(data['datetime'], unit='ms')
            elif not pd.api.types.is_datetime64_any_dtype(data['datetime']):
                data['datetime'] = pd.to_datetime(data['datetime'])
        elif 'timestamp' in data.columns:
            if data['timestamp'].dtype in ['int64', 'float64']:
                data['datetime'] = pd.to_datetime(data['timestamp'], unit='ms')
            else:
                data['datetime'] = pd.to_datetime(data['timestamp'])
        
        return data
    
    def _assess_data_quality(self, data: pd.DataFrame) -> Dict[str, any]:
        """Evalúa la calidad de los datos."""
        if data.empty:
            return {'status': 'empty'}
        
        # Normalizar columnas para evaluación
        data = self._normalize_column_names(data)
        
        quality = {
            'completeness': (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100,
            'duplicates': data.duplicated().sum(),
            'date_continuity': 'good' if len(data) > 1 else 'insufficient',
            'numeric_validity': all(pd.api.types.is_numeric_dtype(data[col]) 
                                 for col in ['open', 'high', 'low', 'close', 'volume'] 
                                 if col in data.columns)
        }
        
        return quality
    
    def get_strategy_info(self) -> Dict[str, Dict]:
        """Obtiene información sobre todas las estrategias disponibles."""
        info = {}
        for pattern, config in self.strategies.items():
            info[pattern.value] = {
                'description': config['description'],
                'data_source_priority': config.get('data_source_priority', []),
                'optimizations': [key for key in config.keys() 
                                if key not in ['handler', 'description']]
            }
        return info

# Importar numpy para cálculos
try:
    import numpy as np
except ImportError:
    logger.warning("NumPy no disponible, algunas funciones de ML pueden fallar")
    np = None