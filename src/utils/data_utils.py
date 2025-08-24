"""Utilidades comunes para procesamiento y validación de datos de trading.

Este módulo consolida funciones duplicadas de normalización, validación
y preparación de datos que se encuentran dispersas en el proyecto.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

# Mapeo estándar de columnas de base de datos a formato normalizado
STANDARD_COLUMN_MAPPING = {
    'timestamp': 'datetime',
    'open_price': 'open',
    'high_price': 'high', 
    'low_price': 'low',
    'close_price': 'close',
    'trades_count': 'trades_count'
}

# Columnas numéricas estándar para datos OHLCV
NUMERIC_COLUMNS = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']

# Columnas requeridas para análisis básico
REQUIRED_OHLCV_COLUMNS = ['open', 'high', 'low', 'close', 'volume']

def normalize_column_names(data: pd.DataFrame, 
                          column_mapping: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    Normaliza nombres de columnas entre diferentes fuentes de datos.
    
    Args:
        data: DataFrame con datos a normalizar
        column_mapping: Mapeo personalizado de columnas (opcional)
        
    Returns:
        DataFrame con columnas normalizadas
    """
    if data.empty:
        return data
    
    # Usar mapeo por defecto si no se proporciona uno personalizado
    mapping = column_mapping or STANDARD_COLUMN_MAPPING
    
    # Resetear índice si hay conflictos con datetime
    if data.index.name == 'datetime' or 'datetime' in data.index.names:
        data = data.reset_index()
    
    # Renombrar columnas si existen y no hay duplicados
    for old_col, new_col in mapping.items():
        if old_col in data.columns and new_col not in data.columns:
            data = data.rename(columns={old_col: new_col})
    
    # Convertir timestamp a datetime si es necesario
    data = _convert_timestamp_columns(data)
    
    return data

def _convert_timestamp_columns(data: pd.DataFrame) -> pd.DataFrame:
    """
    Convierte columnas de timestamp a datetime.
    
    Args:
        data: DataFrame con posibles columnas de timestamp
        
    Returns:
        DataFrame con timestamps convertidos
    """
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

def ensure_numeric_types(data: pd.DataFrame, 
                        numeric_columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Asegura que las columnas numéricas tengan el tipo correcto.
    
    Args:
        data: DataFrame a procesar
        numeric_columns: Lista de columnas a convertir (opcional)
        
    Returns:
        DataFrame con tipos numéricos correctos
    """
    if data.empty:
        return data
    
    columns_to_convert = numeric_columns or NUMERIC_COLUMNS
    
    for col in columns_to_convert:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    return data

def prepare_data_for_ml(data: pd.DataFrame, 
                       include_features: bool = True,
                       drop_na: bool = True) -> pd.DataFrame:
    """
    Prepara datos para análisis de Machine Learning.
    
    Args:
        data: DataFrame con datos OHLCV
        include_features: Si incluir features adicionales
        drop_na: Si eliminar valores nulos
        
    Returns:
        DataFrame preparado para ML
    """
    if data.empty:
        return data
    
    # Normalizar columnas
    data = normalize_column_names(data)
    
    # Asegurar tipos numéricos
    data = ensure_numeric_types(data)
    
    # Ordenar por fecha si existe columna datetime
    if 'datetime' in data.columns:
        data = data.sort_values('datetime').reset_index(drop=True)
    
    # Agregar features básicas si se solicita
    if include_features and 'close' in data.columns:
        data = add_basic_features(data)
    
    # Eliminar valores nulos si se solicita
    if drop_na:
        data = data.dropna()
    
    return data

def add_basic_features(data: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega features básicas para análisis.
    
    Args:
        data: DataFrame con datos OHLCV
        
    Returns:
        DataFrame con features adicionales
    """
    if data.empty or 'close' not in data.columns:
        return data
    
    try:
        # Returns y log returns
        data['returns'] = data['close'].pct_change()
        data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
        
        # Volatilidad rolling
        data['volatility'] = data['returns'].rolling(window=20, min_periods=1).std()
        
        # RSI básico
        data['rsi'] = calculate_rsi(data['close'])
        
        # Cambio de precio y volumen promedio
        data['price_change'] = data['close'].pct_change()
        
        if 'volume' in data.columns:
            data['volume_ma'] = data['volume'].rolling(window=20, min_periods=1).mean()
            
    except Exception as e:
        logger.warning(f"Error agregando features básicas: {e}")
    
    return data

def calculate_rsi(prices: pd.Series, window: int = 14) -> pd.Series:
    """
    Calcula el RSI (Relative Strength Index).
    
    Args:
        prices: Serie de precios
        window: Ventana para el cálculo
        
    Returns:
        Serie con valores RSI
    """
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window, min_periods=1).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi.fillna(50)  # Valor neutral para NaN
    except Exception:
        return pd.Series([50] * len(prices), index=prices.index)

def assess_data_quality(data: pd.DataFrame, 
                       required_columns: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Evalúa la calidad de los datos OHLCV.
    
    Args:
        data: DataFrame a evaluar
        required_columns: Columnas requeridas (opcional)
        
    Returns:
        Diccionario con métricas de calidad
    """
    if data.empty:
        return {'status': 'empty', 'quality_score': 0.0}
    
    # Normalizar datos para evaluación
    normalized_data = normalize_column_names(data)
    required_cols = required_columns or REQUIRED_OHLCV_COLUMNS
    
    # Calcular métricas básicas
    total_cells = len(normalized_data) * len(normalized_data.columns)
    null_cells = normalized_data.isnull().sum().sum()
    completeness = (1 - null_cells / total_cells) * 100 if total_cells > 0 else 0
    
    # Detectar duplicados por timestamp o índice
    duplicates_count = 0
    if 'timestamp' in normalized_data.columns:
        duplicates_count = normalized_data.duplicated(subset=['timestamp']).sum()
    elif 'datetime' in normalized_data.columns:
        duplicates_count = normalized_data.duplicated(subset=['datetime']).sum()
    elif hasattr(normalized_data.index, 'duplicated'):
        duplicates_count = normalized_data.index.duplicated().sum()
    else:
        duplicates_count = normalized_data.duplicated().sum()
    
    quality_metrics = {
        'total_rows': len(normalized_data),
        'total_columns': len(normalized_data.columns),
        'completeness_pct': round(completeness, 2),
        'duplicates_count': duplicates_count,
        'null_values_count': int(null_cells),
        'date_continuity': 'good' if len(normalized_data) > 1 else 'insufficient',
        'has_required_columns': all(col in normalized_data.columns for col in required_cols),
        'numeric_validity': _check_numeric_validity(normalized_data),
        'quality_issues': []
    }
    
    # Detectar problemas de calidad
    issues = []
    
    if completeness < 95:
        issues.append(f"Completeness baja: {completeness:.1f}%")
    
    if quality_metrics['duplicates_count'] > 0:
        issues.append(f"Duplicados encontrados: {quality_metrics['duplicates_count']}")
    
    if not quality_metrics['has_required_columns']:
        missing_cols = [col for col in required_cols if col not in normalized_data.columns]
        issues.append(f"Columnas faltantes: {missing_cols}")
    
    if not quality_metrics['numeric_validity']:
        issues.append("Problemas de validez numérica en columnas OHLCV")
    
    quality_metrics['quality_issues'] = issues
    
    # Calcular score general de calidad
    quality_score = _calculate_quality_score(quality_metrics)
    quality_metrics['quality_score'] = quality_score
    quality_metrics['status'] = _get_quality_status(quality_score)
    
    return quality_metrics

def _check_numeric_validity(data: pd.DataFrame) -> bool:
    """
    Verifica que las columnas numéricas sean válidas.
    
    Args:
        data: DataFrame a verificar
        
    Returns:
        True si las columnas numéricas son válidas
    """
    try:
        numeric_cols = [col for col in NUMERIC_COLUMNS if col in data.columns]
        return all(pd.api.types.is_numeric_dtype(data[col]) for col in numeric_cols)
    except Exception:
        return False

def _calculate_quality_score(metrics: Dict[str, Any]) -> float:
    """
    Calcula un score general de calidad de datos.
    
    Args:
        metrics: Métricas de calidad
        
    Returns:
        Score de calidad entre 0 y 1
    """
    score = 0.0
    
    # Completeness (40% del score)
    completeness_score = metrics['completeness_pct'] / 100
    score += completeness_score * 0.4
    
    # Ausencia de duplicados (20% del score)
    if metrics['duplicates_count'] == 0:
        score += 0.2
    
    # Columnas requeridas (20% del score)
    if metrics['has_required_columns']:
        score += 0.2
    
    # Validez numérica (20% del score)
    if metrics['numeric_validity']:
        score += 0.2
    
    return round(score, 3)

def _get_quality_status(score: float) -> str:
    """
    Determina el status de calidad basado en el score.
    
    Args:
        score: Score de calidad
        
    Returns:
        Status de calidad
    """
    if score >= 0.9:
        return 'excellent'
    elif score >= 0.7:
        return 'good'
    elif score >= 0.5:
        return 'fair'
    else:
        return 'poor'

def validate_ohlcv_data(data: pd.DataFrame, 
                       symbol: str = "Unknown",
                       strict: bool = False) -> Dict[str, Any]:
    """
    Valida datos OHLCV con reglas específicas de trading.
    
    Args:
        data: DataFrame con datos OHLCV
        symbol: Símbolo del activo (para logging)
        strict: Si aplicar validaciones estrictas
        
    Returns:
        Resultado de validación
    """
    validation_result = {
        'is_valid': True,
        'errors': [],
        'warnings': [],
        'symbol': symbol,
        'total_rows': len(data)
    }
    
    if data.empty:
        validation_result['is_valid'] = False
        validation_result['errors'].append("Dataset vacío")
        return validation_result
    
    # Normalizar datos
    normalized_data = normalize_column_names(data)
    
    # Validaciones básicas
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    missing_cols = [col for col in required_cols if col not in normalized_data.columns]
    
    if missing_cols:
        validation_result['is_valid'] = False
        validation_result['errors'].append(f"Columnas faltantes: {missing_cols}")
        return validation_result
    
    # Validaciones de integridad OHLCV
    try:
        # High >= Low
        invalid_hl = (normalized_data['high'] < normalized_data['low']).sum()
        if invalid_hl > 0:
            validation_result['errors'].append(f"High < Low en {invalid_hl} filas")
            if strict:
                validation_result['is_valid'] = False
        
        # Open, Close dentro del rango High-Low
        invalid_open = ((normalized_data['open'] > normalized_data['high']) | 
                       (normalized_data['open'] < normalized_data['low'])).sum()
        invalid_close = ((normalized_data['close'] > normalized_data['high']) | 
                        (normalized_data['close'] < normalized_data['low'])).sum()
        
        if invalid_open > 0:
            validation_result['warnings'].append(f"Open fuera de rango H-L en {invalid_open} filas")
        
        if invalid_close > 0:
            validation_result['warnings'].append(f"Close fuera de rango H-L en {invalid_close} filas")
        
        # Volumen no negativo
        negative_volume = (normalized_data['volume'] < 0).sum()
        if negative_volume > 0:
            validation_result['errors'].append(f"Volumen negativo en {negative_volume} filas")
            if strict:
                validation_result['is_valid'] = False
        
        # Precios no negativos
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            negative_prices = (normalized_data[col] <= 0).sum()
            if negative_prices > 0:
                validation_result['errors'].append(f"Precios no positivos en {col}: {negative_prices} filas")
                if strict:
                    validation_result['is_valid'] = False
        
    except Exception as e:
        validation_result['is_valid'] = False
        validation_result['errors'].append(f"Error en validación: {str(e)}")
    
    return validation_result

def clean_ohlcv_data(data: pd.DataFrame, 
                     remove_duplicates: bool = True,
                     fill_missing: bool = False,
                     validate_integrity: bool = True) -> pd.DataFrame:
    """
    Limpia y prepara datos OHLCV para análisis.
    
    Args:
        data: DataFrame con datos OHLCV
        remove_duplicates: Si eliminar duplicados
        fill_missing: Si rellenar valores faltantes
        validate_integrity: Si validar integridad OHLCV
        
    Returns:
        DataFrame limpio
    """
    if data.empty:
        return data
    
    # Normalizar columnas
    cleaned_data = normalize_column_names(data.copy())
    
    # Asegurar tipos numéricos
    cleaned_data = ensure_numeric_types(cleaned_data)
    
    # Eliminar duplicados si se solicita
    if remove_duplicates:
        initial_rows = len(cleaned_data)
        cleaned_data = cleaned_data.drop_duplicates()
        removed_duplicates = initial_rows - len(cleaned_data)
        if removed_duplicates > 0:
            logger.info(f"Eliminados {removed_duplicates} registros duplicados")
    
    # Ordenar por datetime si existe
    if 'datetime' in cleaned_data.columns:
        cleaned_data = cleaned_data.sort_values('datetime').reset_index(drop=True)
    
    # Validar integridad OHLCV y corregir si es posible
    if validate_integrity and all(col in cleaned_data.columns for col in ['open', 'high', 'low', 'close']):
        cleaned_data = _fix_ohlcv_integrity(cleaned_data)
    
    # Rellenar valores faltantes si se solicita
    if fill_missing:
        cleaned_data = _fill_missing_values(cleaned_data)
    
    return cleaned_data

def _fix_ohlcv_integrity(data: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige problemas básicos de integridad en datos OHLCV.
    
    Args:
        data: DataFrame con datos OHLCV
        
    Returns:
        DataFrame con integridad corregida
    """
    try:
        # Corregir High < Low intercambiando valores
        invalid_hl_mask = data['high'] < data['low']
        if invalid_hl_mask.any():
            logger.warning(f"Corrigiendo {invalid_hl_mask.sum()} filas con High < Low")
            data.loc[invalid_hl_mask, ['high', 'low']] = data.loc[invalid_hl_mask, ['low', 'high']].values
        
        # Ajustar Open y Close al rango High-Low
        data['open'] = data['open'].clip(lower=data['low'], upper=data['high'])
        data['close'] = data['close'].clip(lower=data['low'], upper=data['high'])
        
    except Exception as e:
        logger.warning(f"Error corrigiendo integridad OHLCV: {e}")
    
    return data

def _fill_missing_values(data: pd.DataFrame) -> pd.DataFrame:
    """
    Rellena valores faltantes con métodos apropiados.
    
    Args:
        data: DataFrame con posibles valores faltantes
        
    Returns:
        DataFrame con valores faltantes rellenados
    """
    try:
        # Para precios, usar forward fill y luego backward fill
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col in data.columns:
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
        
        # Para volumen, usar 0 o promedio móvil
        if 'volume' in data.columns:
            data['volume'] = data['volume'].fillna(0)
        
    except Exception as e:
        logger.warning(f"Error rellenando valores faltantes: {e}")
    
    return data