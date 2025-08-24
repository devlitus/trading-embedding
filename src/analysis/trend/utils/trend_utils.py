"""Utilidades para análisis de tendencias.

Funciones auxiliares y utilidades comunes para el análisis
de tendencias y detección de patrones.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from scipy import stats
from scipy.signal import find_peaks


def filter_significant_changes(df: pd.DataFrame, 
                             threshold: float = 0.02,
                             column: str = 'close') -> pd.Series:
    """
    Filtra cambios significativos en una serie de precios
    
    Args:
        df: DataFrame con datos de precios
        threshold: Umbral mínimo de cambio (porcentaje)
        column: Columna a analizar
        
    Returns:
        Serie booleana indicando cambios significativos
    """
    if column not in df.columns:
        raise ValueError(f"Columna '{column}' no encontrada en el DataFrame")
    
    prices = df[column]
    pct_change = prices.pct_change().abs()
    
    return pct_change >= threshold


def calculate_trend_consistency(trend_series: pd.Series, window: int = 10) -> pd.Series:
    """
    Calcula la consistencia de una tendencia en una ventana móvil
    
    Args:
        trend_series: Serie con valores de tendencia
        window: Tamaño de la ventana
        
    Returns:
        Serie con scores de consistencia (0-1)
    """
    consistency = trend_series.rolling(window=window).apply(
        lambda x: (x == x.iloc[-1]).sum() / len(x) if len(x) > 0 else 0
    )
    
    return consistency.fillna(0)


def detect_trend_exhaustion(df: pd.DataFrame, 
                          volume_threshold: float = 0.8,
                          price_threshold: float = 0.02) -> pd.Series:
    """
    Detecta señales de agotamiento de tendencia
    
    Args:
        df: DataFrame con datos OHLC y volumen
        volume_threshold: Umbral de volumen decreciente
        price_threshold: Umbral de momentum decreciente
        
    Returns:
        Serie booleana indicando agotamiento
    """
    exhaustion = pd.Series(False, index=df.index)
    
    if len(df) < 10:
        return exhaustion
    
    # Calcular momentum de precio
    price_momentum = df['close'].pct_change(5)
    
    # Calcular tendencia de volumen si está disponible
    if 'volume' in df.columns:
        volume_trend = df['volume'].rolling(window=5).mean() / df['volume'].rolling(window=20).mean()
        
        # Agotamiento: momentum decreciente + volumen decreciente
        exhaustion = (
            (price_momentum.abs() < price_threshold) & 
            (volume_trend < volume_threshold)
        )
    else:
        # Solo basado en momentum de precio
        exhaustion = price_momentum.abs() < price_threshold
    
    return exhaustion


def calculate_volatility_adjusted_trend(prices: pd.Series, 
                                      window: int = 20) -> pd.Series:
    """
    Calcula tendencia ajustada por volatilidad
    
    Args:
        prices: Serie de precios
        window: Ventana para calcular volatilidad
        
    Returns:
        Serie con tendencia ajustada
    """
    # Calcular retornos
    returns = prices.pct_change()
    
    # Calcular volatilidad móvil
    volatility = returns.rolling(window=window).std()
    
    # Calcular tendencia (pendiente)
    trend = prices.rolling(window=window).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else 0
    )
    
    # Ajustar por volatilidad
    adjusted_trend = trend / (volatility + 1e-8)  # Evitar división por cero
    
    return adjusted_trend


def identify_trend_phases(df: pd.DataFrame, 
                        trend_column: str = 'trend') -> pd.Series:
    """
    Identifica fases de tendencia (inicio, desarrollo, agotamiento)
    
    Args:
        df: DataFrame con columna de tendencia
        trend_column: Nombre de la columna de tendencia
        
    Returns:
        Serie con fases identificadas
    """
    if trend_column not in df.columns:
        raise ValueError(f"Columna '{trend_column}' no encontrada")
    
    phases = pd.Series('undefined', index=df.index)
    trend = df[trend_column]
    
    # Detectar cambios de tendencia
    trend_changes = trend != trend.shift(1)
    
    current_phase = 'inicio'
    phase_length = 0
    
    for i, (idx, is_change) in enumerate(trend_changes.items()):
        if is_change and i > 0:
            current_phase = 'inicio'
            phase_length = 0
        else:
            phase_length += 1
            
            # Determinar fase basada en duración
            if phase_length <= 5:
                current_phase = 'inicio'
            elif phase_length <= 20:
                current_phase = 'desarrollo'
            else:
                current_phase = 'maduracion'
        
        phases[idx] = current_phase
    
    return phases


def calculate_trend_velocity(prices: pd.Series, window: int = 5) -> pd.Series:
    """
    Calcula la velocidad de cambio de tendencia
    
    Args:
        prices: Serie de precios
        window: Ventana para calcular velocidad
        
    Returns:
        Serie con velocidades de tendencia
    """
    # Calcular cambio de precio en la ventana
    price_change = prices.diff(window)
    
    # Velocidad = cambio de precio / tiempo
    velocity = price_change / window
    
    # Normalizar por precio actual
    normalized_velocity = velocity / prices
    
    return normalized_velocity


def detect_trend_convergence(trend1: pd.Series, 
                           trend2: pd.Series,
                           threshold: float = 0.01) -> pd.Series:
    """
    Detecta convergencia entre dos indicadores de tendencia
    
    Args:
        trend1: Primera serie de tendencia
        trend2: Segunda serie de tendencia
        threshold: Umbral de convergencia
        
    Returns:
        Serie booleana indicando convergencia
    """
    # Normalizar ambas series
    norm_trend1 = (trend1 - trend1.mean()) / trend1.std()
    norm_trend2 = (trend2 - trend2.mean()) / trend2.std()
    
    # Calcular diferencia
    difference = abs(norm_trend1 - norm_trend2)
    
    # Convergencia cuando la diferencia es menor al umbral
    convergence = difference <= threshold
    
    return convergence


def calculate_support_resistance_strength(df: pd.DataFrame,
                                        levels: List[float],
                                        tolerance: float = 0.01) -> Dict[float, int]:
    """
    Calcula la fuerza de niveles de soporte/resistencia
    
    Args:
        df: DataFrame con datos OHLC
        levels: Lista de niveles de precio
        tolerance: Tolerancia para considerar un toque
        
    Returns:
        Diccionario con nivel y número de toques
    """
    strength = {}
    
    for level in levels:
        touches = 0
        
        # Contar toques en máximos y mínimos
        high_touches = abs(df['high'] - level) / level <= tolerance
        low_touches = abs(df['low'] - level) / level <= tolerance
        
        touches = high_touches.sum() + low_touches.sum()
        strength[level] = touches
    
    return strength


def smooth_trend_signal(signal: pd.Series, 
                       method: str = 'ema',
                       window: int = 5) -> pd.Series:
    """
    Suaviza señales de tendencia para reducir ruido
    
    Args:
        signal: Serie de señales
        method: Método de suavizado ('ema', 'sma', 'median')
        window: Ventana para el suavizado
        
    Returns:
        Serie suavizada
    """
    if method == 'ema':
        return signal.ewm(span=window).mean()
    elif method == 'sma':
        return signal.rolling(window=window).mean()
    elif method == 'median':
        return signal.rolling(window=window).median()
    else:
        raise ValueError(f"Método de suavizado no soportado: {method}")


def calculate_trend_correlation(df: pd.DataFrame,
                              price_col: str = 'close',
                              volume_col: str = 'volume',
                              window: int = 20) -> pd.Series:
    """
    Calcula correlación móvil entre precio y volumen
    
    Args:
        df: DataFrame con datos
        price_col: Columna de precios
        volume_col: Columna de volumen
        window: Ventana para correlación
        
    Returns:
        Serie con correlaciones móviles
    """
    if volume_col not in df.columns:
        return pd.Series(0, index=df.index)
    
    correlation = df[price_col].rolling(window=window).corr(df[volume_col])
    
    return correlation.fillna(0)


def detect_trend_breakout_confirmation(df: pd.DataFrame,
                                     breakout_level: float,
                                     direction: str = 'up',
                                     confirmation_periods: int = 3,
                                     volume_confirmation: bool = True) -> pd.Series:
    """
    Detecta confirmación de rupturas de tendencia
    
    Args:
        df: DataFrame con datos OHLC
        breakout_level: Nivel de ruptura
        direction: Dirección de ruptura ('up' o 'down')
        confirmation_periods: Períodos para confirmar
        volume_confirmation: Si usar volumen para confirmación
        
    Returns:
        Serie booleana con confirmaciones
    """
    confirmed = pd.Series(False, index=df.index)
    
    if direction == 'up':
        breakout = df['close'] > breakout_level
    else:
        breakout = df['close'] < breakout_level
    
    # Confirmar con períodos consecutivos
    for i in range(confirmation_periods, len(df)):
        if breakout.iloc[i-confirmation_periods:i+1].all():
            confirmation = True
            
            # Confirmación adicional con volumen
            if volume_confirmation and 'volume' in df.columns:
                recent_volume = df['volume'].iloc[i-confirmation_periods:i+1].mean()
                avg_volume = df['volume'].iloc[:i+1].mean()
                
                if recent_volume <= avg_volume:
                    confirmation = False
            
            confirmed.iloc[i] = confirmation
    
    return confirmed


def calculate_trend_momentum_divergence(price: pd.Series,
                                      momentum: pd.Series,
                                      window: int = 10) -> pd.Series:
    """
    Detecta divergencias entre precio y momentum
    
    Args:
        price: Serie de precios
        momentum: Serie de momentum
        window: Ventana para análisis
        
    Returns:
        Serie con tipos de divergencia
    """
    divergence = pd.Series('none', index=price.index)
    
    # Calcular pendientes
    price_slope = price.rolling(window=window).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else 0
    )
    
    momentum_slope = momentum.rolling(window=window).apply(
        lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else 0
    )
    
    # Detectar divergencias
    bullish_div = (price_slope < 0) & (momentum_slope > 0)
    bearish_div = (price_slope > 0) & (momentum_slope < 0)
    
    divergence[bullish_div] = 'bullish'
    divergence[bearish_div] = 'bearish'
    
    return divergence