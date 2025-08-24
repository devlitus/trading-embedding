"""Utilidades consolidadas para análisis técnico.

Este módulo contiene funciones comunes utilizadas en análisis técnico
para evitar duplicación de código.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


def calculate_trend_strength(df: pd.DataFrame, method: str = 'sma') -> pd.Series:
    """
    Calcula la fuerza de la tendencia usando diferentes métodos.
    
    Args:
        df: DataFrame con datos OHLC
        method: Método de cálculo ('sma', 'regression', 'combined')
        
    Returns:
        Series con valores de fuerza de tendencia (0-100)
    """
    try:
        if method == 'sma':
            return _calculate_sma_trend_strength(df)
        elif method == 'regression':
            return _calculate_regression_trend_strength(df)
        elif method == 'combined':
            sma_strength = _calculate_sma_trend_strength(df)
            reg_strength = _calculate_regression_trend_strength(df)
            return (sma_strength + reg_strength) / 2
        else:
            raise ValueError(f"Método no soportado: {method}")
    except Exception as e:
        logger.error(f"Error calculando fuerza de tendencia: {e}")
        return pd.Series([50] * len(df), index=df.index)


def _calculate_sma_trend_strength(df: pd.DataFrame) -> pd.Series:
    """
    Calcula fuerza de tendencia basada en SMA.
    """
    # Calcular SMA si no existe
    if 'sma_20' not in df.columns:
        df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # Pendiente de la SMA
    sma_slope = df['sma_20'].diff(5) / df['sma_20'].shift(5)
    
    # Distancia del precio a la SMA
    price_distance = abs(df['close'] - df['sma_20']) / df['sma_20']
    
    # Combinar factores y normalizar a 0-100
    trend_strength = (abs(sma_slope) + price_distance) / 2
    return np.clip(trend_strength * 100, 0, 100)


def _calculate_regression_trend_strength(df: pd.DataFrame) -> pd.Series:
    """
    Calcula fuerza de tendencia basada en regresión lineal.
    """
    window = 20
    strength_values = []
    
    for i in range(len(df)):
        if i < window:
            strength_values.append(50)  # Valor neutral
            continue
            
        # Datos de la ventana
        window_data = df['close'].iloc[i-window:i+1]
        x = np.arange(len(window_data))
        
        # Regresión lineal
        try:
            slope, intercept = np.polyfit(x, window_data, 1)
            y_pred = slope * x + intercept
            
            # R-cuadrado
            ss_res = np.sum((window_data - y_pred) ** 2)
            ss_tot = np.sum((window_data - np.mean(window_data)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            # Normalizar pendiente
            normalized_slope = abs(slope) * 1000
            
            # Combinar factores
            strength = min(normalized_slope * r_squared, 100)
            strength_values.append(strength)
            
        except Exception:
            strength_values.append(50)
    
    return pd.Series(strength_values, index=df.index)


def calculate_support_resistance_levels(df: pd.DataFrame, 
                                      window: int = 20,
                                      min_touches: int = 2) -> Dict[str, List[float]]:
    """
    Calcula niveles de soporte y resistencia.
    
    Args:
        df: DataFrame con datos OHLC
        window: Ventana para buscar extremos
        min_touches: Mínimo número de toques para validar nivel
        
    Returns:
        Dict con listas de niveles de soporte y resistencia
    """
    try:
        # Encontrar máximos y mínimos locales
        highs = df['high'].rolling(window=window, center=True).max()
        lows = df['low'].rolling(window=window, center=True).min()
        
        # Identificar extremos
        resistance_candidates = df[df['high'] == highs]['high'].dropna()
        support_candidates = df[df['low'] == lows]['low'].dropna()
        
        # Agrupar niveles similares (tolerancia del 0.5%)
        tolerance = 0.005
        
        resistance_levels = _group_similar_levels(resistance_candidates, tolerance, min_touches)
        support_levels = _group_similar_levels(support_candidates, tolerance, min_touches)
        
        return {
            'resistance': sorted(resistance_levels, reverse=True),
            'support': sorted(support_levels)
        }
        
    except Exception as e:
        logger.error(f"Error calculando soporte/resistencia: {e}")
        return {'resistance': [], 'support': []}


def _group_similar_levels(levels: pd.Series, tolerance: float, min_touches: int) -> List[float]:
    """
    Agrupa niveles similares y filtra por número mínimo de toques.
    """
    if len(levels) == 0:
        return []
    
    grouped_levels = []
    levels_list = sorted(levels.tolist())
    
    current_group = [levels_list[0]]
    
    for level in levels_list[1:]:
        # Si el nivel está dentro de la tolerancia, agregarlo al grupo actual
        if abs(level - np.mean(current_group)) / np.mean(current_group) <= tolerance:
            current_group.append(level)
        else:
            # Si el grupo tiene suficientes toques, agregarlo a los resultados
            if len(current_group) >= min_touches:
                grouped_levels.append(np.mean(current_group))
            current_group = [level]
    
    # Procesar el último grupo
    if len(current_group) >= min_touches:
        grouped_levels.append(np.mean(current_group))
    
    return grouped_levels


def calculate_volatility_metrics(df: pd.DataFrame, window: int = 20) -> Dict[str, pd.Series]:
    """
    Calcula métricas de volatilidad.
    
    Args:
        df: DataFrame con datos OHLC
        window: Ventana para cálculos
        
    Returns:
        Dict con diferentes métricas de volatilidad
    """
    try:
        metrics = {}
        
        # Volatilidad de retornos
        returns = df['close'].pct_change()
        metrics['returns_volatility'] = returns.rolling(window=window).std() * np.sqrt(252)
        
        # True Range
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        metrics['atr'] = true_range.rolling(window=window).mean()
        
        # Volatilidad normalizada
        metrics['normalized_volatility'] = metrics['atr'] / df['close'] * 100
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error calculando volatilidad: {e}")
        return {}


def detect_price_patterns(df: pd.DataFrame, pattern_type: str = 'all') -> Dict[str, pd.Series]:
    """
    Detecta patrones básicos de precios.
    
    Args:
        df: DataFrame con datos OHLC
        pattern_type: Tipo de patrón ('doji', 'hammer', 'engulfing', 'all')
        
    Returns:
        Dict con series booleanas indicando presencia de patrones
    """
    try:
        patterns = {}
        
        # Calcular métricas básicas
        body = abs(df['close'] - df['open'])
        upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
        lower_shadow = df[['open', 'close']].min(axis=1) - df['low']
        total_range = df['high'] - df['low']
        
        if pattern_type in ['doji', 'all']:
            # Doji: cuerpo pequeño relativo al rango total
            patterns['doji'] = body <= (total_range * 0.1)
        
        if pattern_type in ['hammer', 'all']:
            # Hammer: sombra inferior larga, cuerpo pequeño en la parte superior
            patterns['hammer'] = (
                (lower_shadow >= body * 2) & 
                (upper_shadow <= body * 0.5) &
                (body <= total_range * 0.3)
            )
        
        if pattern_type in ['engulfing', 'all']:
            # Patrón envolvente alcista
            bullish_engulfing = (
                (df['close'].shift(1) < df['open'].shift(1)) &  # Vela anterior bajista
                (df['close'] > df['open']) &  # Vela actual alcista
                (df['open'] < df['close'].shift(1)) &  # Abre por debajo del cierre anterior
                (df['close'] > df['open'].shift(1))  # Cierra por encima de la apertura anterior
            )
            patterns['bullish_engulfing'] = bullish_engulfing
        
        return patterns
        
    except Exception as e:
        logger.error(f"Error detectando patrones: {e}")
        return {}


def calculate_momentum_indicators(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calcula indicadores de momentum.
    
    Args:
        df: DataFrame con datos OHLC
        
    Returns:
        Dict con indicadores de momentum
    """
    try:
        indicators = {}
        
        # Rate of Change (ROC)
        indicators['roc_10'] = ((df['close'] / df['close'].shift(10)) - 1) * 100
        
        # Momentum
        indicators['momentum_10'] = df['close'] - df['close'].shift(10)
        
        # Williams %R
        high_14 = df['high'].rolling(window=14).max()
        low_14 = df['low'].rolling(window=14).min()
        indicators['williams_r'] = -100 * (high_14 - df['close']) / (high_14 - low_14)
        
        # Stochastic %K
        indicators['stoch_k'] = 100 * (df['close'] - low_14) / (high_14 - low_14)
        indicators['stoch_d'] = indicators['stoch_k'].rolling(window=3).mean()
        
        return indicators
        
    except Exception as e:
        logger.error(f"Error calculando momentum: {e}")
        return {}


def format_analysis_summary(results: Dict[str, Any]) -> str:
    """
    Formatea un resumen de análisis técnico.
    
    Args:
        results: Dict con resultados de análisis
        
    Returns:
        String formateado con el resumen
    """
    try:
        summary_lines = []
        summary_lines.append("=" * 50)
        summary_lines.append("RESUMEN DE ANÁLISIS TÉCNICO")
        summary_lines.append("=" * 50)
        
        if 'symbol' in results:
            summary_lines.append(f"Símbolo: {results['symbol']}")
        
        if 'timeframe' in results:
            summary_lines.append(f"Marco temporal: {results['timeframe']}")
        
        if 'trend_strength' in results:
            strength = results['trend_strength']
            if isinstance(strength, (int, float)):
                summary_lines.append(f"Fuerza de tendencia: {strength:.1f}/100")
        
        if 'support_levels' in results:
            levels = results['support_levels']
            if levels:
                summary_lines.append(f"Niveles de soporte: {', '.join([f'{l:.4f}' for l in levels[:3]])}")
        
        if 'resistance_levels' in results:
            levels = results['resistance_levels']
            if levels:
                summary_lines.append(f"Niveles de resistencia: {', '.join([f'{l:.4f}' for l in levels[:3]])}")
        
        if 'patterns_detected' in results:
            patterns = results['patterns_detected']
            if patterns:
                summary_lines.append(f"Patrones detectados: {len(patterns)}")
        
        summary_lines.append("=" * 50)
        
        return "\n".join(summary_lines)
        
    except Exception as e:
        logger.error(f"Error formateando resumen: {e}")
        return "Error generando resumen de análisis"