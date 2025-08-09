"""Módulo de Detección de Tendencias

Implementa algoritmos para detectar cambios de tendencia, puntos de inflexión
y estructuras de mercado relevantes para el análisis Wyckoff.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.signal import find_peaks, argrelextrema
from scipy.stats import linregress
import warnings
warnings.filterwarnings('ignore')


class TrendDetector:
    """Clase principal para detección de tendencias"""
    
    def __init__(self, min_trend_length: int = 10, significance_threshold: float = 0.02):
        """
        Args:
            min_trend_length: Longitud mínima para considerar una tendencia válida
            significance_threshold: Umbral mínimo de cambio de precio para considerar significativo
        """
        self.min_trend_length = min_trend_length
        self.significance_threshold = significance_threshold
    
    def detect_trend_changes(self, df: pd.DataFrame, method: str = 'moving_average') -> pd.DataFrame:
        """
        Detecta cambios de tendencia usando diferentes métodos
        
        Args:
            df: DataFrame con datos OHLC
            method: Método a usar ('moving_average', 'zigzag', 'regression')
            
        Returns:
            DataFrame con columnas de tendencia añadidas
        """
        result_df = df.copy()
        
        if method == 'moving_average':
            result_df = self._detect_ma_trend_changes(result_df)
        elif method == 'zigzag':
            result_df = self._detect_zigzag_trend_changes(result_df)
        elif method == 'regression':
            result_df = self._detect_regression_trend_changes(result_df)
        else:
            raise ValueError(f"Método no soportado: {method}")
        
        return result_df
    
    def _detect_ma_trend_changes(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detecta cambios de tendencia usando medias móviles"""
        
        # Calcular medias móviles si no existen
        if 'sma_20' not in df.columns:
            df['sma_20'] = df['close'].rolling(window=20).mean()
        if 'sma_50' not in df.columns:
            df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Determinar tendencia basada en medias móviles
        conditions = [
            (df['close'] > df['sma_20']) & (df['sma_20'] > df['sma_50']),
            (df['close'] < df['sma_20']) & (df['sma_20'] < df['sma_50'])
        ]
        choices = ['uptrend', 'downtrend']
        
        df['trend'] = np.select(conditions, choices, default='sideways')
        
        # Detectar cambios de tendencia
        df['trend_change'] = df['trend'] != df['trend'].shift(1)
        
        # Filtrar cambios menores
        df['significant_trend_change'] = self._filter_significant_changes(df)
        
        return df
    
    def _detect_zigzag_trend_changes(self, df: pd.DataFrame, threshold: float = 0.05) -> pd.DataFrame:
        """Detecta cambios de tendencia usando algoritmo ZigZag"""
        
        highs, lows = self._calculate_zigzag_points(df['close'], threshold)
        
        # Crear series de tendencia
        df['zigzag_trend'] = 'sideways'
        df['zigzag_points'] = False
        
        # Marcar puntos ZigZag
        for high_idx in highs:
            if high_idx < len(df):
                df.iloc[high_idx, df.columns.get_loc('zigzag_points')] = True
                
        for low_idx in lows:
            if low_idx < len(df):
                df.iloc[low_idx, df.columns.get_loc('zigzag_points')] = True
        
        # Determinar tendencia entre puntos
        df['zigzag_trend'] = self._interpolate_zigzag_trend(df, highs, lows)
        
        return df
    
    def _detect_regression_trend_changes(self, df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """Detecta cambios de tendencia usando regresión lineal móvil"""
        
        slopes = []
        r_values = []
        
        for i in range(len(df)):
            start_idx = max(0, i - window + 1)
            end_idx = i + 1
            
            if end_idx - start_idx >= 5:  # Mínimo 5 puntos para regresión
                y = df['close'].iloc[start_idx:end_idx].values
                x = np.arange(len(y))
                
                try:
                    slope, intercept, r_value, p_value, std_err = linregress(x, y)
                    slopes.append(slope)
                    r_values.append(abs(r_value))
                except:
                    slopes.append(0)
                    r_values.append(0)
            else:
                slopes.append(0)
                r_values.append(0)
        
        df['regression_slope'] = slopes
        df['regression_r_value'] = r_values
        
        # Determinar tendencia basada en pendiente y correlación
        conditions = [
            (df['regression_slope'] > 0) & (df['regression_r_value'] > 0.7),
            (df['regression_slope'] < 0) & (df['regression_r_value'] > 0.7)
        ]
        choices = ['uptrend', 'downtrend']
        
        df['regression_trend'] = np.select(conditions, choices, default='sideways')
        
        return df
    
    def _calculate_zigzag_points(self, prices: pd.Series, threshold: float) -> Tuple[List[int], List[int]]:
        """Calcula puntos altos y bajos del ZigZag"""
        
        highs = []
        lows = []
        
        # Encontrar máximos y mínimos locales
        high_indices = argrelextrema(prices.values, np.greater, order=5)[0]
        low_indices = argrelextrema(prices.values, np.less, order=5)[0]
        
        # Filtrar por umbral de significancia
        for i in high_indices:
            if i > 0 and i < len(prices) - 1:
                price_change = abs(prices.iloc[i] - prices.iloc[i-5:i+5].mean()) / prices.iloc[i]
                if price_change >= threshold:
                    highs.append(i)
        
        for i in low_indices:
            if i > 0 and i < len(prices) - 1:
                price_change = abs(prices.iloc[i] - prices.iloc[i-5:i+5].mean()) / prices.iloc[i]
                if price_change >= threshold:
                    lows.append(i)
        
        return highs, lows
    
    def _interpolate_zigzag_trend(self, df: pd.DataFrame, highs: List[int], lows: List[int]) -> pd.Series:
        """Interpola la tendencia entre puntos ZigZag"""
        
        trend = pd.Series('sideways', index=df.index)
        
        # Combinar y ordenar puntos
        all_points = [(i, 'high') for i in highs] + [(i, 'low') for i in lows]
        all_points.sort(key=lambda x: x[0])
        
        # Determinar tendencia entre puntos consecutivos
        for i in range(len(all_points) - 1):
            start_idx, start_type = all_points[i]
            end_idx, end_type = all_points[i + 1]
            
            if start_type == 'low' and end_type == 'high':
                trend.iloc[start_idx:end_idx] = 'uptrend'
            elif start_type == 'high' and end_type == 'low':
                trend.iloc[start_idx:end_idx] = 'downtrend'
        
        return trend
    
    def _filter_significant_changes(self, df: pd.DataFrame) -> pd.Series:
        """Filtra cambios de tendencia significativos"""
        
        significant_changes = pd.Series(False, index=df.index)
        
        trend_changes = df[df['trend_change']].index
        
        for i, change_idx in enumerate(trend_changes):
            if i == 0:
                significant_changes.loc[change_idx] = True
                continue
            
            # Calcular cambio de precio desde el último cambio significativo
            prev_change_idx = trend_changes[i-1]
            price_change = abs(df.loc[change_idx, 'close'] - df.loc[prev_change_idx, 'close'])
            price_change_pct = price_change / df.loc[prev_change_idx, 'close']
            
            # Verificar duración mínima (número de períodos)
            duration = (change_idx - prev_change_idx).days if hasattr((change_idx - prev_change_idx), 'days') else int(change_idx - prev_change_idx)
            
            if price_change_pct >= self.significance_threshold and duration >= self.min_trend_length:
                significant_changes.loc[change_idx] = True
        
        return significant_changes
    
    def detect_support_resistance_levels(self, df: pd.DataFrame, window: int = 20, min_touches: int = 2) -> Dict[str, List[float]]:
        """
        Detecta niveles de soporte y resistencia
        
        Args:
            df: DataFrame con datos OHLC
            window: Ventana para buscar niveles
            min_touches: Número mínimo de toques para considerar un nivel válido
            
        Returns:
            Diccionario con listas de niveles de soporte y resistencia
        """
        
        # Encontrar máximos y mínimos locales
        highs = find_peaks(df['high'].values, distance=window//2)[0]
        lows = find_peaks(-df['low'].values, distance=window//2)[0]
        
        # Agrupar niveles similares
        resistance_levels = self._group_similar_levels(df['high'].iloc[highs].values, min_touches)
        support_levels = self._group_similar_levels(df['low'].iloc[lows].values, min_touches)
        
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }
    
    def _group_similar_levels(self, levels: np.ndarray, min_touches: int, tolerance: float = 0.01) -> List[float]:
        """Agrupa niveles similares"""
        
        if len(levels) == 0:
            return []
        
        grouped_levels = []
        levels_sorted = np.sort(levels)
        
        current_group = [levels_sorted[0]]
        
        for level in levels_sorted[1:]:
            # Si el nivel está dentro de la tolerancia, añadir al grupo actual
            if abs(level - np.mean(current_group)) / np.mean(current_group) <= tolerance:
                current_group.append(level)
            else:
                # Si el grupo tiene suficientes toques, añadir a la lista
                if len(current_group) >= min_touches:
                    grouped_levels.append(np.mean(current_group))
                current_group = [level]
        
        # Procesar el último grupo
        if len(current_group) >= min_touches:
            grouped_levels.append(np.mean(current_group))
        
        return grouped_levels
    
    def calculate_trend_strength_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula un score de fuerza de tendencia (0-100)
        
        Returns:
            Series con scores de fuerza de tendencia
        """
        
        scores = []
        
        for i in range(len(df)):
            score = 0
            
            # Factor 1: Consistencia de dirección (30 puntos)
            if i >= 10:
                recent_closes = df['close'].iloc[i-10:i+1]
                if recent_closes.is_monotonic_increasing:
                    score += 30
                elif recent_closes.is_monotonic_decreasing:
                    score += 30
                else:
                    # Calcular consistencia parcial
                    increases = (recent_closes.diff() > 0).sum()
                    decreases = (recent_closes.diff() < 0).sum()
                    consistency = abs(increases - decreases) / 10
                    score += consistency * 30
            
            # Factor 2: Volumen confirmatorio (25 puntos)
            if 'volume' in df.columns and i >= 5:
                recent_volume = df['volume'].iloc[i-5:i+1].mean()
                avg_volume = df['volume'].iloc[:i+1].mean() if i > 20 else recent_volume
                if recent_volume > avg_volume:
                    score += 25
                else:
                    score += (recent_volume / avg_volume) * 25
            
            # Factor 3: Momentum (25 puntos)
            if i >= 5:
                price_momentum = (df['close'].iloc[i] - df['close'].iloc[i-5]) / df['close'].iloc[i-5]
                score += min(abs(price_momentum) * 500, 25)  # Normalizar
            
            # Factor 4: Volatilidad (20 puntos) - menor volatilidad = mayor score
            if i >= 10:
                volatility = df['close'].iloc[i-10:i+1].std() / df['close'].iloc[i-10:i+1].mean()
                score += max(0, 20 - (volatility * 1000))
            
            scores.append(min(100, max(0, score)))
        
        return pd.Series(scores, index=df.index)
    
    def identify_market_structure(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Identifica estructuras de mercado (Higher Highs, Lower Lows, etc.)
        
        Returns:
            Diccionario con diferentes estructuras identificadas
        """
        
        # Encontrar máximos y mínimos locales
        highs_idx = find_peaks(df['high'].values, distance=10)[0]
        lows_idx = find_peaks(-df['low'].values, distance=10)[0]
        
        # Inicializar series
        higher_highs = pd.Series(False, index=df.index)
        lower_highs = pd.Series(False, index=df.index)
        higher_lows = pd.Series(False, index=df.index)
        lower_lows = pd.Series(False, index=df.index)
        
        # Analizar máximos
        for i in range(1, len(highs_idx)):
            current_idx = highs_idx[i]
            prev_idx = highs_idx[i-1]
            
            current_high = df['high'].iloc[current_idx]
            prev_high = df['high'].iloc[prev_idx]
            
            if current_high > prev_high:
                higher_highs.iloc[current_idx] = True
            else:
                lower_highs.iloc[current_idx] = True
        
        # Analizar mínimos
        for i in range(1, len(lows_idx)):
            current_idx = lows_idx[i]
            prev_idx = lows_idx[i-1]
            
            current_low = df['low'].iloc[current_idx]
            prev_low = df['low'].iloc[prev_idx]
            
            if current_low > prev_low:
                higher_lows.iloc[current_idx] = True
            else:
                lower_lows.iloc[current_idx] = True
        
        return {
            'higher_highs': higher_highs,
            'lower_highs': lower_highs,
            'higher_lows': higher_lows,
            'lower_lows': lower_lows
        }


def detect_trend_changes(df: pd.DataFrame, method: str = 'moving_average') -> pd.DataFrame:
    """
    Función de conveniencia para detectar cambios de tendencia
    
    Args:
        df: DataFrame con datos OHLC
        method: Método de detección
        
    Returns:
        DataFrame con información de tendencias
    """
    detector = TrendDetector()
    return detector.detect_trend_changes(df, method)


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Crear datos de ejemplo
    dates = pd.date_range('2023-01-01', periods=200, freq='1H')
    np.random.seed(42)
    
    # Simular datos OHLC con tendencia
    trend = np.linspace(0, 20, 200) + np.sin(np.linspace(0, 4*np.pi, 200)) * 5
    noise = np.random.randn(200) * 2
    close_prices = 100 + trend + noise
    
    high_prices = close_prices + np.random.rand(200) * 3
    low_prices = close_prices - np.random.rand(200) * 3
    open_prices = close_prices + np.random.randn(200) * 1
    volumes = np.random.randint(1000, 10000, 200)
    
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    # Detectar tendencias
    detector = TrendDetector()
    result = detector.detect_trend_changes(sample_df, method='moving_average')
    
    print("Detección de tendencias:")
    print(result[['close', 'trend', 'trend_change', 'significant_trend_change']].tail(10))
    
    # Calcular fuerza de tendencia
    strength = detector.calculate_trend_strength_score(result)
    print(f"\nFuerza de tendencia actual: {strength.iloc[-1]:.1f}/100")
    
    # Detectar niveles de soporte y resistencia
    levels = detector.detect_support_resistance_levels(result)
    print(f"\nNiveles de soporte: {levels['support'][:3]}")
    print(f"Niveles de resistencia: {levels['resistance'][:3]}")