"""Detector principal de tendencias.

Implementa la clase TrendDetector que coordina diferentes algoritmos
de detección de tendencias y proporciona una interfaz unificada.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from ..algorithms.moving_average import MovingAverageDetector
from ..algorithms.zigzag import ZigZagDetector
from ..algorithms.regression import RegressionDetector
from ..utils.trend_utils import filter_significant_changes


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
        
        # Inicializar detectores específicos
        self.ma_detector = MovingAverageDetector()
        self.zigzag_detector = ZigZagDetector()
        self.regression_detector = RegressionDetector()
    
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
            result_df = self.ma_detector.detect(result_df)
        elif method == 'zigzag':
            result_df = self.zigzag_detector.detect(result_df)
        elif method == 'regression':
            result_df = self.regression_detector.detect(result_df)
        else:
            raise ValueError(f"Método no soportado: {method}")
        
        # Filtrar cambios significativos
        result_df['significant_trend_change'] = filter_significant_changes(
            result_df, self.significance_threshold
        )
        
        return result_df
    
    def detect_support_resistance_levels(self, df: pd.DataFrame, 
                                       window: int = 20, 
                                       min_touches: int = 2) -> Dict[str, List[float]]:
        """
        Detecta niveles de soporte y resistencia
        
        Args:
            df: DataFrame con datos OHLC
            window: Ventana para buscar máximos/mínimos locales
            min_touches: Número mínimo de toques para considerar un nivel válido
            
        Returns:
            Diccionario con niveles de soporte y resistencia
        """
        from scipy.signal import find_peaks
        
        # Encontrar máximos y mínimos locales
        highs_idx = find_peaks(df['high'].values, distance=window//2)[0]
        lows_idx = find_peaks(-df['low'].values, distance=window//2)[0]
        
        # Extraer precios de máximos y mínimos
        resistance_prices = df['high'].iloc[highs_idx].values
        support_prices = df['low'].iloc[lows_idx].values
        
        # Agrupar niveles similares
        resistance_levels = self._group_similar_levels(resistance_prices, min_touches)
        support_levels = self._group_similar_levels(support_prices, min_touches)
        
        return {
            'resistance': sorted(resistance_levels, reverse=True),
            'support': sorted(support_levels)
        }
    
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
        from scipy.signal import find_peaks
        
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
    
    def _group_similar_levels(self, prices: np.ndarray, min_touches: int, 
                            tolerance: float = 0.01) -> List[float]:
        """
        Agrupa niveles de precios similares
        
        Args:
            prices: Array de precios
            min_touches: Número mínimo de toques para considerar válido
            tolerance: Tolerancia para agrupar niveles (porcentaje)
            
        Returns:
            Lista de niveles válidos
        """
        if len(prices) == 0:
            return []
        
        levels = []
        used_indices = set()
        
        for i, price in enumerate(prices):
            if i in used_indices:
                continue
            
            # Encontrar precios similares
            similar_prices = []
            for j, other_price in enumerate(prices):
                if j not in used_indices and abs(price - other_price) / price <= tolerance:
                    similar_prices.append(other_price)
                    used_indices.add(j)
            
            # Si hay suficientes toques, agregar el nivel
            if len(similar_prices) >= min_touches:
                levels.append(np.mean(similar_prices))
        
        return levels