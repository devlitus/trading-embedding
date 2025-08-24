"""Detector de patrones geométricos (triángulos, canales, rectángulos)."""

import pandas as pd
import numpy as np
from typing import List, Optional
from scipy.stats import linregress
from scipy.signal import find_peaks

from ..types import PatternResult


class GeometricPatternDetector:
    """Detector especializado en patrones geométricos"""
    
    def __init__(self, min_pattern_length: int = 20):
        self.min_pattern_length = min_pattern_length
    
    def detect_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detecta todos los patrones geométricos"""
        patterns = []
        
        patterns.extend(self.detect_triangles(df))
        patterns.extend(self.detect_channels(df))
        patterns.extend(self.detect_rectangles(df))
        
        return patterns
    
    def detect_triangles(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detecta patrones triangulares (ascendente, descendente, simétrico)"""
        patterns = []
        
        # Buscar formaciones triangulares en ventanas deslizantes
        for i in range(len(df) - self.min_pattern_length):
            end_idx = i + self.min_pattern_length
            window = df.iloc[i:end_idx]
            
            if len(window) < self.min_pattern_length:
                continue
            
            # Encontrar máximos y mínimos locales
            highs_idx = find_peaks(window['high'].values, distance=5)[0]
            lows_idx = find_peaks(-window['low'].values, distance=5)[0]
            
            if len(highs_idx) >= 2 and len(lows_idx) >= 2:
                triangle_pattern = self._analyze_triangle_pattern(window, highs_idx, lows_idx, i)
                if triangle_pattern:
                    patterns.append(triangle_pattern)
        
        return patterns
    
    def _analyze_triangle_pattern(self, window: pd.DataFrame, highs_idx: np.ndarray, 
                                lows_idx: np.ndarray, start_offset: int) -> Optional[PatternResult]:
        """Analiza si los puntos forman un triángulo"""
        
        # Obtener precios de máximos y mínimos
        highs_prices = window['high'].iloc[highs_idx].values
        lows_prices = window['low'].iloc[lows_idx].values
        
        # Calcular líneas de tendencia
        if len(highs_idx) >= 2:
            high_slope, high_intercept, high_r, _, _ = linregress(highs_idx, highs_prices)
        else:
            return None
            
        if len(lows_idx) >= 2:
            low_slope, low_intercept, low_r, _, _ = linregress(lows_idx, lows_prices)
        else:
            return None
        
        # Determinar tipo de triángulo
        pattern_type = "triangle_symmetric"
        confidence = (abs(high_r) + abs(low_r)) / 2
        
        if abs(high_slope) < 0.1 and low_slope > 0.1:  # Línea superior horizontal, inferior ascendente
            pattern_type = "triangle_ascending"
            confidence *= 1.1  # Bonus por patrón más definido
        elif high_slope < -0.1 and abs(low_slope) < 0.1:  # Línea superior descendente, inferior horizontal
            pattern_type = "triangle_descending"
            confidence *= 1.1
        elif high_slope < -0.05 and low_slope > 0.05:  # Ambas líneas convergen
            pattern_type = "triangle_symmetric"
        else:
            return None  # No es un triángulo válido
        
        # Verificar convergencia
        convergence_point = self._calculate_convergence_point(high_slope, high_intercept, low_slope, low_intercept)
        if convergence_point is None or convergence_point < len(window):
            confidence *= 0.8  # Penalizar si no converge apropiadamente
        
        # Calcular parámetros adicionales
        parameters = {
            'high_slope': high_slope,
            'low_slope': low_slope,
            'high_r_value': high_r,
            'low_r_value': low_r,
            'convergence_point': convergence_point,
            'breakout_level': window['close'].iloc[-1]
        }
        
        description = f"Triángulo {pattern_type.split('_')[1]} con R² = {confidence:.2f}"
        
        return PatternResult(
            pattern_type=pattern_type,
            start_idx=start_offset,
            end_idx=start_offset + len(window) - 1,
            confidence=min(confidence, 1.0),
            parameters=parameters,
            description=description
        )
    
    def _calculate_convergence_point(self, slope1: float, intercept1: float, 
                                   slope2: float, intercept2: float) -> Optional[float]:
        """Calcula el punto de convergencia de dos líneas"""
        if abs(slope1 - slope2) < 1e-6:  # Líneas paralelas
            return None
        
        x_convergence = (intercept2 - intercept1) / (slope1 - slope2)
        return x_convergence
    
    def detect_channels(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detecta canales (paralelos ascendentes, descendentes, horizontales)"""
        patterns = []
        
        for i in range(len(df) - self.min_pattern_length):
            end_idx = i + self.min_pattern_length
            window = df.iloc[i:end_idx]
            
            # Encontrar máximos y mínimos
            highs_idx = find_peaks(window['high'].values, distance=5)[0]
            lows_idx = find_peaks(-window['low'].values, distance=5)[0]
            
            if len(highs_idx) >= 2 and len(lows_idx) >= 2:
                channel_pattern = self._analyze_channel_pattern(window, highs_idx, lows_idx, i)
                if channel_pattern:
                    patterns.append(channel_pattern)
        
        return patterns
    
    def _analyze_channel_pattern(self, window: pd.DataFrame, highs_idx: np.ndarray,
                               lows_idx: np.ndarray, start_offset: int) -> Optional[PatternResult]:
        """Analiza si los puntos forman un canal"""
        
        highs_prices = window['high'].iloc[highs_idx].values
        lows_prices = window['low'].iloc[lows_idx].values
        
        # Calcular líneas de tendencia
        high_slope, high_intercept, high_r, _, _ = linregress(highs_idx, highs_prices)
        low_slope, low_intercept, low_r, _, _ = linregress(lows_idx, lows_prices)
        
        # Verificar paralelismo (pendientes similares)
        slope_diff = abs(high_slope - low_slope)
        if slope_diff > 0.5:  # No es paralelo
            return None
        
        # Determinar tipo de canal
        avg_slope = (high_slope + low_slope) / 2
        
        if avg_slope > 0.1:
            pattern_type = "channel_ascending"
        elif avg_slope < -0.1:
            pattern_type = "channel_descending"
        else:
            pattern_type = "channel_horizontal"
        
        # Calcular confianza basada en R² y paralelismo
        confidence = (abs(high_r) + abs(low_r)) / 2
        parallelism_bonus = max(0, 1 - slope_diff)  # Bonus por paralelismo
        confidence *= parallelism_bonus
        
        # Calcular ancho del canal
        channel_width = np.mean(highs_prices) - np.mean(lows_prices)
        
        parameters = {
            'high_slope': high_slope,
            'low_slope': low_slope,
            'high_r_value': high_r,
            'low_r_value': low_r,
            'channel_width': channel_width,
            'slope_difference': slope_diff
        }
        
        description = f"Canal {pattern_type.split('_')[1]} con ancho {channel_width:.2f}"
        
        return PatternResult(
            pattern_type=pattern_type,
            start_idx=start_offset,
            end_idx=start_offset + len(window) - 1,
            confidence=min(confidence, 1.0),
            parameters=parameters,
            description=description
        )
    
    def detect_rectangles(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detecta patrones rectangulares (consolidación horizontal)"""
        patterns = []
        
        for i in range(len(df) - self.min_pattern_length):
            end_idx = i + self.min_pattern_length
            window = df.iloc[i:end_idx]
            
            rectangle_pattern = self._analyze_rectangle_pattern(window, i)
            if rectangle_pattern:
                patterns.append(rectangle_pattern)
        
        return patterns
    
    def _analyze_rectangle_pattern(self, window: pd.DataFrame, start_offset: int) -> Optional[PatternResult]:
        """Analiza si la ventana forma un rectángulo"""
        
        # Calcular niveles de soporte y resistencia
        highs = window['high'].values
        lows = window['low'].values
        
        # Encontrar niveles horizontales
        resistance_level = np.percentile(highs, 90)
        support_level = np.percentile(lows, 10)
        
        # Verificar que los precios respeten estos niveles
        resistance_touches = np.sum(highs >= resistance_level * 0.99)
        support_touches = np.sum(lows <= support_level * 1.01)
        
        # Debe haber al menos 2 toques en cada nivel
        if resistance_touches < 2 or support_touches < 2:
            return None
        
        # Calcular variabilidad de los niveles
        high_variability = np.std(highs[highs >= resistance_level * 0.99]) / resistance_level
        low_variability = np.std(lows[lows <= support_level * 1.01]) / support_level
        
        # Confianza basada en la consistencia de los niveles
        confidence = 1 - (high_variability + low_variability)
        confidence = max(0, min(1, confidence))
        
        # Bonus por más toques
        touch_bonus = min(0.2, (resistance_touches + support_touches - 4) * 0.05)
        confidence += touch_bonus
        
        rectangle_height = resistance_level - support_level
        
        parameters = {
            'resistance_level': resistance_level,
            'support_level': support_level,
            'rectangle_height': rectangle_height,
            'resistance_touches': resistance_touches,
            'support_touches': support_touches,
            'high_variability': high_variability,
            'low_variability': low_variability
        }
        
        description = f"Rectángulo con {resistance_touches + support_touches} toques"
        
        return PatternResult(
            pattern_type="rectangle",
            start_idx=start_offset,
            end_idx=start_offset + len(window) - 1,
            confidence=min(confidence, 1.0),
            parameters=parameters,
            description=description
        )