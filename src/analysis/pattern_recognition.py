"""Módulo de Reconocimiento de Patrones

Implementa algoritmos para detectar patrones técnicos básicos como triángulos,
canales, patrones de velas y formaciones que son precursores de patrones Wyckoff.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
from scipy.stats import linregress
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')


class PatternResult(NamedTuple):
    """Resultado de detección de patrón"""
    pattern_type: str
    start_idx: int
    end_idx: int
    confidence: float
    parameters: Dict
    description: str


class PatternRecognizer:
    """Clase principal para reconocimiento de patrones técnicos"""
    
    def __init__(self, min_pattern_length: int = 20, confidence_threshold: float = 0.6):
        """
        Args:
            min_pattern_length: Longitud mínima para considerar un patrón válido
            confidence_threshold: Umbral mínimo de confianza para reportar patrones
        """
        self.min_pattern_length = min_pattern_length
        self.confidence_threshold = confidence_threshold
    
    def detect_all_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """
        Detecta todos los patrones disponibles
        
        Args:
            df: DataFrame con datos OHLC
            
        Returns:
            Lista de patrones detectados
        """
        patterns = []
        
        # Patrones geométricos
        patterns.extend(self.detect_triangles(df))
        patterns.extend(self.detect_channels(df))
        patterns.extend(self.detect_rectangles(df))
        
        # Patrones de velas
        patterns.extend(self.detect_candlestick_patterns(df))
        
        # Patrones de volumen
        patterns.extend(self.detect_volume_patterns(df))
        
        # Filtrar por confianza
        patterns = [p for p in patterns if p.confidence >= self.confidence_threshold]
        
        # Ordenar por confianza
        patterns.sort(key=lambda x: x.confidence, reverse=True)
        
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
        confidence = ((abs(high_r) + abs(low_r)) / 2) * (1 - slope_diff)
        
        parameters = {
            'high_slope': high_slope,
            'low_slope': low_slope,
            'slope_difference': slope_diff,
            'channel_width': np.mean(highs_prices) - np.mean(lows_prices),
            'high_r_value': high_r,
            'low_r_value': low_r
        }
        
        description = f"Canal {pattern_type.split('_')[1]} con ancho {parameters['channel_width']:.2f}"
        
        return PatternResult(
            pattern_type=pattern_type,
            start_idx=start_offset,
            end_idx=start_offset + len(window) - 1,
            confidence=min(confidence, 1.0),
            parameters=parameters,
            description=description
        )
    
    def detect_rectangles(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detecta patrones rectangulares (consolidación)"""
        patterns = []
        
        for i in range(len(df) - self.min_pattern_length):
            end_idx = i + self.min_pattern_length
            window = df.iloc[i:end_idx]
            
            rectangle_pattern = self._analyze_rectangle_pattern(window, i)
            if rectangle_pattern:
                patterns.append(rectangle_pattern)
        
        return patterns
    
    def _analyze_rectangle_pattern(self, window: pd.DataFrame, start_offset: int) -> Optional[PatternResult]:
        """Analiza si la ventana forma un rectángulo (consolidación)"""
        
        # Calcular niveles de soporte y resistencia
        resistance_level = window['high'].quantile(0.95)
        support_level = window['low'].quantile(0.05)
        
        # Verificar que el precio se mantenga dentro del rango
        price_range = resistance_level - support_level
        avg_price = (resistance_level + support_level) / 2
        
        # Calcular qué porcentaje del tiempo el precio está cerca de los extremos
        near_resistance = (window['high'] >= resistance_level * 0.98).sum()
        near_support = (window['low'] <= support_level * 1.02).sum()
        
        # Verificar volatilidad baja (característica de consolidación)
        volatility = window['close'].std() / avg_price
        
        # Calcular confianza
        range_consistency = 1 - (price_range / avg_price)  # Menor rango = mayor confianza
        boundary_touches = (near_resistance + near_support) / len(window)
        low_volatility_score = max(0, 1 - volatility * 10)
        
        confidence = (range_consistency + boundary_touches + low_volatility_score) / 3
        
        if confidence < 0.5:
            return None
        
        parameters = {
            'resistance_level': resistance_level,
            'support_level': support_level,
            'price_range': price_range,
            'volatility': volatility,
            'boundary_touches': boundary_touches
        }
        
        description = f"Rectángulo entre {support_level:.2f} y {resistance_level:.2f}"
        
        return PatternResult(
            pattern_type="rectangle",
            start_idx=start_offset,
            end_idx=start_offset + len(window) - 1,
            confidence=min(confidence, 1.0),
            parameters=parameters,
            description=description
        )
    
    def detect_candlestick_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detecta patrones de velas japonesas"""
        patterns = []
        
        for i in range(2, len(df)):
            # Patrones de una vela
            single_patterns = self._detect_single_candlestick_patterns(df, i)
            patterns.extend(single_patterns)
            
            # Patrones de múltiples velas
            if i >= 3:
                multi_patterns = self._detect_multi_candlestick_patterns(df, i)
                patterns.extend(multi_patterns)
        
        return patterns
    
    def _detect_single_candlestick_patterns(self, df: pd.DataFrame, idx: int) -> List[PatternResult]:
        """Detecta patrones de una sola vela"""
        patterns = []
        
        current = df.iloc[idx]
        prev = df.iloc[idx-1]
        
        # Calcular características de la vela
        body_size = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        upper_shadow = current['high'] - max(current['open'], current['close'])
        lower_shadow = min(current['open'], current['close']) - current['low']
        
        if total_range == 0:
            return patterns
        
        body_ratio = body_size / total_range
        upper_shadow_ratio = upper_shadow / total_range
        lower_shadow_ratio = lower_shadow / total_range
        
        # Doji
        if body_ratio < 0.1:
            confidence = 1 - body_ratio * 10
            patterns.append(PatternResult(
                pattern_type="doji",
                start_idx=idx,
                end_idx=idx,
                confidence=confidence,
                parameters={'body_ratio': body_ratio},
                description="Doji - Indecisión del mercado"
            ))
        
        # Hammer/Hanging Man
        if lower_shadow_ratio > 0.6 and upper_shadow_ratio < 0.1 and body_ratio < 0.3:
            is_bullish = current['close'] > prev['close']
            pattern_type = "hammer" if is_bullish else "hanging_man"
            confidence = lower_shadow_ratio * (1 - upper_shadow_ratio)
            
            patterns.append(PatternResult(
                pattern_type=pattern_type,
                start_idx=idx,
                end_idx=idx,
                confidence=confidence,
                parameters={
                    'lower_shadow_ratio': lower_shadow_ratio,
                    'body_ratio': body_ratio
                },
                description=f"{pattern_type.title()} - Posible reversión"
            ))
        
        # Shooting Star/Inverted Hammer
        if upper_shadow_ratio > 0.6 and lower_shadow_ratio < 0.1 and body_ratio < 0.3:
            is_bearish = current['close'] < prev['close']
            pattern_type = "shooting_star" if is_bearish else "inverted_hammer"
            confidence = upper_shadow_ratio * (1 - lower_shadow_ratio)
            
            patterns.append(PatternResult(
                pattern_type=pattern_type,
                start_idx=idx,
                end_idx=idx,
                confidence=confidence,
                parameters={
                    'upper_shadow_ratio': upper_shadow_ratio,
                    'body_ratio': body_ratio
                },
                description=f"{pattern_type.replace('_', ' ').title()} - Posible reversión"
            ))
        
        return patterns
    
    def _detect_multi_candlestick_patterns(self, df: pd.DataFrame, idx: int) -> List[PatternResult]:
        """Detecta patrones de múltiples velas"""
        patterns = []
        
        # Engulfing Pattern
        if idx >= 1:
            current = df.iloc[idx]
            prev = df.iloc[idx-1]
            
            # Bullish Engulfing
            if (prev['close'] < prev['open'] and  # Vela anterior bajista
                current['close'] > current['open'] and  # Vela actual alcista
                current['open'] < prev['close'] and  # Abre por debajo del cierre anterior
                current['close'] > prev['open']):  # Cierra por encima de la apertura anterior
                
                confidence = min(1.0, (current['close'] - current['open']) / (prev['open'] - prev['close']))
                
                patterns.append(PatternResult(
                    pattern_type="bullish_engulfing",
                    start_idx=idx-1,
                    end_idx=idx,
                    confidence=confidence,
                    parameters={},
                    description="Envolvente Alcista - Reversión alcista"
                ))
            
            # Bearish Engulfing
            elif (prev['close'] > prev['open'] and  # Vela anterior alcista
                  current['close'] < current['open'] and  # Vela actual bajista
                  current['open'] > prev['close'] and  # Abre por encima del cierre anterior
                  current['close'] < prev['open']):  # Cierra por debajo de la apertura anterior
                
                confidence = min(1.0, (current['open'] - current['close']) / (prev['close'] - prev['open']))
                
                patterns.append(PatternResult(
                    pattern_type="bearish_engulfing",
                    start_idx=idx-1,
                    end_idx=idx,
                    confidence=confidence,
                    parameters={},
                    description="Envolvente Bajista - Reversión bajista"
                ))
        
        return patterns
    
    def detect_volume_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detecta patrones relacionados con el volumen"""
        patterns = []
        
        if 'volume' not in df.columns:
            return patterns
        
        # Calcular volumen promedio
        df_temp = df.copy()
        df_temp['volume_ma'] = df_temp['volume'].rolling(window=20).mean()
        
        for i in range(20, len(df_temp)):
            current_volume = df_temp['volume'].iloc[i]
            avg_volume = df_temp['volume_ma'].iloc[i]
            
            # Volume Spike
            if current_volume > avg_volume * 2:
                price_change = abs(df_temp['close'].iloc[i] - df_temp['close'].iloc[i-1]) / df_temp['close'].iloc[i-1]
                confidence = min(1.0, (current_volume / avg_volume - 1) * price_change * 10)
                
                patterns.append(PatternResult(
                    pattern_type="volume_spike",
                    start_idx=i,
                    end_idx=i,
                    confidence=confidence,
                    parameters={
                        'volume_ratio': current_volume / avg_volume,
                        'price_change': price_change
                    },
                    description=f"Pico de volumen {current_volume/avg_volume:.1f}x promedio"
                ))
            
            # Volume Dry Up (volumen bajo)
            elif current_volume < avg_volume * 0.5:
                confidence = 1 - (current_volume / avg_volume)
                
                patterns.append(PatternResult(
                    pattern_type="volume_dry_up",
                    start_idx=i,
                    end_idx=i,
                    confidence=confidence,
                    parameters={
                        'volume_ratio': current_volume / avg_volume
                    },
                    description=f"Volumen bajo {current_volume/avg_volume:.1f}x promedio"
                ))
        
        return patterns
    
    def get_pattern_summary(self, patterns: List[PatternResult]) -> Dict[str, int]:
        """Obtiene un resumen de los patrones detectados"""
        summary = {}
        for pattern in patterns:
            pattern_type = pattern.pattern_type
            summary[pattern_type] = summary.get(pattern_type, 0) + 1
        return summary
    
    def filter_overlapping_patterns(self, patterns: List[PatternResult]) -> List[PatternResult]:
        """Filtra patrones que se solapan, manteniendo el de mayor confianza"""
        if not patterns:
            return patterns
        
        # Ordenar por confianza descendente
        sorted_patterns = sorted(patterns, key=lambda x: x.confidence, reverse=True)
        filtered_patterns = []
        
        for pattern in sorted_patterns:
            # Verificar si se solapa con algún patrón ya seleccionado
            overlaps = False
            for selected in filtered_patterns:
                if self._patterns_overlap(pattern, selected):
                    overlaps = True
                    break
            
            if not overlaps:
                filtered_patterns.append(pattern)
        
        return filtered_patterns
    
    def _patterns_overlap(self, pattern1: PatternResult, pattern2: PatternResult) -> bool:
        """Verifica si dos patrones se solapan"""
        return not (pattern1.end_idx < pattern2.start_idx or pattern2.end_idx < pattern1.start_idx)


def detect_patterns(df: pd.DataFrame, pattern_types: Optional[List[str]] = None) -> List[PatternResult]:
    """
    Función de conveniencia para detectar patrones
    
    Args:
        df: DataFrame con datos OHLC
        pattern_types: Lista de tipos de patrones a detectar (None = todos)
        
    Returns:
        Lista de patrones detectados
    """
    recognizer = PatternRecognizer()
    all_patterns = recognizer.detect_all_patterns(df)
    
    if pattern_types:
        all_patterns = [p for p in all_patterns if p.pattern_type in pattern_types]
    
    return recognizer.filter_overlapping_patterns(all_patterns)


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Crear datos de ejemplo
    dates = pd.date_range('2023-01-01', periods=100, freq='1H')
    np.random.seed(42)
    
    # Simular datos OHLC con patrones
    base_price = 100
    prices = []
    volumes = []
    
    for i in range(100):
        # Simular movimiento de precio con algunos patrones
        if i < 30:
            # Tendencia alcista
            price = base_price + i * 0.5 + np.random.randn() * 2
        elif i < 60:
            # Consolidación (rectángulo)
            price = base_price + 15 + np.random.randn() * 1
        else:
            # Triángulo
            price = base_price + 15 - (i - 60) * 0.2 + np.random.randn() * 1
        
        prices.append(price)
        volumes.append(np.random.randint(1000, 5000))
    
    # Crear OHLC
    close_prices = np.array(prices)
    high_prices = close_prices + np.random.rand(100) * 2
    low_prices = close_prices - np.random.rand(100) * 2
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    # Detectar patrones
    recognizer = PatternRecognizer()
    patterns = recognizer.detect_all_patterns(sample_df)
    
    print(f"Patrones detectados: {len(patterns)}")
    for pattern in patterns[:5]:  # Mostrar los primeros 5
        print(f"- {pattern.pattern_type}: {pattern.description} (Confianza: {pattern.confidence:.2f})")
    
    # Resumen
    summary = recognizer.get_pattern_summary(patterns)
    print(f"\nResumen de patrones: {summary}")