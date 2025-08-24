"""Detector de tendencias basado en algoritmo ZigZag.

Implementa el algoritmo ZigZag para identificar puntos de inflexión
y cambios de tendencia significativos en los precios.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import interpolate


class ZigZagDetector:
    """Detector de tendencias basado en algoritmo ZigZag"""
    
    def __init__(self, threshold: float = 0.05):
        """
        Args:
            threshold: Umbral mínimo de cambio para considerar un punto ZigZag
        """
        self.threshold = threshold
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta cambios de tendencia usando algoritmo ZigZag
        
        Args:
            df: DataFrame con datos OHLC
            
        Returns:
            DataFrame con columnas de tendencia añadidas
        """
        result_df = df.copy()
        
        # Calcular puntos ZigZag
        zigzag_points = self._calculate_zigzag_points(df['close'].values)
        
        # Interpolar tendencia ZigZag
        zigzag_trend = self._interpolate_zigzag_trend(zigzag_points, len(df))
        
        # Añadir columnas al DataFrame
        result_df['zigzag_points'] = zigzag_points
        result_df['zigzag_trend'] = zigzag_trend
        
        # Detectar cambios de tendencia
        result_df['zigzag_trend_change'] = (
            result_df['zigzag_trend'] != result_df['zigzag_trend'].shift(1)
        )
        
        # Clasificar tendencia
        result_df['zigzag_direction'] = np.where(
            result_df['zigzag_trend'] > 0, 'uptrend',
            np.where(result_df['zigzag_trend'] < 0, 'downtrend', 'sideways')
        )
        
        return result_df
    
    def _calculate_zigzag_points(self, prices: np.ndarray) -> np.ndarray:
        """
        Calcula los puntos ZigZag basados en el umbral
        
        Args:
            prices: Array de precios
            
        Returns:
            Array con puntos ZigZag (NaN donde no hay punto)
        """
        zigzag = np.full(len(prices), np.nan)
        
        if len(prices) < 3:
            return zigzag
        
        # Inicializar con el primer precio
        last_pivot_idx = 0
        last_pivot_price = prices[0]
        zigzag[0] = prices[0]
        
        # Determinar dirección inicial
        direction = 1 if prices[1] > prices[0] else -1
        
        for i in range(1, len(prices)):
            current_price = prices[i]
            
            # Calcular cambio porcentual desde el último pivot
            pct_change = (current_price - last_pivot_price) / last_pivot_price
            
            if direction == 1:  # Buscando máximo
                if pct_change >= self.threshold:
                    # Nuevo máximo, actualizar último pivot
                    last_pivot_idx = i
                    last_pivot_price = current_price
                    zigzag[i] = current_price
                elif pct_change <= -self.threshold:
                    # Cambio de dirección a bajista
                    direction = -1
                    last_pivot_idx = i
                    last_pivot_price = current_price
                    zigzag[i] = current_price
            else:  # direction == -1, buscando mínimo
                if pct_change <= -self.threshold:
                    # Nuevo mínimo, actualizar último pivot
                    last_pivot_idx = i
                    last_pivot_price = current_price
                    zigzag[i] = current_price
                elif pct_change >= self.threshold:
                    # Cambio de dirección a alcista
                    direction = 1
                    last_pivot_idx = i
                    last_pivot_price = current_price
                    zigzag[i] = current_price
        
        return zigzag
    
    def _interpolate_zigzag_trend(self, zigzag_points: np.ndarray, 
                                 length: int) -> np.ndarray:
        """
        Interpola la tendencia entre puntos ZigZag
        
        Args:
            zigzag_points: Array con puntos ZigZag
            length: Longitud del array resultado
            
        Returns:
            Array con tendencia interpolada
        """
        # Encontrar índices de puntos válidos
        valid_indices = np.where(~np.isnan(zigzag_points))[0]
        
        if len(valid_indices) < 2:
            return np.zeros(length)
        
        # Valores de los puntos válidos
        valid_values = zigzag_points[valid_indices]
        
        # Interpolar linealmente
        f = interpolate.interp1d(
            valid_indices, valid_values, 
            kind='linear', 
            bounds_error=False, 
            fill_value='extrapolate'
        )
        
        interpolated = f(np.arange(length))
        
        # Calcular pendiente (tendencia)
        trend = np.gradient(interpolated)
        
        return trend
    
    def identify_swing_points(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Identifica puntos de swing (máximos y mínimos locales)
        
        Returns:
            Diccionario con swing highs y swing lows
        """
        if 'zigzag_points' not in df.columns:
            df = self.detect(df)
        
        zigzag_points = df['zigzag_points']
        prices = df['close']
        
        swing_highs = pd.Series(False, index=df.index)
        swing_lows = pd.Series(False, index=df.index)
        
        # Encontrar puntos ZigZag válidos
        valid_points = ~zigzag_points.isna()
        valid_indices = df.index[valid_points]
        
        for i in range(1, len(valid_indices) - 1):
            current_idx = valid_indices[i]
            prev_idx = valid_indices[i-1]
            next_idx = valid_indices[i+1]
            
            current_price = prices[current_idx]
            prev_price = prices[prev_idx]
            next_price = prices[next_idx]
            
            # Swing high: precio actual > anterior y siguiente
            if current_price > prev_price and current_price > next_price:
                swing_highs[current_idx] = True
            
            # Swing low: precio actual < anterior y siguiente
            elif current_price < prev_price and current_price < next_price:
                swing_lows[current_idx] = True
        
        return {
            'swing_highs': swing_highs,
            'swing_lows': swing_lows
        }
    
    def calculate_zigzag_retracement(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula niveles de retroceso entre puntos ZigZag
        
        Returns:
            Series con porcentajes de retroceso
        """
        if 'zigzag_points' not in df.columns:
            df = self.detect(df)
        
        zigzag_points = df['zigzag_points']
        prices = df['close']
        retracements = pd.Series(0.0, index=df.index)
        
        # Encontrar puntos ZigZag válidos
        valid_indices = df.index[~zigzag_points.isna()]
        
        if len(valid_indices) < 2:
            return retracements
        
        for i in range(1, len(valid_indices)):
            start_idx = valid_indices[i-1]
            end_idx = valid_indices[i]
            
            start_price = prices[start_idx]
            end_price = prices[end_idx]
            
            # Calcular retroceso para el segmento
            segment_indices = df.index[
                (df.index >= start_idx) & (df.index <= end_idx)
            ]
            
            for idx in segment_indices:
                current_price = prices[idx]
                
                if start_price != end_price:
                    # Porcentaje de retroceso desde el inicio del movimiento
                    retracement = (current_price - start_price) / (end_price - start_price)
                    retracements[idx] = retracement
        
        return retracements
    
    def detect_zigzag_patterns(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Detecta patrones comunes en la estructura ZigZag
        
        Returns:
            Diccionario con diferentes patrones detectados
        """
        swing_points = self.identify_swing_points(df)
        swing_highs = swing_points['swing_highs']
        swing_lows = swing_points['swing_lows']
        
        # Obtener secuencias de máximos y mínimos
        high_indices = df.index[swing_highs]
        low_indices = df.index[swing_lows]
        high_prices = df['close'][high_indices]
        low_prices = df['close'][low_indices]
        
        # Inicializar series de patrones
        higher_highs = pd.Series(False, index=df.index)
        lower_highs = pd.Series(False, index=df.index)
        higher_lows = pd.Series(False, index=df.index)
        lower_lows = pd.Series(False, index=df.index)
        
        # Analizar máximos consecutivos
        for i in range(1, len(high_indices)):
            current_idx = high_indices[i]
            prev_price = high_prices.iloc[i-1]
            current_price = high_prices.iloc[i]
            
            if current_price > prev_price:
                higher_highs[current_idx] = True
            else:
                lower_highs[current_idx] = True
        
        # Analizar mínimos consecutivos
        for i in range(1, len(low_indices)):
            current_idx = low_indices[i]
            prev_price = low_prices.iloc[i-1]
            current_price = low_prices.iloc[i]
            
            if current_price > prev_price:
                higher_lows[current_idx] = True
            else:
                lower_lows[current_idx] = True
        
        return {
            'higher_highs': higher_highs,
            'lower_highs': lower_highs,
            'higher_lows': higher_lows,
            'lower_lows': lower_lows
        }
    
    def calculate_zigzag_momentum(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula el momentum basado en la velocidad de cambio ZigZag
        
        Returns:
            Series con valores de momentum
        """
        if 'zigzag_points' not in df.columns:
            df = self.detect(df)
        
        zigzag_points = df['zigzag_points']
        momentum = pd.Series(0.0, index=df.index)
        
        # Encontrar puntos ZigZag válidos
        valid_indices = df.index[~zigzag_points.isna()]
        
        if len(valid_indices) < 2:
            return momentum
        
        for i in range(1, len(valid_indices)):
            start_idx = valid_indices[i-1]
            end_idx = valid_indices[i]
            
            start_price = df['close'][start_idx]
            end_price = df['close'][end_idx]
            
            # Calcular tiempo transcurrido (en períodos)
            time_diff = df.index.get_loc(end_idx) - df.index.get_loc(start_idx)
            
            if time_diff > 0:
                # Momentum = cambio de precio / tiempo
                price_change = (end_price - start_price) / start_price
                segment_momentum = price_change / time_diff
                
                # Aplicar momentum al segmento
                segment_indices = df.index[
                    (df.index >= start_idx) & (df.index <= end_idx)
                ]
                momentum[segment_indices] = segment_momentum
        
        return momentum