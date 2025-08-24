"""Detector de tendencias basado en medias móviles.

Implementa algoritmos de detección de tendencias usando diferentes
tipos de medias móviles y cruces entre ellas.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class MovingAverageDetector:
    """Detector de tendencias basado en medias móviles"""
    
    def __init__(self, short_window: int = 10, long_window: int = 30):
        """
        Args:
            short_window: Período para la media móvil corta
            long_window: Período para la media móvil larga
        """
        self.short_window = short_window
        self.long_window = long_window
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta cambios de tendencia usando medias móviles
        
        Args:
            df: DataFrame con datos OHLC
            
        Returns:
            DataFrame con columnas de tendencia añadidas
        """
        result_df = df.copy()
        
        # Calcular medias móviles
        result_df[f'ma_{self.short_window}'] = result_df['close'].rolling(
            window=self.short_window
        ).mean()
        result_df[f'ma_{self.long_window}'] = result_df['close'].rolling(
            window=self.long_window
        ).mean()
        
        # Detectar cruces
        short_ma = result_df[f'ma_{self.short_window}']
        long_ma = result_df[f'ma_{self.long_window}']
        
        # Señales de cruce
        result_df['ma_cross_up'] = (
            (short_ma > long_ma) & 
            (short_ma.shift(1) <= long_ma.shift(1))
        )
        result_df['ma_cross_down'] = (
            (short_ma < long_ma) & 
            (short_ma.shift(1) >= long_ma.shift(1))
        )
        
        # Tendencia actual
        result_df['ma_trend'] = np.where(
            short_ma > long_ma, 'uptrend',
            np.where(short_ma < long_ma, 'downtrend', 'sideways')
        )
        
        # Cambios de tendencia
        result_df['ma_trend_change'] = (
            result_df['ma_cross_up'] | result_df['ma_cross_down']
        )
        
        return result_df
    
    def calculate_ma_slope(self, df: pd.DataFrame, window: int = 5) -> pd.Series:
        """
        Calcula la pendiente de la media móvil
        
        Args:
            df: DataFrame con datos que incluyen medias móviles
            window: Ventana para calcular la pendiente
            
        Returns:
            Series con las pendientes
        """
        ma_col = f'ma_{self.short_window}'
        if ma_col not in df.columns:
            df = self.detect(df)
        
        slopes = []
        for i in range(len(df)):
            if i < window:
                slopes.append(0)
            else:
                y_values = df[ma_col].iloc[i-window+1:i+1].values
                x_values = np.arange(len(y_values))
                
                # Calcular pendiente usando regresión lineal simple
                if len(y_values) > 1 and not np.isnan(y_values).any():
                    slope = np.polyfit(x_values, y_values, 1)[0]
                    slopes.append(slope)
                else:
                    slopes.append(0)
        
        return pd.Series(slopes, index=df.index)
    
    def detect_ma_divergence(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta divergencias entre precio y media móvil
        
        Returns:
            Series con señales de divergencia
        """
        if f'ma_{self.short_window}' not in df.columns:
            df = self.detect(df)
        
        price_slope = df['close'].rolling(window=10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0
        )
        
        ma_slope = self.calculate_ma_slope(df, window=10)
        
        # Divergencia alcista: precio baja pero MA sube
        bullish_div = (price_slope < 0) & (ma_slope > 0)
        
        # Divergencia bajista: precio sube pero MA baja
        bearish_div = (price_slope > 0) & (ma_slope < 0)
        
        divergence = pd.Series('none', index=df.index)
        divergence[bullish_div] = 'bullish'
        divergence[bearish_div] = 'bearish'
        
        return divergence
    
    def calculate_ma_envelope(self, df: pd.DataFrame, 
                            envelope_pct: float = 0.025) -> Dict[str, pd.Series]:
        """
        Calcula envolventes de media móvil
        
        Args:
            df: DataFrame con datos OHLC
            envelope_pct: Porcentaje para las envolventes
            
        Returns:
            Diccionario con bandas superior e inferior
        """
        if f'ma_{self.short_window}' not in df.columns:
            df = self.detect(df)
        
        ma = df[f'ma_{self.short_window}']
        
        upper_band = ma * (1 + envelope_pct)
        lower_band = ma * (1 - envelope_pct)
        
        return {
            'upper_envelope': upper_band,
            'lower_envelope': lower_band,
            'ma_center': ma
        }
    
    def detect_ma_support_resistance(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Detecta cuando las medias móviles actúan como soporte/resistencia
        
        Returns:
            Diccionario con señales de soporte y resistencia
        """
        if f'ma_{self.short_window}' not in df.columns:
            df = self.detect(df)
        
        short_ma = df[f'ma_{self.short_window}']
        long_ma = df[f'ma_{self.long_window}']
        
        # MA como soporte (precio rebota desde abajo)
        ma_support = (
            (df['low'] <= short_ma * 1.005) &  # Precio toca MA
            (df['close'] > short_ma) &          # Pero cierra arriba
            (df['close'].shift(1) < short_ma.shift(1))  # Venía de abajo
        )
        
        # MA como resistencia (precio rebota desde arriba)
        ma_resistance = (
            (df['high'] >= short_ma * 0.995) &  # Precio toca MA
            (df['close'] < short_ma) &          # Pero cierra abajo
            (df['close'].shift(1) > short_ma.shift(1))  # Venía de arriba
        )
        
        return {
            'ma_support': ma_support,
            'ma_resistance': ma_resistance
        }
    
    def calculate_ma_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula la fuerza de la tendencia basada en MAs
        
        Returns:
            Series con scores de fuerza (0-100)
        """
        if f'ma_{self.short_window}' not in df.columns:
            df = self.detect(df)
        
        short_ma = df[f'ma_{self.short_window}']
        long_ma = df[f'ma_{self.long_window}']
        
        # Distancia entre MAs (normalizada)
        ma_distance = abs(short_ma - long_ma) / long_ma
        
        # Pendiente de la MA larga
        long_ma_slope = self.calculate_ma_slope(
            pd.DataFrame({'ma_30': long_ma}), window=5
        )
        
        # Consistencia de la tendencia
        trend_consistency = df['ma_trend'].rolling(window=10).apply(
            lambda x: (x == x.iloc[-1]).sum() / len(x) if len(x) > 0 else 0
        )
        
        # Combinar factores
        strength = (
            (ma_distance * 100 * 30) +  # 30% peso a distancia
            (abs(long_ma_slope) * 1000 * 40) +  # 40% peso a pendiente
            (trend_consistency * 30)  # 30% peso a consistencia
        )
        
        return np.clip(strength, 0, 100)