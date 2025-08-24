"""Detector de tendencias basado en regresión lineal.

Implementa algoritmos de detección de tendencias usando regresión lineal
y análisis estadístico de las tendencias de precios.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
from src.utils.analysis_utils import calculate_trend_strength


class RegressionDetector:
    """Detector de tendencias basado en regresión lineal"""
    
    def __init__(self, window: int = 20, min_r_squared: float = 0.7):
        """
        Args:
            window: Ventana para calcular la regresión
            min_r_squared: R² mínimo para considerar una tendencia válida
        """
        self.window = window
        self.min_r_squared = min_r_squared
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detecta cambios de tendencia usando regresión lineal
        
        Args:
            df: DataFrame con datos OHLC
            
        Returns:
            DataFrame con columnas de tendencia añadidas
        """
        result_df = df.copy()
        
        # Calcular regresión móvil
        regression_data = self._calculate_rolling_regression(df['close'])
        
        # Añadir columnas al DataFrame
        result_df['regression_slope'] = regression_data['slope']
        result_df['regression_r_squared'] = regression_data['r_squared']
        result_df['regression_trend_line'] = regression_data['trend_line']
        result_df['regression_upper_band'] = regression_data['upper_band']
        result_df['regression_lower_band'] = regression_data['lower_band']
        
        # Clasificar tendencia basada en pendiente y R²
        result_df['regression_trend'] = self._classify_trend(
            regression_data['slope'], 
            regression_data['r_squared']
        )
        
        # Detectar cambios de tendencia
        result_df['regression_trend_change'] = (
            result_df['regression_trend'] != result_df['regression_trend'].shift(1)
        )
        
        # Detectar breakouts de las bandas de regresión
        result_df['regression_breakout_up'] = (
            df['close'] > result_df['regression_upper_band']
        )
        result_df['regression_breakout_down'] = (
            df['close'] < result_df['regression_lower_band']
        )
        
        return result_df
    
    def _calculate_rolling_regression(self, prices: pd.Series) -> Dict[str, pd.Series]:
        """
        Calcula regresión lineal móvil
        
        Args:
            prices: Serie de precios
            
        Returns:
            Diccionario con datos de regresión
        """
        slopes = []
        r_squared_values = []
        trend_lines = []
        upper_bands = []
        lower_bands = []
        
        for i in range(len(prices)):
            if i < self.window - 1:
                slopes.append(0)
                r_squared_values.append(0)
                trend_lines.append(prices.iloc[i] if i < len(prices) else np.nan)
                upper_bands.append(prices.iloc[i] if i < len(prices) else np.nan)
                lower_bands.append(prices.iloc[i] if i < len(prices) else np.nan)
            else:
                # Obtener ventana de datos
                window_prices = prices.iloc[i-self.window+1:i+1]
                x_values = np.arange(len(window_prices))
                y_values = window_prices.values
                
                try:
                    # Calcular regresión lineal usando scipy.stats
                    slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
                    
                    # Predicciones
                    y_pred = slope * x_values + intercept
                    
                    # Métricas
                    r_squared = r_value ** 2
                    
                    # Línea de tendencia (último punto)
                    trend_line = y_pred[-1]
                    
                    # Bandas basadas en desviación estándar de residuos
                    residuals = y_values - y_pred
                    std_residuals = np.std(residuals)
                    
                    upper_band = trend_line + 2 * std_residuals
                    lower_band = trend_line - 2 * std_residuals
                    
                    slopes.append(slope)
                    r_squared_values.append(r_squared)
                    trend_lines.append(trend_line)
                    upper_bands.append(upper_band)
                    lower_bands.append(lower_band)
                except:
                    slopes.append(0)
                    r_squared_values.append(0)
                    trend_lines.append(prices.iloc[i])
                    upper_bands.append(prices.iloc[i])
                    lower_bands.append(prices.iloc[i])
        
        return {
            'slope': pd.Series(slopes, index=prices.index),
            'r_squared': pd.Series(r_squared_values, index=prices.index),
            'trend_line': pd.Series(trend_lines, index=prices.index),
            'upper_band': pd.Series(upper_bands, index=prices.index),
            'lower_band': pd.Series(lower_bands, index=prices.index)
        }
    
    def _classify_trend(self, slopes: pd.Series, r_squared: pd.Series) -> pd.Series:
        """
        Clasifica la tendencia basada en pendiente y R²
        
        Args:
            slopes: Serie de pendientes
            r_squared: Serie de valores R²
            
        Returns:
            Serie con clasificación de tendencia
        """
        trend = pd.Series('sideways', index=slopes.index)
        
        # Tendencia alcista: pendiente positiva y R² alto
        uptrend_mask = (slopes > 0) & (r_squared >= self.min_r_squared)
        trend[uptrend_mask] = 'uptrend'
        
        # Tendencia bajista: pendiente negativa y R² alto
        downtrend_mask = (slopes < 0) & (r_squared >= self.min_r_squared)
        trend[downtrend_mask] = 'downtrend'
        
        return trend
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula la fuerza de la tendencia basada en regresión
        
        Returns:
            Serie con scores de fuerza (0-100)
        """
        # Usar utilidad consolidada con método de regresión
        return calculate_trend_strength(df, method='regression')
    
    def detect_regression_divergence(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta divergencias entre precio y línea de regresión
        
        Returns:
            Serie con señales de divergencia
        """
        if 'regression_trend_line' not in df.columns:
            df = self.detect(df)
        
        price = df['close']
        trend_line = df['regression_trend_line']
        
        # Calcular pendientes
        price_slope = price.rolling(window=10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0
        )
        
        trend_slope = trend_line.rolling(window=10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0
        )
        
        # Detectar divergencias
        divergence = pd.Series('none', index=df.index)
        
        # Divergencia alcista: precio baja, tendencia sube
        bullish_div = (price_slope < 0) & (trend_slope > 0)
        divergence[bullish_div] = 'bullish'
        
        # Divergencia bajista: precio sube, tendencia baja
        bearish_div = (price_slope > 0) & (trend_slope < 0)
        divergence[bearish_div] = 'bearish'
        
        return divergence
    
    def calculate_polynomial_trend(self, df: pd.DataFrame, degree: int = 2) -> Dict[str, pd.Series]:
        """
        Calcula tendencia usando regresión polinomial
        
        Args:
            df: DataFrame con datos OHLC
            degree: Grado del polinomio
            
        Returns:
            Diccionario con datos de regresión polinomial
        """
        prices = df['close']
        
        poly_trends = []
        poly_r_squared = []
        
        for i in range(len(prices)):
            if i < self.window - 1:
                poly_trends.append(prices.iloc[i] if i < len(prices) else np.nan)
                poly_r_squared.append(0)
            else:
                # Obtener ventana de datos
                window_prices = prices.iloc[i-self.window+1:i+1]
                x_values = np.arange(len(window_prices))
                y_values = window_prices.values
                
                try:
                    # Ajustar polinomio usando numpy
                    poly_coeffs = np.polyfit(x_values, y_values, degree)
                    poly_func = np.poly1d(poly_coeffs)
                    
                    # Predicción para el último punto
                    poly_trend = poly_func(len(window_prices) - 1)
                    
                    # Calcular R² manualmente
                    y_pred = poly_func(x_values)
                    ss_res = np.sum((y_values - y_pred) ** 2)
                    ss_tot = np.sum((y_values - np.mean(y_values)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                except:
                    poly_trend = prices.iloc[i]
                    r_squared = 0
                
                poly_trends.append(poly_trend)
                poly_r_squared.append(r_squared)
        
        return {
            'polynomial_trend': pd.Series(poly_trends, index=prices.index),
            'polynomial_r_squared': pd.Series(poly_r_squared, index=prices.index)
        }
    
    def detect_trend_acceleration(self, df: pd.DataFrame) -> pd.Series:
        """
        Detecta aceleración/desaceleración de tendencias
        
        Returns:
            Serie con valores de aceleración
        """
        if 'regression_slope' not in df.columns:
            df = self.detect(df)
        
        # Calcular segunda derivada de la pendiente
        slope_change = df['regression_slope'].diff()
        acceleration = slope_change.diff()
        
        # Normalizar
        acceleration_normalized = acceleration / df['close'] * 10000
        
        return acceleration_normalized
    
    def calculate_regression_channels(self, df: pd.DataFrame, 
                                   std_multiplier: float = 2.0) -> Dict[str, pd.Series]:
        """
        Calcula canales de regresión
        
        Args:
            df: DataFrame con datos OHLC
            std_multiplier: Multiplicador para las bandas
            
        Returns:
            Diccionario con canales de regresión
        """
        if 'regression_trend_line' not in df.columns:
            df = self.detect(df)
        
        prices = df['close']
        trend_line = df['regression_trend_line']
        
        # Calcular desviación estándar móvil de residuos
        residuals = prices - trend_line
        rolling_std = residuals.rolling(window=self.window).std()
        
        # Canales
        upper_channel = trend_line + (std_multiplier * rolling_std)
        lower_channel = trend_line - (std_multiplier * rolling_std)
        middle_channel = trend_line
        
        return {
            'upper_channel': upper_channel,
            'middle_channel': middle_channel,
            'lower_channel': lower_channel,
            'channel_width': upper_channel - lower_channel
        }
    
    def detect_channel_breakouts(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Detecta rupturas de canales de regresión
        
        Returns:
            Diccionario con señales de ruptura
        """
        channels = self.calculate_regression_channels(df)
        
        price = df['close']
        upper_channel = channels['upper_channel']
        lower_channel = channels['lower_channel']
        
        # Rupturas
        breakout_up = price > upper_channel
        breakout_down = price < lower_channel
        
        # Confirmación de ruptura (cierre fuera del canal)
        confirmed_breakout_up = breakout_up & (price.shift(1) <= upper_channel.shift(1))
        confirmed_breakout_down = breakout_down & (price.shift(1) >= lower_channel.shift(1))
        
        return {
            'channel_breakout_up': breakout_up,
            'channel_breakout_down': breakout_down,
            'confirmed_breakout_up': confirmed_breakout_up,
            'confirmed_breakout_down': confirmed_breakout_down
        }