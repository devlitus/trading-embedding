"""Módulo de Indicadores Técnicos

Implementa los indicadores técnicos básicos necesarios para el análisis de mercado
y la detección de patrones Wyckoff.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator, MFIIndicator


class TechnicalIndicators:
    """Clase principal para calcular indicadores técnicos"""
    
    def __init__(self):
        self.indicators_cache = {}
    
    def calculate_all_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calcula todos los indicadores técnicos básicos
        
        Args:
            df: DataFrame con columnas ['open', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame con todos los indicadores calculados
        """
        result_df = df.copy()
        
        # Indicadores de tendencia
        result_df = self._add_trend_indicators(result_df)
        
        # Indicadores de momentum
        result_df = self._add_momentum_indicators(result_df)
        
        # Indicadores de volatilidad
        result_df = self._add_volatility_indicators(result_df)
        
        # Indicadores de volumen
        result_df = self._add_volume_indicators(result_df)
        
        return result_df
    
    def _add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade indicadores de tendencia"""
        
        # Medias móviles simples
        df['sma_20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
        df['sma_50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
        df['sma_200'] = SMAIndicator(close=df['close'], window=200).sma_indicator()
        
        # Medias móviles exponenciales
        df['ema_12'] = EMAIndicator(close=df['close'], window=12).ema_indicator()
        df['ema_26'] = EMAIndicator(close=df['close'], window=26).ema_indicator()
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_histogram'] = macd.macd_diff()
        
        return df
    
    def _add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade indicadores de momentum"""
        
        # RSI
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
        
        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['high'], low=df['low'], close=df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # Rate of Change
        df['roc'] = df['close'].pct_change(periods=10) * 100
        
        return df
    
    def _add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade indicadores de volatilidad"""
        
        # Bollinger Bands
        bb = BollingerBands(close=df['close'], window=20, window_dev=2)
        df['bb_upper'] = bb.bollinger_hband()
        df['bb_middle'] = bb.bollinger_mavg()
        df['bb_lower'] = bb.bollinger_lband()
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Average True Range
        df['atr'] = AverageTrueRange(high=df['high'], low=df['low'], close=df['close']).average_true_range()
        
        # Volatilidad histórica (desviación estándar de retornos)
        df['volatility'] = df['close'].pct_change().rolling(window=20).std() * np.sqrt(252)
        
        return df
    
    def _add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Añade indicadores de volumen"""
        
        # Volumen promedio
        df['volume_sma'] = df['volume'].rolling(window=20).mean()
        
        # On Balance Volume
        df['obv'] = OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()
        
        # Volume Rate of Change
        df['volume_roc'] = df['volume'].pct_change(periods=10) * 100
        
        # Relative Volume
        df['relative_volume'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        return df
    
    def get_market_regime(self, df: pd.DataFrame) -> pd.Series:
        """
        Determina el régimen de mercado basado en indicadores técnicos
        
        Returns:
            Series con valores: 'bullish', 'bearish', 'sideways'
        """
        conditions = []
        choices = []
        
        # Condiciones alcistas
        bullish_condition = (
            (df['close'] > df['sma_20']) &
            (df['sma_20'] > df['sma_50']) &
            (df['rsi'] > 50) &
            (df['macd'] > df['macd_signal'])
        )
        conditions.append(bullish_condition)
        choices.append('bullish')
        
        # Condiciones bajistas
        bearish_condition = (
            (df['close'] < df['sma_20']) &
            (df['sma_20'] < df['sma_50']) &
            (df['rsi'] < 50) &
            (df['macd'] < df['macd_signal'])
        )
        conditions.append(bearish_condition)
        choices.append('bearish')
        
        # Por defecto: lateral
        return pd.Series(
            np.select(conditions, choices, default='sideways'),
            index=df.index
        )
    
    def detect_divergences(self, df: pd.DataFrame, periods: int = 20) -> Dict[str, pd.Series]:
        """
        Detecta divergencias entre precio e indicadores de momentum
        
        Args:
            df: DataFrame con datos y indicadores
            periods: Períodos para buscar divergencias
            
        Returns:
            Diccionario con series de divergencias detectadas
        """
        divergences = {}
        
        # Divergencia RSI
        price_highs = df['high'].rolling(window=periods).max()
        price_lows = df['low'].rolling(window=periods).min()
        rsi_highs = df['rsi'].rolling(window=periods).max()
        rsi_lows = df['rsi'].rolling(window=periods).min()
        
        # Divergencia alcista: precio hace mínimos más bajos, RSI hace mínimos más altos
        bullish_div = (
            (df['low'] == price_lows) &
            (df['low'] < df['low'].shift(periods)) &
            (df['rsi'] > df['rsi'].shift(periods))
        )
        
        # Divergencia bajista: precio hace máximos más altos, RSI hace máximos más bajos
        bearish_div = (
            (df['high'] == price_highs) &
            (df['high'] > df['high'].shift(periods)) &
            (df['rsi'] < df['rsi'].shift(periods))
        )
        
        divergences['rsi_bullish'] = bullish_div
        divergences['rsi_bearish'] = bearish_div
        
        return divergences
    
    def calculate_support_resistance(self, df: pd.DataFrame, window: int = 20) -> Tuple[pd.Series, pd.Series]:
        """
        Calcula niveles de soporte y resistencia dinámicos
        
        Args:
            df: DataFrame con datos OHLC
            window: Ventana para calcular niveles
            
        Returns:
            Tupla con series de soporte y resistencia
        """
        # Soporte: mínimo móvil de los mínimos
        support = df['low'].rolling(window=window).min()
        
        # Resistencia: máximo móvil de los máximos
        resistance = df['high'].rolling(window=window).max()
        
        return support, resistance
    
    def get_overbought_oversold_signals(self, df: pd.DataFrame) -> Dict[str, pd.Series]:
        """
        Identifica señales de sobrecompra y sobreventa
        
        Returns:
            Diccionario con señales de trading
        """
        signals = {}
        
        # RSI
        signals['rsi_overbought'] = df['rsi'] > 70
        signals['rsi_oversold'] = df['rsi'] < 30
        
        # Stochastic
        signals['stoch_overbought'] = (df['stoch_k'] > 80) & (df['stoch_d'] > 80)
        signals['stoch_oversold'] = (df['stoch_k'] < 20) & (df['stoch_d'] < 20)
        
        # Bollinger Bands
        signals['bb_overbought'] = df['close'] > df['bb_upper']
        signals['bb_oversold'] = df['close'] < df['bb_lower']
        
        return signals
    
    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.Series:
        """
        Calcula la fuerza de la tendencia
        
        Returns:
            Series con valores de 0 (sin tendencia) a 1 (tendencia fuerte)
        """
        # Basado en la pendiente de la media móvil y la distancia del precio
        sma_slope = df['sma_20'].diff(5) / df['sma_20'].shift(5)
        price_distance = abs(df['close'] - df['sma_20']) / df['sma_20']
        
        # Normalizar entre 0 y 1
        trend_strength = (abs(sma_slope) + price_distance) / 2
        return np.clip(trend_strength, 0, 1)


def calculate_basic_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Función de conveniencia para calcular indicadores básicos
    
    Args:
        df: DataFrame con columnas OHLC
        
    Returns:
        DataFrame con indicadores calculados
    """
    ti = TechnicalIndicators()
    return ti.calculate_all_indicators(df)


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Crear datos de ejemplo
    dates = pd.date_range('2023-01-01', periods=100, freq='1H')
    np.random.seed(42)
    
    # Simular datos OHLC
    close_prices = 100 + np.cumsum(np.random.randn(100) * 0.5)
    high_prices = close_prices + np.random.rand(100) * 2
    low_prices = close_prices - np.random.rand(100) * 2
    open_prices = close_prices + np.random.randn(100) * 0.5
    volumes = np.random.randint(1000, 10000, 100)
    
    sample_df = pd.DataFrame({
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    })
    
    # Calcular indicadores
    ti = TechnicalIndicators()
    result = ti.calculate_all_indicators(sample_df)
    
    print("Indicadores calculados:")
    print(result[['close', 'rsi', 'macd', 'bb_upper', 'bb_lower']].tail())
    
    # Régimen de mercado
    regime = ti.get_market_regime(result)
    print(f"\nRégimen actual: {regime.iloc[-1]}")