"""Servicio de análisis técnico para el dashboard."""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
import streamlit as st
from datetime import datetime, timedelta

class AnalysisService:
    """Servicio para análisis técnico en el dashboard."""
    
    def __init__(self, data_service):
        """Inicializa el servicio de análisis.
        
        Args:
            data_service: Instancia del DataService
        """
        self.data_service = data_service
    
    def calculate_technical_indicators(self, data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calcula indicadores técnicos básicos.
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            Diccionario con indicadores técnicos
        """
        if data.empty or 'close' not in data.columns:
            return {}
        
        indicators = {}
        
        try:
            # Medias móviles simples
            indicators['sma_20'] = data['close'].rolling(window=20).mean()
            indicators['sma_50'] = data['close'].rolling(window=50).mean()
            indicators['sma_200'] = data['close'].rolling(window=200).mean()
            
            # Medias móviles exponenciales
            indicators['ema_12'] = data['close'].ewm(span=12).mean()
            indicators['ema_26'] = data['close'].ewm(span=26).mean()
            
            # RSI
            indicators['rsi'] = self._calculate_rsi(data['close'])
            
            # MACD
            macd_data = self._calculate_macd(data['close'])
            indicators.update(macd_data)
            
            # Bandas de Bollinger
            bb_data = self._calculate_bollinger_bands(data['close'])
            indicators.update(bb_data)
            
            # Estocástico
            if 'high' in data.columns and 'low' in data.columns:
                stoch_data = self._calculate_stochastic(data['high'], data['low'], data['close'])
                indicators.update(stoch_data)
            
            # ATR (Average True Range)
            if all(col in data.columns for col in ['high', 'low', 'close']):
                indicators['atr'] = self._calculate_atr(data['high'], data['low'], data['close'])
            
        except Exception as e:
            st.error(f"Error calculando indicadores técnicos: {e}")
        
        return indicators
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calcula el RSI (Relative Strength Index).
        
        Args:
            prices: Serie de precios
            period: Período para el cálculo
            
        Returns:
            Serie con valores RSI
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """Calcula MACD.
        
        Args:
            prices: Serie de precios
            fast: Período EMA rápida
            slow: Período EMA lenta
            signal: Período línea de señal
            
        Returns:
            Diccionario con MACD, señal e histograma
        """
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_histogram = macd - macd_signal
        
        return {
            'macd': macd,
            'macd_signal': macd_signal,
            'macd_histogram': macd_histogram
        }
    
    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: int = 2) -> Dict[str, pd.Series]:
        """Calcula Bandas de Bollinger.
        
        Args:
            prices: Serie de precios
            period: Período para la media móvil
            std_dev: Número de desviaciones estándar
            
        Returns:
            Diccionario con bandas superior, media e inferior
        """
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()
        
        return {
            'bb_upper': sma + (std * std_dev),
            'bb_middle': sma,
            'bb_lower': sma - (std * std_dev)
        }
    
    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                            k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calcula el oscilador estocástico.
        
        Args:
            high: Serie de precios máximos
            low: Serie de precios mínimos
            close: Serie de precios de cierre
            k_period: Período para %K
            d_period: Período para %D
            
        Returns:
            Diccionario con %K y %D
        """
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'stoch_k': k_percent,
            'stoch_d': d_percent
        }
    
    def _calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calcula Average True Range.
        
        Args:
            high: Serie de precios máximos
            low: Serie de precios mínimos
            close: Serie de precios de cierre
            period: Período para el cálculo
            
        Returns:
            Serie con valores ATR
        """
        prev_close = close.shift(1)
        
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = true_range.rolling(window=period).mean()
        
        return atr
    
    def generate_trading_signals(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Dict[str, pd.Series]:
        """Genera señales de trading basadas en indicadores técnicos.
        
        Args:
            data: DataFrame con datos OHLCV
            indicators: Diccionario con indicadores técnicos
            
        Returns:
            Diccionario con señales de trading
        """
        signals = {}
        
        try:
            # Señales RSI
            if 'rsi' in indicators:
                rsi_buy = indicators['rsi'] < 30
                rsi_sell = indicators['rsi'] > 70
                signals['rsi_buy'] = rsi_buy
                signals['rsi_sell'] = rsi_sell
            
            # Señales MACD
            if 'macd' in indicators and 'macd_signal' in indicators:
                macd_buy = (indicators['macd'] > indicators['macd_signal']) & \
                          (indicators['macd'].shift(1) <= indicators['macd_signal'].shift(1))
                macd_sell = (indicators['macd'] < indicators['macd_signal']) & \
                           (indicators['macd'].shift(1) >= indicators['macd_signal'].shift(1))
                signals['macd_buy'] = macd_buy
                signals['macd_sell'] = macd_sell
            
            # Señales de medias móviles
            if 'sma_20' in indicators and 'sma_50' in indicators:
                ma_buy = (indicators['sma_20'] > indicators['sma_50']) & \
                        (indicators['sma_20'].shift(1) <= indicators['sma_50'].shift(1))
                ma_sell = (indicators['sma_20'] < indicators['sma_50']) & \
                         (indicators['sma_20'].shift(1) >= indicators['sma_50'].shift(1))
                signals['ma_buy'] = ma_buy
                signals['ma_sell'] = ma_sell
            
            # Señales de Bandas de Bollinger
            if all(key in indicators for key in ['bb_upper', 'bb_lower']):
                bb_buy = data['close'] < indicators['bb_lower']
                bb_sell = data['close'] > indicators['bb_upper']
                signals['bb_buy'] = bb_buy
                signals['bb_sell'] = bb_sell
            
        except Exception as e:
            st.error(f"Error generando señales de trading: {e}")
        
        return signals
    
    def analyze_trend(self, data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Analiza la tendencia del mercado.
        
        Args:
            data: DataFrame con datos OHLCV
            indicators: Diccionario con indicadores técnicos
            
        Returns:
            Diccionario con análisis de tendencia
        """
        if data.empty:
            return {'error': 'No hay datos para analizar'}
        
        analysis = {}
        
        try:
            current_price = data['close'].iloc[-1]
            
            # Análisis de medias móviles
            if 'sma_20' in indicators and 'sma_50' in indicators and 'sma_200' in indicators:
                sma_20_current = indicators['sma_20'].iloc[-1]
                sma_50_current = indicators['sma_50'].iloc[-1]
                sma_200_current = indicators['sma_200'].iloc[-1]
                
                if pd.notna(sma_20_current) and pd.notna(sma_50_current) and pd.notna(sma_200_current):
                    if current_price > sma_20_current > sma_50_current > sma_200_current:
                        trend = "Fuertemente Alcista"
                    elif current_price > sma_20_current > sma_50_current:
                        trend = "Alcista"
                    elif current_price < sma_20_current < sma_50_current < sma_200_current:
                        trend = "Fuertemente Bajista"
                    elif current_price < sma_20_current < sma_50_current:
                        trend = "Bajista"
                    else:
                        trend = "Lateral"
                    
                    analysis['trend'] = trend
                    analysis['price_vs_sma20'] = ((current_price - sma_20_current) / sma_20_current * 100)
            
            # Análisis de momentum (RSI)
            if 'rsi' in indicators:
                rsi_current = indicators['rsi'].iloc[-1]
                if pd.notna(rsi_current):
                    if rsi_current > 70:
                        momentum = "Sobrecomprado"
                    elif rsi_current < 30:
                        momentum = "Sobrevendido"
                    else:
                        momentum = "Neutral"
                    
                    analysis['momentum'] = momentum
                    analysis['rsi_value'] = rsi_current
            
            # Análisis de volatilidad
            if len(data) >= 20:
                returns = data['close'].pct_change().dropna()
                volatility = returns.std() * np.sqrt(252) * 100  # Volatilidad anualizada
                analysis['volatility'] = volatility
                
                if volatility > 50:
                    vol_level = "Alta"
                elif volatility > 25:
                    vol_level = "Media"
                else:
                    vol_level = "Baja"
                
                analysis['volatility_level'] = vol_level
            
            # Análisis de soporte y resistencia
            if len(data) >= 50:
                recent_highs = data['high'].rolling(window=20).max()
                recent_lows = data['low'].rolling(window=20).min()
                
                resistance = recent_highs.iloc[-1]
                support = recent_lows.iloc[-1]
                
                analysis['resistance'] = resistance
                analysis['support'] = support
                analysis['distance_to_resistance'] = ((resistance - current_price) / current_price * 100)
                analysis['distance_to_support'] = ((current_price - support) / current_price * 100)
            
        except Exception as e:
            analysis['error'] = f"Error en análisis de tendencia: {str(e)}"
        
        return analysis
    
    def get_market_sentiment(self, symbols: List[str]) -> Dict[str, Any]:
        """Analiza el sentimiento general del mercado.
        
        Args:
            symbols: Lista de símbolos a analizar
            
        Returns:
            Diccionario con análisis de sentimiento
        """
        sentiment_data = []
        
        for symbol in symbols[:10]:  # Limitar para performance
            try:
                data = self.data_service.get_market_data(symbol, '1h', 24)
                if not data.empty:
                    # Calcular cambio de precio en 24h
                    price_change = ((data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0] * 100)
                    sentiment_data.append({
                        'symbol': symbol,
                        'price_change_24h': price_change,
                        'volume': data['volume'].sum() if 'volume' in data.columns else 0
                    })
            except Exception:
                continue
        
        if not sentiment_data:
            return {'error': 'No se pudo obtener datos de sentimiento'}
        
        # Calcular métricas de sentimiento
        df = pd.DataFrame(sentiment_data)
        
        bullish_count = (df['price_change_24h'] > 0).sum()
        bearish_count = (df['price_change_24h'] < 0).sum()
        neutral_count = (df['price_change_24h'] == 0).sum()
        
        avg_change = df['price_change_24h'].mean()
        
        if avg_change > 2:
            overall_sentiment = "Muy Alcista"
        elif avg_change > 0.5:
            overall_sentiment = "Alcista"
        elif avg_change > -0.5:
            overall_sentiment = "Neutral"
        elif avg_change > -2:
            overall_sentiment = "Bajista"
        else:
            overall_sentiment = "Muy Bajista"
        
        return {
            'overall_sentiment': overall_sentiment,
            'average_change': avg_change,
            'bullish_symbols': bullish_count,
            'bearish_symbols': bearish_count,
            'neutral_symbols': neutral_count,
            'total_symbols': len(df),
            'bullish_percentage': (bullish_count / len(df) * 100) if len(df) > 0 else 0
        }