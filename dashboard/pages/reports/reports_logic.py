#!/usr/bin/env python3
"""
Lógica de negocio para generación de reportes.

Este módulo contiene la clase ReportGenerator y toda la lógica
relacionada con la generación de diferentes tipos de reportes.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, Optional, Tuple
import streamlit as st

from .reports_config import ReportsConfig


class ReportGenerator:
    """Generador de reportes para análisis de trading"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.config = ReportsConfig()
    
    def get_data_for_report(self, symbol: str, date_range: str) -> Optional[pd.DataFrame]:
        """Obtiene datos para el reporte"""
        try:
            start_date, end_date = self.config.calculate_date_range(date_range)
            
            # Obtener datos históricos
            df = self.data_manager.get_historical_data(
                symbol=symbol,
                interval='1h',
                start_date=start_date,
                end_date=end_date
            )
            
            if df is None or df.empty:
                st.warning(f"⚠️ No hay datos disponibles para {symbol} en el rango seleccionado")
                return None
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error obteniendo datos: {e}")
            return None
    
    def calculate_performance_metrics(self, df: pd.DataFrame) -> Dict:
        """Calcula métricas de rendimiento"""
        if df.empty:
            return {}
        
        initial_price = df['close'].iloc[0]
        final_price = df['close'].iloc[-1]
        total_return = ((final_price - initial_price) / initial_price) * 100
        avg_volume = df['volume'].mean()
        
        # Calcular retornos diarios
        df['returns'] = df['close'].pct_change()
        
        # Métricas adicionales
        max_drawdown = self._calculate_max_drawdown(df)
        sharpe_ratio = self._calculate_sharpe_ratio(df['returns'])
        volatility = df['returns'].std() * np.sqrt(24 * 365)  # Anualizada para datos horarios
        
        return {
            'initial_price': initial_price,
            'final_price': final_price,
            'total_return': total_return,
            'avg_volume': avg_volume,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'volatility': volatility,
            'max_price': df['high'].max(),
            'min_price': df['low'].min(),
            'total_volume': df['volume'].sum()
        }
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores técnicos"""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Media móvil simple (SMA)
        df['sma_20'] = df['close'].rolling(window=20).mean()
        df['sma_50'] = df['close'].rolling(window=50).mean()
        
        # Media móvil exponencial (EMA)
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Bandas de Bollinger
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        
        return df
    
    def analyze_trading_signals(self, df: pd.DataFrame) -> Dict:
        """Analiza señales de trading"""
        if df.empty or 'rsi' not in df.columns:
            return {}
        
        # Señales RSI
        oversold_signals = len(df[df['rsi'] < 30])
        overbought_signals = len(df[df['rsi'] > 70])
        
        # Señales MACD
        macd_bullish = len(df[(df['macd'] > df['macd_signal']) & 
                             (df['macd'].shift(1) <= df['macd_signal'].shift(1))])
        macd_bearish = len(df[(df['macd'] < df['macd_signal']) & 
                             (df['macd'].shift(1) >= df['macd_signal'].shift(1))])
        
        # Señales de cruce de medias móviles
        sma_bullish = len(df[(df['sma_20'] > df['sma_50']) & 
                            (df['sma_20'].shift(1) <= df['sma_50'].shift(1))])
        sma_bearish = len(df[(df['sma_20'] < df['sma_50']) & 
                            (df['sma_20'].shift(1) >= df['sma_50'].shift(1))])
        
        return {
            'oversold_signals': oversold_signals,
            'overbought_signals': overbought_signals,
            'macd_bullish': macd_bullish,
            'macd_bearish': macd_bearish,
            'sma_bullish': sma_bullish,
            'sma_bearish': sma_bearish
        }
    
    def calculate_volatility_metrics(self, df: pd.DataFrame) -> Tuple[pd.Series, Dict]:
        """Calcula métricas de volatilidad"""
        if df.empty:
            return pd.Series(), {}
        
        # Calcular retornos
        df['returns'] = df['close'].pct_change()
        
        # Volatilidad rolling
        rolling_vol = df['returns'].rolling(window=self.config.VOLATILITY_WINDOW).std() * np.sqrt(24)
        
        # Métricas de volatilidad
        metrics = {
            'avg_volatility': rolling_vol.mean(),
            'max_volatility': rolling_vol.max(),
            'min_volatility': rolling_vol.min(),
            'current_volatility': rolling_vol.iloc[-1] if not rolling_vol.empty else 0,
            'volatility_trend': 'Creciente' if rolling_vol.iloc[-5:].mean() > rolling_vol.iloc[-10:-5].mean() else 'Decreciente'
        }
        
        return rolling_vol, metrics
    
    def get_system_health(self) -> Dict:
        """Obtiene el estado de salud del sistema"""
        try:
            health = {
                'database': 'Conectado' if self.data_manager.database else 'Desconectado',
                'cache': 'Activo' if hasattr(self.data_manager, 'cache') and self.data_manager.cache else 'Inactivo',
                'api': 'Disponible',  # Simplificado
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Verificar disponibilidad de datos
            try:
                symbols = self.data_manager.database.get_available_symbols()
                health['available_symbols'] = len(symbols) if symbols else 0
            except:
                health['available_symbols'] = 0
            
            return health
            
        except Exception as e:
            return {
                'database': 'Error',
                'cache': 'Error',
                'api': 'Error',
                'error': str(e),
                'last_update': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
    
    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calcula el máximo drawdown"""
        if df.empty:
            return 0.0
        
        cumulative = (1 + df['returns'].fillna(0)).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return abs(drawdown.min()) * 100
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = None) -> float:
        """Calcula el ratio de Sharpe"""
        if returns.empty:
            return 0.0
        
        if risk_free_rate is None:
            risk_free_rate = self.config.RISK_FREE_RATE
        
        excess_returns = returns.mean() * 24 * 365 - risk_free_rate  # Anualizado
        volatility = returns.std() * np.sqrt(24 * 365)  # Anualizado
        
        return excess_returns / volatility if volatility != 0 else 0.0