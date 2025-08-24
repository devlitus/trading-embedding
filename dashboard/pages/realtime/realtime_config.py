#!/usr/bin/env python3
"""
Configuración para el módulo de monitoreo en tiempo real.

Este módulo centraliza todas las configuraciones, constantes y parámetros
relacionados con el monitoreo en tiempo real de criptomonedas.
"""

from typing import Dict, List, Any


class RealtimeConfig:
    """Configuración centralizada para monitoreo en tiempo real"""
    
    # Símbolos disponibles para monitoreo
    AVAILABLE_SYMBOLS = [
        "BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", 
        "XRPUSDT", "SOLUSDT", "DOTUSDT", "LINKUSDT",
        "LTCUSDT", "UNIUSDT", "AVAXUSDT", "MATICUSDT"
    ]
    
    # Símbolos por defecto
    DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
    
    # Intervalos de actualización disponibles
    REFRESH_INTERVALS = {
        "5 segundos": 5,
        "10 segundos": 10,
        "30 segundos": 30,
        "1 minuto": 60,
        "2 minutos": 120,
        "5 minutos": 300
    }
    
    # Configuración de gráficos
    CHART_CONFIG = {
        'mini_chart_height': 150,
        'mini_chart_points': 20,
        'main_chart_height': 400,
        'main_chart_points': 100
    }
    
    # Configuración de alertas
    ALERT_THRESHOLDS = {
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'trend_strength_high': 0.8,
        'pattern_confidence_high': 0.7,
        'volume_spike_multiplier': 2.0,
        'price_change_alert': 5.0  # Porcentaje
    }
    
    # Configuración de colores para alertas
    ALERT_COLORS = {
        'bullish': '#00C851',
        'bearish': '#FF4444',
        'neutral': '#33B5E5',
        'warning': '#FF8800'
    }
    
    # Configuración de métricas
    METRICS_CONFIG = {
        'price_decimals': 2,
        'percentage_decimals': 2,
        'volume_format': 'compact',
        'update_animation': True
    }
    
    # Configuración de datos históricos para análisis
    DATA_CONFIG = {
        'quick_analysis_periods': {
            '1m': 50,   # 50 períodos de 1 minuto
            '5m': 50,   # 50 períodos de 5 minutos
            '15m': 30,  # 30 períodos de 15 minutos
            '1h': 24    # 24 períodos de 1 hora
        },
        'mini_chart_interval': '1m',
        'alert_analysis_interval': '5m'
    }
    
    # Configuración de UI
    UI_CONFIG = {
        'max_symbols_per_row': 4,
        'show_mini_charts': True,
        'show_volume_info': True,
        'auto_refresh_default': False,
        'default_refresh_interval': "30 segundos"
    }
    
    # Emojis para diferentes tipos de alertas
    ALERT_EMOJIS = {
        'overbought': '⚠️',
        'oversold': '💚',
        'bullish_trend': '📈',
        'bearish_trend': '📉',
        'pattern_detected': '🎯',
        'volume_spike': '📊',
        'price_alert': '💰',
        'error': '❌',
        'info': 'ℹ️'
    }
    
    @staticmethod
    def get_refresh_seconds(interval_text: str) -> int:
        """Convierte texto de intervalo a segundos"""
        return RealtimeConfig.REFRESH_INTERVALS.get(interval_text, 30)
    
    @staticmethod
    def format_price(price: float, decimals: int = None) -> str:
        """Formatea precio con decimales apropiados"""
        if decimals is None:
            decimals = RealtimeConfig.METRICS_CONFIG['price_decimals']
        return f"${price:.{decimals}f}"
    
    @staticmethod
    def format_percentage(percentage: float, decimals: int = None) -> str:
        """Formatea porcentaje con signo y decimales"""
        if decimals is None:
            decimals = RealtimeConfig.METRICS_CONFIG['percentage_decimals']
        sign = "+" if percentage >= 0 else ""
        return f"{sign}{percentage:.{decimals}f}%"
    
    @staticmethod
    def get_delta_color(change: float) -> str:
        """Obtiene color para delta basado en cambio"""
        return "normal" if change >= 0 else "inverse"
    
    @staticmethod
    def get_alert_color(alert_type: str) -> str:
        """Obtiene color para tipo de alerta"""
        color_map = {
            'bullish': RealtimeConfig.ALERT_COLORS['bullish'],
            'bearish': RealtimeConfig.ALERT_COLORS['bearish'],
            'warning': RealtimeConfig.ALERT_COLORS['warning'],
            'neutral': RealtimeConfig.ALERT_COLORS['neutral']
        }
        return color_map.get(alert_type, RealtimeConfig.ALERT_COLORS['neutral'])