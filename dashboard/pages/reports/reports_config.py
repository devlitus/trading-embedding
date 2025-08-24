#!/usr/bin/env python3
"""
Configuración y constantes para el módulo de reportes.

Este módulo centraliza toda la configuración relacionada con la generación
de reportes, incluyendo configuraciones de Plotly, rangos de fechas y
formatos de exportación.
"""

from datetime import datetime, timedelta
from typing import Dict, Tuple


class ReportsConfig:
    """Configuración centralizada para reportes"""
    
    # Tipos de reportes disponibles
    REPORT_TYPES = [
        "📈 Performance",
        "📊 Análisis Técnico", 
        "📉 Análisis de Volatilidad",
        "📋 Estado del Sistema"
    ]
    
    # Rangos de fechas disponibles
    DATE_RANGES = [
        "Últimos 7 días", 
        "Últimos 30 días", 
        "Últimos 90 días", 
        "Último año"
    ]
    
    # Formatos de exportación
    EXPORT_FORMATS = [
        "Ver en pantalla", 
        "Descargar CSV", 
        "Descargar JSON"
    ]
    
    # Configuración de Plotly
    PLOTLY_CONFIG = {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': [
            'pan2d', 'lasso2d', 'select2d', 'autoScale2d',
            'hoverClosestCartesian', 'hoverCompareCartesian'
        ],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'trading_chart',
            'height': 600,
            'width': 1200,
            'scale': 2
        }
    }
    
    # Configuración de colores para gráficos
    COLORS = {
        'primary': '#1f77b4',
        'secondary': '#ff7f0e',
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ff7f0e',
        'info': '#17a2b8',
        'background': '#f8f9fa',
        'grid': '#e9ecef'
    }
    
    # Configuración de métricas
    RISK_FREE_RATE = 0.02  # 2% anual
    VOLATILITY_WINDOW = 20  # Ventana para volatilidad rolling
    
    @staticmethod
    def calculate_date_range(range_option: str) -> Tuple[datetime, datetime]:
        """Calcula el rango de fechas basado en la opción seleccionada"""
        end_date = datetime.now()
        date_mapping = {
            "Últimos 7 días": timedelta(days=7),
            "Últimos 30 días": timedelta(days=30),
            "Últimos 90 días": timedelta(days=90),
            "Último año": timedelta(days=365)
        }
        
        delta = date_mapping.get(range_option, timedelta(days=30))
        start_date = end_date - delta
        return start_date, end_date
    
    @staticmethod
    def get_plotly_config() -> Dict:
        """Retorna la configuración de Plotly"""
        return ReportsConfig.PLOTLY_CONFIG.copy()
    
    @staticmethod
    def get_color_scheme() -> Dict[str, str]:
        """Retorna el esquema de colores para gráficos"""
        return ReportsConfig.COLORS.copy()