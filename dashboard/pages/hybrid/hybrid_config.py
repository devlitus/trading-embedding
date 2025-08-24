"""Configuración e imports para la página de estrategia híbrida de datos."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime
import sys
from pathlib import Path

# Configurar path del proyecto
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

# Imports del proyecto
from data.data_strategy import DataStrategy, DataUsagePattern
from data.data_access_layer import DataAccessLayer

# Configuración de la página
PAGE_TITLE = "🔄 Estrategia Híbrida de Datos"
PAGE_DESCRIPTION = """
Esta página demuestra cómo la **DataStrategy** optimiza automáticamente el acceso a datos 
basándose en diferentes patrones de uso, combinando inteligentemente múltiples fuentes 
(API, caché, base de datos) para maximizar rendimiento y confiabilidad.
"""

# Configuración de símbolos e intervalos
DEFAULT_SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'DOTUSDT', 'LINKUSDT']
DEFAULT_INTERVALS = ['1m', '5m', '15m', '1h', '4h', '1d']

# Configuración de patrones de uso
USAGE_PATTERNS = {
    "Realtime Trading": {
        "pattern": DataUsagePattern.REALTIME_TRADING,
        "description": "**Optimizado para:** Decisiones rápidas, datos más recientes, baja latencia",
        "source": "**Fuente de datos:** API → Cache → Base de datos"
    },
    "ML Training": {
        "pattern": DataUsagePattern.ML_TRAINING,
        "description": "**Optimizado para:** Datasets completos, features avanzadas, consistencia",
        "source": "**Fuente de datos:** Base de datos → CSV → API"
    },
    "Backtesting": {
        "pattern": DataUsagePattern.BACKTESTING,
        "description": "**Optimizado para:** Datos históricos completos, validación OHLC, continuidad temporal",
        "source": "**Fuente de datos:** CSV → Base de datos"
    },
    "API Serving": {
        "pattern": DataUsagePattern.API_SERVING,
        "description": "**Optimizado para:** Respuestas rápidas, formato JSON, datos compactos",
        "source": "**Fuente de datos:** Cache → Base de datos"
    },
    "Dashboard": {
        "pattern": DataUsagePattern.DASHBOARD,
        "description": "**Optimizado para:** Visualizaciones enriquecidas, múltiples métricas, interactividad",
        "source": "**Fuente de datos:** Híbrido optimizado"
    }
}

def get_data_strategy():
    """Inicializa y retorna una instancia de DataStrategy."""
    try:
        dal = DataAccessLayer()
        return DataStrategy(dal)
    except Exception as e:
        st.error(f"Error inicializando DataStrategy: {e}")
        return None

def create_sidebar_controls():
    """Crea los controles del sidebar y retorna los valores seleccionados."""
    st.sidebar.header("⚙️ Configuración")
    
    symbol = st.sidebar.selectbox(
        "Símbolo",
        DEFAULT_SYMBOLS,
        index=0
    )
    
    interval = st.sidebar.selectbox(
        "Intervalo",
        DEFAULT_INTERVALS,
        index=3  # 1h por defecto
    )
    
    return symbol, interval

def show_page_header():
    """Muestra el encabezado de la página."""
    st.title(PAGE_TITLE)
    st.markdown(PAGE_DESCRIPTION)
    
    st.markdown("""
    ### 🎯 Patrones de Uso Disponibles
    Cada patrón optimiza automáticamente:
    - **Fuente de datos** (API, caché, base de datos, CSV)
    - **Formato de respuesta** (DataFrame, JSON, enriquecido)
    - **Estrategia de caché** (TTL, invalidación, prioridad)
    - **Validación de datos** (completitud, consistencia, calidad)
    """)

def create_tabs():
    """Crea y retorna las pestañas de la interfaz."""
    return st.tabs([
        "Realtime Trading",
        "ML Training", 
        "Backtesting",
        "API Serving",
        "Dashboard",
        "Comparación de Rendimiento"
    ])