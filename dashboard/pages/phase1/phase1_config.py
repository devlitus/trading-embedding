#!/usr/bin/env python3
"""
Configuración para Fase 1 - Adquisición de Datos
"""

import streamlit as st
from datetime import date
from pages.common import CUSTOM_CSS

# Configuración de símbolos y intervalos
DEFAULT_SYMBOLS = ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"]
DEFAULT_INTERVALS = ["1m", "5m", "15m", "1h", "4h", "1d", "1w"]
DEFAULT_INTERVAL_INDEX = 3  # "1h"

# Configuración de visualización
CHART_HEIGHT = 700
STATS_TABLE_HEIGHT = 200
DATA_LIMIT = 500
FALLBACK_LIMIT = 50

def apply_custom_css():
    """Aplicar CSS personalizado."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def show_page_header():
    """Mostrar encabezado de la página."""
    st.title("📊 Fase 1 - Adquisición de Datos")

def create_control_panel():
    """Crear panel de control con selectores."""
    st.markdown("### ⚙️ Panel de Control y Estado")
    st.markdown("**🔧 Configuración**")
    
    control_col1, control_col2, control_col3 = st.columns(3)
    
    with control_col1:
        symbol = st.selectbox(
            "Símbolo",
            DEFAULT_SYMBOLS,
            index=0,
            key="phase1_symbol"
        )
    
    with control_col2:
        interval = st.selectbox(
            "Intervalo",
            DEFAULT_INTERVALS,
            index=DEFAULT_INTERVAL_INDEX,
            key="phase1_interval"
        )
    
    with control_col3:
        st.markdown("**📅 Período**")
        st.info(f"Solo día actual: {date.today().strftime('%Y-%m-%d')}")
    
    st.markdown("---")  # Separador visual
    
    return symbol, interval

def show_system_status_header(data_manager):
    """Mostrar métricas principales del sistema."""
    if data_manager:
        try:
            db_stats = data_manager.get_database_stats()
            cache_stats = data_manager.get_cache_stats()
            
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Símbolos", db_stats.get('total_symbols', 0))
            with metric_col2:
                st.metric("Registros", f"{db_stats.get('total_records', 0):,}")
            with metric_col3:
                st.metric("Caché", cache_stats.get('total_keys', 0))
        except:
            pass