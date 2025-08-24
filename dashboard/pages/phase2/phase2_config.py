#!/usr/bin/env python3
"""
Configuración para Fase 2 - Análisis Técnico
"""

import streamlit as st
from pages.common import CUSTOM_CSS

def apply_custom_css():
    """Aplicar CSS personalizado"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def show_page_header():
    """Mostrar encabezado de la página"""
    st.title("🔍 Fase 2 - Análisis Técnico")

def create_analysis_controls():
    """Crear controles de configuración del análisis"""
    st.subheader("⚙️ Configuración del Análisis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.selectbox(
            "Símbolo para Análisis",
            ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"],
            index=0,
            key="analysis_symbol"
        )
    
    with col2:
        interval = st.selectbox(
            "Intervalo de Tiempo",
            ["1m", "5m", "15m", "1h", "4h", "1d", "1w"],
            index=3,
            key="analysis_interval"
        )
    
    return symbol, interval

def create_analysis_button():
    """Crear botón de análisis"""
    return st.button("🔍 Ejecutar Análisis Técnico", type="primary")

def safe_float(value):
    """Extraer valores escalares de Series de forma segura"""
    if hasattr(value, 'iloc'):
        return float(value.iloc[-1])
    elif hasattr(value, '__len__') and len(value) > 0:
        return float(value[-1])
    else:
        return float(value)