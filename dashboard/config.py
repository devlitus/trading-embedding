#!/usr/bin/env python3
"""
Configuración del Dashboard - Trading Embedding System
Contiene configuración de rutas, importaciones y estilos CSS
"""

import sys
import os
from pathlib import Path
import streamlit as st
from datetime import datetime

# Configurar rutas del proyecto
project_root = Path(r"c:\dev\trading-embedding")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'src' / 'data'))
sys.path.insert(0, str(project_root / 'src' / 'analysis'))
sys.path.insert(0, str(project_root / 'dashboard' / 'pages'))

# CSS personalizado para el dashboard
CUSTOM_CSS = """
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.success-card {
    background-color: #d4edda;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28a745;
    color: black;
}
.error-card {
    background-color: #f8d7da;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #dc3545;
}
.warning-card {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ffc107;
}
</style>
"""

# Configuración de páginas del dashboard
PAGE_CONFIG = {
    "page_title": "Trading Dashboard - Completo",
    "page_icon": "📈",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Opciones del menú de navegación
PAGE_OPTIONS = {
    "🏠 Inicio": "home",
    "📊 Fase 1 - Datos": "phase1_data",
    "🔍 Fase 2 - Análisis": "phase2_analysis",
    "🏷️ Fase 3 - Etiquetado": "phase3_labeling",
    "✅ Verificación Sistema": "system_verification",
    "📈 Análisis Técnico": "technical_analysis",
    "🎯 Monitoreo en Tiempo Real": "realtime_monitoring",
    "🔄 Estrategia Híbrida de Datos": "hybrid_data_strategy",
    "📋 Reportes": "reports"
}

# Footer HTML
FOOTER_HTML = """
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 <strong>Trading Dashboard</strong> - Sistema Integrado de Análisis de Mercados</p>
    <p>Desarrollado con ❤️ usando Streamlit, Plotly y Python</p>
    <p>Fase 1: Adquisición de Datos | Fase 2: Análisis Técnico | Fase 3: Etiquetado | Sistema de Verificación</p>
</div>
"""

def apply_page_config():
    """Aplica la configuración de página de Streamlit"""
    st.set_page_config(**PAGE_CONFIG)

def apply_custom_css():
    """Aplica el CSS personalizado"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

def get_current_time():
    """Retorna la hora actual formateada"""
    return datetime.now().strftime('%H:%M:%S')