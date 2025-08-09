import sys
import os
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import time
import json

# Configurar rutas del proyecto
project_root = Path(r"c:\dev\trading_embbeding")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'src' / 'data'))
sys.path.insert(0, str(project_root / 'src' / 'analysis'))
sys.path.insert(0, str(project_root / 'dashboard' / 'pages'))

# Importar DataManager y Sistema de Verificación
try:
    from pages.common import DataManager
    sys.path.insert(0, str(project_root))
    from verification_system import Phase2VerificationSystem
except ImportError as e:
    st.error(f"Error importando DataManager o Sistema de Verificación: {e}")
    st.stop()

# Importar páginas
try:
    from pages.home import show_home_page
    from pages.phase1_data import show_phase1_data_page
    from pages.phase2_analysis import show_phase2_analysis_page
    from pages.system_verification import show_system_verification_page
    from pages.technical_analysis import show_technical_analysis_page
    from pages.realtime_monitoring import show_realtime_monitoring_page
    from pages.reports import show_reports_page
except ImportError as e:
    st.error(f"Error importando páginas: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error cargando páginas: {e}")
    st.stop()

# Configuración de la página
st.set_page_config(
    page_title="Trading Dashboard - Completo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicializar DataManager
@st.cache_resource
def get_data_manager():
    return DataManager()

@st.cache_resource
def get_verification_system():
    return Phase2VerificationSystem()

data_manager = get_data_manager()
verification_system = get_verification_system()

# CSS personalizado
st.markdown("""
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
""", unsafe_allow_html=True)

# Sidebar para navegación
st.sidebar.title("🚀 Trading Dashboard")
st.sidebar.markdown("---")

# Menú de navegación
page_options = {
    "🏠 Inicio": "home",
    "📊 Fase 1 - Datos": "phase1_data",
    "🔍 Fase 2 - Análisis": "phase2_analysis",
    "✅ Verificación Sistema": "system_verification",
    "📈 Análisis Técnico": "technical_analysis",
    "🎯 Monitoreo en Tiempo Real": "realtime_monitoring",
    "📋 Reportes": "reports"
}

selected_page = st.sidebar.selectbox(
    "Selecciona una página:",
    list(page_options.keys()),
    index=0
)

# Información del sistema en sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Sistema")
st.sidebar.info(f"Última actualización: {datetime.now().strftime('%H:%M:%S')}")

# Mostrar página seleccionada
page_key = page_options[selected_page]

if page_key == "home":
    show_home_page(data_manager)
elif page_key == "phase1_data":
    show_phase1_data_page(data_manager)
elif page_key == "phase2_analysis":
    show_phase2_analysis_page(data_manager)
elif page_key == "system_verification":
    show_system_verification_page(data_manager, verification_system)
elif page_key == "technical_analysis":
    show_technical_analysis_page(data_manager)
elif page_key == "realtime_monitoring":
    show_realtime_monitoring_page(data_manager)
elif page_key == "reports":
    show_reports_page(data_manager)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    <p>🚀 <strong>Trading Dashboard</strong> - Sistema Integrado de Análisis de Mercados</p>
    <p>Desarrollado con ❤️ usando Streamlit, Plotly y Python</p>
    <p>Fase 1: Adquisición de Datos | Fase 2: Análisis Técnico | Sistema de Verificación</p>
</div>
""", unsafe_allow_html=True)

if __name__ == "__main__":
    pass