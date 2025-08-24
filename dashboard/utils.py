#!/usr/bin/env python3
"""
Utilidades del Dashboard - Trading Embedding System
Contiene funciones de inicialización y utilidades comunes
"""

import streamlit as st
from pathlib import Path
import sys

# Configurar rutas del proyecto
project_root = Path(r"c:\dev\trading-embedding")
sys.path.insert(0, str(project_root))

# Importar componentes del sistema
try:
    from src.data.data_manager import DataManager
    from verification_system import Phase2VerificationSystem
    from src.data.data_strategy import DataStrategy
    from src.data.data_access_layer import DataAccessLayer
except ImportError as e:
    st.error(f"Error importando componentes del sistema: {e}")
    st.stop()

@st.cache_resource
def get_data_manager():
    """Inicializa y retorna el DataManager"""
    return DataManager()

@st.cache_resource
def get_data_strategy():
    """Inicializa la estrategia híbrida de datos."""
    try:
        data_access = DataAccessLayer()
        return DataStrategy(data_access)
    except Exception as e:
        st.error(f"Error inicializando estrategia de datos: {e}")
        return None

@st.cache_resource
def get_verification_system():
    """Inicializa y retorna el sistema de verificación"""
    return Phase2VerificationSystem()

def initialize_components():
    """Inicializa todos los componentes del sistema"""
    data_manager = get_data_manager()
    data_strategy = get_data_strategy()
    verification_system = get_verification_system()
    
    return data_manager, data_strategy, verification_system

def import_pages():
    """Importa todas las páginas del dashboard"""
    try:
        from pages.home import show_home_page
        from pages.phase1_data import show_phase1_data_page
        from pages.phase2_analysis import show_phase2_analysis_page
        from pages.phase3_labeling import show_phase3_labeling_page
        from pages.system_verification import show_system_verification_page
        from pages.technical_analysis import show_technical_analysis_page
        from pages.realtime_monitoring import show_realtime_monitoring_page
        from pages.reports import show_reports_page
        from pages.hybrid_data_strategy import show_hybrid_data_strategy_page
        
        return {
            'home': show_home_page,
            'phase1_data': show_phase1_data_page,
            'phase2_analysis': show_phase2_analysis_page,
            'phase3_labeling': show_phase3_labeling_page,
            'system_verification': show_system_verification_page,
            'technical_analysis': show_technical_analysis_page,
            'realtime_monitoring': show_realtime_monitoring_page,
            'hybrid_data_strategy': show_hybrid_data_strategy_page,
            'reports': show_reports_page
        }
    except ImportError as e:
        st.error(f"Error importando páginas: {e}")
        st.stop()
    except Exception as e:
        st.error(f"Error cargando páginas: {e}")
        st.stop()