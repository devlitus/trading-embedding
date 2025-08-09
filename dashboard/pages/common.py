#!/usr/bin/env python3
"""
Common utilities and configurations for dashboard pages
"""

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

# Agregar los directorios necesarios al path
project_root = Path(r"c:\dev\trading_embbeding")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'src' / 'data'))
sys.path.insert(0, str(project_root / 'src' / 'analysis'))

try:
    # Importar módulos usando rutas absolutas
    import importlib.util
    
    # Cargar BinanceClient
    binance_spec = importlib.util.spec_from_file_location("binance_client", project_root / "src" / "data" / "binance_client.py")
    binance_module = importlib.util.module_from_spec(binance_spec)
    binance_spec.loader.exec_module(binance_module)
    BinanceClient = binance_module.BinanceClient
    
    # Cargar TradingDatabase
    database_spec = importlib.util.spec_from_file_location("database", project_root / "src" / "data" / "database.py")
    database_module = importlib.util.module_from_spec(database_spec)
    database_spec.loader.exec_module(database_module)
    TradingDatabase = database_module.TradingDatabase
    
    # Cargar TradingCache
    cache_spec = importlib.util.spec_from_file_location("cache", project_root / "src" / "data" / "cache.py")
    cache_module = importlib.util.module_from_spec(cache_spec)
    cache_spec.loader.exec_module(cache_module)
    TradingCache = cache_module.TradingCache
    
    # Cargar DataManager
    data_manager_spec = importlib.util.spec_from_file_location("data_manager", project_root / "src" / "data" / "data_manager.py")
    data_manager_module = importlib.util.module_from_spec(data_manager_spec)
    data_manager_spec.loader.exec_module(data_manager_module)
    DataManager = data_manager_module.DataManager
    
    # Cargar Phase2VerificationSystem
    verification_spec = importlib.util.spec_from_file_location("verification_system", project_root / "verification_system.py")
    verification_module = importlib.util.module_from_spec(verification_spec)
    verification_spec.loader.exec_module(verification_module)
    Phase2VerificationSystem = verification_module.Phase2VerificationSystem
    
    # Cargar análisis técnico
    analysis_spec = importlib.util.spec_from_file_location("technical_analysis", project_root / "src" / "analysis" / "technical_analysis.py")
    analysis_module = importlib.util.module_from_spec(analysis_spec)
    analysis_spec.loader.exec_module(analysis_module)
    analyze_symbol = analysis_module.analyze_symbol
    
except ImportError as e:
    st.error(f"Error importando módulos: {e}")
    st.stop()
except Exception as e:
    st.error(f"Error cargando módulos: {e}")
    st.stop()

# Configuración de Plotly
def get_plotly_config():
    """Configuración estándar para gráficos Plotly"""
    return {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': [
            'pan2d', 'lasso2d', 'select2d', 'autoScale2d',
            'hoverClosestCartesian', 'hoverCompareCartesian',
            'toggleSpikelines'
        ],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'trading_chart',
            'height': 500,
            'width': 1200,
            'scale': 1
        }
    }

# CSS personalizado
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

# Inicializar componentes
@st.cache_resource
def init_components():
    """Inicializar componentes del sistema"""
    try:
        # Usar ruta absoluta hacia la base de datos principal
        db_path = str(project_root / "data" / "trading.db")
        data_manager = DataManager(db_path=db_path)
        verification_system = Phase2VerificationSystem()
        return data_manager, verification_system
    except Exception as e:
        st.error(f"Error inicializando componentes: {e}")
        return None, None

# Función para mostrar métricas del sistema
def show_system_metrics(data_manager):
    """Mostrar métricas generales del sistema"""
    if data_manager:
        health = data_manager.health_check()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            try:
                # Verificar el estado de todos los componentes
                components = health.get('components', {})
                all_healthy = all(
                    comp.get('status') == 'healthy' 
                    for comp in components.values()
                )
                status = "🟢 Online" if all_healthy else "🔴 Error"
            except (KeyError, TypeError):
                status = "🔴 Error"
            st.metric("Estado Sistema", status)
        
        with col2:
            db_stats = data_manager.get_database_stats()
            st.metric("Símbolos en BD", db_stats.get('total_symbols', 0))
        
        with col3:
            cache_stats = data_manager.get_cache_stats()
            st.metric("Elementos en Caché", cache_stats.get('total_keys', 0))
        
        with col4:
            st.metric("Última Actualización", datetime.now().strftime("%H:%M:%S"))