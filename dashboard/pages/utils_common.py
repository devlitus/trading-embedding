#!/usr/bin/env python3
"""
Utilidades comunes para páginas del dashboard
"""

import streamlit as st
from datetime import datetime
from .config_common import project_root
from .imports_common import DataManager, Phase2VerificationSystem, load_modules

# Inicializar componentes
@st.cache_resource
def init_components():
    """Inicializar componentes del sistema"""
    # Cargar módulos si no están cargados
    if not load_modules():
        return None, None
        
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