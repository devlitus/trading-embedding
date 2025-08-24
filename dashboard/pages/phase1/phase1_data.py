#!/usr/bin/env python3
"""
Fase 1 - Adquisición de Datos
Muestra el estado del sistema de adquisición de datos en tiempo real
"""

import streamlit as st
from .phase1_config import apply_custom_css, show_page_header, create_control_panel, show_system_status_header
from .phase1_ui import show_status_row, create_main_chart, show_quick_stats, show_health_check_details, show_symbols_table
from .phase1_logic import get_data_manager, get_current_day_data, refresh_system_status, truncate_database, get_system_stats, show_system_stats_details, show_data_quality_report
from .common import get_plotly_config

def show_phase1_data_page(data_manager):
    """Mostrar página de Fase 1 - Adquisición de Datos."""
    
    # Configuración inicial y verificaciones
    apply_custom_css()
    show_page_header()
    
    if not data_manager:
        data_manager = get_data_manager()
        if not data_manager:
            st.error("❌ DataManager no disponible")
            return
    
    show_system_status_header(data_manager)
    symbol, interval = create_control_panel()
    st.markdown("---")
    show_status_row(data_manager, symbol, interval)
    
    # Visualización de datos del día actual
    try:
        df_display = get_current_day_data(data_manager, symbol, interval)
        
        if not df_display.empty:
            fig = create_main_chart(df_display, symbol, interval)
            st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())
            show_quick_stats(df_display)
            show_data_quality_report(df_display, symbol, interval)
            show_health_check_details(data_manager)
            
            st.markdown("---")
            st.markdown("### 📊 Estado del Sistema")
            
            stats = get_system_stats(data_manager)
            show_system_stats_details(stats)
            
            col1, col2 = st.columns(2)
            with col1:
                refresh_system_status(data_manager)
            with col2:
                truncate_database(data_manager)
            
            if stats and stats.get('database'):
                show_symbols_table(stats['database'])
            
        else:
            st.warning(f"⚠️ No hay datos disponibles para {symbol} en intervalo {interval}")
            st.info("💡 Los datos se están obteniendo automáticamente. Intenta cambiar el símbolo o intervalo.")
            
            stats = get_system_stats(data_manager)
            if stats:
                show_system_stats_details(stats)
                col1, col2 = st.columns(2)
                with col1:
                    refresh_system_status(data_manager)
                with col2:
                    truncate_database(data_manager)
                if stats.get('database'):
                    show_symbols_table(stats['database'])
                    
    except Exception as e:
        st.error(f"❌ Error al obtener datos: {e}")
        st.info("💡 Intenta refrescar la página o cambiar la configuración.")