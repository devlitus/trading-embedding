#!/usr/bin/env python3
"""
Layout Principal del Dashboard - Trading Embedding System
Contiene la lógica de navegación y renderizado de páginas
"""

import streamlit as st
from config import PAGE_OPTIONS, FOOTER_HTML, apply_page_config, apply_custom_css, get_current_time
from utils import initialize_components, import_pages

def create_sidebar(page_functions):
    """Crea la barra lateral con navegación"""
    st.sidebar.title("🚀 Trading Dashboard")
    st.sidebar.markdown("---")
    
    # Menú de navegación
    selected_page = st.sidebar.selectbox(
        "Selecciona una página:",
        list(PAGE_OPTIONS.keys()),
        index=0
    )
    
    # Información del sistema en sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Sistema")
    st.sidebar.info(f"Última actualización: {get_current_time()}")
    
    return selected_page

def render_page(selected_page, page_functions, data_manager, data_strategy, verification_system):
    """Renderiza la página seleccionada"""
    page_key = PAGE_OPTIONS[selected_page]
    
    if page_key == "home":
        page_functions['home'](data_manager)
    elif page_key == "phase1_data":
        page_functions['phase1_data'](data_manager)
    elif page_key == "phase2_analysis":
        page_functions['phase2_analysis'](data_manager)
    elif page_key == "phase3_labeling":
        page_functions['phase3_labeling'](data_manager)
    elif page_key == "system_verification":
        page_functions['system_verification'](data_manager, verification_system)
    elif page_key == "technical_analysis":
        page_functions['technical_analysis'](data_manager)
    elif page_key == "realtime_monitoring":
        page_functions['realtime_monitoring'](data_manager)
    elif page_key == "hybrid_data_strategy":
        page_functions['hybrid_data_strategy']()
    elif page_key == "reports":
        page_functions['reports'](data_manager)

def render_footer():
    """Renderiza el footer del dashboard"""
    st.markdown("---")
    st.markdown(FOOTER_HTML, unsafe_allow_html=True)

def main():
    """Función principal del dashboard"""
    # Aplicar configuración
    apply_page_config()
    apply_custom_css()
    
    # Inicializar componentes
    data_manager, data_strategy, verification_system = initialize_components()
    
    # Importar páginas
    page_functions = import_pages()
    
    # Crear sidebar y obtener página seleccionada
    selected_page = create_sidebar(page_functions)
    
    # Renderizar página seleccionada
    render_page(selected_page, page_functions, data_manager, data_strategy, verification_system)
    
    # Renderizar footer
    render_footer()

if __name__ == "__main__":
    main()