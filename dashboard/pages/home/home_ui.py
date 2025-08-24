#!/usr/bin/env python3
"""
Componentes de interfaz de usuario para la página de inicio
"""

import streamlit as st
from ..common import show_system_metrics, CUSTOM_CSS
from .home_config import HomeConfig
from .home_logic import HomeDataProcessor

class HomeUI:
    """Componentes de UI para la página de inicio"""
    
    def __init__(self, data_manager=None):
        """Inicializar componentes de UI"""
        self.data_manager = data_manager
        self.config = HomeConfig()
        self.data_processor = HomeDataProcessor(data_manager) if data_manager else None
    
    def render_page_header(self):
        """Renderizar encabezado de la página"""
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        st.title(self.config.PAGE_TITLE)
    
    def render_welcome_section(self):
        """Renderizar sección de bienvenida y descripción"""
        st.markdown(self.config.WELCOME_TITLE)
        st.markdown(self.config.SYSTEM_DESCRIPTION)
    
    def render_system_metrics(self):
        """Renderizar métricas del sistema"""
        st.subheader(self.config.SECTIONS['system_status'])
        show_system_metrics(self.data_manager)
    
    def render_database_info(self):
        """Renderizar información de la base de datos"""
        st.subheader(self.config.SECTIONS['database_info'])
        
        if not self.data_processor:
            st.error(self.config.MESSAGES['no_connection'])
            return
        
        connection_status = self.data_processor.get_connection_status()
        
        if connection_status['status'] == 'error':
            st.error(connection_status['message'])
        elif connection_status['status'] == 'warning':
            st.warning(connection_status['message'])
        elif connection_status['status'] == 'success':
            st.success(connection_status['message'])
            self._render_database_details(connection_status['data'])
    
    def _render_database_details(self, db_stats):
        """Renderizar detalles de la base de datos"""
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(self.config.STATS_LABELS['symbols_available'])
            symbol_info = self.data_processor.format_symbol_info(db_stats['symbols'])
            for info in symbol_info:
                st.write(info)
        
        with col2:
            st.markdown(self.config.STATS_LABELS['general_stats'])
            general_stats = self.data_processor.get_general_stats_info(db_stats)
            
            for stat_key, stat_value in general_stats.items():
                st.write(stat_value)
    
    def render_system_info(self):
        """Renderizar información del sistema"""
        st.subheader(self.config.SECTIONS['system_info'])
        
        info_col1, info_col2 = st.columns(2)
        
        with info_col1:
            st.markdown(self.config.TECHNOLOGIES)
        
        with info_col2:
            st.markdown(self.config.FEATURES)
    
    def render_divider(self):
        """Renderizar separador"""
        st.divider()
    
    def render_main_page(self):
        """
        Renderizar la página principal de inicio
        
        Orquesta todos los componentes de la página de inicio
        """
        # Encabezado
        self.render_page_header()
        
        # Sección de bienvenida
        self.render_welcome_section()
        
        # Separador
        self.render_divider()
        
        # Métricas del sistema
        self.render_system_metrics()
        
        # Separador
        self.render_divider()
        
        # Información de la base de datos
        self.render_database_info()
        
        # Separador
        self.render_divider()
        
        # Información del sistema
        self.render_system_info()