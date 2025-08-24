"""Componentes de layout para el dashboard."""

import streamlit as st
from datetime import datetime
from typing import Dict, List, Optional

class DashboardLayout:
    """Clase para manejar el layout del dashboard."""
    
    @staticmethod
    def setup_page_config():
        """Configura la página de Streamlit."""
        st.set_page_config(
            page_title="Trading Dashboard - Completo",
            page_icon="📈",
            layout="wide",
            initial_sidebar_state="expanded"
        )
    
    @staticmethod
    def apply_custom_css():
        """Aplica CSS personalizado al dashboard."""
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
    
    @staticmethod
    def create_sidebar(page_options: Dict[str, str]) -> str:
        """Crea la barra lateral de navegación.
        
        Args:
            page_options: Diccionario con opciones de página
            
        Returns:
            Clave de la página seleccionada
        """
        st.sidebar.title("🚀 Trading Dashboard")
        st.sidebar.markdown("---")
        
        selected_page = st.sidebar.selectbox(
            "Selecciona una página:",
            list(page_options.keys()),
            index=0
        )
        
        # Información del sistema en sidebar
        st.sidebar.markdown("---")
        st.sidebar.markdown("### 📊 Sistema")
        st.sidebar.info(f"Última actualización: {datetime.now().strftime('%H:%M:%S')}")
        
        return page_options[selected_page]
    
    @staticmethod
    def show_footer():
        """Muestra el footer del dashboard."""
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 20px;'>
            <p>🚀 <strong>Trading Dashboard</strong> - Sistema Integrado de Análisis de Mercados</p>
            <p>Desarrollado con ❤️ usando Streamlit, Plotly y Python</p>
            <p>Fase 1: Adquisición de Datos | Fase 2: Análisis Técnico | Fase 3: Etiquetado | Sistema de Verificación</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def show_metric_card(title: str, value: str, card_type: str = "metric"):
        """Muestra una tarjeta de métrica.
        
        Args:
            title: Título de la métrica
            value: Valor de la métrica
            card_type: Tipo de tarjeta (metric, success, error, warning)
        """
        card_class = f"{card_type}-card"
        st.markdown(f"""
        <div class="{card_class}">
            <h4>{title}</h4>
            <p>{value}</p>
        </div>
        """, unsafe_allow_html=True)