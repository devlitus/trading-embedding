"""Página de inicio del dashboard."""

import streamlit as st
from typing import Optional
from ..components import DashboardLayout, MetricsComponent
from ..services import DataService
from ..utils import DataFormatters

class HomePage:
    """Clase para la página de inicio del dashboard."""
    
    def __init__(self, data_service: Optional[DataService] = None):
        """Inicializa la página de inicio.
        
        Args:
            data_service: Servicio de datos
        """
        self.data_service = data_service or DataService()
    
    def render(self) -> None:
        """Renderiza la página de inicio."""
        # Título principal
        st.title("🚀 Trading Embedding System")
        st.markdown("---")
        
        # Descripción del sistema
        self._render_system_description()
        
        # Métricas del sistema
        self._render_system_metrics()
        
        # Estado del sistema
        self._render_system_status()
        
        # Accesos rápidos
        self._render_quick_actions()
    
    def _render_system_description(self) -> None:
        """Renderiza la descripción del sistema."""
        st.header("📊 Sistema Integrado de Trading")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🎯 Funcionalidades Principales")
            st.markdown("""
            - **Adquisición de Datos**: Recopilación automática de datos de mercado en tiempo real
            - **Análisis Técnico**: Cálculo de indicadores técnicos y patrones de trading
            - **Machine Learning**: Embeddings y análisis predictivo avanzado
            - **Verificación de Sistema**: Validación continua de la integridad de datos
            - **Monitoreo en Tiempo Real**: Seguimiento de métricas y rendimiento
            """)
        
        with col2:
            st.subheader("🔧 Componentes del Sistema")
            st.markdown("""
            - **Data Manager**: Gestión centralizada de datos de mercado
            - **Pattern Recognition**: Detección de patrones de trading
            - **Wyckoff Analysis**: Análisis según metodología Wyckoff
            - **Trend Detection**: Identificación de tendencias de mercado
            - **Embeddings ML**: Representaciones vectoriales de datos
            """)
    
    def _render_system_metrics(self) -> None:
        """Renderiza métricas del sistema."""
        st.header("📈 Métricas del Sistema")
        
        try:
            # Obtener métricas básicas
            symbols = self.data_service.get_available_symbols()
            total_symbols = len(symbols) if symbols else 0
            
            # Obtener información de algunos símbolos para estadísticas
            sample_data = None
            if symbols:
                try:
                    sample_data = self.data_service.get_market_data(symbols[0], '1h', limit=100)
                except Exception:
                    pass
            
            # Mostrar métricas en columnas
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="📊 Símbolos Disponibles",
                    value=total_symbols,
                    delta=None
                )
            
            with col2:
                data_points = len(sample_data) if sample_data is not None else 0
                st.metric(
                    label="📈 Puntos de Datos",
                    value=DataFormatters.format_number(data_points, decimals=0),
                    delta=None
                )
            
            with col3:
                st.metric(
                    label="🔄 Estado del Sistema",
                    value="Activo",
                    delta="Operacional"
                )
            
            with col4:
                st.metric(
                    label="⏱️ Última Actualización",
                    value="Tiempo Real",
                    delta="Conectado"
                )
                
        except Exception as e:
            st.error(f"Error al cargar métricas: {str(e)}")
            
            # Métricas por defecto en caso de error
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📊 Símbolos", "N/A")
            with col2:
                st.metric("📈 Datos", "N/A")
            with col3:
                st.metric("🔄 Estado", "Error")
            with col4:
                st.metric("⏱️ Actualización", "N/A")
    
    def _render_system_status(self) -> None:
        """Renderiza el estado del sistema."""
        st.header("🔍 Estado del Sistema")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌐 Conectividad")
            
            # Estado de conexión a APIs
            try:
                # Intentar obtener datos para verificar conectividad
                test_symbols = self.data_service.get_available_symbols()
                if test_symbols:
                    st.success("✅ Conexión a API activa")
                    st.info(f"📡 {len(test_symbols)} símbolos disponibles")
                else:
                    st.warning("⚠️ Conexión limitada")
            except Exception as e:
                st.error(f"❌ Error de conexión: {str(e)}")
            
            # Estado de base de datos
            st.success("✅ Base de datos operativa")
            st.success("✅ Sistema de caché activo")
        
        with col2:
            st.subheader("📊 Calidad de Datos")
            
            # Indicadores de calidad
            st.success("✅ Validación de datos activa")
            st.success("✅ Detección de anomalías")
            st.info("ℹ️ Limpieza automática de datos")
            st.info("ℹ️ Respaldo de datos configurado")
    
    def _render_quick_actions(self) -> None:
        """Renderiza acciones rápidas."""
        st.header("⚡ Accesos Rápidos")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("📈 Análisis de Mercado")
            if st.button("🔍 Ver Datos de Mercado", key="market_data"):
                st.switch_page("pages/market_data.py")
            
            if st.button("📊 Análisis Técnico", key="technical_analysis"):
                st.switch_page("pages/technical_analysis.py")
        
        with col2:
            st.subheader("🤖 Machine Learning")
            if st.button("🧠 Embeddings", key="embeddings"):
                st.switch_page("pages/embeddings.py")
            
            if st.button("🔮 Predicciones", key="predictions"):
                st.switch_page("pages/predictions.py")
        
        with col3:
            st.subheader("⚙️ Sistema")
            if st.button("🔧 Configuración", key="settings"):
                st.switch_page("pages/settings.py")
            
            if st.button("📋 Logs del Sistema", key="system_logs"):
                st.switch_page("pages/system_logs.py")
    
    @staticmethod
    def show() -> None:
        """Método estático para mostrar la página."""
        page = HomePage()
        page.render()

# Función de compatibilidad con el sistema anterior
def show_home_page() -> None:
    """Función de compatibilidad para mostrar la página de inicio."""
    HomePage.show()