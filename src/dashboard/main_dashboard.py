"""Dashboard principal usando estructura modular."""

import streamlit as st
import sys
from pathlib import Path

# Agregar el directorio raíz al path para imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Imports del sistema
try:
    from src.data.data_manager import DataManager
    from src.data.data_strategy import DataStrategy
    from src.verification.phase2_verification import Phase2VerificationSystem
except ImportError as e:
    st.error(f"Error al importar módulos del sistema: {e}")
    st.stop()

# Imports del dashboard modular
from .components import DashboardLayout
from .pages import (
    HomePage,
    MarketDataPage
)

class MainDashboard:
    """Clase principal del dashboard modular."""
    
    def __init__(self):
        """Inicializa el dashboard principal."""
        self.setup_page_config()
        self.initialize_systems()
        self.setup_navigation()
    
    def setup_page_config(self) -> None:
        """Configura la página de Streamlit."""
        DashboardLayout.setup_page_config(
            title="Trading Embedding System",
            icon="📈",
            layout="wide"
        )
        
        # Aplicar CSS personalizado
        DashboardLayout.apply_custom_css()
    
    def initialize_systems(self) -> None:
        """Inicializa los sistemas principales."""
        try:
            # Inicializar sistemas en session_state si no existen
            if 'data_manager' not in st.session_state:
                with st.spinner("Inicializando sistema de datos..."):
                    st.session_state.data_manager = DataManager()
            
            if 'data_strategy' not in st.session_state:
                with st.spinner("Configurando estrategia de datos..."):
                    st.session_state.data_strategy = DataStrategy()
            
            if 'verification_system' not in st.session_state:
                with st.spinner("Inicializando sistema de verificación..."):
                    st.session_state.verification_system = Phase2VerificationSystem()
                    
        except Exception as e:
            st.error(f"Error al inicializar sistemas: {e}")
            # Continuar sin los sistemas para permitir navegación básica
    
    def setup_navigation(self) -> None:
        """Configura la navegación del dashboard."""
        # Definir páginas disponibles
        self.pages = {
            "🏠 Inicio": HomePage,
            "📈 Datos de Mercado": MarketDataPage,
            "📊 Análisis Técnico": self.placeholder_page,
            "🧠 Embeddings": self.placeholder_page,
            "🔮 Predicciones": self.placeholder_page,
            "📋 Verificación": self.placeholder_page,
            "⚙️ Configuración": self.placeholder_page,
            "📝 Logs del Sistema": self.placeholder_page
        }
    
    def placeholder_page(self) -> None:
        """Página placeholder para funcionalidades en desarrollo."""
        st.title("🚧 En Desarrollo")
        st.info("Esta funcionalidad está siendo migrada a la nueva estructura modular.")
        st.markdown("""
        ### 📋 Estado de Migración
        
        ✅ **Completado:**
        - Estructura modular del dashboard
        - Página de inicio
        - Página de datos de mercado
        - Componentes base (layout, métricas, gráficos)
        - Servicios de datos y análisis
        - Utilidades de formateo y validación
        
        🚧 **En Progreso:**
        - Migración de páginas restantes
        - Integración con sistemas existentes
        - Pruebas y optimización
        
        📋 **Pendiente:**
        - Página de análisis técnico
        - Página de embeddings
        - Página de predicciones
        - Página de verificación
        - Página de configuración
        - Página de logs del sistema
        """)
    
    def render_sidebar(self) -> str:
        """Renderiza la barra lateral y retorna la página seleccionada.
        
        Returns:
            Nombre de la página seleccionada
        """
        with st.sidebar:
            st.title("🚀 Trading System")
            st.markdown("---")
            
            # Información del sistema
            self.render_system_info()
            
            # Navegación
            st.subheader("📋 Navegación")
            selected_page = st.radio(
                "Seleccionar página:",
                options=list(self.pages.keys()),
                key="page_selector"
            )
            
            st.markdown("---")
            
            # Footer
            DashboardLayout.show_footer()
            
            return selected_page
    
    def render_system_info(self) -> None:
        """Renderiza información del sistema en la barra lateral."""
        st.subheader("ℹ️ Sistema")
        
        # Estado de los sistemas
        data_manager_status = "✅" if 'data_manager' in st.session_state else "❌"
        verification_status = "✅" if 'verification_system' in st.session_state else "❌"
        
        st.markdown(f"""
        **Estado de Componentes:**
        - Data Manager: {data_manager_status}
        - Verificación: {verification_status}
        - Dashboard: ✅
        """)
        
        # Botón de reinicio
        if st.button("🔄 Reiniciar Sistema", key="restart_system"):
            # Limpiar session state
            for key in list(st.session_state.keys()):
                if key.startswith(('data_manager', 'verification_system', 'data_strategy')):
                    del st.session_state[key]
            st.rerun()
    
    def run(self) -> None:
        """Ejecuta el dashboard principal."""
        try:
            # Renderizar barra lateral y obtener página seleccionada
            selected_page = self.render_sidebar()
            
            # Renderizar página seleccionada
            page_class = self.pages.get(selected_page)
            if page_class:
                if hasattr(page_class, 'show'):
                    page_class.show()
                else:
                    page_class()
            else:
                st.error(f"Página no encontrada: {selected_page}")
                
        except Exception as e:
            st.error(f"Error en el dashboard: {e}")
            st.exception(e)

def main():
    """Función principal para ejecutar el dashboard."""
    dashboard = MainDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()