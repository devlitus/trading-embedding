#!/usr/bin/env python3
"""
Página de Monitoreo en Tiempo Real

Este módulo ha sido refactorizado y modularizado en:
- realtime_config.py: Configuración y constantes
- realtime_logic.py: Lógica de negocio y procesamiento de datos
- realtime_ui.py: Componentes de interfaz de usuario
"""

from .realtime_ui import RealtimeUI

def show_realtime_monitoring_page(data_manager):
    """Mostrar la página de Monitoreo en Tiempo Real"""
    # Crear instancia de UI y renderizar página principal
    realtime_ui = RealtimeUI(data_manager)
    realtime_ui.render_main_page()
    
    # Todas las funcionalidades han sido modularizadas en:
    # - realtime_config.py: Configuración y constantes
    # - realtime_logic.py: Lógica de negocio y procesamiento de datos  
    # - realtime_ui.py: Componentes de interfaz de usuario