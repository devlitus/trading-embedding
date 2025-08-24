#!/usr/bin/env python3
"""
Módulo de página de inicio
"""

from .home_ui import HomeUI

def show_home_page(data_manager=None):
    """
    Función principal para mostrar la página de inicio
    
    Args:
        data_manager: Instancia del gestor de datos (opcional)
    """
    home_ui = HomeUI(data_manager)
    home_ui.render_main_page()

# Exportar la función principal
__all__ = ['show_home_page']