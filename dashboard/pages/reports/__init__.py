#!/usr/bin/env python3
"""
Módulo de reportes - Punto de entrada principal.

Este módulo exporta la función principal para mostrar la página de reportes.
"""

from .reports_ui import ReportsUI


def show_reports_page(data_manager=None):
    """
    Función principal para mostrar la página de reportes.
    
    Args:
        data_manager: Gestor de datos opcional
    """
    reports_ui = ReportsUI(data_manager)
    reports_ui.render_main_page()


# Exportar la función principal
__all__ = ['show_reports_page']