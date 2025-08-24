#!/usr/bin/env python3
"""
Página de Fase 2 - Análisis Técnico
"""

from .phase2_config import apply_custom_css, show_page_header, create_analysis_controls, create_analysis_button
from .phase2_logic import check_data_manager, process_analysis_request
from .phase2_ui import show_analysis_summary, show_technical_indicators, show_detected_patterns, create_analysis_chart, show_detailed_analysis

def show_phase2_analysis_page(data_manager):
    """Mostrar la página de Fase 2 - Análisis Técnico"""
    apply_custom_css()
    show_page_header()
    
    check_data_manager(data_manager)
    
    # Configuración del análisis
    symbol, interval = create_analysis_controls()
    
    if create_analysis_button():
        # Procesar solicitud de análisis
        df, analysis_result = process_analysis_request(data_manager, symbol, interval)
        
        if df is not None and analysis_result is not None:
            # Mostrar resultados del análisis
            show_analysis_summary(analysis_result, df, symbol)
            show_technical_indicators(analysis_result.indicators)
            show_detected_patterns(analysis_result.patterns)
            create_analysis_chart(df, symbol, interval, analysis_result)
            show_detailed_analysis(analysis_result)