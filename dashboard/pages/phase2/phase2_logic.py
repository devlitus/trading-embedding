#!/usr/bin/env python3
"""
Lógica de análisis para Fase 2 - Análisis Técnico
"""

import streamlit as st
from pages.common import analyze_symbol

def check_data_manager(data_manager):
    """Verificar disponibilidad del DataManager"""
    if not data_manager:
        st.error("Error: DataManager no disponible")
        st.stop()
        return False
    return True

def get_analysis_data(data_manager, symbol, interval, limit=200):
    """Obtener datos para análisis"""
    try:
        df = data_manager.get_data(symbol, interval, limit=limit)
        
        if df.empty:
            st.warning(f"No hay datos disponibles para {symbol}. Obtén datos primero en la Fase 1.")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error al obtener datos: {e}")
        return None

def perform_technical_analysis(df):
    """Ejecutar análisis técnico completo"""
    try:
        analysis_result = analyze_symbol(df)
        return analysis_result
    except Exception as e:
        st.error(f"Error en el análisis técnico: {e}")
        raise e

def validate_analysis_result(analysis_result):
    """Validar resultado del análisis"""
    if not analysis_result:
        st.error("No se pudo completar el análisis")
        return False
    
    # Verificar que tenga los componentes básicos
    if not hasattr(analysis_result, 'trend_analysis'):
        st.warning("Análisis de tendencia no disponible")
    
    if not hasattr(analysis_result, 'indicators'):
        st.warning("Indicadores técnicos no disponibles")
    
    if not hasattr(analysis_result, 'patterns'):
        st.warning("Patrones no disponibles")
    
    return True

def process_analysis_request(data_manager, symbol, interval):
    """Procesar solicitud completa de análisis"""
    with st.spinner(f"Analizando {symbol}..."):
        try:
            # Obtener datos
            df = get_analysis_data(data_manager, symbol, interval)
            if df is None:
                return None, None
            
            # Ejecutar análisis técnico
            analysis_result = perform_technical_analysis(df)
            
            # Validar resultado
            if not validate_analysis_result(analysis_result):
                return None, None
            
            return df, analysis_result
            
        except Exception as e:
            st.error(f"Error en el análisis: {e}")
            st.exception(e)
            return None, None