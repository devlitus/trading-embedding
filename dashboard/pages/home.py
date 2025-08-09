#!/usr/bin/env python3
"""
Página de inicio del dashboard
"""

import streamlit as st
from .common import show_system_metrics, CUSTOM_CSS

def show_home_page(data_manager):
    """Mostrar la página de inicio"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Título principal
    st.title("🏠 Dashboard de Trading - Inicio")
    
    # Descripción del sistema
    st.markdown("""
    ### Bienvenido al Sistema de Trading Integrado
    
    Este dashboard proporciona un análisis completo de datos de trading con las siguientes funcionalidades:
    
    **📊 Fase 1 - Adquisición de Datos:**
    - Conexión en tiempo real con Binance
    - Almacenamiento en base de datos SQLite
    - Sistema de caché para optimización
    
    **🔍 Fase 2 - Análisis Técnico:**
    - Indicadores técnicos avanzados
    - Análisis de patrones de precios
    - Visualizaciones interactivas
    
    **✅ Verificación del Sistema:**
    - Monitoreo de salud de componentes
    - Validación de datos
    - Métricas de rendimiento
    
    **📈 Análisis Técnico:**
    - RSI, MACD, Bandas de Bollinger
    - Medias móviles y tendencias
    - Señales de compra/venta
    
    **🎯 Monitoreo en Tiempo Real:**
    - Datos en vivo de mercado
    - Alertas automáticas
    - Dashboard interactivo
    
    **📋 Reportes:**
    - Análisis histórico
    - Métricas de rendimiento
    - Exportación de datos
    """)
    
    # Separador
    st.divider()
    
    # Métricas del sistema
    st.subheader("📊 Estado del Sistema")
    show_system_metrics(data_manager)
    
    # Separador
    st.divider()
    
    # Información de la base de datos
    st.subheader("💾 Información de la Base de Datos")
    
    if data_manager:
        try:
            db_stats = data_manager.get_database_stats()
            
            if db_stats and db_stats.get('symbols'):
                st.success(f"✅ Base de datos conectada con {db_stats['total_symbols']} símbolos")
                
                # Mostrar información de símbolos
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**Símbolos disponibles:**")
                    for symbol_info in db_stats['symbols']:
                        symbol = symbol_info['symbol']
                        count = symbol_info['count']
                        st.write(f"• {symbol}: {count:,} registros")
                
                with col2:
                    st.markdown("**Estadísticas generales:**")
                    st.write(f"• Total de registros: {db_stats.get('total_records', 0):,}")
                    st.write(f"• Símbolos únicos: {db_stats.get('total_symbols', 0)}")
                    
                    # Mostrar último registro si existe
                    if db_stats.get('latest_timestamp'):
                        st.write(f"• Último registro: {db_stats['latest_timestamp']}")
            else:
                st.warning("⚠️ No hay datos disponibles en la base de datos")
                
        except Exception as e:
            st.error(f"❌ Error accediendo a la base de datos: {str(e)}")
    else:
        st.error("❌ No se pudo conectar al gestor de datos")
    
    # Información adicional
    st.divider()
    st.subheader("ℹ️ Información del Sistema")
    
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        **Tecnologías utilizadas:**
        - Python 3.x
        - Streamlit
        - SQLite
        - Plotly
        - Pandas
        - Binance API
        """)
    
    with info_col2:
        st.markdown("""
        **Características:**
        - Datos en tiempo real
        - Análisis técnico avanzado
        - Interfaz web interactiva
        - Sistema de caché optimizado
        - Monitoreo de salud
        """)