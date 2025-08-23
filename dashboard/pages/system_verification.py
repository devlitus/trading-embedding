#!/usr/bin/env python3
"""
Página de Verificación del Sistema
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from dashboard.pages.common import get_plotly_config, CUSTOM_CSS

def show_system_verification_page(data_manager, verification_system):
    """Mostrar la página de Verificación del Sistema"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    st.title("✅ Verificación del Sistema")
    
    if not verification_system:
        st.error("Error: Sistema de verificación no disponible")
        st.stop()
    
    # Tabs para diferentes tipos de verificación
    tab1, tab2, tab3, tab4 = st.tabs(["🔍 Verificación Rápida", "🔧 Verificación Completa", "🧪 Pruebas de Escenarios", "📊 Estado del Sistema"])
    
    with tab1:
        st.subheader("🔍 Verificación Rápida")
        st.write("Ejecuta una verificación básica de todos los componentes principales.")
        
        if st.button("▶️ Ejecutar Verificación Rápida", type="primary"):
            with st.spinner("Ejecutando verificación rápida..."):
                try:
                    result = verification_system.run_quick_verification()
                    
                    if result['success']:
                        st.success("✅ Verificación rápida completada exitosamente")
                        
                        # Mostrar resultados
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Componentes OK", f"{result.get('passed', 0)}/{result.get('total', 0)}")
                        
                        with col2:
                            st.metric("Tiempo Total", f"{result.get('execution_time', 0):.2f}s")
                        
                        with col3:
                            status = "🟢 Saludable" if result.get('passed', 0) == result.get('total', 0) else "🟡 Advertencias"
                            st.metric("Estado General", status)
                        
                        # Detalles de cada componente
                        st.subheader("📋 Detalles por Componente")
                        
                        for component in result.get('components', []):
                            if component.get('passed', False):
                                st.success(f"✅ {component.get('name', 'Componente')}: {component.get('message', 'OK')}")
                            else:
                                st.error(f"❌ {component.get('name', 'Componente')}: {component.get('error', 'Error')}")
                    
                    else:
                        st.error(f"❌ Error en verificación rápida: {result.get('error', 'Error desconocido')}")
                        
                except Exception as e:
                    st.error(f"Error ejecutando verificación rápida: {e}")
    
    with tab2:
        st.subheader("🔧 Verificación Completa")
        st.write("Ejecuta una verificación exhaustiva de todos los sistemas, incluyendo pruebas de rendimiento y integridad.")
        
        # Opciones de verificación
        include_performance = st.checkbox("Incluir pruebas de rendimiento", value=True)
        include_data_integrity = st.checkbox("Verificar integridad de datos", value=True)
        include_api_tests = st.checkbox("Probar conexiones API", value=True)
        
        if st.button("🔧 Ejecutar Verificación Completa", type="primary"):
            with st.spinner("Ejecutando verificación completa... Esto puede tomar varios minutos."):
                try:
                    result = verification_system.run_complete_verification(
                        include_performance=include_performance,
                        include_data_quality=include_data_integrity,
                        include_ml_models=include_api_tests
                    )
                    
                    if result['success']:
                        st.success("✅ Verificación completa finalizada")
                        
                        # Métricas principales
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Score General", f"{result.get('overall_score', 0):.1f}/10")
                        
                        with col2:
                            st.metric("Componentes OK", f"{result.get('components_passed', 0)}/{result.get('total_components', 0)}")
                        
                        with col3:
                            st.metric("Tiempo Total", f"{result.get('total_time', 0):.1f}s")
                        
                        with col4:
                            status = "🟢 Saludable" if result.get('overall_score', 0) >= 8 else "🟡 Advertencias" if result.get('overall_score', 0) >= 6 else "🔴 Crítico"
                            st.metric("Estado", status)
                        
                        # Resultados por categoría
                        st.subheader("📊 Resultados por Categoría")
                        
                        categories = result.get('categories', {})
                        for category, data in categories.items():
                            with st.expander(f"📁 {category.title()} - Score: {data.get('score', 0):.1f}/10"):
                                
                                tests = data.get('tests', [])
                                for test in tests:
                                    if test.get('passed', False):
                                        st.success(f"✅ {test.get('name', 'Test')}: {test.get('message', 'OK')}")
                                    else:
                                        st.error(f"❌ {test.get('name', 'Test')}: {test.get('error', 'Error')}")
                                
                                # Métricas específicas
                                if 'metrics' in data:
                                    st.write("**Métricas:**")
                                    metrics_df = pd.DataFrame([data['metrics']])
                                    st.dataframe(metrics_df, use_container_width=True)
                        
                        # Recomendaciones
                        if 'recommendations' in result:
                            st.subheader("💡 Recomendaciones")
                            for rec in result['recommendations']:
                                st.info(f"💡 {rec}")
                    
                    else:
                        st.error(f"❌ Error en verificación completa: {result.get('error', 'Error desconocido')}")
                        
                except Exception as e:
                    st.error(f"Error ejecutando verificación completa: {e}")
    
    with tab3:
        st.subheader("🧪 Pruebas de Escenarios")
        st.write("Ejecuta pruebas específicas de escenarios de trading.")
        
        scenario_type = st.selectbox(
            "Tipo de Escenario",
            ["Mercado Alcista", "Mercado Bajista", "Mercado Lateral", "Alta Volatilidad", "Baja Volatilidad"]
        )
        
        symbol_for_scenario = st.selectbox(
            "Símbolo para Prueba",
            ["BTCUSDT", "ETHUSDT", "ADAUSDT"],
            key="scenario_symbol"
        )
        
        if st.button("🧪 Ejecutar Prueba de Escenario", type="primary"):
            with st.spinner(f"Ejecutando escenario: {scenario_type}..."):
                try:
                    result = verification_system.run_scenario_test(
                        scenario_type=scenario_type.lower().replace(" ", "_"),
                        symbol=symbol_for_scenario
                    )
                    
                    if result['success']:
                        st.success(f"✅ Escenario '{scenario_type}' ejecutado exitosamente")
                        
                        # Resultados del escenario
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Precisión", f"{result.get('accuracy', 0):.2f}%")
                        
                        with col2:
                            st.metric("Señales Generadas", result.get('signals_generated', 0))
                        
                        with col3:
                            st.metric("Tiempo de Respuesta", f"{result.get('response_time', 0):.3f}s")
                        
                        # Detalles del análisis
                        if 'analysis' in result:
                            st.subheader("📊 Análisis del Escenario")
                            analysis = result['analysis']
                            
                            # Gráfico de resultados
                            if 'price_data' in analysis:
                                fig = go.Figure()
                                
                                price_data = analysis['price_data']
                                fig.add_trace(go.Scatter(
                                    x=list(range(len(price_data))),
                                    y=price_data,
                                    mode='lines',
                                    name='Precio',
                                    line=dict(color='blue')
                                ))
                                
                                # Agregar señales si existen
                                if 'signals' in analysis:
                                    signals = analysis['signals']
                                    buy_signals = [i for i, s in enumerate(signals) if s == 'buy']
                                    sell_signals = [i for i, s in enumerate(signals) if s == 'sell']
                                    
                                    if buy_signals:
                                        fig.add_trace(go.Scatter(
                                            x=buy_signals,
                                            y=[price_data[i] for i in buy_signals],
                                            mode='markers',
                                            name='Señales Compra',
                                            marker=dict(color='green', size=10, symbol='triangle-up')
                                        ))
                                    
                                    if sell_signals:
                                        fig.add_trace(go.Scatter(
                                            x=sell_signals,
                                            y=[price_data[i] for i in sell_signals],
                                            mode='markers',
                                            name='Señales Venta',
                                            marker=dict(color='red', size=10, symbol='triangle-down')
                                        ))
                                
                                fig.update_layout(
                                    title=f"Análisis de Escenario: {scenario_type}",
                                    xaxis_title="Tiempo",
                                    yaxis_title="Precio",
                                    height=400
                                )
                                
                                st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())
                            
                            # Métricas detalladas
                            if 'metrics' in analysis:
                                st.write("**Métricas Detalladas:**")
                                metrics_df = pd.DataFrame([analysis['metrics']])
                                st.dataframe(metrics_df, use_container_width=True)
                    
                    else:
                        st.error(f"❌ Error en escenario: {result.get('error', 'Error desconocido')}")
                        
                except Exception as e:
                    st.error(f"Error ejecutando escenario: {e}")
    
    with tab4:
        st.subheader("📊 Estado del Sistema")
        st.write("Monitoreo en tiempo real del estado de todos los componentes.")
        
        # Auto-refresh
        auto_refresh = st.checkbox("Auto-actualizar cada 30 segundos", value=False)
        
        if auto_refresh:
            time.sleep(30)
            st.rerun()
        
        if st.button("🔄 Actualizar Estado") or auto_refresh:
            try:
                # Estado de componentes
                component_status = verification_system.get_component_status()
                
                st.subheader("🔧 Estado de Componentes")
                
                for component, status in component_status.items():
                    col1, col2, col3 = st.columns([2, 1, 1])
                    
                    with col1:
                        st.write(f"**{component}**")
                    
                    with col2:
                        if status.get('healthy', False):
                            st.success("🟢 Online")
                        else:
                            st.error("🔴 Error")
                    
                    with col3:
                        response_time = status.get('response_time', 0)
                        st.write(f"{response_time:.3f}s")
                    
                    # Detalles adicionales
                    if not status.get('healthy', False) and 'error' in status:
                        st.error(f"Error: {status['error']}")
                
                # Métricas del sistema
                st.subheader("📈 Métricas del Sistema")
                
                system_metrics = verification_system.get_system_metrics()
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("CPU Usage", f"{system_metrics.get('cpu_percent', 0):.1f}%")
                
                with col2:
                    st.metric("Memory Usage", f"{system_metrics.get('memory_percent', 0):.1f}%")
                
                with col3:
                    st.metric("Disk Usage", f"{system_metrics.get('disk_percent', 0):.1f}%")
                
                with col4:
                    st.metric("Uptime", f"{system_metrics.get('uptime_hours', 0):.1f}h")
                
                # Gráfico de métricas históricas
                if 'historical_metrics' in system_metrics:
                    st.subheader("📊 Métricas Históricas")
                    
                    historical = system_metrics['historical_metrics']
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=historical['timestamps'],
                        y=historical['cpu'],
                        mode='lines',
                        name='CPU %',
                        line=dict(color='red')
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=historical['timestamps'],
                        y=historical['memory'],
                        mode='lines',
                        name='Memory %',
                        line=dict(color='blue')
                    ))
                    
                    fig.update_layout(
                        title="Uso de Recursos del Sistema",
                        xaxis_title="Tiempo",
                        yaxis_title="Porcentaje (%)",
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())
                
            except Exception as e:
                st.error(f"Error obteniendo estado del sistema: {e}")