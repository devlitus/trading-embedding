"""Funciones de demostración para dashboard y comparación de rendimiento."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

from .hybrid_config import DataUsagePattern

def show_dashboard_demo(strategy, symbol, interval):
    """Demo de dashboard."""
    st.subheader("📈 Dashboard")
    st.markdown("""
    **Optimizado para:** Visualizaciones enriquecidas, múltiples métricas, interactividad
    **Fuente de datos:** Híbrido optimizado
    """)
    
    if st.button("Obtener Datos para Dashboard", key="dashboard"):
        with st.spinner("Preparando datos para visualización..."):
            start_time = time.time()
            
            try:
                data = strategy.get_data(
                    pattern=DataUsagePattern.DASHBOARD,
                    symbol=symbol,
                    interval=interval,
                    enrich_data=True
                )
                
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000
                
                if not data.empty:
                    # Métricas principales
                    col1, col2, col3, col4 = st.columns(4)
                    
                    close_col = 'close' if 'close' in data.columns else 'close_price'
                    
                    with col1:
                        st.metric("Registros", len(data))
                    with col2:
                        st.metric("Precio Actual", f"${data[close_col].iloc[-1]:.2f}")
                    with col3:
                        if len(data) > 1:
                            price_change = ((data[close_col].iloc[-1] / data[close_col].iloc[-2]) - 1) * 100
                            st.metric("Cambio %", f"{price_change:.2f}%")
                    with col4:
                        st.metric("Tiempo (ms)", f"{execution_time:.1f}")
                    
                    # Gráfico principal con múltiples indicadores
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=['Precio y Indicadores', 'Volumen'],
                        row_heights=[0.7, 0.3]
                    )
                    
                    time_col = 'datetime' if 'datetime' in data.columns else data.index
                    
                    # Precio
                    fig.add_trace(
                        go.Scatter(
                            x=time_col,
                            y=data[close_col],
                            mode='lines',
                            name='Precio',
                            line=dict(color='#1f77b4', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # Volumen
                    volume_col = 'volume' if 'volume' in data.columns else 'volume'
                    if volume_col in data.columns:
                        fig.add_trace(
                            go.Bar(
                                x=time_col,
                                y=data[volume_col],
                                name='Volumen',
                                marker_color='lightblue'
                            ),
                            row=2, col=1
                        )
                    
                    fig.update_layout(
                        title=f"Dashboard Completo - {symbol} ({interval})",
                        height=600,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.warning("No se encontraron datos")
                    
            except Exception as e:
                st.error(f"Error preparando datos de dashboard: {e}")

def show_performance_comparison(strategy, symbol, interval):
    """Muestra comparación de rendimiento entre patrones."""
    
    if st.button("Ejecutar Comparación de Rendimiento", key="performance"):
        with st.spinner("Ejecutando benchmarks..."):
            results = []
            patterns = [
                (DataUsagePattern.REALTIME_TRADING, "Trading Tiempo Real"),
                (DataUsagePattern.ML_TRAINING, "Entrenamiento ML"),
                (DataUsagePattern.BACKTESTING, "Backtesting"),
                (DataUsagePattern.API_SERVING, "API Serving"),
                (DataUsagePattern.DASHBOARD, "Dashboard")
            ]
            
            for pattern, name in patterns:
                try:
                    start_time = time.time()
                    
                    data = strategy.get_data(
                        pattern=pattern,
                        symbol=symbol,
                        interval=interval,
                        limit=100 if pattern in [DataUsagePattern.REALTIME_TRADING, DataUsagePattern.API_SERVING] else None
                    )
                    
                    end_time = time.time()
                    execution_time = (end_time - start_time) * 1000
                    
                    results.append({
                        'Patrón': name,
                        'Tiempo (ms)': execution_time,
                        'Registros': len(data) if not data.empty else 0,
                        'Columnas': len(data.columns) if not data.empty else 0
                    })
                    
                except Exception as e:
                    results.append({
                        'Patrón': name,
                        'Tiempo (ms)': 0,
                        'Registros': 0,
                        'Columnas': 0,
                        'Error': str(e)
                    })
            
            # Mostrar resultados
            df_results = pd.DataFrame(results)
            
            # Tabla de resultados
            st.subheader("Tabla de Resultados")
            st.dataframe(df_results)
            
            # Gráfico de rendimiento debajo de la tabla
            st.subheader("Gráfico de Rendimiento")
            fig = px.bar(
                df_results,
                x='Patrón',
                y='Tiempo (ms)',
                title="Tiempo de Ejecución por Patrón",
                color='Tiempo (ms)',
                color_continuous_scale='viridis'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recomendaciones
            fastest = df_results.loc[df_results['Tiempo (ms)'].idxmin()]
            st.success(f"🏆 **Patrón más rápido:** {fastest['Patrón']} ({fastest['Tiempo (ms)']:.1f}ms)")
            
            st.info("""
            **💡 Recomendaciones de uso:**
            - **Trading en tiempo real:** Usa para decisiones rápidas con datos recientes
            - **Entrenamiento ML:** Usa para datasets completos con features avanzadas
            - **Backtesting:** Usa para análisis histórico con validación de datos
            - **API Serving:** Usa para respuestas web optimizadas
            - **Dashboard:** Usa para visualizaciones enriquecidas
            """)