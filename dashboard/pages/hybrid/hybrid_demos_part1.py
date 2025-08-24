"""Funciones de demostración para trading en tiempo real y entrenamiento ML."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime

from .hybrid_config import DataUsagePattern

def show_realtime_trading_demo(strategy, symbol, interval):
    """Demo de trading en tiempo real."""
    st.subheader("⚡ Trading en Tiempo Real")
    st.markdown("""
    **Optimizado para:** Decisiones rápidas, datos más recientes, baja latencia
    **Fuente de datos:** API → Cache → Base de datos
    """)
    
    if st.button("Obtener Datos en Tiempo Real", key="realtime"):
        with st.spinner("Obteniendo datos más recientes..."):
            start_time = time.time()
            
            try:
                data = strategy.get_data(
                    pattern=DataUsagePattern.REALTIME_TRADING,
                    symbol=symbol,
                    interval=interval,
                    limit=100
                )
                
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000
                
                if not data.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    # Determinar nombres de columnas
                    close_col = 'close' if 'close' in data.columns else 'close_price'
                    volume_col = 'volume' if 'volume' in data.columns else 'volume'
                    
                    with col1:
                        st.metric("Registros", len(data))
                    with col2:
                        st.metric("Precio Actual", f"${data[close_col].iloc[-1]:.2f}")
                    with col3:
                        st.metric("Tiempo (ms)", f"{execution_time:.1f}")
                    with col4:
                        if len(data) > 1:
                            price_change = ((data[close_col].iloc[-1] / data[close_col].iloc[-2]) - 1) * 100
                            st.metric("Cambio %", f"{price_change:.2f}%")
                    
                    # Gráfico de precio y volumen
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=['Precio', 'Volumen'],
                        row_heights=[0.7, 0.3]
                    )
                    
                    time_col = 'datetime' if 'datetime' in data.columns else data.index
                    
                    # Precio
                    fig.add_trace(
                        go.Scatter(
                            x=time_col,
                            y=data[close_col],
                            mode='lines+markers',
                            name='Precio',
                            line=dict(color='#00ff00', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # Volumen
                    if volume_col in data.columns:
                        fig.add_trace(
                            go.Bar(
                                x=time_col,
                                y=data[volume_col],
                                name='Volumen',
                                marker_color='lightgreen'
                            ),
                            row=2, col=1
                        )
                    
                    fig.update_layout(
                        title=f"Trading en Tiempo Real - {symbol} ({interval})",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar muestra de datos
                    st.subheader("Muestra de Datos")
                    st.dataframe(data.tail(10))
                    
                else:
                    st.warning("No se encontraron datos")
                    
            except Exception as e:
                st.error(f"Error obteniendo datos en tiempo real: {e}")

def show_ml_training_demo(strategy, symbol, interval):
    """Demo de entrenamiento ML."""
    st.subheader("🤖 Entrenamiento ML")
    st.markdown("""
    **Optimizado para:** Datasets completos, features avanzadas, consistencia
    **Fuente de datos:** Base de datos → CSV → API
    """)
    
    if st.button("Preparar Datos para ML", key="ml"):
        with st.spinner("Preparando dataset completo..."):
            start_time = time.time()
            
            try:
                data = strategy.get_data(
                    pattern=DataUsagePattern.ML_TRAINING,
                    symbol=symbol,
                    interval=interval,
                    include_features=True,
                    include_labels=True
                )
                
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000
                
                if not data.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Registros", len(data))
                    with col2:
                        st.metric("Features", len(data.columns))
                    with col3:
                        st.metric("Tiempo (ms)", f"{execution_time:.1f}")
                    with col4:
                        # Calcular completitud de datos
                        completeness = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
                        st.metric("Completitud %", f"{completeness:.1f}")
                    
                    # Análisis de features
                    st.subheader("Análisis de Features")
                    
                    # Distribución de features numéricas
                    numeric_cols = data.select_dtypes(include=['float64', 'int64']).columns
                    if len(numeric_cols) > 0:
                        selected_features = st.multiselect(
                            "Seleccionar features para visualizar:",
                            numeric_cols.tolist(),
                            default=numeric_cols[:4].tolist()
                        )
                        
                        if selected_features:
                            fig = make_subplots(
                                rows=2, cols=2,
                                subplot_titles=selected_features[:4]
                            )
                            
                            for i, feature in enumerate(selected_features[:4]):
                                row = (i // 2) + 1
                                col = (i % 2) + 1
                                
                                fig.add_trace(
                                    go.Histogram(
                                        x=data[feature].dropna(),
                                        name=feature,
                                        showlegend=False
                                    ),
                                    row=row, col=col
                                )
                            
                            fig.update_layout(
                                title="Distribución de Features",
                                height=400
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Información del dataset
                    st.subheader("Información del Dataset")
                    info_data = {
                        'Columna': data.columns,
                        'Tipo': [str(dtype) for dtype in data.dtypes],
                        'No Nulos': [data[col].count() for col in data.columns],
                        'Nulos': [data[col].isnull().sum() for col in data.columns]
                    }
                    st.dataframe(pd.DataFrame(info_data))
                    
                else:
                    st.warning("No se encontraron datos")
                    
            except Exception as e:
                st.error(f"Error preparando datos ML: {e}")