"""Funciones de demostración para backtesting y API serving."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time

from .hybrid_config import DataUsagePattern

def show_backtesting_demo(strategy, symbol, interval):
    """Demo de backtesting."""
    st.subheader("📊 Backtesting")
    st.markdown("""
    **Optimizado para:** Datos históricos completos, validación OHLC, continuidad temporal
    **Fuente de datos:** CSV → Base de datos
    """)
    
    if st.button("Obtener Datos para Backtesting", key="backtest"):
        with st.spinner("Preparando datos históricos..."):
            start_time = time.time()
            
            try:
                data = strategy.get_data(
                    pattern=DataUsagePattern.BACKTESTING,
                    symbol=symbol,
                    interval=interval
                )
                
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000
                
                if not data.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Registros", len(data))
                    with col2:
                        st.metric("Período", f"{len(data)} {interval}")
                    with col3:
                        st.metric("Tiempo (ms)", f"{execution_time:.1f}")
                    with col4:
                        if 'forward_return' in data.columns:
                            avg_return = data['forward_return'].mean() * 100
                            st.metric("Return Promedio %", f"{avg_return:.3f}")
                    
                    # Gráfico de velas
                    fig = go.Figure(data=go.Candlestick(
                        x=data['datetime'] if 'datetime' in data.columns else data.index,
                        open=data['open'] if 'open' in data.columns else data['open_price'],
                        high=data['high'] if 'high' in data.columns else data['high_price'],
                        low=data['low'] if 'low' in data.columns else data['low_price'],
                        close=data['close'] if 'close' in data.columns else data['close_price'],
                        name=symbol
                    ))
                    
                    fig.update_layout(
                        title=f"Datos Históricos - {symbol} ({interval})",
                        xaxis_title="Tiempo",
                        yaxis_title="Precio",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.warning("No se encontraron datos históricos")
                    
            except Exception as e:
                st.error(f"Error preparando datos de backtesting: {e}")

def show_api_serving_demo(strategy, symbol, interval):
    """Demo de API serving."""
    st.subheader("🌐 API Serving")
    st.markdown("""
    **Optimizado para:** Respuestas rápidas, formato JSON, datos compactos
    **Fuente de datos:** Cache → Base de datos
    """)
    
    if st.button("Obtener Datos para API", key="api"):
        with st.spinner("Preparando respuesta API..."):
            start_time = time.time()
            
            try:
                data = strategy.get_data(
                    pattern=DataUsagePattern.API_SERVING,
                    symbol=symbol,
                    interval=interval,
                    limit=50,
                    format_for_json=True
                )
                
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000
                
                if not data.empty:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Registros", len(data))
                    with col2:
                        st.metric("Tiempo (ms)", f"{execution_time:.1f}")
                    with col3:
                        # Estimar tamaño JSON
                        json_size = len(data.to_json()) / 1024
                        st.metric("Tamaño JSON (KB)", f"{json_size:.1f}")
                    
                    # Mostrar formato JSON
                    st.subheader("Formato de Respuesta API")
                    sample_data = data.head(3).to_dict('records')
                    st.json({
                        "status": "success",
                        "symbol": symbol,
                        "interval": interval,
                        "count": len(data),
                        "data": sample_data
                    })
                    
                    # Tabla de datos
                    st.subheader("Datos Completos")
                    st.dataframe(data)
                    
                else:
                    st.warning("No se encontraron datos")
                    
            except Exception as e:
                st.error(f"Error preparando respuesta API: {e}")