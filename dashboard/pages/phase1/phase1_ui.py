#!/usr/bin/env python3
"""
Componentes de UI para Fase 1 - Adquisición de Datos
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, date
from pages.common import get_plotly_config
from .phase1_config import CHART_HEIGHT, STATS_TABLE_HEIGHT

def show_status_row(data_manager, symbol, interval):
    """Mostrar fila de estado del sistema."""
    st.markdown("**📊 Estado del Sistema**")
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        _show_data_status(data_manager, symbol, interval)
    
    with status_col2:
        _show_visualization_status(data_manager, symbol, interval)
    
    with status_col3:
        _show_health_check_status(data_manager)

def _show_data_status(data_manager, symbol, interval):
    """Mostrar estado de los datos."""
    st.markdown("**🔄 Estado de Datos**")
    try:
        result = data_manager.fetch_and_store_data(
            symbol=symbol,
            interval=interval,
            days_back=1
        )
        
        if 'error' in result:
            st.error(f"❌ Error")
            st.caption(f"Error: {result['error'][:30]}...")
        else:
            records_count = result.get('records_count', 0)
            inserted_count = result.get('inserted_count', records_count)
            st.success(f"✅ {records_count:,} registros")
            st.caption(f"({inserted_count:,} nuevos)")
    except Exception as e:
        st.error("❌ Error")
        st.caption(f"Error: {str(e)[:30]}...")

def _show_visualization_status(data_manager, symbol, interval):
    """Mostrar estado de visualización."""
    st.markdown("**📊 Visualización**")
    try:
        df = data_manager.get_data(symbol, interval, limit=200)
        if not df.empty:
            current_price = df['close'].iloc[-1]
            change_pct = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100)
            st.success(f"✅ ${current_price:.4f}")
            color = "green" if change_pct >= 0 else "red"
            st.markdown(f"<span style='color: {color}'>{change_pct:+.2f}%</span>", unsafe_allow_html=True)
        else:
            st.warning("⚠️ Sin datos")
            st.caption("No disponible")
    except Exception as e:
        st.error("❌ Error")
        st.caption(f"Error: {str(e)[:30]}...")

def _show_health_check_status(data_manager):
    """Mostrar estado del health check."""
    st.markdown("**🔍 Health Check**")
    try:
        health = data_manager.health_check()
        all_healthy = all(component['status'] == 'healthy' for component in health['components'].values())
        
        if all_healthy:
            st.success("✅ Sistema OK")
            st.caption("Todo operativo")
        else:
            st.error("❌ Problemas")
            st.caption("Ver detalles abajo")
    except Exception as e:
        st.error("❌ Error")
        st.caption(f"Error: {str(e)[:30]}...")

def create_main_chart(df, symbol, interval):
    """Crear gráfico principal de velas con volumen."""
    fig = go.Figure()
    
    # Candlestick principal
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name=symbol,
        increasing_line_color='#00ff88',
        decreasing_line_color='#ff4444'
    ))
    
    # Agregar volumen como gráfico secundario
    fig.add_trace(go.Bar(
        x=df.index,
        y=df['volume'],
        name='Volumen',
        yaxis='y2',
        opacity=0.3,
        marker_color='#1f77b4'
    ))
    
    # Layout optimizado
    fig.update_layout(
        title={
            'text': f"{symbol} - {interval} | Precio: ${df['close'].iloc[-1]:.4f} | Cambio: {((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100):.2f}%",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20}
        },
        yaxis=dict(
            title="Precio (USDT)",
            side='left',
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)'
        ),
        yaxis2=dict(
            title="Volumen",
            overlaying='y',
            side='right',
            showgrid=False
        ),
        xaxis=dict(
            title="Tiempo",
            showgrid=True,
            gridcolor='rgba(128,128,128,0.2)',
            rangeslider=dict(visible=False)
        ),
        height=CHART_HEIGHT,
        hovermode='x unified',
        template='plotly_white',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    return fig

def show_quick_stats(df):
    """Mostrar estadísticas rápidas del dataset."""
    stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
    
    with stats_col1:
        st.metric("Precio Actual", f"${df['close'].iloc[-1]:.4f}")
    with stats_col2:
        change_pct = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100)
        st.metric("Cambio Total", f"{change_pct:.2f}%", delta=f"{change_pct:.2f}%")
    with stats_col3:
        st.metric("Máximo", f"${df['high'].max():.4f}")
    with stats_col4:
        st.metric("Mínimo", f"${df['low'].min():.4f}")
    with stats_col5:
        st.metric("Volumen Prom.", f"{df['volume'].mean():,.0f}")

def show_health_check_details(data_manager):
    """Mostrar detalles del health check si hay problemas."""
    try:
        health = data_manager.health_check()
        all_healthy = all(component['status'] == 'healthy' for component in health['components'].values())
        
        if not all_healthy:
            st.warning("⚠️ Se detectaron problemas en el sistema")
            with st.expander("Detalles del Health Check"):
                st.json(health)
    except Exception as e:
        st.error(f"Error en health check: {e}")

def show_symbols_table(db_stats):
    """Mostrar tabla de símbolos disponibles."""
    if db_stats.get('symbol_details'):
        st.markdown("#### 📋 Símbolos Disponibles")
        
        symbols_data = []
        for sym, info in db_stats['symbol_details'].items():
            symbols_data.append({
                'Símbolo': sym,
                'Registros': f"{info['total_records']:,}",
                'Último': info.get('latest_timestamp', 'N/A')[:10] if info.get('latest_timestamp') != 'N/A' else 'N/A'
            })
        
        symbols_df = pd.DataFrame(symbols_data)
        st.dataframe(symbols_df, use_container_width=True, height=STATS_TABLE_HEIGHT)