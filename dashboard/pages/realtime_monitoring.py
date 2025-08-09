#!/usr/bin/env python3
"""
Página de Monitoreo en Tiempo Real
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import time
from .common import get_plotly_config, CUSTOM_CSS, analyze_symbol

def show_realtime_monitoring_page(data_manager):
    """Mostrar la página de Monitoreo en Tiempo Real"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    st.title("🎯 Monitoreo en Tiempo Real")
    
    if not data_manager:
        st.error("Error: DataManager no disponible")
        st.stop()
    
    # Configuración del monitoreo
    st.subheader("⚙️ Configuración del Monitoreo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbols_to_monitor = st.multiselect(
            "Símbolos a Monitorear",
            ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"],
            default=["BTCUSDT", "ETHUSDT"]
        )
    
    with col2:
        refresh_interval = st.selectbox(
            "Intervalo de Actualización",
            ["5 segundos", "10 segundos", "30 segundos", "1 minuto"],
            index=2
        )
    
    with col3:
        auto_refresh = st.checkbox("Auto-actualizar", value=False)
    
    # Convertir intervalo a segundos
    interval_seconds = {
        "5 segundos": 5,
        "10 segundos": 10,
        "30 segundos": 30,
        "1 minuto": 60
    }[refresh_interval]
    
    # Placeholder para datos en tiempo real
    if symbols_to_monitor:
        
        # Botón de actualización manual
        if st.button("🔄 Actualizar Datos") or auto_refresh:
            
            # Contenedor para métricas en tiempo real
            metrics_container = st.container()
            
            with metrics_container:
                st.subheader("📊 Precios en Tiempo Real")
                
                # Crear columnas dinámicamente
                cols = st.columns(len(symbols_to_monitor))
                
                for i, symbol in enumerate(symbols_to_monitor):
                    with cols[i]:
                        try:
                            # Obtener último precio
                            df = data_manager.get_data(symbol, '1m', limit=2)
                            
                            if not df.empty:
                                current_price = df['close'].iloc[-1]
                                prev_price = df['close'].iloc[-2] if len(df) > 1 else current_price
                                change = current_price - prev_price
                                change_pct = (change / prev_price) * 100 if prev_price != 0 else 0
                                
                                # Color basado en cambio
                                delta_color = "normal" if change >= 0 else "inverse"
                                
                                st.metric(
                                    label=symbol,
                                    value=f"${current_price:.2f}",
                                    delta=f"{change_pct:+.2f}%",
                                    delta_color=delta_color
                                )
                                
                                # Mini gráfico
                                mini_df = data_manager.get_data(symbol, '1m', limit=20)
                                if not mini_df.empty:
                                    fig = go.Figure()
                                    fig.add_trace(go.Scatter(
                                        x=mini_df.index,
                                        y=mini_df['close'],
                                        mode='lines',
                                        name=symbol,
                                        line=dict(width=2)
                                    ))
                                    
                                    fig.update_layout(
                                        height=150,
                                        margin=dict(l=0, r=0, t=0, b=0),
                                        showlegend=False,
                                        xaxis=dict(showticklabels=False),
                                        yaxis=dict(showticklabels=False)
                                    )
                                    
                                    st.plotly_chart(fig, use_container_width=True, key=f"mini_{symbol}", config=get_plotly_config())
                            
                            else:
                                st.warning(f"No hay datos para {symbol}")
                                
                        except Exception as e:
                            st.error(f"Error obteniendo datos para {symbol}: {e}")
            
            # Alertas y señales
            st.subheader("🚨 Alertas y Señales")
            
            alerts_container = st.container()
            
            with alerts_container:
                for symbol in symbols_to_monitor:
                    try:
                        # Obtener datos recientes
                        df = data_manager.get_data(symbol, '5m', limit=50)
                        
                        if not df.empty:
                            # Análisis rápido
                            analysis = analyze_symbol(df)
                            
                            # Generar alertas
                            alerts = []
                            
                            # Alerta de RSI
                            if 'rsi' in analysis.indicators:
                                rsi = analysis.indicators['rsi']
                                # Handle Series/scalar values
                                if hasattr(rsi, 'iloc'):
                                    rsi_value = rsi.iloc[-1]
                                elif hasattr(rsi, '__len__') and len(rsi) > 0:
                                    rsi_value = rsi[-1]
                                else:
                                    rsi_value = rsi
                                
                                if rsi_value > 70:
                                    alerts.append(f"⚠️ {symbol}: RSI sobrecomprado ({rsi_value:.1f})")
                                elif rsi_value < 30:
                                    alerts.append(f"💚 {symbol}: RSI sobrevendido ({rsi_value:.1f})")
                            
                            # Alerta de tendencia
                            trend = analysis.trend_analysis.get('current_trend')
                            strength = analysis.trend_analysis.get('trend_strength', 0)
                            
                            if trend == 'Bullish' and strength > 0.8:
                                alerts.append(f"📈 {symbol}: Tendencia alcista fuerte")
                            elif trend == 'Bearish' and strength > 0.8:
                                alerts.append(f"📉 {symbol}: Tendencia bajista fuerte")
                            
                            # Alerta de patrones
                            if analysis.patterns:
                                high_conf_patterns = [p for p in analysis.patterns if p.confidence > 0.7]
                                if high_conf_patterns:
                                    alerts.append(f"🎯 {symbol}: {len(high_conf_patterns)} patrón(es) de alta confianza")
                            
                            # Mostrar alertas
                            for alert in alerts:
                                if "sobrecomprado" in alert or "bajista" in alert:
                                    st.warning(alert)
                                elif "sobrevendido" in alert or "alcista" in alert:
                                    st.success(alert)
                                else:
                                    st.info(alert)
                    
                    except Exception as e:
                        st.error(f"Error generando alertas para {symbol}: {e}")
            
            # Auto-refresh
            if auto_refresh:
                time.sleep(interval_seconds)
                st.rerun()
    
    else:
        st.info("Selecciona al menos un símbolo para monitorear")