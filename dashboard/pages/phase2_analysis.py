#!/usr/bin/env python3
"""
Página de Fase 2 - Análisis Técnico
"""

import streamlit as st
import plotly.graph_objects as go
from .common import get_plotly_config, CUSTOM_CSS, analyze_symbol

def show_phase2_analysis_page(data_manager):
    """Mostrar la página de Fase 2 - Análisis Técnico"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    st.title("🔍 Fase 2 - Análisis Técnico")
    
    if not data_manager:
        st.error("Error: DataManager no disponible")
        st.stop()
    
    # Configuración del análisis
    st.subheader("⚙️ Configuración del Análisis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        symbol = st.selectbox(
            "Símbolo para Análisis",
            ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"],
            index=0,
            key="analysis_symbol"
        )
    
    with col2:
        interval = st.selectbox(
            "Intervalo de Tiempo",
            ["1m", "5m", "15m", "1h", "4h", "1d"],
            index=3,
            key="analysis_interval"
        )
    
    if st.button("🔍 Ejecutar Análisis Técnico", type="primary"):
        with st.spinner(f"Analizando {symbol}..."):
            try:
                # Obtener datos
                df = data_manager.get_data(symbol, interval, limit=200)
                
                if df.empty:
                    st.warning(f"No hay datos disponibles para {symbol}. Obtén datos primero en la Fase 1.")
                else:
                    # Ejecutar análisis técnico
                    analysis_result = analyze_symbol(df)
                    
                    # Mostrar resultados
                    st.success("✅ Análisis completado")
                    
                    # Métricas principales
                    st.subheader("📊 Resumen del Análisis")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        trend = analysis_result.trend_analysis.get('current_trend', 'N/A')
                        st.metric("Tendencia Actual", trend)
                    
                    with col2:
                        strength = analysis_result.trend_analysis.get('trend_strength', 0)
                        st.metric("Fuerza de Tendencia", f"{strength:.2f}")
                    
                    with col3:
                        patterns_count = len(analysis_result.patterns)
                        st.metric("Patrones Detectados", patterns_count)
                    
                    with col4:
                        last_price = df['close'].iloc[-1]
                        st.metric("Último Precio", f"${last_price:.2f}")
                    
                    # Indicadores técnicos
                    st.subheader("📈 Indicadores Técnicos")
                    
                    indicators = analysis_result.indicators
                    
                    # Helper function to safely extract scalar values from Series
                    def safe_float(value):
                        if hasattr(value, 'iloc'):
                            return float(value.iloc[-1])
                        elif hasattr(value, '__len__') and len(value) > 0:
                            return float(value[-1])
                        else:
                            return float(value)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Medias Móviles:**")
                        if 'sma_20' in indicators:
                            st.write(f"SMA 20: ${safe_float(indicators['sma_20']):.2f}")
                        if 'sma_50' in indicators:
                            st.write(f"SMA 50: ${safe_float(indicators['sma_50']):.2f}")
                        if 'ema_12' in indicators:
                            st.write(f"EMA 12: ${safe_float(indicators['ema_12']):.2f}")
                        if 'ema_26' in indicators:
                            st.write(f"EMA 26: ${safe_float(indicators['ema_26']):.2f}")
                    
                    with col2:
                        st.write("**Osciladores:**")
                        if 'rsi' in indicators:
                            rsi_value = safe_float(indicators['rsi'])
                            rsi_status = "Sobrecomprado" if rsi_value > 70 else "Sobrevendido" if rsi_value < 30 else "Neutral"
                            st.write(f"RSI: {rsi_value:.2f} ({rsi_status})")
                        if 'macd' in indicators:
                            st.write(f"MACD: {safe_float(indicators['macd']):.4f}")
                        if 'macd_signal' in indicators:
                            st.write(f"MACD Signal: {safe_float(indicators['macd_signal']):.4f}")
                    
                    # Patrones detectados
                    if analysis_result.patterns:
                        st.subheader("🎯 Patrones Detectados")
                        
                        for pattern in analysis_result.patterns:
                            pattern_type = pattern.pattern_type
                            confidence = pattern.confidence
                            
                            if confidence > 0.7:
                                st.success(f"🟢 {pattern_type} (Confianza: {confidence:.2f})")
                            elif confidence > 0.5:
                                st.warning(f"🟡 {pattern_type} (Confianza: {confidence:.2f})")
                            else:
                                st.info(f"🔵 {pattern_type} (Confianza: {confidence:.2f})")
                    
                    # Gráfico con indicadores
                    st.subheader("📊 Gráfico con Indicadores")
                    
                    fig = go.Figure()
                    
                    # Velas japonesas
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name=symbol
                    ))
                    
                    # Agregar medias móviles si están disponibles
                    if 'sma_20' in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['sma_20'],
                            mode='lines',
                            name='SMA 20',
                            line=dict(color='orange')
                        ))
                    
                    if 'sma_50' in df.columns:
                        fig.add_trace(go.Scatter(
                            x=df.index,
                            y=df['sma_50'],
                            mode='lines',
                            name='SMA 50',
                            line=dict(color='red')
                        ))
                    
                    fig.update_layout(
                        title=f"{symbol} - Análisis Técnico ({interval})",
                        yaxis_title="Precio (USDT)",
                        xaxis_title="Tiempo",
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())
                    
                    # Análisis detallado en JSON
                    with st.expander("🔍 Ver Análisis Completo (JSON)"):
                        st.json({
                            "trend_analysis": analysis_result.trend_analysis,
                            "indicators": analysis_result.indicators,
                            "patterns": [{
                                "type": p.pattern_type,
                                "confidence": p.confidence,
                                "start_idx": p.start_idx,
                                "end_idx": p.end_idx
                            } for p in analysis_result.patterns],
                            "timestamp": analysis_result.timestamp.isoformat()
                        })
                    
            except Exception as e:
                st.error(f"Error en el análisis: {e}")
                st.exception(e)