#!/usr/bin/env python3
"""
Página de Fase 2 - Análisis Técnico
"""

import streamlit as st
import plotly.graph_objects as go
from pages.common import get_plotly_config, CUSTOM_CSS, analyze_symbol

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
            ["1m", "5m", "15m", "1h", "4h", "1d", "1w"],
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
                        
                        # Diccionario con información detallada de cada patrón
                        pattern_info = {
                            "triangle": {
                                "name": "Triángulo",
                                "description": "Patrón de consolidación donde el precio se mueve entre líneas de tendencia convergentes.",
                                "criteria": [
                                    "Mínimo 4 puntos de contacto (2 máximos y 2 mínimos)",
                                    "Líneas de tendencia convergentes",
                                    "Volumen decreciente durante la formación",
                                    "Ruptura con incremento de volumen"
                                ],
                                "theory": "Basado en la teoría de Dow y análisis técnico clásico. Representa un período de indecisión del mercado antes de una ruptura direccional.",
                                "reliability": "Alta (70-80% de efectividad)",
                                "timeframe": "Funciona mejor en marcos temporales de 1h o superiores"
                            },
                            "rectangle": {
                                "name": "Rectángulo",
                                "description": "Patrón de consolidación horizontal donde el precio oscila entre niveles de soporte y resistencia paralelos.",
                                "criteria": [
                                    "Mínimo 4 puntos de contacto en niveles horizontales",
                                    "Soporte y resistencia claramente definidos",
                                    "Rango de precio relativamente estable",
                                    "Volumen variable durante la formación"
                                ],
                                "theory": "Representa equilibrio entre compradores y vendedores. Basado en conceptos de soporte/resistencia de la teoría técnica clásica.",
                                "reliability": "Media-Alta (60-75% de efectividad)",
                                "timeframe": "Efectivo en todos los marcos temporales"
                            },
                            "channel": {
                                "name": "Canal",
                                "description": "Patrón donde el precio se mueve entre dos líneas de tendencia paralelas (canal alcista, bajista o lateral).",
                                "criteria": [
                                    "Dos líneas de tendencia paralelas",
                                    "Mínimo 3 puntos de contacto por línea",
                                    "Precio respeta los límites del canal",
                                    "Tendencia direccional clara"
                                ],
                                "theory": "Basado en la teoría de tendencias de Charles Dow. Los canales representan movimientos ordenados del mercado.",
                                "reliability": "Alta (75-85% de efectividad)",
                                "timeframe": "Más confiable en marcos temporales largos (4h+)"
                            },
                            "head_and_shoulders": {
                                "name": "Cabeza y Hombros",
                                "description": "Patrón de reversión que indica el final de una tendencia alcista, formado por tres picos con el central más alto.",
                                "criteria": [
                                    "Tres picos: hombro izquierdo, cabeza, hombro derecho",
                                    "La cabeza debe ser el pico más alto",
                                    "Línea de cuello conecta los mínimos",
                                    "Volumen decreciente en la formación"
                                ],
                                "theory": "Patrón clásico de reversión identificado por Richard Schabacker y popularizado por Edwards & Magee.",
                                "reliability": "Muy Alta (80-90% de efectividad)",
                                "timeframe": "Más efectivo en marcos temporales diarios o semanales"
                            },
                            "double_top": {
                                "name": "Doble Techo",
                                "description": "Patrón de reversión bajista formado por dos picos de altura similar separados por un valle.",
                                "criteria": [
                                    "Dos picos de altura similar (±3%)",
                                    "Valle intermedio claramente definido",
                                    "Ruptura del soporte del valle",
                                    "Volumen confirmatorio en la ruptura"
                                ],
                                "theory": "Indica agotamiento de la presión compradora. Concepto desarrollado en el análisis técnico clásico.",
                                "reliability": "Alta (70-80% de efectividad)",
                                "timeframe": "Funciona en todos los marcos temporales"
                            },
                            "double_bottom": {
                                "name": "Doble Suelo",
                                "description": "Patrón de reversión alcista formado por dos mínimos de altura similar separados por un pico.",
                                "criteria": [
                                    "Dos mínimos de altura similar (±3%)",
                                    "Pico intermedio claramente definido",
                                    "Ruptura de la resistencia del pico",
                                    "Volumen confirmatorio en la ruptura"
                                ],
                                "theory": "Indica agotamiento de la presión vendedora. Patrón complementario al doble techo.",
                                "reliability": "Alta (70-80% de efectividad)",
                                "timeframe": "Funciona en todos los marcos temporales"
                            }
                        }
                        
                        for pattern in analysis_result.patterns:
                            pattern_type = pattern.pattern_type
                            confidence = pattern.confidence
                            
                            # Mostrar el patrón con color según confianza
                            if confidence > 0.7:
                                st.success(f"🟢 {pattern_type} (Confianza: {confidence:.2f})")
                            elif confidence > 0.5:
                                st.warning(f"🟡 {pattern_type} (Confianza: {confidence:.2f})")
                            else:
                                st.info(f"🔵 {pattern_type} (Confianza: {confidence:.2f})")
                            
                            # Crear desplegable con información detallada
                            with st.expander(f"📋 {pattern_type.upper()} - Definición y Criterios de Identificación"):
                                if pattern_type in pattern_info:
                                    info = pattern_info[pattern_type]
                                    
                                    # Información del patrón detectado (lo más importante primero)
                                    st.markdown("**🎯 PATRÓN DETECTADO**")
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric("Confianza", f"{confidence:.1%}")
                                    with col2:
                                        st.metric("Inicio", f"Vela {pattern.start_idx}")
                                    with col3:
                                        st.metric("Fin", f"Vela {pattern.end_idx}")
                                    
                                    st.divider()
                                    
                                    # Descripción concisa
                                    st.markdown("**📖 ¿Qué es este patrón?**")
                                    st.info(info["description"])
                                    
                                    # Criterios de identificación
                                    st.markdown("**✅ ¿Cómo se identifica?**")
                                    for i, criterion in enumerate(info["criteria"], 1):
                                        st.write(f"{i}. {criterion}")
                                    
                                    # Información técnica en columnas
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        st.markdown("**📊 Efectividad**")
                                        st.success(info["reliability"])
                                    
                                    with col2:
                                        st.markdown("**⏰ Mejor Timeframe**")
                                        st.info(info["timeframe"])
                                    
                                    # Base teórica (menos prominente)
                                    with st.expander("🎓 Fundamento Teórico"):
                                        st.write(info["theory"])
                                    
                                else:
                                    st.write("Información detallada no disponible para este patrón.")
                                    st.write(f"Patrón detectado desde el índice {pattern.start_idx} hasta {pattern.end_idx}")
                    
                    # Gráfico con indicadores
                    st.subheader("📊 Gráfico con Indicadores y Patrones")
                    
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
                    
                    # Marcar patrones detectados en el gráfico
                    if analysis_result.patterns:
                        for i, pattern in enumerate(analysis_result.patterns):
                            start_idx = pattern.start_idx
                            end_idx = pattern.end_idx
                            
                            # Asegurar que los índices estén dentro del rango
                            if start_idx < len(df) and end_idx < len(df) and start_idx >= 0 and end_idx >= 0:
                                # Obtener los datos del patrón
                                pattern_data = df.iloc[start_idx:end_idx+1]
                                
                                # Colores altamente contrastantes y accesibles para daltonismo
                                if pattern.confidence > 0.7:
                                    color = 'rgba(0, 0, 139, 0.3)'  # Azul marino transparente (alta confianza)
                                    border_color = '#00008B'  # Azul marino sólido
                                    text_color = '#FFFFFF'  # Texto blanco
                                    bg_color = 'rgba(0, 0, 139, 0.9)'  # Fondo azul marino semi-opaco
                                    dash_pattern = 'solid'
                                    symbol = '🔵'  # Círculo azul
                                elif pattern.confidence > 0.5:
                                    color = 'rgba(255, 140, 0, 0.3)'  # Naranja oscuro transparente (media confianza)
                                    border_color = '#FF8C00'  # Naranja oscuro sólido
                                    text_color = '#000000'  # Texto negro
                                    bg_color = 'rgba(255, 255, 255, 0.95)'  # Fondo blanco
                                    dash_pattern = 'dash'
                                    symbol = '🔶'  # Rombo naranja
                                else:
                                    color = 'rgba(220, 20, 60, 0.3)'  # Carmesí transparente (baja confianza)
                                    border_color = '#DC143C'  # Carmesí sólido
                                    text_color = '#FFFFFF'  # Texto blanco
                                    bg_color = 'rgba(220, 20, 60, 0.9)'  # Fondo carmesí semi-opaco
                                    dash_pattern = 'dot'
                                    symbol = '🔴'  # Círculo rojo
                                
                                # Añadir área sombreada para el patrón con máxima accesibilidad
                                fig.add_shape(
                                    type="rect",
                                    x0=pattern_data.index[0],
                                    y0=pattern_data['low'].min() * 0.999,  # Ligeramente por debajo del mínimo
                                    x1=pattern_data.index[-1],
                                    y1=pattern_data['high'].max() * 1.001,  # Ligeramente por encima del máximo
                                    fillcolor=color,
                                    line=dict(color=border_color, width=4, dash=dash_pattern),
                                    opacity=0.7
                                )
                                
                                # Añadir anotación del patrón con máxima accesibilidad y contraste
                                fig.add_annotation(
                                    x=pattern_data.index[len(pattern_data)//2],  # Punto medio del patrón
                                    y=pattern_data['high'].max() * 1.008,
                                    text=f"<b>{symbol} {pattern.pattern_type}</b><br><b>Confianza: {pattern.confidence:.1%}</b>",
                                    showarrow=True,
                                    arrowhead=3,
                                    arrowsize=1.5,
                                    arrowcolor=border_color,
                                    arrowwidth=3,
                                    bgcolor=bg_color,
                                    bordercolor=border_color,
                                    borderwidth=4,
                                    font=dict(
                                        size=14,
                                        color=text_color,
                                        family="Arial Black"
                                    ),
                                    opacity=1.0
                                )
                    
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