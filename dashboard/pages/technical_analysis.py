#!/usr/bin/env python3
"""
Página de Análisis Técnico Avanzado
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pages.common import get_plotly_config, CUSTOM_CSS, analyze_symbol

def show_technical_analysis_page(data_manager):
    """Mostrar la página de Análisis Técnico Avanzado"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    st.title("📈 Análisis Técnico Avanzado")
    
    if not data_manager:
        st.error("Error: DataManager no disponible")
        st.stop()
    
    # Configuración
    col1, col2, col3 = st.columns(3)
    
    with col1:
        symbol = st.selectbox(
            "Símbolo",
            ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"],
            key="tech_analysis_symbol"
        )
    
    with col2:
        interval = st.selectbox(
            "Intervalo",
            ["1m", "5m", "15m", "1h", "4h", "1d", "1w"],
            index=5,
            key="tech_analysis_interval"
        )
    
    with col3:
        lookback_periods = st.number_input("Períodos de análisis", min_value=50, max_value=500, value=200)
    
    if st.button("📊 Generar Análisis Completo", type="primary"):
        with st.spinner("Generando análisis técnico completo..."):
            try:
                # Obtener datos
                df = data_manager.get_data(symbol, interval, limit=lookback_periods)
                
                if df.empty:
                    st.warning(f"No hay datos suficientes para {symbol}")
                    st.stop()
                
                # Realizar análisis técnico
                analysis = analyze_symbol(df)
                
                if not analysis:
                    st.error("Error al realizar el análisis técnico")
                    st.stop()
                
                # === RESUMEN EJECUTIVO ===
                st.header("📊 Resumen Ejecutivo")
                
                col1, col2, col3, col4, col5 = st.columns(5)
                
                with col1:
                    trend = analysis.trend_analysis.get('current_trend', 'Neutral')
                    trend_emoji = "📈" if trend == "Bullish" else "📉" if trend == "Bearish" else "➡️"
                    st.metric("Tendencia", f"{trend_emoji} {trend}")
                
                with col2:
                    strength = analysis.trend_analysis.get('trend_strength', 0)
                    st.metric("Fuerza", f"{strength:.3f}")
                
                with col3:
                    volatility = analysis.indicators.get('volatility', 0)
                    if hasattr(volatility, 'iloc'):
                        volatility = volatility.iloc[-1]
                    st.metric("Volatilidad", f"{volatility:.2f}%")
                
                with col4:
                    # Calcular tendencia de volumen desde los indicadores
                    volume_trend = "Neutral"
                    if 'relative_volume' in analysis.indicators:
                        rel_vol = analysis.indicators['relative_volume']
                        if hasattr(rel_vol, 'iloc'):
                            rel_vol_value = rel_vol.iloc[-1]
                        else:
                            rel_vol_value = rel_vol
                        
                        if rel_vol_value > 1.2:
                            volume_trend = "Increasing"
                        elif rel_vol_value < 0.8:
                            volume_trend = "Decreasing"
                    
                    volume_emoji = "📊" if volume_trend == "Increasing" else "📉" if volume_trend == "Decreasing" else "➡️"
                    st.metric("Volumen", f"{volume_emoji} {volume_trend}")
                
                with col5:
                    patterns_count = len(analysis.patterns)
                    st.metric("Patrones", f"🎯 {patterns_count}")
                
                # === GRÁFICO TÉCNICO PRINCIPAL ===
                st.header("📈 Gráfico Técnico Principal")
                
                # Crear subplots
                fig = make_subplots(
                    rows=3, cols=1,
                    shared_xaxes=True,
                    vertical_spacing=0.05,
                    subplot_titles=('Precio y Medias Móviles', 'RSI', 'MACD'),
                    row_heights=[0.6, 0.2, 0.2]
                )
                
                # Gráfico de velas
                fig.add_trace(
                    go.Candlestick(
                        x=df.index,
                        open=df['open'],
                        high=df['high'],
                        low=df['low'],
                        close=df['close'],
                        name='Precio'
                    ),
                    row=1, col=1
                )
                
                # Medias móviles
                if 'sma_20' in analysis.indicators:
                    sma_20 = analysis.indicators['sma_20']
                    fig.add_trace(
                        go.Scatter(x=df.index, y=sma_20, name='SMA 20', line=dict(color='orange')),
                        row=1, col=1
                    )
                
                if 'sma_50' in analysis.indicators:
                    sma_50 = analysis.indicators['sma_50']
                    fig.add_trace(
                        go.Scatter(x=df.index, y=sma_50, name='SMA 50', line=dict(color='blue')),
                        row=1, col=1
                    )
                
                # RSI
                if 'rsi' in analysis.indicators:
                    rsi = analysis.indicators['rsi']
                    fig.add_trace(
                        go.Scatter(x=df.index, y=rsi, name='RSI', line=dict(color='purple')),
                        row=2, col=1
                    )
                    
                    # Líneas de sobrecompra/sobreventa
                    fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
                    fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
                
                # MACD
                if 'macd' in analysis.indicators and 'macd_signal' in analysis.indicators:
                    macd = analysis.indicators['macd']
                    macd_signal = analysis.indicators['macd_signal']
                    
                    fig.add_trace(
                        go.Scatter(x=df.index, y=macd, name='MACD', line=dict(color='blue')),
                        row=3, col=1
                    )
                    fig.add_trace(
                        go.Scatter(x=df.index, y=macd_signal, name='Signal', line=dict(color='red')),
                        row=3, col=1
                    )
                
                fig.update_layout(
                    title=f"Análisis Técnico - {symbol} ({interval})",
                    height=800,
                    showlegend=True,
                    xaxis_rangeslider_visible=False
                )
                
                st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())
                
                # === ANÁLISIS DE PATRONES DETALLADO ===
                st.header("🎯 Análisis de Patrones Detallado")
                
                if analysis.patterns:
                    # Clasificar patrones por tipo
                    bullish_patterns = [p for p in analysis.patterns if 'bullish' in p.pattern_type.lower() or 'bull' in p.pattern_type.lower()]
                    bearish_patterns = [p for p in analysis.patterns if 'bearish' in p.pattern_type.lower() or 'bear' in p.pattern_type.lower()]
                    neutral_patterns = [p for p in analysis.patterns if p not in bullish_patterns and p not in bearish_patterns]
                    
                    # Tabs para diferentes tipos de patrones
                    tab1, tab2, tab3, tab4 = st.tabs(["📈 Alcistas", "📉 Bajistas", "➡️ Neutrales", "📊 Resumen"])
                    
                    with tab1:
                        if bullish_patterns:
                            st.success(f"Se encontraron {len(bullish_patterns)} patrón(es) alcista(s)")
                            for i, pattern in enumerate(bullish_patterns):
                                _show_pattern_details(pattern, df, i, "bullish")
                        else:
                            st.info("No se encontraron patrones alcistas en el período analizado")
                    
                    with tab2:
                        if bearish_patterns:
                            st.error(f"Se encontraron {len(bearish_patterns)} patrón(es) bajista(s)")
                            for i, pattern in enumerate(bearish_patterns):
                                _show_pattern_details(pattern, df, i, "bearish")
                        else:
                            st.info("No se encontraron patrones bajistas en el período analizado")
                    
                    with tab3:
                        if neutral_patterns:
                            st.warning(f"Se encontraron {len(neutral_patterns)} patrón(es) neutral(es)")
                            for i, pattern in enumerate(neutral_patterns):
                                _show_pattern_details(pattern, df, i, "neutral")
                        else:
                            st.info("No se encontraron patrones neutrales en el período analizado")
                    
                    with tab4:
                        # Resumen estadístico de patrones
                        st.write("**📊 Estadísticas de Patrones:**")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Total Patrones", len(analysis.patterns))
                        with col2:
                            st.metric("Alcistas", len(bullish_patterns))
                        with col3:
                            st.metric("Bajistas", len(bearish_patterns))
                        with col4:
                            st.metric("Neutrales", len(neutral_patterns))
                        
                        # Confianza promedio
                        if analysis.patterns:
                            avg_confidence = sum(p.confidence for p in analysis.patterns) / len(analysis.patterns)
                            st.write(f"**🎯 Confianza Promedio:** {avg_confidence:.3f}")
                            
                            # Patrón con mayor confianza
                            best_pattern = max(analysis.patterns, key=lambda p: p.confidence)
                            st.write(f"**🏆 Mejor Patrón:** {best_pattern.pattern_type} (Confianza: {best_pattern.confidence:.3f})")
                else:
                    st.info("No se detectaron patrones técnicos en el período analizado")
                
                # === RECOMENDACIONES DE TRADING ===
                st.header("💡 Recomendaciones de Trading")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📈 Análisis de Tendencia:**")
                    trend = analysis.trend_analysis.get('current_trend', 'Neutral')
                    strength = analysis.trend_analysis.get('trend_strength', 0)
                    
                    if trend == "Bullish" and strength > 0.6:
                        st.success(f"Tendencia alcista fuerte (Fuerza: {strength:.3f})")
                        st.write("• Considerar posiciones largas")
                        st.write("• Buscar retrocesos para entrar")
                    elif trend == "Bearish" and strength > 0.6:
                        st.error(f"Tendencia bajista fuerte (Fuerza: {strength:.3f})")
                        st.write("• Considerar posiciones cortas")
                        st.write("• Evitar compras")
                    else:
                        st.warning(f"Tendencia {trend.lower()} débil (Fuerza: {strength:.3f})")
                        st.write("• Mercado lateral")
                        st.write("• Esperar confirmación direccional")
                
                with col2:
                    st.write("**📊 Análisis de RSI:**")
                    if 'rsi' in analysis.indicators:
                        rsi_value = analysis.indicators['rsi']
                        if hasattr(rsi_value, 'iloc'):
                            rsi_value = rsi_value.iloc[-1]
                        
                        if rsi_value > 70:
                            st.error(f"RSI Sobrecomprado ({rsi_value:.2f})")
                            st.write("• Posible corrección a la baja")
                            st.write("• Considerar tomar ganancias")
                        elif rsi_value < 30:
                            st.success(f"RSI Sobrevendido ({rsi_value:.2f})")
                            st.write("• Posible rebote al alza")
                            st.write("• Oportunidad de compra")
                        else:
                            st.info(f"RSI Neutral ({rsi_value:.2f})")
                            st.write("• Condiciones normales")
                            st.write("• Seguir la tendencia principal")
                
                # Recomendaciones basadas en patrones
                if analysis.patterns:
                    st.write("**🎯 Recomendaciones por Patrones:**")
                    high_confidence_patterns = [p for p in analysis.patterns if p.confidence > 0.7]
                    
                    if high_confidence_patterns:
                        for pattern in high_confidence_patterns[:3]:  # Top 3
                            pattern_type = "alcista" if any(word in pattern.pattern_type.lower() for word in ['bullish', 'bull']) else "bajista" if any(word in pattern.pattern_type.lower() for word in ['bearish', 'bear']) else "neutral"
                            
                            if pattern_type == "alcista":
                                st.success(f"✅ {pattern.pattern_type} (Confianza: {pattern.confidence:.3f}) - Señal de compra")
                            elif pattern_type == "bajista":
                                st.error(f"❌ {pattern.pattern_type} (Confianza: {pattern.confidence:.3f}) - Señal de venta")
                            else:
                                st.warning(f"⚠️ {pattern.pattern_type} (Confianza: {pattern.confidence:.3f}) - Esperar confirmación")
                
                # === ANÁLISIS TÉCNICO AVANZADO ===
                st.header("🔬 Análisis Técnico Avanzado")
                
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Indicadores", "🎯 Soporte/Resistencia", "🌀 Fibonacci", "🚨 Alertas"])
                
                with tab1:
                    # Análisis detallado de indicadores
                    st.write("**📈 Indicadores Técnicos Detallados**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if 'rsi' in analysis.indicators:
                            rsi = analysis.indicators['rsi']
                            if hasattr(rsi, 'iloc'):
                                current_rsi = rsi.iloc[-1]
                                prev_rsi = rsi.iloc[-2] if len(rsi) > 1 else current_rsi
                            else:
                                current_rsi = rsi
                                prev_rsi = rsi
                            
                            rsi_change = current_rsi - prev_rsi
                            st.metric("RSI", f"{current_rsi:.2f}", f"{rsi_change:+.2f}")
                            
                            if current_rsi > 80:
                                st.error("Extremadamente sobrecomprado")
                            elif current_rsi > 70:
                                st.warning("Sobrecomprado")
                            elif current_rsi < 20:
                                st.error("Extremadamente sobrevendido")
                            elif current_rsi < 30:
                                st.success("Sobrevendido")
                            else:
                                st.info("Rango normal")
                    
                    with col2:
                        if 'macd' in analysis.indicators and 'macd_signal' in analysis.indicators:
                            macd = analysis.indicators['macd']
                            macd_signal = analysis.indicators['macd_signal']
                            
                            if hasattr(macd, 'iloc'):
                                current_macd = macd.iloc[-1]
                                current_signal = macd_signal.iloc[-1]
                            else:
                                current_macd = macd
                                current_signal = macd_signal
                            
                            macd_diff = current_macd - current_signal
                            st.metric("MACD", f"{current_macd:.6f}", f"{macd_diff:+.6f}")
                            
                            if current_macd > current_signal:
                                st.success("Señal alcista (MACD > Signal)")
                            else:
                                st.error("Señal bajista (MACD < Signal)")
                    
                    # Bandas de Bollinger
                    if 'bb_upper' in analysis.indicators and 'bb_lower' in analysis.indicators:
                        st.write("**📊 Bandas de Bollinger:**")
                        bb_upper = analysis.indicators['bb_upper']
                        bb_lower = analysis.indicators['bb_lower']
                        current_price = df['close'].iloc[-1]
                        
                        if hasattr(bb_upper, 'iloc'):
                            upper_band = bb_upper.iloc[-1]
                            lower_band = bb_lower.iloc[-1]
                        else:
                            upper_band = bb_upper
                            lower_band = bb_lower
                        
                        bb_position = (current_price - lower_band) / (upper_band - lower_band)
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Banda Superior", f"${upper_band:.4f}")
                        with col2:
                            st.metric("Banda Inferior", f"${lower_band:.4f}")
                        with col3:
                            st.metric("Posición %", f"{bb_position*100:.1f}%")
                        
                        if bb_position > 0.8:
                            st.warning("Precio cerca de la banda superior - Posible sobrecompra")
                        elif bb_position < 0.2:
                            st.success("Precio cerca de la banda inferior - Posible sobreventa")
                        else:
                            st.info("Precio en rango normal de las bandas")
                
                with tab2:
                    # Análisis de soporte y resistencia
                    st.write("**🎯 Niveles de Soporte y Resistencia**")
                    support_resistance = _calculate_support_resistance(df)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**🟢 Niveles de Soporte:**")
                        for i, support in enumerate(support_resistance['support'][:5]):
                            distance = ((df['close'].iloc[-1] - support) / support) * 100
                            st.write(f"S{i+1}: ${support:.4f} ({distance:+.2f}%)")
                    
                    with col2:
                        st.write("**🔴 Niveles de Resistencia:**")
                        for i, resistance in enumerate(support_resistance['resistance'][:5]):
                            distance = ((resistance - df['close'].iloc[-1]) / df['close'].iloc[-1]) * 100
                            st.write(f"R{i+1}: ${resistance:.4f} (+{distance:.2f}%)")
                    
                    # Gráfico de soporte y resistencia
                    fig_sr = go.Figure()
                    
                    # Precio
                    fig_sr.add_trace(
                        go.Candlestick(
                            x=df.index[-50:],  # Últimos 50 períodos
                            open=df['open'].iloc[-50:],
                            high=df['high'].iloc[-50:],
                            low=df['low'].iloc[-50:],
                            close=df['close'].iloc[-50:],
                            name='Precio'
                        )
                    )
                    
                    # Líneas de soporte
                    for i, support in enumerate(support_resistance['support'][:3]):
                        fig_sr.add_hline(y=support, line_dash="dash", line_color="green", 
                                       annotation_text=f"S{i+1}: ${support:.4f}")
                    
                    # Líneas de resistencia
                    for i, resistance in enumerate(support_resistance['resistance'][:3]):
                        fig_sr.add_hline(y=resistance, line_dash="dash", line_color="red", 
                                       annotation_text=f"R{i+1}: ${resistance:.4f}")
                    
                    fig_sr.update_layout(
                        title="Niveles de Soporte y Resistencia",
                        height=400,
                        showlegend=False,
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig_sr, use_container_width=True, config=get_plotly_config())
                
                with tab3:
                    # Análisis de Fibonacci
                    st.write("**🌀 Retrocesos de Fibonacci**")
                    fib_levels = _calculate_fibonacci_levels(df)
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📊 Niveles de Fibonacci:**")
                        current_price = df['close'].iloc[-1]
                        
                        for level_name, level_price in fib_levels.items():
                            distance = ((current_price - level_price) / level_price) * 100
                            if abs(distance) < 1:  # Cerca del nivel
                                st.success(f"**{level_name}**: ${level_price:.4f} ({distance:+.2f}%) 🎯")
                            else:
                                st.write(f"{level_name}: ${level_price:.4f} ({distance:+.2f}%)")
                    
                    with col2:
                        st.write("**💡 Interpretación:**")
                        st.write("• Los niveles de Fibonacci actúan como soporte/resistencia")
                        st.write("• 38.2% y 61.8% son los más importantes")
                        st.write("• Buscar rebotes en estos niveles")
                        st.write("• Ruptura confirma continuación de tendencia")
                    
                    # Gráfico de Fibonacci
                    fig_fib = go.Figure()
                    
                    # Precio
                    fig_fib.add_trace(
                        go.Candlestick(
                            x=df.index[-100:],
                            open=df['open'].iloc[-100:],
                            high=df['high'].iloc[-100:],
                            low=df['low'].iloc[-100:],
                            close=df['close'].iloc[-100:],
                            name='Precio'
                        )
                    )
                    
                    # Líneas de Fibonacci
                    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'purple', 'red']
                    for i, (level_name, level_price) in enumerate(fib_levels.items()):
                        fig_fib.add_hline(y=level_price, line_dash="dot", 
                                        line_color=colors[i % len(colors)],
                                        annotation_text=f"{level_name}: ${level_price:.4f}")
                    
                    fig_fib.update_layout(
                        title="Retrocesos de Fibonacci",
                        height=400,
                        showlegend=False,
                        xaxis_rangeslider_visible=False
                    )
                    
                    st.plotly_chart(fig_fib, use_container_width=True, config=get_plotly_config())
                
                with tab4:
                    # Sistema de alertas
                    st.write("**🚨 Alertas de Trading**")
                    alerts = _generate_trading_alerts(df, analysis, support_resistance, fib_levels)
                    
                    if alerts:
                        for alert in alerts:
                            if alert['type'] == 'success':
                                st.success(f"✅ {alert['message']}")
                            elif alert['type'] == 'error':
                                st.error(f"❌ {alert['message']}")
                            elif alert['type'] == 'warning':
                                st.warning(f"⚠️ {alert['message']}")
                            else:
                                st.info(f"ℹ️ {alert['message']}")
                    else:
                        st.info("No hay alertas activas en este momento")
                    
                    # Configuración de alertas personalizadas
                    st.write("**⚙️ Configurar Alertas Personalizadas**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        alert_price = st.number_input("Precio de Alerta", value=float(df['close'].iloc[-1]), format="%.4f")
                    with col2:
                        alert_type = st.selectbox("Tipo de Alerta", ["Precio Mayor", "Precio Menor", "Cruce de Media"])
                    with col3:
                        if st.button("Crear Alerta"):
                            st.success(f"Alerta creada: {alert_type} a ${alert_price:.4f}")
                
            except Exception as e:
                st.error(f"Error en análisis técnico: {e}")
                st.exception(e)


def _show_pattern_details(pattern, df, index, pattern_type):
    """Muestra detalles específicos de un patrón"""
    with st.expander(f"🔍 {pattern.pattern_type} - Confianza: {pattern.confidence:.3f}", expanded=index == 0):
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Gráfico específico del patrón
            pattern_start = max(0, pattern.start_idx - 10)
            pattern_end = min(len(df), pattern.end_idx + 10)
            pattern_df = df.iloc[pattern_start:pattern_end]
            
            fig_pattern = go.Figure()
            
            # Velas del patrón
            fig_pattern.add_trace(
                go.Candlestick(
                    x=pattern_df.index,
                    open=pattern_df['open'],
                    high=pattern_df['high'],
                    low=pattern_df['low'],
                    close=pattern_df['close'],
                    name='Precio'
                )
            )
            
            # Resaltar el área del patrón
            pattern_area = df.iloc[pattern.start_idx:pattern.end_idx + 1]
            fig_pattern.add_vrect(
                x0=pattern_area.index[0],
                x1=pattern_area.index[-1],
                fillcolor="rgba(255, 0, 0, 0.1)" if pattern_type == "bearish" else "rgba(0, 255, 0, 0.1)" if pattern_type == "bullish" else "rgba(255, 255, 0, 0.1)",
                layer="below",
                line_width=0,
            )
            
            fig_pattern.update_layout(
                title=f"Patrón: {pattern.pattern_type}",
                height=300,
                showlegend=False,
                xaxis_rangeslider_visible=False
            )
            
            st.plotly_chart(fig_pattern, use_container_width=True, config=get_plotly_config())
        
        with col2:
            # Detalles del patrón
            st.write("**📊 Detalles del Patrón:**")
            st.write(f"• **Tipo:** {pattern.pattern_type}")
            st.write(f"• **Confianza:** {pattern.confidence:.3f}")
            st.write(f"• **Inicio:** Período {pattern.start_idx}")
            st.write(f"• **Fin:** Período {pattern.end_idx}")
            st.write(f"• **Duración:** {pattern.end_idx - pattern.start_idx} períodos")
            
            # Descripción del patrón
            if hasattr(pattern, 'description') and pattern.description:
                st.write(f"• **Descripción:** {pattern.description}")
            
            # Parámetros específicos del patrón
            if hasattr(pattern, 'parameters') and pattern.parameters:
                st.write("**🔧 Parámetros:**")
                for key, value in pattern.parameters.items():
                    if isinstance(value, float):
                        st.write(f"• {key}: {value:.4f}")
                    else:
                        st.write(f"• {key}: {value}")
            
            # Análisis de precios en el patrón
            pattern_prices = df.iloc[pattern.start_idx:pattern.end_idx + 1]
            if not pattern_prices.empty:
                price_change = ((pattern_prices['close'].iloc[-1] - pattern_prices['close'].iloc[0]) / pattern_prices['close'].iloc[0]) * 100
                st.write(f"**📈 Cambio de Precio:** {price_change:+.2f}%")
                
                # Volatilidad durante el patrón
                pattern_volatility = pattern_prices['close'].pct_change().std() * 100
                st.write(f"**📊 Volatilidad:** {pattern_volatility:.2f}%")
            
            # Recomendación específica
            st.write("**💡 Implicación:**")
            if pattern_type == "bullish":
                st.success("Señal alcista - Posible continuación al alza")
            elif pattern_type == "bearish":
                st.error("Señal bajista - Posible continuación a la baja")
            else:
                st.warning("Patrón neutral - Esperar confirmación direccional")


def _calculate_support_resistance(df):
    """Calcula niveles de soporte y resistencia usando máximos y mínimos locales"""
    try:
        from scipy.signal import find_peaks
        
        # Encontrar máximos y mínimos locales
        highs = df['high'].values
        lows = df['low'].values
        
        # Encontrar picos (resistencias)
        resistance_peaks, _ = find_peaks(highs, distance=10, prominence=highs.std()*0.5)
        resistance_levels = sorted(highs[resistance_peaks], reverse=True)
        
        # Encontrar valles (soportes)
        support_peaks, _ = find_peaks(-lows, distance=10, prominence=lows.std()*0.5)
        support_levels = sorted(lows[support_peaks])
        
        return {
            'support': support_levels[:5],  # Top 5 support levels
            'resistance': resistance_levels[:5]  # Top 5 resistance levels
        }
    except Exception:
        # Fallback simple
        recent_data = df.tail(50)
        support_levels = [recent_data['low'].min(), recent_data['low'].quantile(0.25)]
        resistance_levels = [recent_data['high'].max(), recent_data['high'].quantile(0.75)]
        return {
            'support': support_levels,
            'resistance': resistance_levels
        }


def _calculate_fibonacci_levels(df):
    """Calcula niveles de retroceso de Fibonacci"""
    try:
        # Usar los últimos 100 períodos para encontrar swing high/low
        recent_data = df.tail(100)
        swing_high = recent_data['high'].max()
        swing_low = recent_data['low'].min()
        
        # Calcular niveles de Fibonacci
        diff = swing_high - swing_low
        
        fib_levels = {
            '0.0% (Swing High)': swing_high,
            '23.6%': swing_high - (diff * 0.236),
            '38.2%': swing_high - (diff * 0.382),
            '50.0%': swing_high - (diff * 0.5),
            '61.8%': swing_high - (diff * 0.618),
            '78.6%': swing_high - (diff * 0.786),
            '100.0% (Swing Low)': swing_low
        }
        
        return fib_levels
    except Exception:
        # Fallback
        current_price = df['close'].iloc[-1]
        return {
            'Current': current_price,
            '23.6%': current_price * 0.764,
            '38.2%': current_price * 0.618,
            '50.0%': current_price * 0.5,
            '61.8%': current_price * 0.382
        }


def _generate_trading_alerts(df, analysis, support_resistance, fib_levels):
    """Genera alertas de trading basadas en el análisis"""
    alerts = []
    current_price = df['close'].iloc[-1]
    
    try:
        # Alertas de soporte y resistencia
        for i, support in enumerate(support_resistance['support'][:2]):
            distance_pct = abs((current_price - support) / support) * 100
            if distance_pct < 2.0:  # Dentro del 2%
                alerts.append({
                    'type': 'warning',
                    'message': f'Precio cerca del soporte S{i+1} (${support:.4f}) - Distancia: {distance_pct:.2f}%'
                })
        
        for i, resistance in enumerate(support_resistance['resistance'][:2]):
            distance_pct = abs((current_price - resistance) / resistance) * 100
            if distance_pct < 2.0:  # Dentro del 2%
                alerts.append({
                    'type': 'warning',
                    'message': f'Precio cerca de la resistencia R{i+1} (${resistance:.4f}) - Distancia: {distance_pct:.2f}%'
                })
        
        # Alertas de RSI
        if 'rsi' in analysis.indicators:
            rsi_value = analysis.indicators['rsi']
            if hasattr(rsi_value, 'iloc'):
                rsi_value = rsi_value.iloc[-1]
            elif hasattr(rsi_value, '__len__') and len(rsi_value) > 0:
                rsi_value = rsi_value[-1]
            
            if rsi_value > 80:
                alerts.append({
                    'type': 'error',
                    'message': f'RSI extremadamente sobrecomprado ({rsi_value:.2f}) - Alta probabilidad de corrección'
                })
            elif rsi_value < 20:
                alerts.append({
                    'type': 'success',
                    'message': f'RSI extremadamente sobrevendido ({rsi_value:.2f}) - Posible oportunidad de compra'
                })
        
        # Alertas de volatilidad
        if 'volatility' in analysis.indicators:
            volatility = analysis.indicators['volatility']
            if hasattr(volatility, 'iloc'):
                volatility_value = volatility.iloc[-1]
            else:
                volatility_value = volatility
            
            if volatility_value > 50:  # Alta volatilidad
                alerts.append({
                    'type': 'warning',
                    'message': f'Alta volatilidad detectada ({volatility_value:.2f}%) - Aumentar gestión de riesgo'
                })
        
        # Alertas de patrones
        high_confidence_patterns = [p for p in analysis.patterns if p.confidence > 0.8]
        if high_confidence_patterns:
            for pattern in high_confidence_patterns[:2]:  # Solo los 2 mejores
                pattern_type = "alcista" if 'bullish' in pattern.pattern_type.lower() else "bajista" if 'bearish' in pattern.pattern_type.lower() else "neutral"
                alerts.append({
                    'type': 'success' if pattern_type == 'alcista' else 'error' if pattern_type == 'bajista' else 'info',
                    'message': f'Patrón {pattern_type} de alta confianza: {pattern.pattern_type} ({pattern.confidence:.3f})'
                })
        
        # Alertas de tendencia
        trend = analysis.trend_analysis.get('current_trend', 'Neutral')
        strength = analysis.trend_analysis.get('trend_strength', 0)
        
        if strength > 0.8:
            alerts.append({
                'type': 'success' if trend == 'Bullish' else 'error' if trend == 'Bearish' else 'info',
                'message': f'Tendencia {trend.lower()} muy fuerte detectada (fuerza: {strength:.3f})'
            })
        
        return alerts
        
    except Exception as e:
        return [{
            'type': 'error',
            'message': f'Error generando alertas: {str(e)}'
        }]