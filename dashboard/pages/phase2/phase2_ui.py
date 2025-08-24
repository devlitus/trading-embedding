#!/usr/bin/env python3
"""
Componentes de UI para Fase 2 - Análisis Técnico
"""

import streamlit as st
import plotly.graph_objects as go
from pages.common import get_plotly_config
from .phase2_config import safe_float
from .phase2_patterns import get_pattern_info, get_pattern_colors

def show_analysis_summary(analysis_result, df, symbol):
    """Mostrar resumen del análisis"""
    st.success("✅ Análisis completado")
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

def show_technical_indicators(indicators):
    """Mostrar indicadores técnicos"""
    st.subheader("📈 Indicadores Técnicos")
    
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

def show_detected_patterns(patterns):
    """Mostrar patrones detectados con información detallada"""
    if not patterns:
        return
    
    st.subheader("🎯 Patrones Detectados")
    pattern_info = get_pattern_info()
    
    for pattern in patterns:
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
                
                # Información del patrón detectado
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
                
                # Base teórica
                with st.expander("🎓 Fundamento Teórico"):
                    st.write(info["theory"])
            else:
                st.write("Información detallada no disponible para este patrón.")
                st.write(f"Patrón detectado desde el índice {pattern.start_idx} hasta {pattern.end_idx}")

def create_analysis_chart(df, symbol, interval, analysis_result):
    """Crear gráfico con indicadores y patrones"""
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
        _add_patterns_to_chart(fig, df, analysis_result.patterns)
    
    fig.update_layout(
        title=f"{symbol} - Análisis Técnico ({interval})",
        yaxis_title="Precio (USDT)",
        xaxis_title="Tiempo",
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())

def _add_patterns_to_chart(fig, df, patterns):
    """Agregar patrones al gráfico"""
    for i, pattern in enumerate(patterns):
        start_idx = pattern.start_idx
        end_idx = pattern.end_idx
        
        # Asegurar que los índices estén dentro del rango
        if start_idx < len(df) and end_idx < len(df) and start_idx >= 0 and end_idx >= 0:
            pattern_data = df.iloc[start_idx:end_idx+1]
            colors = get_pattern_colors(pattern.confidence)
            
            # Añadir área sombreada para el patrón
            fig.add_shape(
                type="rect",
                x0=pattern_data.index[0],
                y0=pattern_data['low'].min() * 0.999,
                x1=pattern_data.index[-1],
                y1=pattern_data['high'].max() * 1.001,
                fillcolor=colors['color'],
                line=dict(color=colors['border_color'], width=4, dash=colors['dash_pattern']),
                opacity=0.7
            )
            
            # Añadir anotación del patrón
            fig.add_annotation(
                x=pattern_data.index[len(pattern_data)//2],
                y=pattern_data['high'].max() * 1.008,
                text=f"<b>{colors['symbol']} {pattern.pattern_type}</b><br><b>Confianza: {pattern.confidence:.1%}</b>",
                showarrow=True,
                arrowhead=3,
                arrowsize=1.5,
                arrowcolor=colors['border_color'],
                arrowwidth=3,
                bgcolor=colors['bg_color'],
                bordercolor=colors['border_color'],
                borderwidth=4,
                font=dict(
                    size=14,
                    color=colors['text_color'],
                    family="Arial Black"
                ),
                opacity=1.0
            )

def show_detailed_analysis(analysis_result):
    """Mostrar análisis detallado en JSON"""
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