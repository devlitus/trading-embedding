import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
import logging

# Configurar rutas del proyecto
project_root = Path(r"c:\dev\trading_embedding")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))

try:
    from data.data_manager import DataManager
    from ml.labeling import (
        WyckoffHeuristicEngine,
        WyckoffScoringSystem,
        DatasetManager,
        LabeledSample
    )
except ImportError as e:
    st.error(f"Error importando componentes de ML: {e}")
    st.stop()

def show_phase3_labeling_page(data_manager):
    """Página principal del sistema de etiquetado Wyckoff - Fase 3 (Rediseñada)."""
    
    st.title("🏷️ Fase 3: Etiquetado Inteligente de Patrones Wyckoff")
    
    # Introducción clara y visual
    st.markdown("""
    <div style="background: linear-gradient(90deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin-bottom: 20px;">
        <h3>🎯 ¿Qué hace la Fase 3?</h3>
        <p><strong>Convierte datos de trading en conocimiento para IA:</strong></p>
        <ul>
            <li>🔍 <strong>Detecta automáticamente</strong> patrones Wyckoff en gráficos de precios</li>
            <li>📊 <strong>Evalúa la calidad</strong> de cada patrón encontrado</li>
            <li>🏷️ <strong>Etiqueta manualmente</strong> para crear datasets de entrenamiento</li>
            <li>🤖 <strong>Prepara datos</strong> para entrenar modelos de IA</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Flujo paso a paso visual
    st.markdown("""
    ### 🔄 Flujo de Trabajo Simplificado
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="text-align: center; padding: 20px; border: 3px solid #2E7D32; border-radius: 12px; background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%); box-shadow: 0 4px 8px rgba(46, 125, 50, 0.2);">
            <h4 style="color: #1B5E20; font-weight: bold; margin-bottom: 8px;">1️⃣ DETECTAR</h4>
            <p style="color: #2E7D32; font-weight: 500; margin: 0;">Busca patrones Wyckoff automáticamente</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 20px; border: 3px solid #1565C0; border-radius: 12px; background: linear-gradient(135deg, #E3F2FD 0%, #E8F4FD 100%); box-shadow: 0 4px 8px rgba(21, 101, 192, 0.2);">
            <h4 style="color: #0D47A1; font-weight: bold; margin-bottom: 8px;">2️⃣ EVALUAR</h4>
            <p style="color: #1565C0; font-weight: 500; margin: 0;">Puntúa la calidad de cada patrón</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="text-align: center; padding: 20px; border: 3px solid #E65100; border-radius: 12px; background: linear-gradient(135deg, #FFF3E0 0%, #FFF8F0 100%); box-shadow: 0 4px 8px rgba(230, 81, 0, 0.2);">
            <h4 style="color: #BF360C; font-weight: bold; margin-bottom: 8px;">3️⃣ ETIQUETAR</h4>
            <p style="color: #E65100; font-weight: 500; margin: 0;">Confirma o corrige manualmente</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div style="text-align: center; padding: 20px; border: 3px solid #6A1B9A; border-radius: 12px; background: linear-gradient(135deg, #F3E5F5 0%, #F8F5F9 100%); box-shadow: 0 4px 8px rgba(106, 27, 154, 0.2);">
            <h4 style="color: #4A148C; font-weight: bold; margin-bottom: 8px;">4️⃣ ENTRENAR</h4>
            <p style="color: #6A1B9A; font-weight: 500; margin: 0;">Crea datasets para IA</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Configuración simplificada en la parte superior
    st.markdown("### ⚙️ Configuración Rápida")
    
    col_config1, col_config2, col_config3 = st.columns(3)
    
    with col_config1:
        symbol = st.selectbox(
            "📈 Criptomoneda",
            ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"],
            index=0,
            help="Selecciona el par de trading a analizar"
        )
    
    with col_config2:
        timeframe = st.selectbox(
            "⏰ Temporalidad",
            ["1h", "4h", "1d", "1w"],
            index=2,
            help="Intervalo de tiempo de las velas"
        )
    
    with col_config3:
        days_back = st.slider(
            "📅 Período (días)",
            min_value=7,
            max_value=360,
            value=30,
            help="Cuántos días hacia atrás analizar"
        )
    
    # Tabs rediseñados con mejor UX
    tab1, tab2, tab3 = st.tabs([
        "🔍 Paso 1: Detectar Patrones",
        "🏷️ Paso 2: Etiquetar y Validar", 
        "📊 Paso 3: Gestionar Datasets"
    ])
    
    with tab1:
        show_pattern_detection_redesigned(data_manager, symbol, timeframe, days_back)
    
    with tab2:
        show_labeling_interface_redesigned(data_manager, symbol, timeframe, days_back)
    
    with tab3:
        show_dataset_management_redesigned(data_manager)

def show_pattern_detection_redesigned(data_manager, symbol, timeframe, days_back):
    """Interfaz rediseñada para detección de patrones - Paso 1."""
    
    st.markdown("""
    <div style="background: #2d5a2d; padding: 20px; border-radius: 12px; border-left: 6px solid #4CAF50; box-shadow: 0 4px 8px rgba(0,0,0,0.2); color: #ffffff;">
        <h4 style="color: #ffffff; margin-bottom: 15px; font-weight: 600;">🔍 Paso 1: Detección Automática de Patrones</h4>
        <p style="color: #e8f5e8; margin-bottom: 15px; font-size: 16px;">El sistema analiza los datos de precios y volumen para identificar automáticamente patrones Wyckoff como:</p>
        <ul style="color: #e8f5e8; font-size: 15px; line-height: 1.6;">
            <li><strong style="color: #ffffff;">Acumulación:</strong> Zonas donde grandes inversores compran discretamente</li>
            <li><strong style="color: #ffffff;">Distribución:</strong> Zonas donde grandes inversores venden discretamente</li>
            <li><strong style="color: #ffffff;">Re-acumulación:</strong> Pausas en tendencias alcistas</li>
            <li><strong style="color: #ffffff;">Re-distribución:</strong> Pausas en tendencias bajistas</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Configuración avanzada en expander
    with st.expander("⚙️ Configuración Avanzada de Detección"):
        col1, col2 = st.columns(2)
        
        with col1:
            min_confidence = st.slider(
                "🎯 Confianza Mínima",
                min_value=0.1,
                max_value=1.0,
                value=0.6,
                step=0.1,
                help="Solo mostrar patrones con esta confianza o mayor"
            )
            
            max_signals = st.number_input(
                "📊 Máximo de Señales",
                min_value=5,
                max_value=50,
                value=15,
                help="Limitar el número de patrones a mostrar"
            )
        
        with col2:
            volume_weight = st.slider(
                "📈 Peso del Volumen",
                min_value=0.1,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Importancia del volumen en la detección"
            )
            
            pattern_types = st.multiselect(
                "🎨 Tipos de Patrones",
                ["Acumulación", "Distribución", "Re-acumulación", "Re-distribución"],
                default=["Acumulación", "Distribución"],
                help="Selecciona qué tipos de patrones detectar"
            )
    
    # Botón principal de detección
    if st.button("🚀 Iniciar Detección de Patrones", type="primary", use_container_width=True):
        with st.spinner("🔍 Analizando datos y detectando patrones Wyckoff..."):
            try:
                # Mostrar progreso
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Paso 1: Obtener datos
                status_text.text("📊 Obteniendo datos históricos...")
                progress_bar.progress(20)
                
                end_time = datetime.now()
                start_time = end_time - timedelta(days=days_back)
                
                df = data_manager.get_data(
                    symbol=symbol,
                    interval=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if df.empty:
                    st.error("❌ No se pudieron obtener datos para el análisis")
                    return
                
                # Paso 2: Inicializar motor de detección
                status_text.text("🤖 Inicializando motor de detección...")
                progress_bar.progress(40)
                
                heuristic_engine = WyckoffHeuristicEngine()
                
                # Paso 3: Detectar patrones
                status_text.text("🔍 Detectando patrones Wyckoff...")
                progress_bar.progress(60)
                
                signals = heuristic_engine.analyze_pattern(df)
                
                # Paso 4: Filtrar resultados
                status_text.text("🎯 Filtrando y clasificando resultados...")
                progress_bar.progress(80)
                
                filtered_signals = [
                    s for s in signals 
                    if s.confidence >= min_confidence
                ][:max_signals]
                
                progress_bar.progress(100)
                status_text.text("✅ ¡Detección completada!")
                
                # Mostrar resultados de forma clara
                st.markdown("---")
                st.markdown("### 📊 Resultados de la Detección")
                
                # Métricas principales
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric(
                        "🎯 Patrones Encontrados",
                        len(filtered_signals),
                        delta=f"de {len(signals)} totales"
                    )
                
                with col2:
                    avg_confidence = np.mean([s.confidence for s in filtered_signals]) if filtered_signals else 0
                    st.metric(
                        "📈 Confianza Promedio",
                        f"{avg_confidence:.1%}",
                        delta=f"Min: {min_confidence:.1%}"
                    )
                
                with col3:
                    st.metric(
                        "📅 Período Analizado",
                        f"{days_back} días",
                        delta=f"{len(df)} velas"
                    )
                
                with col4:
                    price_range = df['high'].max() - df['low'].min()
                    st.metric(
                        "💰 Rango de Precios",
                        f"${price_range:.2f}",
                        delta=f"{symbol}"
                    )
                
                if filtered_signals:
                    # Gráfico interactivo mejorado
                    st.markdown("### 📈 Gráfico de Precios con Patrones Detectados")
                    
                    fig = create_enhanced_price_chart(df, filtered_signals, symbol)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabla de señales mejorada
                    st.markdown("### 📋 Detalle de Patrones Detectados")
                    
                    signals_data = []
                    for i, signal in enumerate(filtered_signals, 1):
                        # Determinar color según el tipo de patrón
                        phase_color = {
                            'accumulation': '🟢',
                            'distribution': '🔴',
                            'reaccumulation': '🟡',
                            'redistribution': '🟠'
                        }.get(signal.phase.value.lower(), '⚪')
                        
                        description = getattr(signal, 'description', 'Patrón Wyckoff detectado')
                        
                        signals_data.append({
                            "#": i,
                            "🕐 Fecha y Hora": signal.timestamp.strftime("%d/%m/%Y %H:%M"),
                            "🎨 Tipo": f"{phase_color} {signal.phase.value.title()}",
                            "💰 Precio": f"${signal.price:.2f}",
                            "🎯 Confianza": f"{signal.confidence:.1%}",
                            "📝 Descripción": description[:60] + "..." if len(description) > 60 else description
                        })
                    
                    signals_df = pd.DataFrame(signals_data)
                    st.dataframe(signals_df, use_container_width=True, hide_index=True)
                    
                    # Análisis estadístico
                    st.markdown("### 📊 Análisis Estadístico")
                    
                    col_stats1, col_stats2 = st.columns(2)
                    
                    with col_stats1:
                        # Distribución por tipos
                        phase_counts = {}
                        for signal in filtered_signals:
                            phase = signal.phase.value.title()
                            phase_counts[phase] = phase_counts.get(phase, 0) + 1
                        
                        if phase_counts:
                            fig_pie = px.pie(
                                values=list(phase_counts.values()),
                                names=list(phase_counts.keys()),
                                title="Distribución por Tipo de Patrón",
                                color_discrete_sequence=px.colors.qualitative.Set3
                            )
                            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
                            st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col_stats2:
                        # Distribución de confianza
                        confidence_values = [s.confidence for s in filtered_signals]
                        fig_hist = px.histogram(
                            x=confidence_values,
                            nbins=8,
                            title="Distribución de Niveles de Confianza",
                            labels={"x": "Confianza", "y": "Cantidad de Patrones"},
                            color_discrete_sequence=['#4CAF50']
                        )
                        fig_hist.update_layout(showlegend=False)
                        st.plotly_chart(fig_hist, use_container_width=True)
                    
                    # Guardar resultados para siguiente paso
                    st.session_state['detected_patterns'] = filtered_signals
                    st.session_state['analysis_data'] = df
                    st.session_state['analysis_symbol'] = symbol
                    st.session_state['analysis_timeframe'] = timeframe
                    
                    st.success("✅ ¡Patrones detectados exitosamente! Ahora puedes ir al **Paso 2** para etiquetarlos.")
                    
                else:
                    st.warning("⚠️ No se encontraron patrones que cumplan con los criterios especificados. Intenta reducir la confianza mínima o aumentar el período de análisis.")
                    
            except Exception as e:
                st.error(f"❌ Error durante la detección: {str(e)}")
                st.exception(e)

def create_enhanced_price_chart(df, signals, symbol):
    """Crea un gráfico de precios mejorado con patrones Wyckoff."""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'Precio de {symbol}', 'Volumen'),
        row_width=[0.7, 0.3]
    )
    
    # Gráfico de velas
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Precio',
            increasing_line_color='#26a69a',
            decreasing_line_color='#ef5350'
        ),
        row=1, col=1
    )
    
    # Agregar señales con colores distintivos
    colors = {
        'accumulation': '#4CAF50',
        'distribution': '#f44336',
        'reaccumulation': '#FF9800',
        'redistribution': '#9C27B0'
    }
    
    symbols_map = {
        'accumulation': 'triangle-up',
        'distribution': 'triangle-down',
        'reaccumulation': 'circle',
        'redistribution': 'square'
    }
    
    for signal in signals:
        phase_key = signal.phase.value.lower()
        color = colors.get(phase_key, '#757575')
        symbol_shape = symbols_map.get(phase_key, 'circle')
        
        fig.add_trace(
            go.Scatter(
                x=[signal.timestamp],
                y=[signal.price],
                mode='markers',
                marker=dict(
                    symbol=symbol_shape,
                    size=15,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                name=f'{signal.phase.value.title()} ({signal.confidence:.1%})',
                hovertemplate=f'<b>{signal.phase.value.title()}</b><br>' +
                             f'Precio: ${signal.price:.2f}<br>' +
                             f'Confianza: {signal.confidence:.1%}<br>' +
                             f'Fecha: {signal.timestamp.strftime("%d/%m/%Y %H:%M")}<extra></extra>'
            ),
            row=1, col=1
        )
    
    # Gráfico de volumen
    colors_volume = ['#26a69a' if close >= open else '#ef5350' 
                    for close, open in zip(df['close'], df['open'])]
    
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name='Volumen',
            marker_color=colors_volume,
            opacity=0.7
        ),
        row=2, col=1
    )
    
    # Configuración del layout
    fig.update_layout(
        title=f'Análisis de Patrones Wyckoff - {symbol}',
        xaxis_title='Fecha',
        yaxis_title='Precio (USD)',
        yaxis2_title='Volumen',
        height=700,
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

def show_labeling_interface_redesigned(data_manager, symbol, timeframe, days_back):
    """Interfaz rediseñada para etiquetado manual - Paso 2."""
    
    st.markdown("""
    <div style="background: #1a365d; padding: 20px; border-radius: 12px; border-left: 6px solid #2196F3; box-shadow: 0 4px 8px rgba(0,0,0,0.2); color: #ffffff;">
        <h4 style="color: #ffffff; margin-bottom: 15px; font-weight: 600;">🏷️ Paso 2: Etiquetado y Validación Manual</h4>
        <p style="color: #e3f2fd; margin-bottom: 15px; font-size: 16px;">Revisa y valida los patrones detectados automáticamente. Tu experiencia como trader es crucial para:</p>
        <ul style="color: #e3f2fd; font-size: 15px; line-height: 1.6;">
            <li><strong style="color: #ffffff;">Confirmar patrones válidos:</strong> Marca como correctos los patrones bien identificados</li>
            <li><strong style="color: #ffffff;">Rechazar falsos positivos:</strong> Descarta patrones incorrectos</li>
            <li><strong style="color: #ffffff;">Ajustar clasificaciones:</strong> Corrige el tipo de patrón si es necesario</li>
            <li><strong style="color: #ffffff;">Añadir contexto:</strong> Agrega notas y observaciones importantes</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar si hay patrones detectados
    if 'detected_patterns' not in st.session_state or not st.session_state['detected_patterns']:
        st.warning("⚠️ No hay patrones detectados. Ve al **Paso 1** para detectar patrones primero.")
        return
    
    patterns = st.session_state['detected_patterns']
    df = st.session_state.get('analysis_data')
    
    st.markdown(f"### 📊 Etiquetando {len(patterns)} patrones detectados")
    
    # Inicializar estado de etiquetado si no existe
    if 'labeling_results' not in st.session_state:
        st.session_state['labeling_results'] = {}
    
    # Selector de patrón para revisar
    pattern_options = []
    for i, pattern in enumerate(patterns):
        status = st.session_state['labeling_results'].get(i, {}).get('status', '⏳ Pendiente')
        pattern_options.append(
            f"{i+1}. {pattern.phase.value.title()} - {pattern.timestamp.strftime('%d/%m %H:%M')} - {status}"
        )
    
    selected_idx = st.selectbox(
        "🎯 Selecciona un patrón para revisar:",
        range(len(patterns)),
        format_func=lambda x: pattern_options[x]
    )
    
    if selected_idx is not None:
        current_pattern = patterns[selected_idx]
        
        # Mostrar detalles del patrón actual
        st.markdown("---")
        st.markdown(f"### 🔍 Revisando Patrón #{selected_idx + 1}")
        
        col_info1, col_info2, col_info3 = st.columns(3)
        
        with col_info1:
            st.metric("🎨 Tipo Detectado", current_pattern.phase.value.title())
            st.metric("💰 Precio", f"${current_pattern.price:.2f}")
        
        with col_info2:
            st.metric("🎯 Confianza IA", f"{current_pattern.confidence:.1%}")
            st.metric("🕐 Timestamp", current_pattern.timestamp.strftime("%d/%m/%Y %H:%M"))
        
        with col_info3:
            # Calcular puntuación si está disponible
            try:
                scoring_system = WyckoffScoringSystem()
                # Obtener ventana de datos alrededor del patrón
                signal_idx = df.index.get_loc(current_pattern.timestamp)
                window_start = max(0, signal_idx - 20)
                window_end = min(len(df), signal_idx + 21)
                window_data = df.iloc[window_start:window_end].copy()
                
                if not window_data.empty:
                    score = scoring_system.score_signal(current_pattern, window_data, signal_idx - window_start)
                    st.metric("⭐ Puntuación IA", f"{score.overall_score:.1f}/5")
                    st.metric("📊 Categoría", score.category)
                else:
                    st.metric("⭐ Puntuación IA", "N/A")
                    st.metric("📊 Categoría", "N/A")
            except:
                st.metric("⭐ Puntuación IA", "N/A")
                st.metric("📊 Categoría", "N/A")
        
        # Gráfico enfocado en el patrón actual
        st.markdown("### 📈 Vista Detallada del Patrón")
        
        # Crear gráfico centrado en el patrón
        if df is not None:
            pattern_time = current_pattern.timestamp
            
            # Obtener ventana de tiempo alrededor del patrón
            try:
                pattern_idx = df.index.get_loc(pattern_time)
                start_idx = max(0, pattern_idx - 50)
                end_idx = min(len(df), pattern_idx + 50)
                
                focused_df = df.iloc[start_idx:end_idx]
                focused_patterns = [current_pattern]  # Solo mostrar el patrón actual
                
                fig_focused = create_enhanced_price_chart(focused_df, focused_patterns, symbol)
                fig_focused.update_layout(title=f"Patrón #{selected_idx + 1}: {current_pattern.phase.value.title()}")
                st.plotly_chart(fig_focused, use_container_width=True)
                
            except Exception as e:
                st.error(f"Error creando gráfico: {e}")
        
        # Interfaz de etiquetado
        st.markdown("### ✅ Validación Manual")
        
        col_label1, col_label2 = st.columns(2)
        
        with col_label1:
            # Estado del patrón
            current_status = st.session_state['labeling_results'].get(selected_idx, {}).get('status', 'pending')
            
            status = st.radio(
                "🎯 ¿Es este patrón válido?",
                options=['valid', 'invalid', 'uncertain'],
                format_func=lambda x: {
                    'valid': '✅ Válido - Patrón correcto',
                    'invalid': '❌ Inválido - Falso positivo', 
                    'uncertain': '❓ Incierto - Necesita más análisis'
                }[x],
                index=['valid', 'invalid', 'uncertain'].index(current_status) if current_status in ['valid', 'invalid', 'uncertain'] else 0
            )
            
            # Tipo de patrón corregido
            current_type = st.session_state['labeling_results'].get(selected_idx, {}).get('corrected_type', current_pattern.phase.value)
            
            corrected_type = st.selectbox(
                "🎨 Tipo de patrón correcto:",
                options=['accumulation', 'distribution', 'reaccumulation', 'redistribution'],
                format_func=lambda x: {
                    'accumulation': '🟢 Acumulación',
                    'distribution': '🔴 Distribución',
                    'reaccumulation': '🟡 Re-acumulación',
                    'redistribution': '🟠 Re-distribución'
                }[x],
                index=['accumulation', 'distribution', 'reaccumulation', 'redistribution'].index(current_type) if current_type in ['accumulation', 'distribution', 'reaccumulation', 'redistribution'] else 0
            )
        
        with col_label2:
            # Confianza del trader
            trader_confidence = st.slider(
                "🎯 Tu nivel de confianza:",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state['labeling_results'].get(selected_idx, {}).get('trader_confidence', 0.5),
                step=0.1,
                help="¿Qué tan seguro estás de tu evaluación?"
            )
            
            # Calidad del patrón
            quality_score = st.slider(
                "⭐ Calidad del patrón (1-5):",
                min_value=1,
                max_value=5,
                value=st.session_state['labeling_results'].get(selected_idx, {}).get('quality_score', 3),
                help="Califica la claridad y fuerza del patrón"
            )
        
        # Notas del trader
        notes = st.text_area(
            "📝 Notas y observaciones:",
            value=st.session_state['labeling_results'].get(selected_idx, {}).get('notes', ''),
            placeholder="Agrega cualquier observación importante sobre este patrón...",
            height=100
        )
        
        # Botones de acción
        col_btn1, col_btn2, col_btn3 = st.columns(3)
        
        with col_btn1:
            if st.button("💾 Guardar Etiqueta", type="primary", use_container_width=True):
                # Guardar resultado del etiquetado
                st.session_state['labeling_results'][selected_idx] = {
                    'status': status,
                    'corrected_type': corrected_type,
                    'trader_confidence': trader_confidence,
                    'quality_score': quality_score,
                    'notes': notes,
                    'timestamp': datetime.now(),
                    'original_pattern': current_pattern
                }
                st.success("✅ Etiqueta guardada exitosamente!")
        
        with col_btn2:
            if st.button("⏭️ Siguiente Patrón", use_container_width=True):
                if selected_idx < len(patterns) - 1:
                    st.rerun()
        
        with col_btn3:
            if st.button("📊 Ver Progreso", use_container_width=True):
                show_labeling_progress(patterns)

def show_labeling_progress(patterns):
    """Muestra el progreso del etiquetado."""
    
    st.markdown("### 📊 Progreso del Etiquetado")
    
    total_patterns = len(patterns)
    labeled_count = len(st.session_state.get('labeling_results', {}))
    
    progress = labeled_count / total_patterns if total_patterns > 0 else 0
    
    st.progress(progress)
    st.write(f"**{labeled_count}/{total_patterns}** patrones etiquetados ({progress:.1%} completado)")
    
    if labeled_count > 0:
        # Estadísticas de etiquetado
        results = st.session_state['labeling_results']
        
        valid_count = sum(1 for r in results.values() if r['status'] == 'valid')
        invalid_count = sum(1 for r in results.values() if r['status'] == 'invalid')
        uncertain_count = sum(1 for r in results.values() if r['status'] == 'uncertain')
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("✅ Válidos", valid_count)
        with col2:
            st.metric("❌ Inválidos", invalid_count)
        with col3:
            st.metric("❓ Inciertos", uncertain_count)

def show_dataset_management_redesigned(data_manager):
    """Interfaz rediseñada para gestión de datasets - Paso 3."""
    
    st.markdown("""
    <div style="background: #4a1a4a; padding: 20px; border-radius: 12px; border-left: 6px solid #9C27B0; box-shadow: 0 4px 8px rgba(0,0,0,0.2); color: #ffffff;">
        <h4 style="color: #ffffff; margin-bottom: 15px; font-weight: 600;">📊 Paso 3: Gestión de Datasets de Entrenamiento</h4>
        <p style="color: #f3e5f5; margin-bottom: 15px; font-size: 16px;">Convierte tus etiquetas validadas en datasets listos para entrenar modelos de IA. Aquí puedes:</p>
        <ul style="color: #f3e5f5; font-size: 15px; line-height: 1.6;">
            <li><strong style="color: #ffffff;">Crear datasets:</strong> Genera conjuntos de datos estructurados desde tus etiquetas</li>
            <li><strong style="color: #ffffff;">Exportar datos:</strong> Descarga en formatos compatibles con frameworks de ML</li>
            <li><strong style="color: #ffffff;">Gestionar versiones:</strong> Mantén control de diferentes versiones de tus datasets</li>
            <li><strong style="color: #ffffff;">Análisis de calidad:</strong> Revisa estadísticas y distribución de tus datos</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # Verificar si hay etiquetas disponibles
    labeling_results = st.session_state.get('labeling_results', {})
    
    if not labeling_results:
        st.warning("⚠️ No hay etiquetas disponibles. Ve al **Paso 2** para etiquetar patrones primero.")
        return
    
    # Estadísticas de etiquetas disponibles
    st.markdown("### 📈 Resumen de Etiquetas Disponibles")
    
    valid_labels = {k: v for k, v in labeling_results.items() if v['status'] == 'valid'}
    total_labels = len(labeling_results)
    valid_count = len(valid_labels)
    
    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)
    
    with col_stats1:
        st.metric("📝 Total Etiquetas", total_labels)
    
    with col_stats2:
        st.metric("✅ Etiquetas Válidas", valid_count)
    
    with col_stats3:
        quality_avg = np.mean([v['quality_score'] for v in valid_labels.values()]) if valid_labels else 0
        st.metric("⭐ Calidad Promedio", f"{quality_avg:.1f}/5")
    
    with col_stats4:
        confidence_avg = np.mean([v['trader_confidence'] for v in valid_labels.values()]) if valid_labels else 0
        st.metric("🎯 Confianza Promedio", f"{confidence_avg:.1%}")
    
    if valid_count == 0:
        st.warning("⚠️ No hay etiquetas válidas para crear datasets. Marca algunos patrones como válidos en el Paso 2.")
        return
    
    # Distribución por tipo de patrón
    st.markdown("### 📊 Distribución de Patrones Válidos")
    
    pattern_counts = {}
    for label_data in valid_labels.values():
        pattern_type = label_data['corrected_type']
        pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
    
    if pattern_counts:
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            # Gráfico de barras
            fig_bar = px.bar(
                x=list(pattern_counts.keys()),
                y=list(pattern_counts.values()),
                title="Distribución por Tipo de Patrón",
                labels={'x': 'Tipo de Patrón', 'y': 'Cantidad'},
                color=list(pattern_counts.values()),
                color_continuous_scale='viridis'
            )
            fig_bar.update_layout(height=400)
            st.plotly_chart(fig_bar, use_container_width=True)
        
        with col_chart2:
            # Gráfico de pastel
            fig_pie = px.pie(
                values=list(pattern_counts.values()),
                names=list(pattern_counts.keys()),
                title="Proporción de Patrones"
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
    
    # Configuración del dataset
    st.markdown("### ⚙️ Configuración del Dataset")
    
    col_config1, col_config2 = st.columns(2)
    
    with col_config1:
        dataset_name = st.text_input(
            "📝 Nombre del Dataset:",
            value=f"wyckoff_patterns_{datetime.now().strftime('%Y%m%d_%H%M')}",
            help="Nombre único para identificar este dataset"
        )
        
        include_uncertain = st.checkbox(
            "❓ Incluir patrones inciertos",
            value=False,
            help="Incluir también los patrones marcados como inciertos"
        )
    
    with col_config2:
        min_quality = st.slider(
            "⭐ Calidad mínima requerida:",
            min_value=1,
            max_value=5,
            value=3,
            help="Solo incluir patrones con esta calidad o superior"
        )
        
        min_confidence = st.slider(
            "🎯 Confianza mínima del trader:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Solo incluir patrones con esta confianza o superior"
        )
    
    # Filtrar datos según configuración
    filtered_labels = {}
    for k, v in labeling_results.items():
        if v['status'] == 'valid' or (include_uncertain and v['status'] == 'uncertain'):
            if v['quality_score'] >= min_quality and v['trader_confidence'] >= min_confidence:
                filtered_labels[k] = v
    
    st.markdown(f"### 🎯 Dataset Resultante: {len(filtered_labels)} muestras")
    
    if len(filtered_labels) == 0:
        st.warning("⚠️ No hay muestras que cumplan los criterios seleccionados. Ajusta los filtros.")
        return
    
    # Mostrar preview del dataset
    with st.expander("👀 Vista Previa del Dataset", expanded=False):
        preview_data = []
        for idx, (k, v) in enumerate(list(filtered_labels.items())[:10]):
            preview_data.append({
                'ID': k,
                'Tipo': v['corrected_type'].title(),
                'Calidad': v['quality_score'],
                'Confianza Trader': f"{v['trader_confidence']:.1%}",
                'Confianza IA': f"{v['original_pattern'].confidence:.1%}",
                'Precio': f"${v['original_pattern'].price:.2f}",
                'Fecha': v['original_pattern'].timestamp.strftime('%d/%m/%Y %H:%M'),
                'Notas': v['notes'][:50] + '...' if len(v['notes']) > 50 else v['notes']
            })
        
        if preview_data:
            df_preview = pd.DataFrame(preview_data)
            st.dataframe(df_preview, use_container_width=True)
            
            if len(filtered_labels) > 10:
                st.info(f"📋 Mostrando 10 de {len(filtered_labels)} muestras totales")
    
    # Botones de acción
    st.markdown("### 🚀 Acciones del Dataset")
    
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    
    with col_btn1:
        if st.button("💾 Crear Dataset", type="primary", use_container_width=True):
            try:
                # Crear dataset usando DatasetManager
                dataset_manager = DatasetManager()
                
                # Convertir etiquetas a LabeledSamples
                labeled_samples = []
                for k, v in filtered_labels.items():
                    sample = LabeledSample(
                        signal=v['original_pattern'],
                        label=v['corrected_type'],
                        confidence=v['trader_confidence'],
                        metadata={
                            'quality_score': v['quality_score'],
                            'notes': v['notes'],
                            'labeling_timestamp': v['timestamp'].isoformat(),
                            'ai_confidence': v['original_pattern'].confidence
                        }
                    )
                    labeled_samples.append(sample)
                
                # Crear dataset
                dataset_id = dataset_manager.create_dataset(
                    name=dataset_name,
                    samples=labeled_samples,
                    metadata={
                        'creation_method': 'manual_labeling',
                        'total_samples': len(labeled_samples),
                        'min_quality': min_quality,
                        'min_confidence': min_confidence,
                        'include_uncertain': include_uncertain
                    }
                )
                
                st.success(f"✅ Dataset '{dataset_name}' creado exitosamente con ID: {dataset_id}")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error creando dataset: {e}")
    
    with col_btn2:
        if st.button("📤 Exportar CSV", use_container_width=True):
            try:
                # Crear DataFrame para exportar
                export_data = []
                for k, v in filtered_labels.items():
                    export_data.append({
                        'pattern_id': k,
                        'pattern_type': v['corrected_type'],
                        'timestamp': v['original_pattern'].timestamp.isoformat(),
                        'price': v['original_pattern'].price,
                        'ai_confidence': v['original_pattern'].confidence,
                        'trader_confidence': v['trader_confidence'],
                        'quality_score': v['quality_score'],
                        'status': v['status'],
                        'notes': v['notes'],
                        'labeling_timestamp': v['timestamp'].isoformat()
                    })
                
                df_export = pd.DataFrame(export_data)
                csv = df_export.to_csv(index=False)
                
                st.download_button(
                    label="⬇️ Descargar CSV",
                    data=csv,
                    file_name=f"{dataset_name}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ Error exportando CSV: {e}")
    
    with col_btn3:
        if st.button("📋 Ver Datasets Existentes", use_container_width=True):
            show_existing_datasets()

def show_existing_datasets():
    """Muestra los datasets existentes en el sistema."""
    
    st.markdown("### 📚 Datasets Existentes")
    
    try:
        dataset_manager = DatasetManager()
        datasets = dataset_manager.list_datasets()
        
        if not datasets:
            st.info("📭 No hay datasets creados aún.")
            return
        
        # Mostrar datasets en una tabla
        dataset_data = []
        for dataset in datasets:
            dataset_data.append({
                'ID': dataset['dataset_id'],
                'Nombre': dataset['name'],
                'Muestras': dataset.get('sample_count', 'N/A'),
                'Fecha Creación': dataset['creation_date'],
                'Método': dataset.get('metadata', {}).get('creation_method', 'N/A')
            })
        
        df_datasets = pd.DataFrame(dataset_data)
        st.dataframe(df_datasets, use_container_width=True)
        
        # Selector para ver detalles
        if len(datasets) > 0:
            selected_dataset = st.selectbox(
                "🔍 Ver detalles del dataset:",
                options=range(len(datasets)),
                format_func=lambda x: f"{datasets[x]['name']} ({datasets[x]['dataset_id']})"
            )
            
            if selected_dataset is not None:
                dataset = datasets[selected_dataset]
                
                st.markdown(f"#### 📊 Detalles: {dataset['name']}")
                
                col_detail1, col_detail2 = st.columns(2)
                
                with col_detail1:
                    st.write(f"**ID:** {dataset['dataset_id']}")
                    st.write(f"**Nombre:** {dataset['name']}")
                    st.write(f"**Fecha:** {dataset['creation_date']}")
                
                with col_detail2:
                    st.write(f"**Muestras:** {dataset.get('sample_count', 'N/A')}")
                    metadata = dataset.get('metadata', {})
                    st.write(f"**Método:** {metadata.get('creation_method', 'N/A')}")
                    st.write(f"**Calidad mín:** {metadata.get('min_quality', 'N/A')}")
                
                # Botón para exportar dataset
                if st.button(f"📤 Exportar {dataset['name']}", key=f"export_{dataset['dataset_id']}"):
                    try:
                        samples = dataset_manager.get_dataset_samples(dataset['dataset_id'])
                        
                        export_data = []
                        for sample in samples:
                            export_data.append({
                                'pattern_type': sample.label,
                                'timestamp': sample.signal.timestamp.isoformat(),
                                'price': sample.signal.price,
                                'confidence': sample.confidence,
                                'metadata': str(sample.metadata)
                            })
                        
                        df_export = pd.DataFrame(export_data)
                        csv = df_export.to_csv(index=False)
                        
                        st.download_button(
                            label=f"⬇️ Descargar {dataset['name']}.csv",
                            data=csv,
                            file_name=f"{dataset['name']}.csv",
                            mime="text/csv",
                            key=f"download_{dataset['dataset_id']}"
                        )
                        
                    except Exception as e:
                        st.error(f"❌ Error exportando dataset: {e}")
    
    except Exception as e:
        st.error(f"❌ Error cargando datasets: {e}")

def show_automatic_detection(data_manager, symbol, timeframe, days_back, min_confidence, max_signals):
    """Muestra la funcionalidad de detección automática de patrones."""
    
    st.header("🔍 Detección Automática de Patrones Wyckoff")
    st.markdown("""
    Utiliza reglas heurísticas avanzadas para detectar automáticamente patrones 
    de acumulación y distribución según la metodología Wyckoff.
    """)
    
    if st.button("🚀 Ejecutar Detección", type="primary"):
        with st.spinner("Analizando datos y detectando patrones..."):
            try:
                # Inicializar componentes
                heuristic_engine = WyckoffHeuristicEngine()
                
                # Obtener datos
                end_time = datetime.now()
                start_time = end_time - timedelta(days=days_back)
                
                df = data_manager.get_data(
                    symbol=symbol,
                    interval=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if df.empty:
                    st.error("❌ No se pudieron obtener datos para el análisis")
                    return
                
                # Detectar patrones
                signals = heuristic_engine.analyze_pattern(df)
                
                # Filtrar por confianza
                filtered_signals = [
                    s for s in signals 
                    if s.confidence >= min_confidence
                ][:max_signals]
                
                # Mostrar resultados
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        "Total Señales",
                        len(signals),
                        delta=f"+{len(filtered_signals)} filtradas"
                    )
                
                with col2:
                    avg_confidence = np.mean([s.confidence for s in filtered_signals]) if filtered_signals else 0
                    st.metric(
                        "Confianza Promedio",
                        f"{avg_confidence:.2f}",
                        delta=f"Min: {min_confidence}"
                    )
                
                with col3:
                    st.metric(
                        "Período Analizado",
                        f"{days_back} días",
                        delta=f"{len(df)} velas"
                    )
                
                if filtered_signals:
                    # Gráfico de precio con señales
                    fig = create_price_chart_with_signals(df, filtered_signals)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabla de señales
                    st.subheader("📋 Señales Detectadas")
                    
                    signals_data = []
                    for signal in filtered_signals:
                        description = getattr(signal, 'description', 'Sin descripción')
                        signals_data.append({
                            "Timestamp": signal.timestamp.strftime("%Y-%m-%d %H:%M"),
                            "Fase": signal.phase.value,
                            "Precio": f"${signal.price:.2f}",
                            "Confianza": f"{signal.confidence:.2f}",
                            "Descripción": description[:50] + "..." if len(description) > 50 else description
                        })
                    
                    signals_df = pd.DataFrame(signals_data)
                    st.dataframe(signals_df, use_container_width=True)
                    
                    # Distribución por fases
                    phase_counts = {}
                    for signal in filtered_signals:
                        phase = signal.phase.value
                        phase_counts[phase] = phase_counts.get(phase, 0) + 1
                    
                    if phase_counts:
                        st.subheader("📊 Distribución por Fases")
                        
                        fig_pie = px.pie(
                            values=list(phase_counts.values()),
                            names=list(phase_counts.keys()),
                            title="Distribución de Señales por Fase Wyckoff"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                
                else:
                    st.warning(f"⚠️ No se encontraron señales con confianza >= {min_confidence}")
                    
            except Exception as e:
                st.error(f"❌ Error en la detección: {str(e)}")
                st.exception(e)

def show_scoring_system(data_manager, symbol, timeframe, days_back, min_confidence):
    """Muestra el sistema de puntuación de señales."""
    
    st.header("📊 Sistema de Puntuación de Señales")
    st.markdown("""
    Evalúa la calidad de las señales detectadas mediante un sistema 
    multi-dimensional que considera múltiples factores técnicos.
    """)
    
    if st.button("📈 Analizar y Puntuar", type="primary"):
        with st.spinner("Ejecutando análisis de puntuación..."):
            try:
                # Inicializar componentes
                heuristic_engine = WyckoffHeuristicEngine()
                scoring_system = WyckoffScoringSystem()
                
                # Obtener datos
                end_time = datetime.now()
                start_time = end_time - timedelta(days=days_back)
                
                df = data_manager.get_data(
                    symbol=symbol,
                    interval=timeframe,
                    start_time=start_time,
                    end_time=end_time
                )
                
                if df.empty:
                    st.error("❌ No se pudieron obtener datos")
                    return
                
                # Detectar señales
                signals = heuristic_engine.analyze_pattern(df)
                filtered_signals = [s for s in signals if s.confidence >= min_confidence]
                
                if not filtered_signals:
                    st.warning("⚠️ No hay señales para puntuar")
                    return
                
                # Puntuar señales
                scored_signals = []
                for signal in filtered_signals[:10]:  # Limitar a 10 para rendimiento
                    try:
                        # Obtener ventana de datos
                        signal_idx = df[df.index <= signal.timestamp].index[-1]
                        signal_index = df.index.get_loc(signal_idx)
                        window_start = max(0, signal_index - 50)
                        window_end = min(len(df), signal_index + 50)
                        window_data = df.iloc[window_start:window_end]
                        
                        # Calcular puntuación
                        score = scoring_system.score_signal(signal, window_data, signal_index - window_start)
                        scored_signals.append((signal, score))
                        
                    except Exception as e:
                        st.warning(f"Error puntuando señal: {e}")
                        continue
                
                if scored_signals:
                    # Mostrar métricas generales
                    scores = [score for _, score in scored_signals]
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric(
                            "Señales Puntuadas",
                            len(scored_signals),
                            delta=f"de {len(filtered_signals)} total"
                        )
                    
                    with col2:
                        avg_score = np.mean([s.overall_score for s in scores])
                        st.metric(
                            "Puntuación Promedio",
                            f"{avg_score:.2f}/5",
                            delta=f"Rango: {min([s.overall_score for s in scores]):.1f}-{max([s.overall_score for s in scores]):.1f}"
                        )
                    
                    with col3:
                        high_quality = len([s for s in scores if s.overall_score >= 4.0])
                        st.metric(
                            "Alta Calidad (≥4.0)",
                            high_quality,
                            delta=f"{(high_quality/len(scores)*100):.1f}%"
                        )
                    
                    # Tabla detallada de puntuaciones
                    st.subheader("🎯 Análisis Detallado de Puntuaciones")
                    
                    scoring_data = []
                    for signal, score in scored_signals:
                        scoring_data.append({
                            "Timestamp": signal.timestamp.strftime("%Y-%m-%d %H:%M"),
                            "Fase": signal.phase.value,
                            "Precio": f"${signal.price:.2f}",
                            "Puntuación Total": f"{score.overall_score:.2f}/5",
                            "Categoría": score.category,
                            "Volumen": f"{score.volume_score:.1f}",
                            "Momentum": f"{score.momentum_score:.1f}",
                            "Estructura": f"{score.structure_score:.1f}",
                            "Sugerencias": ", ".join(score.improvement_suggestions[:2])
                        })
                    
                    scoring_df = pd.DataFrame(scoring_data)
                    st.dataframe(scoring_df, use_container_width=True)
                    
                    # Gráfico de distribución de puntuaciones
                    st.subheader("📊 Distribución de Puntuaciones")
                    
                    fig_hist = px.histogram(
                        x=[s.overall_score for s in scores],
                        nbins=10,
                        title="Distribución de Puntuaciones Generales",
                        labels={"x": "Puntuación", "y": "Frecuencia"}
                    )
                    st.plotly_chart(fig_hist, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error en el sistema de puntuación: {str(e)}")
                st.exception(e)

def show_dataset_management(data_manager):
    """Muestra la gestión de datasets etiquetados."""
    
    st.header("💾 Gestión de Datasets Etiquetados")
    st.markdown("""
    Administra datasets de patrones Wyckoff etiquetados para entrenamiento 
    de modelos de machine learning.
    """)
    
    try:
        dataset_manager = DatasetManager()
        
        # Mostrar datasets existentes
        st.subheader("📚 Datasets Existentes")
        
        datasets = dataset_manager.list_datasets()
        
        if datasets:
            for dataset in datasets:
                with st.expander(f"📊 {dataset['name']} (ID: {dataset['dataset_id']})"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Descripción:** {dataset['description']}")
                        st.write(f"**Creado:** {dataset['creation_date']}")
                        st.write(f"**Sesiones:** {len(dataset.get('annotation_sessions', []))}")
                    
                    with col2:
                        # Obtener estadísticas del dataset
                        try:
                            summary = dataset_manager.get_dataset_summary(dataset['dataset_id'])
                            if summary:
                                st.metric("Total Muestras", summary['total_samples'])
                                st.metric("Muestras Válidas", summary['valid_samples'])
                                st.metric("Calidad Promedio", f"{summary['average_quality_score']:.2f}/5")
                        except:
                            st.write("Estadísticas no disponibles")
                    
                    # Botones de acción
                    col_btn1, col_btn2, col_btn3 = st.columns(3)
                    
                    with col_btn1:
                        if st.button(f"📥 Exportar", key=f"export_{dataset['id']}"):
                            try:
                                export_path = dataset_manager.export_dataset(
                                    dataset['id'], 
                                    format='csv'
                                )
                                st.success(f"✅ Dataset exportado a: {export_path}")
                            except Exception as e:
                                st.error(f"❌ Error exportando: {e}")
                    
                    with col_btn2:
                        if st.button(f"🤖 Preparar ML", key=f"ml_{dataset['id']}"):
                            try:
                                (X_train, y_train), (X_test, y_test) = dataset_manager.get_training_data(dataset['id'])
                                st.success(f"✅ Datos ML preparados: {X_train.shape[0]} muestras")
                                
                                # Mostrar información de las características
                                st.write(f"**Características:** {X_train.shape[1]}")
                                st.write(f"**Entrenamiento:** {X_train.shape[0]} muestras")
                                st.write(f"**Prueba:** {X_test.shape[0]} muestras")
                                
                            except Exception as e:
                                st.error(f"❌ Error preparando ML: {e}")
                    
                    with col_btn3:
                        if st.button(f"🗑️ Eliminar", key=f"delete_{dataset['id']}"):
                            if st.session_state.get(f"confirm_delete_{dataset['id']}", False):
                                try:
                                    dataset_manager.delete_dataset(dataset['id'])
                                    st.success("✅ Dataset eliminado")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"❌ Error eliminando: {e}")
                            else:
                                st.session_state[f"confirm_delete_{dataset['id']}"] = True
                                st.warning("⚠️ Haz clic nuevamente para confirmar")
        else:
            st.info("📝 No hay datasets creados aún. Usa el flujo integrado para crear uno.")
        
        # Estadísticas generales
        st.subheader("📈 Estadísticas Generales")
        
        total_datasets = len(datasets)
        total_samples = sum(dataset_manager.get_dataset_summary(d['id'])['total_samples'] 
                          for d in datasets 
                          if dataset_manager.get_dataset_summary(d['id']))
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Datasets", total_datasets)
        
        with col2:
            st.metric("Total Muestras", total_samples)
        
        with col3:
            avg_quality = 0
            if datasets:
                qualities = []
                for d in datasets:
                    summary = dataset_manager.get_dataset_summary(d['id'])
                    if summary and summary['average_quality_score']:
                        qualities.append(summary['average_quality_score'])
                avg_quality = np.mean(qualities) if qualities else 0
            
            st.metric("Calidad Promedio", f"{avg_quality:.2f}/5")
        
    except Exception as e:
        st.error(f"❌ Error en gestión de datasets: {str(e)}")
        st.exception(e)

def show_integrated_workflow(data_manager, symbol, timeframe, days_back):
    """Muestra el flujo de trabajo integrado completo."""
    
    st.header("🔄 Flujo de Trabajo Integrado")
    st.markdown("""
    Ejecuta el proceso completo desde la detección automática hasta la 
    preparación de datos para machine learning.
    """)
    
    # Configuración del flujo
    with st.expander("⚙️ Configuración del Flujo", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            dataset_name = st.text_input(
                "Nombre del Dataset",
                value=f"Dataset {symbol} {timeframe} {datetime.now().strftime('%Y%m%d')}"
            )
            
            min_confidence_flow = st.slider(
                "Confianza mínima para incluir",
                min_value=0.1,
                max_value=1.0,
                value=0.6,
                step=0.1
            )
        
        with col2:
            dataset_description = st.text_area(
                "Descripción del Dataset",
                value=f"Dataset generado automáticamente para {symbol} en timeframe {timeframe}"
            )
            
            max_samples = st.number_input(
                "Máximo de muestras",
                min_value=10,
                max_value=1000,
                value=100,
                step=10
            )
    
    if st.button("🚀 Ejecutar Flujo Completo", type="primary", use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            # Paso 1: Inicialización
            status_text.text("🔧 Paso 1/5: Inicializando componentes...")
            progress_bar.progress(0.1)
            
            heuristic_engine = WyckoffHeuristicEngine()
            scoring_system = WyckoffScoringSystem()
            dataset_manager = DatasetManager()
            
            # Paso 2: Obtención de datos
            status_text.text("📊 Paso 2/5: Obteniendo datos históricos...")
            progress_bar.progress(0.2)
            
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days_back)
            
            df = data_manager.get_data(
                symbol=symbol,
                interval=timeframe,
                start_time=start_time,
                end_time=end_time
            )
            
            if df.empty:
                st.error("❌ No se pudieron obtener datos")
                return
            
            st.success(f"✅ Datos obtenidos: {len(df)} velas")
            
            # Paso 3: Detección y puntuación
            status_text.text("🔍 Paso 3/5: Detectando y puntuando patrones...")
            progress_bar.progress(0.4)
            
            signals = heuristic_engine.analyze_pattern(df)
            filtered_signals = [s for s in signals if s.confidence >= min_confidence_flow][:max_samples]
            
            if not filtered_signals:
                st.warning("⚠️ No se encontraron señales con la confianza especificada")
                return
            
            st.info(f"🎯 Señales detectadas: {len(filtered_signals)}")
            
            # Puntuar señales
            scored_samples = []
            for i, signal in enumerate(filtered_signals):
                try:
                    # Obtener ventana de datos
                    signal_idx = df[df.index <= signal.timestamp].index[-1]
                    signal_index = df.index.get_loc(signal_idx)
                    window_start = max(0, signal_index - 50)
                    window_end = min(len(df), signal_index + 50)
                    window_data = df.iloc[window_start:window_end]
                    
                    # Calcular puntuación
                    score = scoring_system.score_signal(signal, window_data, signal_index - window_start)
                    
                    # Crear muestra etiquetada
                    sample = LabeledSample(
                        timestamp=signal.timestamp,
                        symbol=symbol,
                        timeframe=timeframe,
                        phase=signal.phase,
                        confidence=signal.confidence,
                        price=signal.price,
                        volume=window_data.loc[signal_idx, 'volume'] if signal_idx in window_data.index else 0,
                        features={
                            'reasoning': getattr(signal, 'description', 'Sin descripción'),
                            'volume_score': score.volume_score,
                            'momentum_score': score.momentum_score,
                            'structure_score': score.structure_score,
                            'overall_score': score.overall_score,
                            'category': score.category
                        }
                    )
                    
                    scored_samples.append(sample)
                    
                    # Actualizar progreso
                    progress = 0.4 + (0.3 * (i + 1) / len(filtered_signals))
                    progress_bar.progress(progress)
                    
                except Exception as e:
                    st.warning(f"Error procesando señal {i+1}: {e}")
                    continue
            
            if not scored_samples:
                st.error("❌ No se pudieron procesar las señales")
                return
            
            # Paso 4: Crear dataset
            status_text.text("💾 Paso 4/5: Creando dataset...")
            progress_bar.progress(0.8)
            
            # Crear sesión de anotación
            session_data = {
                'session_id': f"auto_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'symbol': symbol,
                'timeframe': timeframe,
                'start_time': start_time,
                'end_time': end_time,
                'samples': scored_samples,
                'metadata': {
                    'detection_method': 'automatic_heuristic',
                    'min_confidence': min_confidence_flow,
                    'total_signals_detected': len(signals),
                    'signals_included': len(scored_samples)
                }
            }
            
            dataset_manager.save_annotations(session_data)
            dataset_id = dataset_manager.create_dataset(
                name=dataset_name,
                description=dataset_description,
                annotation_sessions=[session_data['session_id']]
            )
            
            st.success(f"✅ Dataset creado con ID: {dataset_id}")
            
            # Paso 5: Preparar para ML
            status_text.text("🤖 Paso 5/5: Preparando datos para ML...")
            progress_bar.progress(0.9)
            
            try:
                (X_train, y_train), (X_test, y_test) = dataset_manager.get_training_data(dataset_id)
                progress_bar.progress(1.0)
                
                # Mostrar resultados finales
                st.success("🎉 ¡Flujo completo ejecutado exitosamente!")
                
                # Métricas finales
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Muestras Procesadas", len(scored_samples))
                
                with col2:
                    avg_confidence = np.mean([s.confidence for s in scored_samples])
                    st.metric("Confianza Promedio", f"{avg_confidence:.2f}")
                
                with col3:
                    st.metric("Datos Entrenamiento", X_train.shape[0])
                
                with col4:
                    st.metric("Datos Prueba", X_test.shape[0])
                
                # Resumen del dataset
                summary = dataset_manager.get_dataset_summary(dataset_id)
                if summary:
                    st.subheader("📊 Resumen del Dataset Creado")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Muestras", summary['total_samples'])
                    
                    with col2:
                        st.metric("Muestras Válidas", summary['valid_samples'])
                    
                    with col3:
                        st.metric("Calidad Promedio", f"{summary['average_quality_score']:.2f}/5")
                
                # Información adicional
                with st.expander("ℹ️ Información del Dataset"):
                    st.write(f"**ID del Dataset:** {dataset_id}")
                    st.write(f"**Nombre:** {dataset_name}")
                    st.write(f"**Descripción:** {dataset_description}")
                    st.write(f"**Símbolo:** {symbol}")
                    st.write(f"**Timeframe:** {timeframe}")
                    st.write(f"**Período:** {start_time.strftime('%Y-%m-%d')} a {end_time.strftime('%Y-%m-%d')}")
                    st.write(f"**Características ML:** {X_train.shape[1]}")
                
            except Exception as e:
                st.error(f"❌ Error preparando datos ML: {e}")
                st.info("💡 El dataset se creó correctamente, pero hubo un problema preparando los datos para ML")
            
        except Exception as e:
            st.error(f"❌ Error en el flujo integrado: {str(e)}")
            st.exception(e)
        
        finally:
            progress_bar.empty()
            status_text.empty()

def create_price_chart_with_signals(df, signals):
    """Crea un gráfico de precio con las señales detectadas."""
    
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Precio con Señales Wyckoff', 'Volumen'),
        row_width=[0.7, 0.3]
    )
    
    # Gráfico de velas
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Precio"
        ),
        row=1, col=1
    )
    
    # Añadir señales
    colors = {
        'Acumulación': 'green',
        'Distribución': 'red',
        'Re-acumulación': 'blue',
        'Re-distribución': 'orange'
    }
    
    for signal in signals:
        color = colors.get(signal.phase.value, 'purple')
        
        fig.add_trace(
            go.Scatter(
                x=[signal.timestamp],
                y=[signal.price],
                mode='markers',
                marker=dict(
                    symbol='triangle-up' if 'acumulación' in signal.phase.value.lower() else 'triangle-down',
                    size=15,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                name=f"{signal.phase.value} ({signal.confidence:.2f})",
                hovertemplate=f"<b>{signal.phase.value}</b><br>" +
                             f"Precio: ${signal.price:.2f}<br>" +
                             f"Confianza: {signal.confidence:.2f}<br>" +
                             f"Descripción: {signal.description}<br>" +
                             "<extra></extra>"
            ),
            row=1, col=1
        )
    
    # Gráfico de volumen
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name="Volumen",
            marker_color='lightblue',
            opacity=0.7
        ),
        row=2, col=1
    )
    
    fig.update_layout(
        title="Análisis de Patrones Wyckoff",
        xaxis_title="Tiempo",
        yaxis_title="Precio",
        height=800,
        showlegend=True
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig