#!/usr/bin/env python3
"""
Componentes de UI para Fase 3 - Etiquetado Wyckoff
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional

# Imports locales
from .phase3_charts import (
    create_enhanced_price_chart,
    create_pattern_distribution_chart,
    create_confidence_distribution_chart,
    create_focused_pattern_chart,
    create_labeling_progress_chart,
    create_quality_score_chart
)
from .phase3_logic import (
    get_labeling_progress,
    save_pattern_label,
    get_pattern_statistics
)

def show_detection_results(signals: List, metrics: Dict, df: pd.DataFrame, symbol: str) -> None:
    """Mostrar resultados de detección de patrones."""
    if not signals:
        st.warning("⚠️ No se detectaron patrones con los criterios especificados")
        st.info("💡 Intenta ajustar los parámetros de detección (reducir confianza mínima o cambiar tipos de patrones)")
        return
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎯 Patrones Detectados", metrics.get('total_patterns', 0))
    
    with col2:
        avg_conf = metrics.get('avg_confidence', 0)
        st.metric("📊 Confianza Promedio", f"{avg_conf:.2f}")
    
    with col3:
        st.metric("🔥 Alta Confianza", metrics.get('high_confidence', 0))
    
    with col4:
        st.metric("📈 Tipos Únicos", metrics.get('pattern_types', 0))
    
    st.markdown("---")
    
    # Gráfico principal
    st.subheader("📈 Gráfico de Precios con Patrones")
    
    try:
        chart = create_enhanced_price_chart(df, signals, symbol)
        if chart:
            st.plotly_chart(chart, use_container_width=True)
        else:
            st.error("❌ Error al crear el gráfico")
    except Exception as e:
        st.error(f"❌ Error al mostrar gráfico: {str(e)}")
    
    # Gráficos de distribución
    col1, col2 = st.columns(2)
    
    with col1:
        dist_chart = create_pattern_distribution_chart(signals)
        if dist_chart:
            st.plotly_chart(dist_chart, use_container_width=True)
    
    with col2:
        conf_chart = create_confidence_distribution_chart(signals)
        if conf_chart:
            st.plotly_chart(conf_chart, use_container_width=True)
    
    # Tabla detallada de señales
    st.subheader("📋 Tabla Detallada de Señales")
    
    signals_data = []
    for i, signal in enumerate(signals):
        signals_data.append({
            'ID': i + 1,
            'Tipo': signal.phase.title(),
            'Timestamp': signal.timestamp.strftime('%Y-%m-%d %H:%M'),
            'Confianza': f"{signal.confidence:.3f}",
            'Categoría': '🔥 Alta' if signal.confidence > 0.7 else '📊 Media' if signal.confidence >= 0.4 else '⚠️ Baja'
        })
    
    signals_df = pd.DataFrame(signals_data)
    st.dataframe(signals_df, use_container_width=True)

def show_pattern_selection_ui(patterns: List) -> Optional[Any]:
    """Mostrar UI de selección de patrones para etiquetado."""
    if not patterns:
        st.warning("⚠️ No hay patrones detectados para etiquetar")
        return None
    
    st.subheader("🎯 Seleccionar Patrón para Revisar")
    
    # Crear opciones para el selectbox
    pattern_options = []
    for i, pattern in enumerate(patterns):
        label = f"{i+1}. {pattern.phase.title()} - Confianza: {pattern.confidence:.3f} - {pattern.timestamp.strftime('%Y-%m-%d %H:%M')}"
        pattern_options.append(label)
    
    selected_index = st.selectbox(
        "Selecciona un patrón:",
        range(len(pattern_options)),
        format_func=lambda x: pattern_options[x],
        key="pattern_selector"
    )
    
    if selected_index is not None:
        return patterns[selected_index], selected_index
    
    return None

def show_pattern_details(pattern: Any, pattern_index: int) -> None:
    """Mostrar detalles del patrón seleccionado."""
    st.subheader(f"📊 Detalles del Patrón #{pattern_index + 1}")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🏷️ Tipo", pattern.phase.title())
    
    with col2:
        st.metric("📊 Confianza IA", f"{pattern.confidence:.3f}")
    
    with col3:
        st.metric("⏰ Timestamp", pattern.timestamp.strftime('%Y-%m-%d %H:%M'))
    
    # Información adicional
    with st.expander("ℹ️ Información Adicional", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Puntuación IA:** {getattr(pattern, 'ai_score', 'N/A')}")
            st.write(f"**Categoría:** {getattr(pattern, 'category', 'N/A')}")
        
        with col2:
            st.write(f"**Precio:** ${getattr(pattern, 'price', 'N/A')}")
            st.write(f"**Volumen:** {getattr(pattern, 'volume', 'N/A')}")

def show_pattern_chart(df: pd.DataFrame, pattern: Any, symbol: str) -> None:
    """Mostrar gráfico enfocado del patrón."""
    st.subheader("📈 Vista Enfocada del Patrón")
    
    try:
        focused_chart = create_focused_pattern_chart(df, pattern, symbol)
        if focused_chart:
            st.plotly_chart(focused_chart, use_container_width=True)
        else:
            st.error("❌ Error al crear el gráfico enfocado")
    except Exception as e:
        st.error(f"❌ Error al mostrar gráfico enfocado: {str(e)}")

def show_manual_validation_ui(pattern: Any, pattern_index: int) -> Optional[Dict]:
    """Mostrar UI de validación manual."""
    st.subheader("✅ Validación Manual")
    
    with st.form(f"validation_form_{pattern_index}"):
        col1, col2 = st.columns(2)
        
        with col1:
            # Estado del patrón
            status = st.radio(
                "¿Es este patrón válido?",
                options=['valid', 'invalid', 'uncertain'],
                format_func=lambda x: {
                    'valid': '✅ Válido',
                    'invalid': '❌ Inválido',
                    'uncertain': '❓ Incierto'
                }[x],
                key=f"status_{pattern_index}"
            )
            
            # Tipo corregido
            corrected_type = st.selectbox(
                "Tipo correcto (si es diferente):",
                options=['', 'Accumulation', 'Distribution', 'Reaccumulation', 'Redistribution'],
                key=f"corrected_type_{pattern_index}"
            )
        
        with col2:
            # Confianza del trader
            trader_confidence = st.slider(
                "Tu nivel de confianza (1-5):",
                min_value=1,
                max_value=5,
                value=3,
                key=f"trader_conf_{pattern_index}"
            )
            
            # Puntuación de calidad
            quality_score = st.slider(
                "Puntuación de calidad (1-5):",
                min_value=1,
                max_value=5,
                value=3,
                key=f"quality_{pattern_index}"
            )
        
        # Notas
        notes = st.text_area(
            "Notas adicionales:",
            placeholder="Observaciones, razones para la validación, etc.",
            key=f"notes_{pattern_index}"
        )
        
        # Botón de guardar
        submitted = st.form_submit_button("💾 Guardar Etiqueta")
        
        if submitted:
            label_data = {
                'status': status,
                'corrected_type': corrected_type if corrected_type else pattern.phase.title(),
                'trader_confidence': trader_confidence,
                'quality_score': quality_score,
                'notes': notes,
                'pattern_data': {
                    'original_type': pattern.phase,
                    'ai_confidence': pattern.confidence,
                    'timestamp': pattern.timestamp,
                    'price': getattr(pattern, 'price', None)
                }
            }
            
            return label_data
    
    return None

def show_labeling_progress_ui() -> None:
    """Mostrar progreso del etiquetado."""
    progress = get_labeling_progress()
    
    if progress['total'] == 0:
        st.info("📋 No hay patrones etiquetados aún")
        return
    
    st.subheader("📊 Progreso del Etiquetado")
    
    # Métricas de progreso
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📋 Total", progress['total'])
    
    with col2:
        st.metric("✅ Completados", progress['completed'])
    
    with col3:
        st.metric("🎯 Válidos", progress['valid'])
    
    with col4:
        st.metric("❌ Inválidos", progress['invalid'])
    
    # Barra de progreso
    st.progress(progress['progress'] / 100)
    st.write(f"**Progreso:** {progress['progress']:.1f}% completado")
    
    # Gráfico de progreso
    if 'labeling_results' in st.session_state and st.session_state.labeling_results:
        progress_chart = create_labeling_progress_chart(st.session_state.labeling_results)
        if progress_chart:
            st.plotly_chart(progress_chart, use_container_width=True)

def show_dataset_management_ui() -> None:
    """Mostrar UI de gestión de datasets."""
    st.subheader("📚 Gestión de Datasets")
    
    progress = get_labeling_progress()
    
    if progress['completed'] == 0:
        st.warning("⚠️ No hay patrones etiquetados para crear un dataset")
        st.info("💡 Primero etiqueta algunos patrones en la pestaña 'Label and Validate'")
        return
    
    # Información del dataset potencial
    st.write("### 📊 Información del Dataset")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("📋 Muestras Totales", progress['completed'])
    
    with col2:
        st.metric("✅ Muestras Válidas", progress['valid'])
    
    with col3:
        st.metric("❌ Muestras Inválidas", progress['invalid'])
    
    # Gráfico de calidad
    if 'labeling_results' in st.session_state and st.session_state.labeling_results:
        quality_chart = create_quality_score_chart(st.session_state.labeling_results)
        if quality_chart:
            st.plotly_chart(quality_chart, use_container_width=True)
    
    # Botones de acción
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("💾 Crear Dataset", type="primary"):
            st.session_state.create_dataset_requested = True
    
    with col2:
        if st.button("📊 Preparar para ML"):
            st.session_state.prepare_ml_requested = True
    
    with col3:
        if st.button("🔄 Reiniciar Sesión"):
            st.session_state.reset_session_requested = True

def show_final_metrics(dataset_result: Dict) -> None:
    """Mostrar métricas finales del dataset creado."""
    st.success("✅ Dataset creado exitosamente!")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Muestras Totales", dataset_result['samples_count'])
    
    with col2:
        st.metric("✅ Muestras Válidas", dataset_result['valid_samples'])
    
    with col3:
        st.metric("⭐ Calidad Promedio", f"{dataset_result['avg_quality']:.2f}")
    
    with col4:
        st.metric("🎯 Confianza Promedio", f"{dataset_result['avg_confidence']:.2f}")
    
    # Información del dataset
    with st.expander("📋 Información del Dataset", expanded=True):
        dataset_info = dataset_result['dataset_info']
        st.write(f"**Nombre:** {dataset_info.get('name', 'N/A')}")
        st.write(f"**Descripción:** {dataset_info.get('description', 'N/A')}")
        st.write(f"**Creado:** {dataset_info.get('created_at', 'N/A')}")
        st.write(f"**ID:** {dataset_info.get('id', 'N/A')}")

def show_ml_preparation_summary(ml_data: Dict) -> None:
    """Mostrar resumen de preparación para ML."""
    st.success("🤖 Datos preparados para Machine Learning!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Entrenamiento", ml_data['train_samples'])
    
    with col2:
        st.metric("🧪 Prueba", ml_data['test_samples'])
    
    with col3:
        st.metric("📊 Características", ml_data['features_count'])
    
    # Clases disponibles
    st.write("### 🏷️ Clases Disponibles")
    classes_text = ", ".join(ml_data['classes'])
    st.write(f"**Tipos de patrones:** {classes_text}")
    
    st.info("💡 Los datos están listos para ser utilizados en el entrenamiento de modelos de ML")