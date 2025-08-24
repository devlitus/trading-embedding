import streamlit as st
from typing import Dict, Any

# Imports de módulos locales
from .phase3_config import (
    apply_custom_css,
    show_page_header,
    create_quick_config,
    create_advanced_config
)
from .phase3_logic import (
    check_data_manager,
    retrieve_data_for_analysis,
    perform_pattern_detection,
    initialize_labeling_session,
    save_pattern_label,
    create_dataset_from_labels,
    save_annotation_session,
    prepare_ml_data,
    reset_labeling_session
)
from .phase3_ui import (
    show_detection_results,
    show_pattern_selection_ui,
    show_pattern_details,
    show_pattern_chart,
    show_manual_validation_ui,
    show_labeling_progress_ui,
    show_dataset_management_ui,
    show_final_metrics,
    show_ml_preparation_summary
)

def show_phase3_labeling_page(data_manager):
    """Página principal del sistema de etiquetado Wyckoff - Fase 3."""
    
    # Aplicar CSS personalizado
    apply_custom_css()
    
    # Mostrar encabezado de la página
    show_page_header()
    
    # Verificar disponibilidad del DataManager
    if not check_data_manager(data_manager):
        return
    
    # Configuración rápida
    symbol, timeframe, period = create_quick_config()
    
    # Tabs principales
    tab1, tab2, tab3 = st.tabs([
        "🔍 Detectar Patrones",
        "🏷️ Etiquetar y Validar", 
        "📊 Gestionar Datasets"
    ])
    
    with tab1:
        # Configuración avanzada
        advanced_config = create_advanced_config()
        
        # Botón de detección
        if st.button("🚀 Iniciar Detección de Patrones", type="primary", use_container_width=True):
            # Realizar detección
            results = perform_pattern_detection(data_manager, symbol, timeframe, period, advanced_config)
            
            if results:
                # Mostrar resultados
                show_detection_results(results, symbol)
    
    with tab2:
        # Verificar si hay patrones detectados
        if 'detected_patterns' in st.session_state and st.session_state['detected_patterns']:
            # Inicializar sesión de etiquetado
            initialize_labeling_session()
            
            # Mostrar interfaz de selección de patrones
            selected_pattern = show_pattern_selection_ui(st.session_state['detected_patterns'])
            
            if selected_pattern is not None:
                # Mostrar detalles del patrón
                show_pattern_details(selected_pattern)
                
                # Mostrar gráfico enfocado
                show_pattern_chart(selected_pattern, st.session_state.get('analysis_data'))
                
                # Interfaz de validación manual
                validation_result = show_manual_validation_ui(selected_pattern)
                
                if validation_result:
                    # Guardar etiqueta
                    save_pattern_label(selected_pattern, validation_result)
            
            # Mostrar progreso de etiquetado
            show_labeling_progress_ui()
        else:
            st.warning("⚠️ No hay patrones detectados. Ve al **Paso 1** para detectar patrones primero.")
    
    with tab3:
        # Mostrar interfaz de gestión de datasets
        show_dataset_management_ui(data_manager)
        
        # Botón para crear dataset
        if st.button("📊 Crear Dataset desde Etiquetas", type="primary"):
            if 'labeling_results' in st.session_state:
                # Crear dataset
                dataset = create_dataset_from_labels(st.session_state['labeling_results'])
                
                if dataset:
                    # Guardar sesión de anotación
                    save_annotation_session(dataset, symbol, timeframe)
                    
                    # Preparar datos para ML
                    ml_data = prepare_ml_data(dataset)
                    
                    # Mostrar métricas finales
                    show_final_metrics(dataset)
                    
                    # Mostrar resumen de preparación ML
                    show_ml_preparation_summary(ml_data)
            else:
                st.warning("⚠️ No hay etiquetas disponibles. Completa el etiquetado en el **Paso 2** primero.")

# Todas las funciones han sido modularizadas en:
# - phase3_config.py: Configuración y CSS
# - phase3_logic.py: Lógica de negocio
# - phase3_ui.py: Componentes de interfaz de usuario
# - phase3_charts.py: Gráficos y visualizaciones