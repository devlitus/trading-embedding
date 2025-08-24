#!/usr/bin/env python3
"""
Configuración y CSS para Fase 3 - Etiquetado Wyckoff
"""

import streamlit as st
from datetime import datetime, timedelta

def apply_custom_css():
    """Aplicar estilos CSS personalizados para la Fase 3."""
    st.markdown("""
    <style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .step-card {
        padding: 20px;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin: 10px 0;
    }
    .step-detect {
        border: 3px solid #2E7D32;
        background: linear-gradient(135deg, #E8F5E8 0%, #F1F8E9 100%);
    }
    .step-evaluate {
        border: 3px solid #1565C0;
        background: linear-gradient(135deg, #E3F2FD 0%, #E8F4FD 100%);
    }
    .step-label {
        border: 3px solid #E65100;
        background: linear-gradient(135deg, #FFF3E0 0%, #FFF8F0 100%);
    }
    .step-train {
        border: 3px solid #6A1B9A;
        background: linear-gradient(135deg, #F3E5F5 0%, #F8F5F9 100%);
    }
    .pattern-info {
        background: #2d5a2d;
        padding: 20px;
        border-radius: 12px;
        border-left: 6px solid #4CAF50;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        color: #ffffff;
    }
    </style>
    """, unsafe_allow_html=True)

def show_page_header():
    """Mostrar el encabezado de la página con información introductoria."""
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

def show_workflow_steps():
    """Mostrar los pasos del flujo de trabajo de forma visual."""
    st.markdown("""
    ### 🔄 Flujo de Trabajo Simplificado
    """)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="step-card step-detect">
            <h4 style="color: #1B5E20; font-weight: bold; margin-bottom: 8px;">1️⃣ DETECTAR</h4>
            <p style="color: #2E7D32; font-weight: 500; margin: 0;">Busca patrones Wyckoff automáticamente</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="step-card step-evaluate">
            <h4 style="color: #0D47A1; font-weight: bold; margin-bottom: 8px;">2️⃣ EVALUAR</h4>
            <p style="color: #1565C0; font-weight: 500; margin: 0;">Puntúa la calidad de cada patrón</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="step-card step-label">
            <h4 style="color: #BF360C; font-weight: bold; margin-bottom: 8px;">3️⃣ ETIQUETAR</h4>
            <p style="color: #E65100; font-weight: 500; margin: 0;">Confirma o corrige manualmente</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="step-card step-train">
            <h4 style="color: #4A148C; font-weight: bold; margin-bottom: 8px;">4️⃣ ENTRENAR</h4>
            <p style="color: #6A1B9A; font-weight: 500; margin: 0;">Crea datasets para IA</p>
        </div>
        """, unsafe_allow_html=True)

def create_quick_config():
    """Crear controles de configuración rápida."""
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
            index=0,
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
    
    return symbol, timeframe, days_back

def create_advanced_detection_config():
    """Crear configuración avanzada para detección de patrones."""
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
    
    return min_confidence, max_signals, volume_weight, pattern_types

def get_time_range(days_back):
    """Obtener rango de tiempo basado en días hacia atrás."""
    end_time = datetime.now()
    start_time = end_time - timedelta(days=days_back)
    return start_time, end_time