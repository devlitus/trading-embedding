#!/usr/bin/env python3
"""
Lógica de negocio para Fase 3 - Etiquetado Wyckoff
"""

import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple

# Imports del sistema
from src.ml.wyckoff.heuristic_engine import WyckoffHeuristicEngine
from src.ml.wyckoff.scoring_system import WyckoffScoringSystem
from src.ml.labeling.dataset_manager import DatasetManager
from src.ml.labeling.labeled_sample import LabeledSample

def check_data_manager(data_manager) -> bool:
    """Verificar disponibilidad del DataManager."""
    if data_manager is None:
        st.error("❌ DataManager no disponible. Por favor, configura la conexión a datos primero.")
        st.info("💡 Ve a la página 'Gestión de Datos' para configurar tu fuente de datos.")
        return False
    return True

def get_time_range(period: str) -> Tuple[datetime, datetime]:
    """Obtener rango de tiempo basado en el período seleccionado."""
    end_time = datetime.now()
    
    if period == "1 semana":
        start_time = end_time - timedelta(days=7)
    elif period == "2 semanas":
        start_time = end_time - timedelta(days=14)
    elif period == "1 mes":
        start_time = end_time - timedelta(days=30)
    elif period == "3 meses":
        start_time = end_time - timedelta(days=90)
    elif period == "6 meses":
        start_time = end_time - timedelta(days=180)
    else:  # 1 año
        start_time = end_time - timedelta(days=365)
    
    return start_time, end_time

def retrieve_data_for_analysis(data_manager, symbol: str, interval: str, period: str) -> Optional[pd.DataFrame]:
    """Recuperar datos para análisis."""
    try:
        start_time, end_time = get_time_range(period)
        
        with st.spinner(f"📊 Obteniendo datos de {symbol} ({interval})..."):
            df = data_manager.get_ohlc_data(
                symbol=symbol,
                interval=interval,
                start_time=start_time,
                end_time=end_time
            )
        
        if df is None or df.empty:
            st.error(f"❌ No se pudieron obtener datos para {symbol}")
            return None
        
        st.success(f"✅ Datos obtenidos: {len(df)} registros desde {start_time.strftime('%Y-%m-%d')}")
        return df
        
    except Exception as e:
        st.error(f"❌ Error al obtener datos: {str(e)}")
        return None

def perform_pattern_detection(df: pd.DataFrame, config: Dict[str, Any]) -> Tuple[List, Dict]:
    """Realizar detección de patrones Wyckoff."""
    try:
        # Inicializar motor heurístico
        engine = WyckoffHeuristicEngine(
            min_confidence=config['min_confidence'],
            volume_weight=config['volume_weight']
        )
        
        # Detectar patrones
        with st.spinner("🔍 Detectando patrones Wyckoff..."):
            signals = engine.detect_patterns(df)
        
        # Filtrar por tipos de patrones seleccionados
        if config['pattern_types'] != ['Todos']:
            signals = [s for s in signals if s.phase.lower() in [p.lower() for p in config['pattern_types']]]
        
        # Limitar número de señales
        if len(signals) > config['max_signals']:
            signals = sorted(signals, key=lambda x: x.confidence, reverse=True)[:config['max_signals']]
        
        # Calcular métricas
        metrics = {
            'total_patterns': len(signals),
            'avg_confidence': sum(s.confidence for s in signals) / len(signals) if signals else 0,
            'pattern_types': len(set(s.phase for s in signals)),
            'high_confidence': len([s for s in signals if s.confidence > 0.7]),
            'medium_confidence': len([s for s in signals if 0.4 <= s.confidence <= 0.7]),
            'low_confidence': len([s for s in signals if s.confidence < 0.4])
        }
        
        return signals, metrics
        
    except Exception as e:
        st.error(f"❌ Error en detección de patrones: {str(e)}")
        return [], {}

def initialize_labeling_session(patterns: List) -> Dict:
    """Inicializar sesión de etiquetado."""
    if 'labeling_results' not in st.session_state:
        st.session_state.labeling_results = {}
    
    if 'current_pattern_index' not in st.session_state:
        st.session_state.current_pattern_index = 0
    
    if 'annotation_session' not in st.session_state:
        st.session_state.annotation_session = {
            'session_id': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'start_time': datetime.now(),
            'patterns_count': len(patterns),
            'completed_count': 0
        }
    
    return st.session_state.labeling_results

def save_pattern_label(pattern_id: str, label_data: Dict) -> None:
    """Guardar etiqueta de patrón."""
    if 'labeling_results' not in st.session_state:
        st.session_state.labeling_results = {}
    
    st.session_state.labeling_results[pattern_id] = {
        'status': label_data['status'],
        'corrected_type': label_data.get('corrected_type'),
        'trader_confidence': label_data.get('trader_confidence', 3),
        'quality_score': label_data.get('quality_score', 3),
        'notes': label_data.get('notes', ''),
        'timestamp': datetime.now(),
        'pattern_data': label_data.get('pattern_data')
    }
    
    # Actualizar contador de completados
    if 'annotation_session' in st.session_state:
        completed = len([r for r in st.session_state.labeling_results.values() if r['status'] != 'uncertain'])
        st.session_state.annotation_session['completed_count'] = completed

def get_labeling_progress() -> Dict:
    """Obtener progreso del etiquetado."""
    if 'labeling_results' not in st.session_state:
        return {'total': 0, 'completed': 0, 'valid': 0, 'invalid': 0, 'uncertain': 0, 'progress': 0}
    
    results = st.session_state.labeling_results
    total = len(results)
    
    if total == 0:
        return {'total': 0, 'completed': 0, 'valid': 0, 'invalid': 0, 'uncertain': 0, 'progress': 0}
    
    valid = len([r for r in results.values() if r['status'] == 'valid'])
    invalid = len([r for r in results.values() if r['status'] == 'invalid'])
    uncertain = len([r for r in results.values() if r['status'] == 'uncertain'])
    completed = valid + invalid
    progress = (completed / total) * 100 if total > 0 else 0
    
    return {
        'total': total,
        'completed': completed,
        'valid': valid,
        'invalid': invalid,
        'uncertain': uncertain,
        'progress': progress
    }

def create_dataset_from_labels(symbol: str, interval: str) -> Optional[Dict]:
    """Crear dataset a partir de las etiquetas."""
    if 'labeling_results' not in st.session_state or not st.session_state.labeling_results:
        st.warning("⚠️ No hay etiquetas disponibles para crear el dataset")
        return None
    
    try:
        # Inicializar sistema de puntuación
        scoring_system = WyckoffScoringSystem()
        
        # Crear muestras etiquetadas
        labeled_samples = []
        
        for pattern_id, label_data in st.session_state.labeling_results.items():
            if label_data['status'] in ['valid', 'invalid']:
                # Crear muestra etiquetada
                sample = LabeledSample(
                    pattern_id=pattern_id,
                    symbol=symbol,
                    interval=interval,
                    pattern_type=label_data.get('corrected_type', 'unknown'),
                    is_valid=label_data['status'] == 'valid',
                    trader_confidence=label_data.get('trader_confidence', 3),
                    quality_score=label_data.get('quality_score', 3),
                    notes=label_data.get('notes', ''),
                    timestamp=label_data['timestamp'],
                    pattern_data=label_data.get('pattern_data', {})
                )
                
                labeled_samples.append(sample)
        
        if not labeled_samples:
            st.warning("⚠️ No hay muestras válidas para crear el dataset")
            return None
        
        # Puntuar muestras
        scored_samples = []
        for sample in labeled_samples:
            try:
                score = scoring_system.score_sample(sample)
                sample.ai_score = score
                scored_samples.append(sample)
            except Exception as e:
                st.warning(f"⚠️ Error al puntuar muestra {sample.pattern_id}: {str(e)}")
                continue
        
        # Crear dataset
        dataset_manager = DatasetManager()
        dataset_info = dataset_manager.create_dataset(
            samples=scored_samples,
            name=f"wyckoff_{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description=f"Dataset Wyckoff para {symbol} ({interval}) - {len(scored_samples)} muestras"
        )
        
        return {
            'dataset_info': dataset_info,
            'samples_count': len(scored_samples),
            'valid_samples': len([s for s in scored_samples if s.is_valid]),
            'invalid_samples': len([s for s in scored_samples if not s.is_valid]),
            'avg_quality': sum(s.quality_score for s in scored_samples) / len(scored_samples),
            'avg_confidence': sum(s.trader_confidence for s in scored_samples) / len(scored_samples)
        }
        
    except Exception as e:
        st.error(f"❌ Error al crear dataset: {str(e)}")
        return None

def save_annotation_session() -> bool:
    """Guardar sesión de anotación."""
    try:
        if 'annotation_session' not in st.session_state:
            return False
        
        session_data = {
            'session_info': st.session_state.annotation_session,
            'labeling_results': st.session_state.labeling_results,
            'saved_at': datetime.now()
        }
        
        # Aquí se podría implementar guardado en base de datos
        # Por ahora, solo actualizamos el estado
        st.session_state.annotation_session['saved'] = True
        st.session_state.annotation_session['saved_at'] = datetime.now()
        
        return True
        
    except Exception as e:
        st.error(f"❌ Error al guardar sesión: {str(e)}")
        return False

def prepare_ml_data(dataset_info: Dict) -> Dict:
    """Preparar datos para machine learning."""
    try:
        # Simular preparación de datos ML
        samples_count = dataset_info['samples_count']
        
        # Dividir en entrenamiento y prueba (80/20)
        train_size = int(samples_count * 0.8)
        test_size = samples_count - train_size
        
        ml_data = {
            'total_samples': samples_count,
            'train_samples': train_size,
            'test_samples': test_size,
            'features_count': 15,  # Número estimado de características
            'classes': ['accumulation', 'distribution', 'reaccumulation', 'redistribution'],
            'prepared_at': datetime.now()
        }
        
        return ml_data
        
    except Exception as e:
        st.error(f"❌ Error al preparar datos ML: {str(e)}")
        return {}

def reset_labeling_session() -> None:
    """Reiniciar sesión de etiquetado."""
    keys_to_reset = [
        'labeling_results',
        'current_pattern_index',
        'annotation_session',
        'selected_pattern_for_review'
    ]
    
    for key in keys_to_reset:
        if key in st.session_state:
            del st.session_state[key]

def get_pattern_statistics(patterns: List) -> Dict:
    """Obtener estadísticas de patrones detectados."""
    if not patterns:
        return {}
    
    stats = {
        'total': len(patterns),
        'by_type': {},
        'by_confidence': {'high': 0, 'medium': 0, 'low': 0},
        'avg_confidence': 0,
        'confidence_range': {'min': 1.0, 'max': 0.0}
    }
    
    confidences = []
    
    for pattern in patterns:
        # Por tipo
        pattern_type = pattern.phase.title()
        stats['by_type'][pattern_type] = stats['by_type'].get(pattern_type, 0) + 1
        
        # Por confianza
        conf = pattern.confidence
        confidences.append(conf)
        
        if conf > 0.7:
            stats['by_confidence']['high'] += 1
        elif conf >= 0.4:
            stats['by_confidence']['medium'] += 1
        else:
            stats['by_confidence']['low'] += 1
    
    if confidences:
        stats['avg_confidence'] = sum(confidences) / len(confidences)
        stats['confidence_range']['min'] = min(confidences)
        stats['confidence_range']['max'] = max(confidences)
    
    return stats