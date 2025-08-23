#!/usr/bin/env python3
"""
Demo simplificado de Machine Learning - Fase 4
Sistema de Trading con Embeddings y Análisis de Patrones Wyckoff

Este demo muestra las capacidades básicas del sistema ML sin dependencias complejas.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.ml.data_preprocessing import DataPreprocessor
    from src.ml.wyckoff_model import WyckoffFeatureExtractor, WyckoffClassifier, WyckoffAnalyzer
    from src.ml.labeling.dataset_manager import DatasetManager, LabeledSample
except ImportError as e:
    logger.error(f"Error importando módulos ML: {e}")
    sys.exit(1)

def generate_sample_data(symbol: str = "BTCUSDT", days: int = 30) -> pd.DataFrame:
    """
    Genera datos de muestra para la demostración
    """
    logger.info(f"Generando datos de muestra para {symbol} ({days} días)")
    
    # Generar fechas
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='1H')
    
    # Generar datos OHLC sintéticos
    np.random.seed(42)  # Para reproducibilidad
    base_price = 45000  # Precio base de BTC
    
    data = []
    current_price = base_price
    
    for i, date in enumerate(dates):
        # Simular movimiento de precio con tendencia y volatilidad
        change = np.random.normal(0, 0.02)  # 2% volatilidad promedio
        current_price *= (1 + change)
        
        # Generar OHLC
        high_factor = 1 + abs(np.random.normal(0, 0.01))
        low_factor = 1 - abs(np.random.normal(0, 0.01))
        
        open_price = current_price
        high_price = current_price * high_factor
        low_price = current_price * low_factor
        close_price = current_price * (1 + np.random.normal(0, 0.005))
        
        # Volumen sintético
        volume = np.random.lognormal(10, 1)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })
        
        current_price = close_price
    
    df = pd.DataFrame(data)
    logger.info(f"Generados {len(df)} registros de datos")
    return df

def demo_data_preprocessing():
    """
    Demuestra las capacidades de preprocesamiento de datos
    """
    logger.info("=== DEMO: Preprocesamiento de Datos ===")
    
    # Generar datos de muestra
    data = generate_sample_data()
    
    # Inicializar preprocessor
    preprocessor = DataPreprocessor()
    
    # Preparar características
    features = preprocessor.prepare_features(data)
    logger.info(f"Características generadas: {features.shape}")
    logger.info(f"Columnas: {list(features.columns)}")
    
    # Crear secuencias temporales
    sequences, targets = preprocessor.create_sequences(features)
    logger.info(f"Secuencias creadas: {sequences.shape}")
    if targets is not None:
        logger.info(f"Targets: {targets.shape}")
    else:
        logger.info("Targets: None (sin etiquetas)")
    
    return features, sequences, targets

def demo_wyckoff_analysis():
    """
    Demuestra el análisis de patrones Wyckoff
    """
    logger.info("=== DEMO: Análisis de Patrones Wyckoff ===")
    
    # Generar datos de muestra
    data = generate_sample_data()
    
    # Inicializar extractor de características
    feature_extractor = WyckoffFeatureExtractor()
    
    # Extraer características Wyckoff
    price_features = feature_extractor.extract_price_action_features(data)
    volume_features = feature_extractor.extract_volume_features(data)
    wyckoff_features = feature_extractor.extract_wyckoff_specific_features(data)
    
    logger.info(f"Características de precio extraídas: {len(price_features)}")
    logger.info(f"Características de volumen extraídas: {len(volume_features)}")
    logger.info(f"Características Wyckoff específicas: {len(wyckoff_features)}")
    
    # Detectar fases específicas (usando métodos privados disponibles)
    phases = {
        'preliminary_support': feature_extractor._detect_preliminary_support(data),
        'selling_climax': feature_extractor._detect_selling_climax(data),
        'automatic_rally': feature_extractor._detect_automatic_rally(data),
        'secondary_test': feature_extractor._detect_secondary_test(data),
        'spring': feature_extractor._calculate_spring_probability(data),
        'upthrust': feature_extractor._calculate_upthrust_probability(data)
    }
    
    logger.info("Fases Wyckoff detectadas:")
    for phase, detected in phases.items():
        count = np.sum(detected) if isinstance(detected, np.ndarray) else (1 if detected else 0)
        logger.info(f"  {phase}: {count} ocurrencias")
    
    return wyckoff_features, phases

def demo_classification():
    """
    Demuestra la clasificación de patrones
    """
    logger.info("=== DEMO: Clasificación de Patrones ===")
    
    # Generar datos de muestra
    data = generate_sample_data()
    
    # Extraer características
    feature_extractor = WyckoffFeatureExtractor()
    price_features = feature_extractor.extract_price_action_features(data)
    volume_features = feature_extractor.extract_volume_features(data)
    wyckoff_features = feature_extractor.extract_wyckoff_specific_features(data)
    
    # Combinar todas las características en un array
    all_features = {**price_features, **volume_features, **wyckoff_features}
    features = np.array(list(all_features.values())).reshape(1, -1)
    
    logger.info(f"Características combinadas: {features.shape}")
    
    # Crear datos sintéticos para entrenamiento en el formato correcto
    n_samples = 20  # Reducir para demo
    patterns = ['accumulation', 'distribution', 'markup', 'markdown', 'neutral']
    
    # Generar datos sintéticos como lista de tuplas (DataFrame, label)
    training_data = []
    for i in range(n_samples):
        # Generar datos OHLC sintéticos
        synthetic_data = generate_sample_data(days=5)  # Datos más pequeños para demo
        pattern_label = np.random.choice(patterns)
        training_data.append((synthetic_data, pattern_label))
    
    # Inicializar clasificador
    classifier = WyckoffClassifier(model_type='random_forest')
    
    # Entrenar
    logger.info("Entrenando clasificador...")
    training_results = classifier.train(training_data, validation_split=0.2)
    
    # Hacer predicciones con datos de prueba
    logger.info("Realizando predicciones...")
    test_data = generate_sample_data(days=7)
    prediction, confidence = classifier.predict(test_data)
    
    logger.info(f"Predicción: {prediction} (confianza: {confidence:.2%})")
    logger.info(f"Métricas de entrenamiento: {training_results}")
    logger.info("✅ Demo de clasificación completado")
    
    return classifier, training_results

def demo_dataset_management():
    """
    Demuestra la gestión de datasets etiquetados
    """
    logger.info("=== DEMO: Gestión de Datasets ===")
    
    # Inicializar gestor de datasets
    dataset_manager = DatasetManager()
    
    # Crear muestras etiquetadas sintéticas
    data = generate_sample_data(days=7)  # Datos más pequeños para demo
    
    patterns = ['accumulation', 'distribution', 'markup']
    for i, pattern in enumerate(patterns):
        sample = LabeledSample(
            timestamp=datetime.now(),
            symbol="BTCUSDT",
            timeframe="1h",
            pattern=pattern,
            confidence=0.8 + i * 0.1,
            score=0.7 + i * 0.1,
            data=data.iloc[i*100:(i+1)*100].to_dict('records'),
            signals={'volume_spike': True, 'price_action': 'bullish'},
            metadata={'source': 'demo', 'version': '1.0'}
        )
        dataset_manager.add_sample(sample)
    
    # Mostrar estadísticas
    stats = dataset_manager.get_dataset_stats()
    logger.info(f"Muestras en dataset: {stats['total_samples']}")
    logger.info(f"Patrones: {stats['patterns']}")
    logger.info(f"Símbolos: {stats['symbols']}")
    logger.info(f"Confianza promedio: {stats['avg_confidence']:.2f}")
    logger.info(f"Score promedio: {stats['avg_score']:.2f}")
    
    # Convertir a DataFrame para análisis
    df = dataset_manager.to_dataframe()
    logger.info(f"DataFrame creado: {df.shape}")
    logger.info(f"Columnas: {list(df.columns)}")
    
    # Obtener muestras filtradas
    high_confidence_samples = dataset_manager.get_samples(min_confidence=0.8)
    logger.info(f"Muestras con alta confianza: {len(high_confidence_samples)}")
    
    logger.info("✅ Demo de gestión de datasets completado")
    
    return dataset_manager

def main():
    """
    Función principal del demo
    """
    parser = argparse.ArgumentParser(description='Demo ML Fase 4 - Simplificado')
    parser.add_argument('--quick', action='store_true', help='Ejecutar demo rápido')
    parser.add_argument('--full', action='store_true', help='Ejecutar demo completo')
    args = parser.parse_args()
    
    logger.info("🚀 Iniciando Demo ML Fase 4 - Simplificado")
    logger.info("=" * 50)
    
    try:
        if args.quick or not args.full:
            logger.info("Ejecutando demo rápido...")
            
            # Demo básico de preprocesamiento
            features, sequences, targets = demo_data_preprocessing()
            
            # Demo básico de análisis Wyckoff
            wyckoff_features, phases = demo_wyckoff_analysis()
            
            logger.info("✅ Demo rápido completado exitosamente")
            
        if args.full:
            logger.info("Ejecutando demo completo...")
            
            # Todos los demos
            demo_data_preprocessing()
            demo_wyckoff_analysis()
            demo_classification()
            demo_dataset_management()
            
            logger.info("✅ Demo completo ejecutado exitosamente")
            
    except Exception as e:
        logger.error(f"❌ Error durante la ejecución del demo: {e}")
        raise
    
    logger.info("=" * 50)
    logger.info("🎉 Demo ML Fase 4 finalizado")
    logger.info("")
    logger.info("Próximos pasos:")
    logger.info("1. Integrar con datos reales de Binance")
    logger.info("2. Entrenar modelos con datos históricos")
    logger.info("3. Implementar sistema de backtesting")
    logger.info("4. Desarrollar API para predicciones en tiempo real")

if __name__ == "__main__":
    main()