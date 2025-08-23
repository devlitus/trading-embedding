#!/usr/bin/env python3
"""
Script de demostración para la Fase 4: Machine Learning
Testing del pipeline completo de entrenamiento ML para patrones Wyckoff.

Este script demuestra:
1. Preparación de datos de entrenamiento
2. Entrenamiento de embeddings
3. Entrenamiento del modelo Wyckoff
4. Validación del pipeline completo
5. Generación de reportes

Autor: DevAgent
Fecha: 2025-01-09
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime

# Agregar el directorio src al path
sys.path.append(str(Path(__file__).parent / 'src'))

from src.ml.training import MLTrainingPipeline
from src.config.config_manager import ConfigManager

def setup_logging():
    """
    Configura el sistema de logging para la demostración.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('demo_ml_fase4.log')
        ]
    )
    
    # Reducir verbosidad de algunas librerías
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)

def print_banner():
    """
    Imprime el banner de inicio de la demostración.
    """
    print("\n" + "="*80)
    print("🤖 DEMOSTRACIÓN FASE 4: MACHINE LEARNING")
    print("📊 Pipeline de Entrenamiento de Patrones Wyckoff")
    print("🚀 Trading Embedding System")
    print("="*80 + "\n")

def print_section(title: str):
    """
    Imprime un separador de sección.
    
    Args:
        title: Título de la sección
    """
    print(f"\n{'='*20} {title} {'='*20}")

def demo_data_preparation(pipeline: MLTrainingPipeline):
    """
    Demuestra la preparación de datos de entrenamiento.
    
    Args:
        pipeline: Pipeline de entrenamiento ML
    """
    print_section("PREPARACIÓN DE DATOS")
    
    # Símbolos de demostración (reducidos para prueba rápida)
    demo_symbols = ['BTCUSDT', 'ETHUSDT']
    demo_timeframes = ['1h', '4h']
    demo_days_back = 30  # Solo 30 días para demostración rápida
    
    print(f"📈 Símbolos: {demo_symbols}")
    print(f"⏰ Timeframes: {demo_timeframes}")
    print(f"📅 Días hacia atrás: {demo_days_back}")
    print("\n🔄 Iniciando preparación de datos...")
    
    try:
        stats = pipeline.prepare_training_data(
            symbols=demo_symbols,
            timeframes=demo_timeframes,
            days_back=demo_days_back
        )
        
        print("\n✅ Preparación de datos completada:")
        print(f"   • Símbolos procesados: {stats['symbols_processed']}")
        print(f"   • Total de muestras: {stats['total_samples']}")
        print(f"   • Distribución por patrón:")
        
        for pattern, count in stats['samples_by_pattern'].items():
            print(f"     - {pattern}: {count} muestras")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en preparación de datos: {e}")
        return False

def demo_embeddings_training(pipeline: MLTrainingPipeline):
    """
    Demuestra el entrenamiento del sistema de embeddings.
    
    Args:
        pipeline: Pipeline de entrenamiento ML
    """
    print_section("ENTRENAMIENTO DE EMBEDDINGS")
    
    print("🧠 Iniciando entrenamiento de embeddings...")
    print("   • Codificación temporal de secuencias OHLC")
    print("   • Compresión de indicadores técnicos")
    print("   • Representación de patrones Wyckoff")
    
    try:
        results = pipeline.train_embeddings(force_retrain=False)
        
        print(f"\n✅ Entrenamiento de embeddings: {results['status']}")
        
        if results['status'] == 'trained':
            print(f"   • Muestras utilizadas: {results['samples_used']}")
            print(f"   • Modelo guardado en: {results['model_path']}")
            if 'metrics' in results:
                print(f"   • Métricas de entrenamiento disponibles")
        elif results['status'] == 'loaded_existing':
            print(f"   • Modelo cargado desde: {results['path']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en entrenamiento de embeddings: {e}")
        return False

def demo_wyckoff_training(pipeline: MLTrainingPipeline):
    """
    Demuestra el entrenamiento del modelo de clasificación Wyckoff.
    
    Args:
        pipeline: Pipeline de entrenamiento ML
    """
    print_section("ENTRENAMIENTO MODELO WYCKOFF")
    
    print("🎯 Iniciando entrenamiento del clasificador Wyckoff...")
    print("   • Extracción de características de precio")
    print("   • Análisis de patrones de volumen")
    print("   • Detección de fases Wyckoff")
    
    try:
        results = pipeline.train_wyckoff_model(force_retrain=False)
        
        print(f"\n✅ Entrenamiento modelo Wyckoff: {results['status']}")
        
        if results['status'] == 'trained':
            print(f"   • Muestras utilizadas: {results['samples_used']}")
            print(f"   • Modelo guardado en: {results['model_path']}")
            if 'metrics' in results:
                print(f"   • Métricas de entrenamiento disponibles")
        elif results['status'] == 'loaded_existing':
            print(f"   • Modelo cargado desde: {results['path']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en entrenamiento modelo Wyckoff: {e}")
        return False

def demo_pipeline_validation(pipeline: MLTrainingPipeline):
    """
    Demuestra la validación del pipeline completo.
    
    Args:
        pipeline: Pipeline de entrenamiento ML
    """
    print_section("VALIDACIÓN DEL PIPELINE")
    
    print("🔍 Validando pipeline completo...")
    print("   • Verificando modelos cargados")
    print("   • Ejecutando predicciones de prueba")
    print("   • Calculando métricas de rendimiento")
    
    try:
        validation_results = pipeline.validate_pipeline()
        
        print("\n✅ Resultados de validación:")
        print(f"   • Embeddings cargados: {'✅' if validation_results['embeddings_loaded'] else '❌'}")
        print(f"   • Modelo Wyckoff cargado: {'✅' if validation_results['wyckoff_model_loaded'] else '❌'}")
        print(f"   • Test end-to-end: {'✅' if validation_results['end_to_end_test'] else '❌'}")
        
        if validation_results['sample_predictions']:
            print(f"   • Predicciones de prueba: {len(validation_results['sample_predictions'])}")
            
            for i, pred in enumerate(validation_results['sample_predictions']):
                status = "✅" if pred['match'] else "❌"
                print(f"     {status} Muestra {i+1}: {pred['true_pattern']} → {pred['predicted_pattern']} (conf: {pred['confidence']:.2f})")
        
        if validation_results['performance_metrics']:
            metrics = validation_results['performance_metrics']
            print(f"   • Precisión: {metrics['accuracy']:.2%}")
            print(f"   • Predicciones correctas: {metrics['correct_predictions']}/{metrics['total_predictions']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error en validación del pipeline: {e}")
        return False

def demo_full_pipeline():
    """
    Ejecuta la demostración completa del pipeline de ML.
    """
    print_banner()
    
    # Configurar logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    logger.info("Iniciando demostración de Fase 4: Machine Learning")
    
    try:
        # Inicializar pipeline
        print("🚀 Inicializando pipeline de Machine Learning...")
        pipeline = MLTrainingPipeline()
        
        # Verificar configuración
        print(f"📁 Directorio de modelos: {pipeline.models_dir}")
        print(f"📁 Directorio de embeddings: {pipeline.embeddings_dir}")
        print(f"⚙️  Configuración de entrenamiento:")
        for key, value in pipeline.training_config.items():
            print(f"   • {key}: {value}")
        
        # Ejecutar pasos de demostración
        steps_success = []
        
        # Paso 1: Preparación de datos
        steps_success.append(demo_data_preparation(pipeline))
        
        # Paso 2: Entrenamiento de embeddings
        if steps_success[-1]:
            steps_success.append(demo_embeddings_training(pipeline))
        else:
            print("⏭️  Saltando entrenamiento de embeddings debido a error anterior")
            steps_success.append(False)
        
        # Paso 3: Entrenamiento modelo Wyckoff
        if steps_success[-1]:
            steps_success.append(demo_wyckoff_training(pipeline))
        else:
            print("⏭️  Saltando entrenamiento Wyckoff debido a error anterior")
            steps_success.append(False)
        
        # Paso 4: Validación
        if any(steps_success):
            steps_success.append(demo_pipeline_validation(pipeline))
        else:
            print("⏭️  Saltando validación debido a errores anteriores")
            steps_success.append(False)
        
        # Resumen final
        print_section("RESUMEN FINAL")
        
        step_names = [
            "Preparación de datos",
            "Entrenamiento embeddings", 
            "Entrenamiento Wyckoff",
            "Validación pipeline"
        ]
        
        print("📊 Resultados por paso:")
        for i, (step_name, success) in enumerate(zip(step_names, steps_success)):
            status = "✅ ÉXITO" if success else "❌ ERROR"
            print(f"   {i+1}. {step_name}: {status}")
        
        total_success = sum(steps_success)
        print(f"\n🎯 Pasos completados exitosamente: {total_success}/{len(steps_success)}")
        
        if total_success == len(steps_success):
            print("\n🎉 ¡DEMOSTRACIÓN COMPLETADA EXITOSAMENTE!")
            print("   El pipeline de Machine Learning está funcionando correctamente.")
            print("   Los modelos están entrenados y listos para usar.")
        elif total_success > 0:
            print("\n⚠️  DEMOSTRACIÓN PARCIALMENTE EXITOSA")
            print("   Algunos componentes funcionan, pero hay errores que resolver.")
        else:
            print("\n❌ DEMOSTRACIÓN FALLIDA")
            print("   Revisar logs y configuración del sistema.")
        
        # Información adicional
        print("\n📚 Información adicional:")
        print(f"   • Log de demostración: demo_ml_fase4.log")
        print(f"   • Directorio de modelos: {pipeline.models_dir}")
        print(f"   • Directorio de embeddings: {pipeline.embeddings_dir}")
        print(f"   • Configuración: config.yaml")
        
        return total_success == len(steps_success)
        
    except Exception as e:
        logger.error(f"Error crítico en demostración: {e}")
        print(f"\n💥 ERROR CRÍTICO: {e}")
        print("   Revisar configuración y dependencias del sistema.")
        return False
    
    finally:
        print("\n" + "="*80)
        print(f"🕐 Demostración finalizada: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)

def demo_quick_test():
    """
    Ejecuta una prueba rápida del pipeline sin entrenamiento completo.
    """
    print("\n🚀 PRUEBA RÁPIDA - FASE 4 ML")
    print("="*50)
    
    try:
        # Solo inicializar y verificar componentes
        pipeline = MLTrainingPipeline()
        
        print("✅ Pipeline inicializado correctamente")
        print(f"📁 Modelos: {pipeline.models_dir}")
        print(f"📁 Embeddings: {pipeline.embeddings_dir}")
        
        # Verificar componentes
        print("\n🔍 Verificando componentes:")
        print(f"   • DataPreprocessor: {'✅' if pipeline.preprocessor else '❌'}")
        print(f"   • TradingEmbeddings: {'✅' if pipeline.embeddings else '❌'}")
        print(f"   • WyckoffAnalyzer: {'✅' if pipeline.wyckoff_analyzer else '❌'}")
        print(f"   • DatasetManager: {'✅' if pipeline.dataset_manager else '❌'}")
        
        print("\n✅ Prueba rápida completada exitosamente")
        return True
        
    except Exception as e:
        print(f"❌ Error en prueba rápida: {e}")
        return False

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Demostración Fase 4: Machine Learning')
    parser.add_argument('--quick', action='store_true', 
                       help='Ejecutar solo prueba rápida sin entrenamiento')
    parser.add_argument('--full', action='store_true',
                       help='Ejecutar demostración completa con entrenamiento')
    
    args = parser.parse_args()
    
    if args.quick:
        success = demo_quick_test()
    elif args.full:
        success = demo_full_pipeline()
    else:
        # Por defecto, mostrar opciones
        print("\n🤖 DEMOSTRACIÓN FASE 4: MACHINE LEARNING")
        print("\nOpciones disponibles:")
        print("  python demo_ml_fase4.py --quick    # Prueba rápida")
        print("  python demo_ml_fase4.py --full     # Demostración completa")
        print("\nEjecutando prueba rápida por defecto...\n")
        success = demo_quick_test()
    
    sys.exit(0 if success else 1)