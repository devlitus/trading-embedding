import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from datetime import datetime, timedelta
import json
import os
from pathlib import Path

# Importar componentes locales
from .data_preprocessing import DataPreprocessor
from .embeddings import TradingEmbeddings
from .wyckoff_model import WyckoffAnalyzer, WyckoffPattern
from .labeling.dataset_manager import DatasetManager, LabeledSample
from ..config.config_manager import ConfigManager
from ..data.data_manager import DataManager

class MLTrainingPipeline:
    """
    Pipeline completo de entrenamiento de machine learning para la Fase 4.
    Coordina el preprocesamiento, embeddings, y entrenamiento del modelo Wyckoff.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Inicializa el pipeline de entrenamiento.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = ConfigManager(config_path)
        self.logger = logging.getLogger(__name__)
        
        # Inicializar componentes
        self.data_manager = DataManager(self.config)
        self.dataset_manager = DatasetManager()
        self.preprocessor = DataPreprocessor()
        self.embeddings = TradingEmbeddings()
        self.wyckoff_analyzer = WyckoffAnalyzer()
        
        # Configuración de entrenamiento
        self.training_config = self.config.get('ml_training', {
            'validation_split': 0.2,
            'test_split': 0.1,
            'min_samples_per_pattern': 10,
            'sequence_length': 50,
            'batch_size': 32,
            'epochs': 100,
            'learning_rate': 0.001
        })
        
        # Directorios de trabajo
        self.models_dir = Path(self.config.get('paths', {}).get('models_dir', 'models'))
        self.models_dir.mkdir(exist_ok=True)
        
        self.embeddings_dir = Path(self.config.get('paths', {}).get('embeddings_dir', 'embeddings'))
        self.embeddings_dir.mkdir(exist_ok=True)
    
    def prepare_training_data(self, symbols: List[str], 
                            timeframes: List[str] = ['1h', '4h', '1d'],
                            days_back: int = 365) -> Dict[str, Any]:
        """
        Prepara los datos de entrenamiento desde múltiples fuentes.
        
        Args:
            symbols: Lista de símbolos a procesar
            timeframes: Lista de timeframes
            days_back: Días hacia atrás para obtener datos
            
        Returns:
            Diccionario con estadísticas de preparación
        """
        self.logger.info(f"Preparando datos de entrenamiento para {len(symbols)} símbolos")
        
        stats = {
            'symbols_processed': 0,
            'total_samples': 0,
            'samples_by_pattern': {},
            'timeframes_processed': timeframes,
            'date_range': {
                'start': (datetime.now() - timedelta(days=days_back)).isoformat(),
                'end': datetime.now().isoformat()
            }
        }
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        for symbol in symbols:
            try:
                self.logger.info(f"Procesando símbolo: {symbol}")
                
                for timeframe in timeframes:
                    # Obtener datos históricos
                    ohlc_data = self.data_manager.get_ohlc_data(
                        symbol=symbol,
                        interval=timeframe,
                        start_time=start_date,
                        end_time=end_date
                    )
                    
                    if ohlc_data is None or len(ohlc_data) < self.training_config['sequence_length']:
                        self.logger.warning(f"Datos insuficientes para {symbol} {timeframe}")
                        continue
                    
                    # Crear ventanas deslizantes para entrenamiento
                    windows = self._create_sliding_windows(
                        ohlc_data, 
                        window_size=self.training_config['sequence_length']
                    )
                    
                    for i, window_data in enumerate(windows):
                        # Generar etiqueta automática (heurística inicial)
                        pattern_label = self._generate_heuristic_label(window_data)
                        
                        if pattern_label != 'no_pattern':
                            # Crear muestra etiquetada
                            sample = LabeledSample(
                                timestamp=window_data.index[-1],
                                symbol=symbol,
                                timeframe=timeframe,
                                pattern=pattern_label,
                                confidence=0.7,  # Confianza heurística
                                score=self._calculate_pattern_score(window_data, pattern_label),
                                data=window_data.to_dict('records'),
                                signals={},
                                metadata={
                                    'window_index': i,
                                    'data_length': len(window_data),
                                    'generation_method': 'heuristic'
                                }
                            )
                            
                            # Agregar al dataset
                            self.dataset_manager.add_sample(sample)
                            
                            # Actualizar estadísticas
                            stats['total_samples'] += 1
                            if pattern_label not in stats['samples_by_pattern']:
                                stats['samples_by_pattern'][pattern_label] = 0
                            stats['samples_by_pattern'][pattern_label] += 1
                
                stats['symbols_processed'] += 1
                
            except Exception as e:
                self.logger.error(f"Error procesando {symbol}: {e}")
                continue
        
        self.logger.info(f"Preparación completada: {stats['total_samples']} muestras generadas")
        return stats
    
    def _create_sliding_windows(self, data: pd.DataFrame, 
                              window_size: int, 
                              step_size: int = None) -> List[pd.DataFrame]:
        """
        Crea ventanas deslizantes de los datos.
        
        Args:
            data: DataFrame con datos OHLC
            window_size: Tamaño de la ventana
            step_size: Tamaño del paso (por defecto window_size // 4)
            
        Returns:
            Lista de DataFrames con las ventanas
        """
        if step_size is None:
            step_size = max(1, window_size // 4)
        
        windows = []
        for i in range(0, len(data) - window_size + 1, step_size):
            window = data.iloc[i:i + window_size].copy()
            windows.append(window)
        
        return windows
    
    def _generate_heuristic_label(self, data: pd.DataFrame) -> str:
        """
        Genera etiquetas heurísticas basadas en patrones simples.
        
        Args:
            data: DataFrame con datos OHLC
            
        Returns:
            Etiqueta del patrón identificado
        """
        try:
            # Usar el extractor de características de Wyckoff
            from .wyckoff_model import WyckoffFeatureExtractor
            extractor = WyckoffFeatureExtractor()
            
            # Extraer características
            price_features = extractor.extract_price_action_features(data)
            volume_features = extractor.extract_volume_features(data)
            wyckoff_features = extractor.extract_wyckoff_specific_features(data)
            
            # Lógica heurística simple
            accumulation_score = wyckoff_features.get('accumulation_score', 0)
            distribution_score = wyckoff_features.get('distribution_score', 0)
            trend_strength = price_features.get('trend_strength', 0)
            
            # Umbrales para clasificación
            if accumulation_score > 0.6:
                return 'accumulation'
            elif distribution_score > 0.6:
                return 'distribution'
            elif trend_strength > 0.1:
                return 'markup'
            elif trend_strength < -0.1:
                return 'markdown'
            else:
                return 'no_pattern'
                
        except Exception as e:
            self.logger.warning(f"Error en etiquetado heurístico: {e}")
            return 'no_pattern'
    
    def _calculate_pattern_score(self, data: pd.DataFrame, pattern: str) -> float:
        """
        Calcula un score de calidad para el patrón identificado.
        
        Args:
            data: DataFrame con datos OHLC
            pattern: Patrón identificado
            
        Returns:
            Score de calidad [0, 1]
        """
        try:
            # Factores de calidad básicos
            data_quality = 1.0 - (data.isnull().sum().sum() / (len(data) * len(data.columns)))
            
            # Volatilidad apropiada
            volatility = data['close'].pct_change().std()
            volatility_score = min(volatility * 10, 1.0)  # Normalizar
            
            # Volumen consistente
            volume_score = 1.0
            if 'volume' in data.columns:
                volume_cv = data['volume'].std() / data['volume'].mean()
                volume_score = max(0.0, 1.0 - volume_cv)
            
            # Score combinado
            total_score = (data_quality * 0.4 + volatility_score * 0.3 + volume_score * 0.3)
            
            return min(max(total_score, 0.0), 1.0)
            
        except Exception as e:
            self.logger.warning(f"Error calculando score: {e}")
            return 0.5
    
    def train_embeddings(self, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Entrena el sistema de embeddings.
        
        Args:
            force_retrain: Forzar reentrenamiento aunque exista modelo
            
        Returns:
            Métricas de entrenamiento
        """
        self.logger.info("Iniciando entrenamiento de embeddings")
        
        # Verificar si ya existe modelo entrenado
        embeddings_path = self.embeddings_dir / 'trading_embeddings.pkl'
        if embeddings_path.exists() and not force_retrain:
            self.logger.info("Cargando embeddings existentes")
            self.embeddings.load_embeddings(str(embeddings_path))
            return {'status': 'loaded_existing', 'path': str(embeddings_path)}
        
        # Obtener datos de entrenamiento
        samples = self.dataset_manager.get_all_samples()
        
        if len(samples) < 10:
            raise ValueError(f"Datos insuficientes para entrenamiento: {len(samples)} muestras")
        
        self.logger.info(f"Entrenando embeddings con {len(samples)} muestras")
        
        # Preparar datos para embeddings
        temporal_data = []
        technical_data = []
        pattern_data = []
        
        for sample in samples:
            try:
                # Convertir datos de muestra a DataFrame
                sample_df = pd.DataFrame(sample.data)
                
                # Preprocesar datos
                processed_data = self.preprocessor.prepare_features(sample_df)
                
                # Crear secuencias temporales
                temporal_sequence = self.preprocessor.create_sequences(
                    processed_data['ohlc_features'], 
                    sequence_length=self.training_config['sequence_length']
                )
                
                if len(temporal_sequence) > 0:
                    temporal_data.append(temporal_sequence[0])  # Tomar primera secuencia
                    technical_data.append(processed_data['technical_features'][-1])  # Últimas características técnicas
                    
                    # Codificar patrón
                    pattern_encoding = self._encode_pattern(sample.pattern)
                    pattern_data.append(pattern_encoding)
            
            except Exception as e:
                self.logger.warning(f"Error procesando muestra: {e}")
                continue
        
        if len(temporal_data) == 0:
            raise ValueError("No se pudieron procesar datos para embeddings")
        
        # Convertir a arrays numpy
        temporal_data = np.array(temporal_data)
        technical_data = np.array(technical_data)
        pattern_data = np.array(pattern_data)
        
        self.logger.info(f"Datos preparados - Temporal: {temporal_data.shape}, Técnico: {technical_data.shape}, Patrones: {pattern_data.shape}")
        
        # Entrenar embeddings
        training_metrics = self.embeddings.train(
            temporal_data=temporal_data,
            technical_data=technical_data,
            pattern_data=pattern_data,
            epochs=self.training_config['epochs'],
            batch_size=self.training_config['batch_size'],
            learning_rate=self.training_config['learning_rate']
        )
        
        # Guardar embeddings entrenados
        self.embeddings.save_embeddings(str(embeddings_path))
        
        self.logger.info("Entrenamiento de embeddings completado")
        
        return {
            'status': 'trained',
            'metrics': training_metrics,
            'samples_used': len(temporal_data),
            'model_path': str(embeddings_path)
        }
    
    def train_wyckoff_model(self, force_retrain: bool = False) -> Dict[str, Any]:
        """
        Entrena el modelo de clasificación de patrones Wyckoff.
        
        Args:
            force_retrain: Forzar reentrenamiento aunque exista modelo
            
        Returns:
            Métricas de entrenamiento
        """
        self.logger.info("Iniciando entrenamiento del modelo Wyckoff")
        
        # Verificar si ya existe modelo entrenado
        model_path = self.models_dir / 'wyckoff_classifier.pkl'
        if model_path.exists() and not force_retrain:
            self.logger.info("Cargando modelo Wyckoff existente")
            self.wyckoff_analyzer.load_analyzer(str(model_path))
            return {'status': 'loaded_existing', 'path': str(model_path)}
        
        # Obtener datos de entrenamiento
        samples = self.dataset_manager.get_all_samples()
        
        # Filtrar muestras por calidad y cantidad mínima por patrón
        filtered_samples = self._filter_training_samples(samples)
        
        if len(filtered_samples) < 20:
            raise ValueError(f"Datos insuficientes para entrenamiento: {len(filtered_samples)} muestras")
        
        self.logger.info(f"Entrenando modelo Wyckoff con {len(filtered_samples)} muestras")
        
        # Preparar datos para entrenamiento
        training_data = []
        
        for sample in filtered_samples:
            try:
                # Convertir datos de muestra a DataFrame
                sample_df = pd.DataFrame(sample.data)
                
                # Asegurar que tenemos las columnas necesarias
                if not all(col in sample_df.columns for col in ['open', 'high', 'low', 'close']):
                    continue
                
                training_data.append((sample_df, sample.pattern))
                
            except Exception as e:
                self.logger.warning(f"Error preparando muestra para entrenamiento: {e}")
                continue
        
        if len(training_data) == 0:
            raise ValueError("No se pudieron preparar datos para entrenamiento")
        
        # Entrenar el analizador
        training_metrics = self.wyckoff_analyzer.train_analyzer(
            training_data=training_data
        )
        
        # Guardar modelo entrenado
        self.wyckoff_analyzer.save_analyzer(str(model_path))
        
        self.logger.info("Entrenamiento del modelo Wyckoff completado")
        
        return {
            'status': 'trained',
            'metrics': training_metrics,
            'samples_used': len(training_data),
            'model_path': str(model_path)
        }
    
    def _encode_pattern(self, pattern: str) -> np.ndarray:
        """
        Codifica un patrón como vector one-hot.
        
        Args:
            pattern: Nombre del patrón
            
        Returns:
            Vector one-hot del patrón
        """
        patterns = ['accumulation', 'distribution', 'markup', 'markdown', 'no_pattern']
        encoding = np.zeros(len(patterns))
        
        if pattern in patterns:
            encoding[patterns.index(pattern)] = 1.0
        else:
            encoding[-1] = 1.0  # no_pattern por defecto
        
        return encoding
    
    def _filter_training_samples(self, samples: List[LabeledSample]) -> List[LabeledSample]:
        """
        Filtra las muestras de entrenamiento por calidad y balance.
        
        Args:
            samples: Lista de muestras etiquetadas
            
        Returns:
            Lista filtrada de muestras
        """
        # Contar muestras por patrón
        pattern_counts = {}
        for sample in samples:
            pattern_counts[sample.pattern] = pattern_counts.get(sample.pattern, 0) + 1
        
        self.logger.info(f"Distribución de patrones: {pattern_counts}")
        
        # Filtrar por calidad mínima
        quality_threshold = 0.5
        high_quality_samples = [
            sample for sample in samples 
            if sample.score >= quality_threshold and sample.confidence >= 0.6
        ]
        
        # Balancear clases (limitar muestras por patrón)
        max_samples_per_pattern = 100
        balanced_samples = []
        pattern_sample_counts = {}
        
        for sample in high_quality_samples:
            current_count = pattern_sample_counts.get(sample.pattern, 0)
            if current_count < max_samples_per_pattern:
                balanced_samples.append(sample)
                pattern_sample_counts[sample.pattern] = current_count + 1
        
        self.logger.info(f"Muestras después del filtrado: {len(balanced_samples)}")
        
        return balanced_samples
    
    def run_full_training_pipeline(self, 
                                 symbols: List[str],
                                 timeframes: List[str] = ['1h', '4h', '1d'],
                                 days_back: int = 365,
                                 force_retrain: bool = False) -> Dict[str, Any]:
        """
        Ejecuta el pipeline completo de entrenamiento de ML.
        
        Args:
            symbols: Lista de símbolos a procesar
            timeframes: Lista de timeframes
            days_back: Días hacia atrás para obtener datos
            force_retrain: Forzar reentrenamiento de todos los modelos
            
        Returns:
            Resumen completo del entrenamiento
        """
        self.logger.info("=== INICIANDO PIPELINE COMPLETO DE ENTRENAMIENTO ML ===")
        
        pipeline_results = {
            'start_time': datetime.now().isoformat(),
            'config': self.training_config,
            'data_preparation': {},
            'embeddings_training': {},
            'wyckoff_training': {},
            'validation_results': {},
            'end_time': None,
            'total_duration': None,
            'success': False
        }
        
        try:
            # Paso 1: Preparar datos de entrenamiento
            self.logger.info("Paso 1/4: Preparando datos de entrenamiento")
            data_stats = self.prepare_training_data(
                symbols=symbols,
                timeframes=timeframes,
                days_back=days_back
            )
            pipeline_results['data_preparation'] = data_stats
            
            # Paso 2: Entrenar embeddings
            self.logger.info("Paso 2/4: Entrenando sistema de embeddings")
            embeddings_results = self.train_embeddings(force_retrain=force_retrain)
            pipeline_results['embeddings_training'] = embeddings_results
            
            # Paso 3: Entrenar modelo Wyckoff
            self.logger.info("Paso 3/4: Entrenando modelo de clasificación Wyckoff")
            wyckoff_results = self.train_wyckoff_model(force_retrain=force_retrain)
            pipeline_results['wyckoff_training'] = wyckoff_results
            
            # Paso 4: Validación del pipeline
            self.logger.info("Paso 4/4: Validando pipeline completo")
            validation_results = self.validate_pipeline()
            pipeline_results['validation_results'] = validation_results
            
            pipeline_results['success'] = True
            self.logger.info("=== PIPELINE DE ENTRENAMIENTO COMPLETADO EXITOSAMENTE ===")
            
        except Exception as e:
            self.logger.error(f"Error en pipeline de entrenamiento: {e}")
            pipeline_results['error'] = str(e)
            
        finally:
            pipeline_results['end_time'] = datetime.now().isoformat()
            start_time = datetime.fromisoformat(pipeline_results['start_time'])
            end_time = datetime.fromisoformat(pipeline_results['end_time'])
            pipeline_results['total_duration'] = str(end_time - start_time)
        
        return pipeline_results
    
    def validate_pipeline(self) -> Dict[str, Any]:
        """
        Valida que el pipeline completo funcione correctamente.
        
        Returns:
            Resultados de validación
        """
        validation_results = {
            'embeddings_loaded': False,
            'wyckoff_model_loaded': False,
            'end_to_end_test': False,
            'sample_predictions': [],
            'performance_metrics': {}
        }
        
        try:
            # Verificar que los embeddings estén cargados
            embeddings_path = self.embeddings_dir / 'trading_embeddings.pkl'
            if embeddings_path.exists():
                validation_results['embeddings_loaded'] = True
            
            # Verificar que el modelo Wyckoff esté cargado
            model_path = self.models_dir / 'wyckoff_classifier.pkl'
            if model_path.exists():
                validation_results['wyckoff_model_loaded'] = True
            
            # Test end-to-end con datos de muestra
            test_samples = self.dataset_manager.get_samples_by_pattern('accumulation', limit=3)
            
            for i, sample in enumerate(test_samples[:3]):
                try:
                    # Convertir muestra a DataFrame
                    sample_df = pd.DataFrame(sample.data)
                    
                    # Predecir con modelo Wyckoff
                    predicted_pattern, confidence = self.wyckoff_analyzer.classifier.predict(sample_df)
                    
                    # Generar embedding
                    processed_data = self.preprocessor.prepare_features(sample_df)
                    
                    prediction_result = {
                        'sample_id': i,
                        'true_pattern': sample.pattern,
                        'predicted_pattern': predicted_pattern,
                        'confidence': confidence,
                        'match': sample.pattern == predicted_pattern
                    }
                    
                    validation_results['sample_predictions'].append(prediction_result)
                    
                except Exception as e:
                    self.logger.warning(f"Error en predicción de muestra {i}: {e}")
            
            # Calcular métricas de rendimiento
            if validation_results['sample_predictions']:
                correct_predictions = sum(1 for pred in validation_results['sample_predictions'] if pred['match'])
                total_predictions = len(validation_results['sample_predictions'])
                
                validation_results['performance_metrics'] = {
                    'accuracy': correct_predictions / total_predictions if total_predictions > 0 else 0,
                    'total_predictions': total_predictions,
                    'correct_predictions': correct_predictions
                }
                
                validation_results['end_to_end_test'] = True
            
        except Exception as e:
            self.logger.error(f"Error en validación del pipeline: {e}")
            validation_results['error'] = str(e)
        
        return validation_results
    
    def save_training_report(self, results: Dict[str, Any], filepath: Optional[str] = None) -> str:
        """
        Guarda un reporte detallado del entrenamiento.
        
        Args:
            results: Resultados del pipeline de entrenamiento
            filepath: Ruta donde guardar el reporte
            
        Returns:
            Ruta del archivo guardado
        """
        if filepath is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filepath = self.models_dir / f'training_report_{timestamp}.json'
        
        # Guardar reporte
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Reporte de entrenamiento guardado en: {filepath}")
        
        return str(filepath)

class ModelValidator:
    """
    Validador de modelos entrenados para asegurar calidad y rendimiento.
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def validate_model_performance(self, model_path: str, 
                                 test_data: List[Tuple[pd.DataFrame, str]]) -> Dict[str, Any]:
        """
        Valida el rendimiento de un modelo entrenado.
        
        Args:
            model_path: Ruta al modelo entrenado
            test_data: Datos de prueba
            
        Returns:
            Métricas de validación
        """
        # Implementar validación detallada
        pass
    
    def cross_validate_model(self, training_data: List[Tuple[pd.DataFrame, str]], 
                           k_folds: int = 5) -> Dict[str, Any]:
        """
        Realiza validación cruzada del modelo.
        
        Args:
            training_data: Datos de entrenamiento
            k_folds: Número de folds para validación cruzada
            
        Returns:
            Resultados de validación cruzada
        """
        # Implementar validación cruzada
        pass

# Función de utilidad para ejecutar entrenamiento desde línea de comandos
def main():
    """
    Función principal para ejecutar el pipeline de entrenamiento.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline de entrenamiento ML - Fase 4')
    parser.add_argument('--symbols', nargs='+', default=['BTCUSDT', 'ETHUSDT'], 
                       help='Símbolos a procesar')
    parser.add_argument('--timeframes', nargs='+', default=['1h', '4h', '1d'],
                       help='Timeframes a usar')
    parser.add_argument('--days-back', type=int, default=365,
                       help='Días hacia atrás para obtener datos')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Forzar reentrenamiento de modelos existentes')
    parser.add_argument('--config', type=str, default=None,
                       help='Ruta al archivo de configuración')
    
    args = parser.parse_args()
    
    # Configurar logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Ejecutar pipeline
    pipeline = MLTrainingPipeline(config_path=args.config)
    
    results = pipeline.run_full_training_pipeline(
        symbols=args.symbols,
        timeframes=args.timeframes,
        days_back=args.days_back,
        force_retrain=args.force_retrain
    )
    
    # Guardar reporte
    report_path = pipeline.save_training_report(results)
    
    print(f"\n=== RESUMEN DEL ENTRENAMIENTO ===")
    print(f"Éxito: {results['success']}")
    print(f"Duración total: {results['total_duration']}")
    print(f"Muestras procesadas: {results.get('data_preparation', {}).get('total_samples', 0)}")
    print(f"Reporte guardado en: {report_path}")
    
    if not results['success']:
        print(f"Error: {results.get('error', 'Error desconocido')}")
        return 1
    
    return 0

if __name__ == '__main__':
    exit(main())