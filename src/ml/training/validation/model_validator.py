import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import logging
import json
from pathlib import Path

# Imports opcionales de sklearn
try:
    from sklearn.model_selection import KFold
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    KFold = None
    accuracy_score = None
    precision_recall_fscore_support = None
    confusion_matrix = None

from ....config.config_manager import ConfigManager
from ...wyckoff_model import WyckoffAnalyzer
from ...labeling.dataset_manager import LabeledSample

class ModelValidator:
    """
    Validador de modelos entrenados para asegurar calidad y rendimiento.
    """
    
    def __init__(self, config: ConfigManager):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Configuración de validación
        self.validation_config = self.config.get('model_validation', {
            'min_accuracy': 0.7,
            'min_precision': 0.6,
            'min_recall': 0.6,
            'cross_validation_folds': 5,
            'test_size': 0.2,
            'random_state': 42
        })
    
    def validate_model_performance(self, model_path: str, 
                                 test_data: List[Tuple[pd.DataFrame, str]]) -> Dict[str, Any]:
        """
        Valida el rendimiento de un modelo entrenado.
        
        Args:
            model_path: Ruta al modelo entrenado
            test_data: Datos de prueba como lista de tuplas (DataFrame, etiqueta)
            
        Returns:
            Métricas de validación
        """
        self.logger.info(f"Validando modelo: {model_path}")
        
        validation_results = {
            'model_path': model_path,
            'test_samples': len(test_data),
            'accuracy': 0.0,
            'precision': {},
            'recall': {},
            'f1_score': {},
            'confusion_matrix': [],
            'classification_report': {},
            'passes_validation': False,
            'validation_errors': []
        }
        
        try:
            # Cargar modelo
            analyzer = WyckoffAnalyzer()
            analyzer.load_analyzer(model_path)
            
            # Realizar predicciones
            y_true = []
            y_pred = []
            prediction_confidences = []
            
            for data, true_label in test_data:
                try:
                    predicted_pattern, confidence = analyzer.classifier.predict(data)
                    y_true.append(true_label)
                    y_pred.append(predicted_pattern)
                    prediction_confidences.append(confidence)
                    
                except Exception as e:
                    self.logger.warning(f"Error en predicción: {e}")
                    continue
            
            if len(y_true) == 0:
                raise ValueError("No se pudieron realizar predicciones")
            
            # Obtener etiquetas únicas
            unique_labels = list(set(y_true + y_pred))
            
            if SKLEARN_AVAILABLE:
                # Calcular métricas con sklearn
                accuracy = accuracy_score(y_true, y_pred)
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true, y_pred, average=None, zero_division=0
                )
                
                # Organizar métricas por clase
                for i, label in enumerate(unique_labels):
                    if i < len(precision):
                        validation_results['precision'][label] = float(precision[i])
                        validation_results['recall'][label] = float(recall[i])
                        validation_results['f1_score'][label] = float(f1[i])
                
                # Matriz de confusión
                cm = confusion_matrix(y_true, y_pred, labels=unique_labels)
                validation_results['confusion_matrix'] = cm.tolist()
                
                # Métricas generales
                validation_results['accuracy'] = float(accuracy)
                validation_results['avg_precision'] = float(np.mean(precision))
                validation_results['avg_recall'] = float(np.mean(recall))
                validation_results['avg_f1'] = float(np.mean(f1))
            else:
                # Calcular métricas manualmente
                accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
                
                # Calcular métricas por clase manualmente
                precision_dict = {}
                recall_dict = {}
                f1_dict = {}
                cm_dict = {}
                
                for label in unique_labels:
                    tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
                    fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
                    fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
                    
                    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                    
                    precision_dict[label] = float(precision)
                    recall_dict[label] = float(recall)
                    f1_dict[label] = float(f1)
                
                validation_results['precision'] = precision_dict
                validation_results['recall'] = recall_dict
                validation_results['f1_score'] = f1_dict
                
                # Matriz de confusión manual
                cm_matrix = []
                for true_label in unique_labels:
                    row = []
                    for pred_label in unique_labels:
                        count = sum(1 for t, p in zip(y_true, y_pred) if t == true_label and p == pred_label)
                        row.append(count)
                    cm_matrix.append(row)
                validation_results['confusion_matrix'] = cm_matrix
                
                # Métricas generales
                validation_results['accuracy'] = float(accuracy)
                validation_results['avg_precision'] = float(np.mean(list(precision_dict.values())))
                validation_results['avg_recall'] = float(np.mean(list(recall_dict.values())))
                validation_results['avg_f1'] = float(np.mean(list(f1_dict.values())))
            
            validation_results['avg_confidence'] = float(np.mean(prediction_confidences))
            
            # Verificar si pasa la validación
            passes_validation = self._check_validation_criteria(validation_results)
            validation_results['passes_validation'] = passes_validation
            
            if not passes_validation:
                validation_results['validation_errors'] = self._get_validation_errors(validation_results)
            
            self.logger.info(f"Validación completada - Accuracy: {accuracy:.3f}, Pasa validación: {passes_validation}")
            
        except Exception as e:
            self.logger.error(f"Error en validación del modelo: {e}")
            validation_results['error'] = str(e)
        
        return validation_results
    
    def cross_validate_model(self, training_data: List[Tuple[pd.DataFrame, str]], 
                           k_folds: int = 5) -> Dict[str, Any]:
        """
        Realiza validación cruzada del modelo.
        
        Args:
            training_data: Datos de entrenamiento como lista de tuplas (DataFrame, etiqueta)
            k_folds: Número de folds para validación cruzada
            
        Returns:
            Resultados de validación cruzada
        """
        self.logger.info(f"Iniciando validación cruzada con {k_folds} folds")
        
        cv_results = {
            'k_folds': k_folds,
            'total_samples': len(training_data),
            'fold_results': [],
            'mean_accuracy': 0.0,
            'std_accuracy': 0.0,
            'mean_precision': 0.0,
            'mean_recall': 0.0,
            'mean_f1': 0.0,
            'best_fold': 0,
            'worst_fold': 0
        }
        
        try:
            if not SKLEARN_AVAILABLE:
                self.logger.warning("scikit-learn no está disponible. Realizando validación simple.")
                # Realizar una validación simple sin KFold
                return self._simple_validation(training_data)
            
            # Preparar datos
            X = [data for data, _ in training_data]
            y = [label for _, label in training_data]
            
            # Crear índices para validación cruzada
            kf = KFold(n_splits=k_folds, shuffle=True, random_state=self.validation_config['random_state'])
            
            fold_accuracies = []
            fold_precisions = []
            fold_recalls = []
            fold_f1s = []
            
            for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
                self.logger.info(f"Procesando fold {fold_idx + 1}/{k_folds}")
                
                # Dividir datos
                train_fold_data = [(X[i], y[i]) for i in train_idx]
                val_fold_data = [(X[i], y[i]) for i in val_idx]
                
                # Entrenar modelo para este fold
                fold_analyzer = WyckoffAnalyzer()
                
                try:
                    # Entrenar
                    training_metrics = fold_analyzer.train_analyzer(train_fold_data)
                    
                    # Validar
                    fold_results = self._validate_fold(fold_analyzer, val_fold_data)
                    
                    fold_results['fold_index'] = fold_idx
                    fold_results['train_samples'] = len(train_fold_data)
                    fold_results['val_samples'] = len(val_fold_data)
                    fold_results['training_metrics'] = training_metrics
                    
                    cv_results['fold_results'].append(fold_results)
                    
                    # Acumular métricas
                    fold_accuracies.append(fold_results['accuracy'])
                    fold_precisions.append(fold_results.get('avg_precision', 0.0))
                    fold_recalls.append(fold_results.get('avg_recall', 0.0))
                    fold_f1s.append(fold_results.get('avg_f1', 0.0))
                    
                except Exception as e:
                    self.logger.error(f"Error en fold {fold_idx}: {e}")
                    continue
            
            # Calcular estadísticas finales
            if fold_accuracies:
                cv_results['mean_accuracy'] = float(np.mean(fold_accuracies))
                cv_results['std_accuracy'] = float(np.std(fold_accuracies))
                cv_results['mean_precision'] = float(np.mean(fold_precisions))
                cv_results['mean_recall'] = float(np.mean(fold_recalls))
                cv_results['mean_f1'] = float(np.mean(fold_f1s))
                
                # Identificar mejor y peor fold
                cv_results['best_fold'] = int(np.argmax(fold_accuracies))
                cv_results['worst_fold'] = int(np.argmin(fold_accuracies))
            
            self.logger.info(f"Validación cruzada completada - Accuracy promedio: {cv_results['mean_accuracy']:.3f} ± {cv_results['std_accuracy']:.3f}")
            
        except Exception as e:
            self.logger.error(f"Error en validación cruzada: {e}")
            cv_results['error'] = str(e)
        
        return cv_results
    
    def validate_data_quality(self, samples: List[LabeledSample]) -> Dict[str, Any]:
        """
        Valida la calidad de los datos de entrenamiento.
        
        Args:
            samples: Lista de muestras etiquetadas
            
        Returns:
            Reporte de calidad de datos
        """
        self.logger.info(f"Validando calidad de {len(samples)} muestras")
        
        quality_report = {
            'total_samples': len(samples),
            'pattern_distribution': {},
            'quality_scores': [],
            'confidence_scores': [],
            'data_issues': [],
            'recommendations': [],
            'overall_quality': 'unknown'
        }
        
        try:
            # Analizar distribución de patrones
            for sample in samples:
                pattern = sample.pattern
                if pattern not in quality_report['pattern_distribution']:
                    quality_report['pattern_distribution'][pattern] = 0
                quality_report['pattern_distribution'][pattern] += 1
                
                quality_report['quality_scores'].append(sample.score)
                quality_report['confidence_scores'].append(sample.confidence)
            
            # Calcular estadísticas
            avg_quality = np.mean(quality_report['quality_scores'])
            avg_confidence = np.mean(quality_report['confidence_scores'])
            
            quality_report['avg_quality_score'] = float(avg_quality)
            quality_report['avg_confidence_score'] = float(avg_confidence)
            quality_report['min_quality_score'] = float(np.min(quality_report['quality_scores']))
            quality_report['max_quality_score'] = float(np.max(quality_report['quality_scores']))
            
            # Detectar problemas
            issues = []
            recommendations = []
            
            # Verificar balance de clases
            pattern_counts = list(quality_report['pattern_distribution'].values())
            if len(pattern_counts) > 1:
                max_count = max(pattern_counts)
                min_count = min(pattern_counts)
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                
                if imbalance_ratio > 5:
                    issues.append(f"Desbalance de clases detectado (ratio: {imbalance_ratio:.1f})")
                    recommendations.append("Considerar técnicas de balanceo de clases")
            
            # Verificar calidad mínima
            low_quality_samples = sum(1 for score in quality_report['quality_scores'] if score < 0.5)
            if low_quality_samples > len(samples) * 0.2:
                issues.append(f"{low_quality_samples} muestras con calidad baja (<0.5)")
                recommendations.append("Revisar y filtrar muestras de baja calidad")
            
            # Verificar confianza
            low_confidence_samples = sum(1 for conf in quality_report['confidence_scores'] if conf < 0.6)
            if low_confidence_samples > len(samples) * 0.3:
                issues.append(f"{low_confidence_samples} muestras con baja confianza (<0.6)")
                recommendations.append("Mejorar proceso de etiquetado para aumentar confianza")
            
            quality_report['data_issues'] = issues
            quality_report['recommendations'] = recommendations
            
            # Determinar calidad general
            if avg_quality >= 0.8 and avg_confidence >= 0.8 and len(issues) == 0:
                quality_report['overall_quality'] = 'excellent'
            elif avg_quality >= 0.7 and avg_confidence >= 0.7 and len(issues) <= 1:
                quality_report['overall_quality'] = 'good'
            elif avg_quality >= 0.6 and avg_confidence >= 0.6:
                quality_report['overall_quality'] = 'acceptable'
            else:
                quality_report['overall_quality'] = 'poor'
            
            self.logger.info(f"Calidad de datos: {quality_report['overall_quality']} (Score: {avg_quality:.3f}, Confianza: {avg_confidence:.3f})")
            
        except Exception as e:
            self.logger.error(f"Error validando calidad de datos: {e}")
            quality_report['error'] = str(e)
        
        return quality_report
    
    def _validate_fold(self, analyzer: WyckoffAnalyzer, val_data: List[Tuple[pd.DataFrame, str]]) -> Dict[str, Any]:
        """
        Valida un fold específico durante la validación cruzada.
        
        Args:
            analyzer: Analizador entrenado
            val_data: Datos de validación
            
        Returns:
            Métricas del fold
        """
        y_true = []
        y_pred = []
        
        for data, true_label in val_data:
            try:
                predicted_pattern, _ = analyzer.classifier.predict(data)
                y_true.append(true_label)
                y_pred.append(predicted_pattern)
            except Exception:
                continue
        
        if len(y_true) == 0:
            return {'accuracy': 0.0, 'error': 'No predictions made'}
        
        if SKLEARN_AVAILABLE:
            accuracy = accuracy_score(y_true, y_pred)
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='macro', zero_division=0
            )
        else:
            # Calcular métricas manualmente
            accuracy = sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
            
            # Métricas simplificadas sin sklearn
            unique_labels = list(set(y_true + y_pred))
            precision_scores = []
            recall_scores = []
            f1_scores = []
            
            for label in unique_labels:
                tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
                fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
                fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
                
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
                
                precision_scores.append(precision)
                recall_scores.append(recall)
                f1_scores.append(f1)
            
            precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0.0
            recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0
            f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
        
        return {
            'accuracy': float(accuracy),
            'avg_precision': float(precision),
            'avg_recall': float(recall),
            'avg_f1': float(f1),
            'predictions_made': len(y_true)
        }
    
    def _check_validation_criteria(self, results: Dict[str, Any]) -> bool:
        """
        Verifica si el modelo cumple con los criterios de validación.
        
        Args:
            results: Resultados de validación
            
        Returns:
            True si pasa todos los criterios
        """
        criteria = [
            results['accuracy'] >= self.validation_config['min_accuracy'],
            results.get('avg_precision', 0) >= self.validation_config['min_precision'],
            results.get('avg_recall', 0) >= self.validation_config['min_recall']
        ]
        
        return all(criteria)
    
    def _get_validation_errors(self, results: Dict[str, Any]) -> List[str]:
        """
        Obtiene lista de errores de validación.
        
        Args:
            results: Resultados de validación
            
        Returns:
            Lista de errores encontrados
        """
        errors = []
        
        if results['accuracy'] < self.validation_config['min_accuracy']:
            errors.append(f"Accuracy insuficiente: {results['accuracy']:.3f} < {self.validation_config['min_accuracy']}")
        
        if results.get('avg_precision', 0) < self.validation_config['min_precision']:
            errors.append(f"Precision insuficiente: {results.get('avg_precision', 0):.3f} < {self.validation_config['min_precision']}")
        
        if results.get('avg_recall', 0) < self.validation_config['min_recall']:
            errors.append(f"Recall insuficiente: {results.get('avg_recall', 0):.3f} < {self.validation_config['min_recall']}")
        
        return errors
    
    def save_validation_report(self, results: Dict[str, Any], filepath: Optional[str] = None) -> str:
        """
        Guarda un reporte de validación.
        
        Args:
            results: Resultados de validación
            filepath: Ruta donde guardar el reporte
            
        Returns:
            Ruta del archivo guardado
        """
        if filepath is None:
            timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
            filepath = f'validation_report_{timestamp}.json'
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        self.logger.info(f"Reporte de validación guardado en: {filepath}")
        
        return str(filepath)
    
    def _simple_validation(self, training_data: List[Tuple[pd.DataFrame, str]]) -> Dict[str, Any]:
        """
        Realiza una validación simple sin sklearn cuando no está disponible.
        
        Args:
            training_data: Datos de entrenamiento
            
        Returns:
            Resultados de validación simplificada
        """
        self.logger.info("Realizando validación simple sin sklearn")
        
        # Dividir datos manualmente (80/20)
        total_samples = len(training_data)
        split_idx = int(total_samples * 0.8)
        
        train_data = training_data[:split_idx]
        val_data = training_data[split_idx:]
        
        # Entrenar modelo
        analyzer = WyckoffAnalyzer()
        training_metrics = analyzer.train_analyzer(train_data)
        
        # Validar
        val_results = self._validate_fold(analyzer, val_data)
        
        return {
            'k_folds': 1,
            'total_samples': total_samples,
            'fold_results': [val_results],
            'mean_accuracy': val_results.get('accuracy', 0.0),
            'std_accuracy': 0.0,
            'mean_precision': val_results.get('precision', 0.0),
            'mean_recall': val_results.get('recall', 0.0),
            'mean_f1': val_results.get('f1_score', 0.0),
            'best_fold': 0,
            'worst_fold': 0,
            'training_metrics': training_metrics,
            'sklearn_available': False
        }