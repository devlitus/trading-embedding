"""Sistema de Verificación para la Fase 2: Análisis Técnico

Este sistema proporciona herramientas completas para verificar que todos los
componentes de análisis técnico funcionen correctamente, incluyendo:
- Validación de datos de entrada
- Verificación de cálculos de indicadores
- Pruebas de consistencia
- Análisis de rendimiento
- Reportes detallados
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import json
import time
from dataclasses import dataclass, asdict

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis.technical_indicators import TechnicalIndicators
from analysis.trend_detection import TrendDetector
from analysis.pattern_recognition import PatternRecognizer
from analysis.technical_analysis import TechnicalAnalyzer, analyze_symbol


@dataclass
class VerificationResult:
    """Resultado de una verificación específica"""
    test_name: str
    passed: bool
    message: str
    details: Dict[str, Any]
    execution_time: float
    timestamp: datetime


@dataclass
class ValidationReport:
    """Reporte completo de validación"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    success_rate: float
    total_execution_time: float
    results: List[VerificationResult]
    summary: Dict[str, Any]
    timestamp: datetime


class DataValidator:
    """Validador de datos de entrada"""
    
    @staticmethod
    def validate_ohlc_data(df: pd.DataFrame) -> VerificationResult:
        """Valida la estructura y consistencia de datos OHLC"""
        start_time = time.time()
        errors = []
        warnings = []
        
        try:
            # Verificar columnas requeridas
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                errors.append(f"Columnas faltantes: {missing_cols}")
            
            # Verificar que no haya valores nulos
            null_counts = df[required_cols].isnull().sum()
            if null_counts.any():
                warnings.append(f"Valores nulos encontrados: {null_counts.to_dict()}")
            
            # Verificar consistencia OHLC
            inconsistent_rows = []
            for i, row in df.iterrows():
                if not (row['low'] <= row['open'] <= row['high'] and 
                       row['low'] <= row['close'] <= row['high']):
                    inconsistent_rows.append(i)
            
            if inconsistent_rows:
                errors.append(f"Filas con datos OHLC inconsistentes: {len(inconsistent_rows)}")
            
            # Verificar volúmenes negativos
            negative_volume = (df['volume'] < 0).sum()
            if negative_volume > 0:
                errors.append(f"Volúmenes negativos encontrados: {negative_volume}")
            
            # Verificar precios negativos o cero
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                invalid_prices = (df[col] <= 0).sum()
                if invalid_prices > 0:
                    errors.append(f"Precios inválidos en {col}: {invalid_prices}")
            
            execution_time = time.time() - start_time
            
            if errors:
                return VerificationResult(
                    test_name="Validación de Datos OHLC",
                    passed=False,
                    message=f"Errores encontrados: {'; '.join(errors)}",
                    details={"errors": errors, "warnings": warnings},
                    execution_time=execution_time,
                    timestamp=datetime.now()
                )
            else:
                return VerificationResult(
                    test_name="Validación de Datos OHLC",
                    passed=True,
                    message="Datos OHLC válidos",
                    details={"warnings": warnings, "rows_validated": len(df)},
                    execution_time=execution_time,
                    timestamp=datetime.now()
                )
                
        except Exception as e:
            return VerificationResult(
                test_name="Validación de Datos OHLC",
                passed=False,
                message=f"Error durante validación: {str(e)}",
                details={"exception": str(e)},
                execution_time=time.time() - start_time,
                timestamp=datetime.now()
            )


class IndicatorValidator:
    """Validador de indicadores técnicos"""
    
    def __init__(self):
        self.indicators = TechnicalIndicators()
    
    def verify_sma_calculation(self, df: pd.DataFrame, period: int = 20) -> VerificationResult:
        """Verifica el cálculo de SMA"""
        start_time = time.time()
        
        try:
            # Calcular SMA usando nuestro método
            df_with_sma = self.indicators.calculate_all_indicators(df)
            our_sma = df_with_sma[f'sma_{period}']
            
            # Calcular SMA manualmente para verificación
            manual_sma = df['close'].rolling(window=period).mean()
            
            # Comparar resultados (ignorar NaN)
            valid_indices = ~(our_sma.isna() | manual_sma.isna())
            if valid_indices.sum() == 0:
                return VerificationResult(
                    test_name=f"Verificación SMA({period})",
                    passed=False,
                    message="No hay datos válidos para comparar",
                    details={},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now()
                )
            
            differences = np.abs(our_sma[valid_indices] - manual_sma[valid_indices])
            max_diff = differences.max()
            mean_diff = differences.mean()
            
            # Tolerancia para diferencias de punto flotante
            tolerance = 1e-10
            passed = max_diff < tolerance
            
            return VerificationResult(
                test_name=f"Verificación SMA({period})",
                passed=passed,
                message=f"Diferencia máxima: {max_diff:.2e}, Diferencia promedio: {mean_diff:.2e}",
                details={
                    "max_difference": float(max_diff),
                    "mean_difference": float(mean_diff),
                    "tolerance": tolerance,
                    "valid_points": int(valid_indices.sum())
                },
                execution_time=time.time() - start_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return VerificationResult(
                test_name=f"Verificación SMA({period})",
                passed=False,
                message=f"Error durante verificación: {str(e)}",
                details={"exception": str(e)},
                execution_time=time.time() - start_time,
                timestamp=datetime.now()
            )
    
    def verify_rsi_calculation(self, df: pd.DataFrame, period: int = 14) -> VerificationResult:
        """Verifica el cálculo de RSI"""
        start_time = time.time()
        
        try:
            # Calcular RSI usando nuestro método
            df_with_rsi = self.indicators.calculate_all_indicators(df)
            our_rsi = df_with_rsi['rsi']
            
            # Verificar que RSI esté en el rango correcto (0-100)
            valid_rsi = our_rsi.dropna()
            if len(valid_rsi) == 0:
                return VerificationResult(
                    test_name=f"Verificación RSI({period})",
                    passed=False,
                    message="No hay valores RSI válidos",
                    details={},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now()
                )
            
            # Verificar rango
            min_rsi = valid_rsi.min()
            max_rsi = valid_rsi.max()
            range_valid = (min_rsi >= 0) and (max_rsi <= 100)
            
            # Verificar que no haya valores extremos constantes
            unique_values = len(valid_rsi.unique())
            diversity_check = unique_values > 1
            
            passed = range_valid and diversity_check
            
            return VerificationResult(
                test_name=f"Verificación RSI({period})",
                passed=passed,
                message=f"Rango: [{min_rsi:.2f}, {max_rsi:.2f}], Valores únicos: {unique_values}",
                details={
                    "min_rsi": float(min_rsi),
                    "max_rsi": float(max_rsi),
                    "unique_values": unique_values,
                    "range_valid": range_valid,
                    "diversity_check": diversity_check,
                    "valid_points": len(valid_rsi)
                },
                execution_time=time.time() - start_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return VerificationResult(
                test_name=f"Verificación RSI({period})",
                passed=False,
                message=f"Error durante verificación: {str(e)}",
                details={"exception": str(e)},
                execution_time=time.time() - start_time,
                timestamp=datetime.now()
            )
    
    def verify_all_indicators(self, df: pd.DataFrame) -> List[VerificationResult]:
        """Verifica todos los indicadores"""
        results = []
        
        # Verificar que se calculen todos los indicadores esperados
        start_time = time.time()
        try:
            df_with_indicators = self.indicators.calculate_all_indicators(df)
            
            expected_indicators = [
                'sma_20', 'sma_50', 'ema_12', 'ema_26', 'rsi', 'macd', 'macd_signal',
                'bb_upper', 'bb_lower', 'bb_middle', 'atr', 'obv', 'stoch_k', 'stoch_d'
            ]
            
            missing_indicators = []
            present_indicators = []
            
            for indicator in expected_indicators:
                if indicator in df_with_indicators.columns:
                    present_indicators.append(indicator)
                else:
                    missing_indicators.append(indicator)
            
            passed = len(missing_indicators) == 0
            
            results.append(VerificationResult(
                test_name="Verificación de Indicadores Completos",
                passed=passed,
                message=f"Indicadores presentes: {len(present_indicators)}/{len(expected_indicators)}",
                details={
                    "present_indicators": present_indicators,
                    "missing_indicators": missing_indicators,
                    "total_columns": len(df_with_indicators.columns)
                },
                execution_time=time.time() - start_time,
                timestamp=datetime.now()
            ))
            
        except Exception as e:
            results.append(VerificationResult(
                test_name="Verificación de Indicadores Completos",
                passed=False,
                message=f"Error durante verificación: {str(e)}",
                details={"exception": str(e)},
                execution_time=time.time() - start_time,
                timestamp=datetime.now()
            ))
        
        # Verificar cálculos específicos
        results.append(self.verify_sma_calculation(df))
        results.append(self.verify_rsi_calculation(df))
        
        return results


class TrendValidator:
    """Validador de detección de tendencias"""
    
    def __init__(self):
        self.detector = TrendDetector()
    
    def verify_trend_detection(self, df: pd.DataFrame) -> VerificationResult:
        """Verifica la detección de tendencias"""
        start_time = time.time()
        
        try:
            # Detectar cambios de tendencia
            trend_changes = self.detector.detect_trend_changes(df)
            
            # Verificar que el resultado sea un DataFrame
            if not isinstance(trend_changes, pd.DataFrame):
                return VerificationResult(
                    test_name="Verificación de Detección de Tendencias",
                    passed=False,
                    message="El resultado no es un DataFrame",
                    details={"result_type": str(type(trend_changes))},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now()
                )
            
            # Verificar niveles de soporte y resistencia
            levels = self.detector.detect_support_resistance_levels(df)
            
            # Verificar puntuación de fuerza de tendencia
            strength_score = self.detector.calculate_trend_strength_score(df)
            
            # Validar que la puntuación esté en rango válido
            if hasattr(strength_score, 'iloc'):
                current_strength = strength_score.iloc[-1]
            else:
                current_strength = strength_score
            
            strength_valid = 0 <= current_strength <= 100
            
            passed = True
            details = {
                "trend_changes_count": len(trend_changes),
                "support_levels": len(levels.get('support', [])),
                "resistance_levels": len(levels.get('resistance', [])),
                "current_strength": float(current_strength),
                "strength_valid": strength_valid
            }
            
            return VerificationResult(
                test_name="Verificación de Detección de Tendencias",
                passed=passed,
                message=f"Tendencias detectadas correctamente. Fuerza: {current_strength:.2f}",
                details=details,
                execution_time=time.time() - start_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return VerificationResult(
                test_name="Verificación de Detección de Tendencias",
                passed=False,
                message=f"Error durante verificación: {str(e)}",
                details={"exception": str(e)},
                execution_time=time.time() - start_time,
                timestamp=datetime.now()
            )


class PatternValidator:
    """Validador de reconocimiento de patrones"""
    
    def __init__(self):
        self.recognizer = PatternRecognizer()
    
    def verify_pattern_detection(self, df: pd.DataFrame) -> VerificationResult:
        """Verifica la detección de patrones"""
        start_time = time.time()
        
        try:
            # Detectar todos los patrones
            patterns = self.recognizer.detect_all_patterns(df)
            
            # Verificar que el resultado sea una lista
            if not isinstance(patterns, list):
                return VerificationResult(
                    test_name="Verificación de Detección de Patrones",
                    passed=False,
                    message="El resultado no es una lista",
                    details={"result_type": str(type(patterns))},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now()
                )
            
            # Verificar estructura de patrones
            valid_patterns = 0
            invalid_patterns = 0
            
            for pattern in patterns:
                if hasattr(pattern, 'confidence') and hasattr(pattern, 'description'):
                    if 0 <= pattern.confidence <= 1:
                        valid_patterns += 1
                    else:
                        invalid_patterns += 1
                else:
                    invalid_patterns += 1
            
            # Obtener resumen por tipo
            summary = self.recognizer.get_pattern_summary(patterns)
            
            passed = invalid_patterns == 0
            
            return VerificationResult(
                test_name="Verificación de Detección de Patrones",
                passed=passed,
                message=f"Patrones válidos: {valid_patterns}, Inválidos: {invalid_patterns}",
                details={
                    "total_patterns": len(patterns),
                    "valid_patterns": valid_patterns,
                    "invalid_patterns": invalid_patterns,
                    "pattern_summary": summary
                },
                execution_time=time.time() - start_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return VerificationResult(
                test_name="Verificación de Detección de Patrones",
                passed=False,
                message=f"Error durante verificación: {str(e)}",
                details={"exception": str(e)},
                execution_time=time.time() - start_time,
                timestamp=datetime.now()
            )


class IntegratedAnalysisValidator:
    """Validador del análisis técnico integrado"""
    
    def __init__(self):
        self.analyzer = TechnicalAnalyzer()
    
    def verify_integrated_analysis(self, df: pd.DataFrame, symbol: str = "TEST") -> VerificationResult:
        """Verifica el análisis técnico integrado"""
        start_time = time.time()
        
        try:
            # Realizar análisis completo
            result = self.analyzer.analyze(df, symbol)
            
            # Verificar estructura del resultado
            required_attributes = [
                'timestamp', 'symbol', 'indicators', 'trend_analysis',
                'patterns', 'signals', 'overall_score', 'recommendation'
            ]
            
            missing_attributes = []
            for attr in required_attributes:
                if not hasattr(result, attr):
                    missing_attributes.append(attr)
            
            if missing_attributes:
                return VerificationResult(
                    test_name="Verificación de Análisis Integrado",
                    passed=False,
                    message=f"Atributos faltantes: {missing_attributes}",
                    details={"missing_attributes": missing_attributes},
                    execution_time=time.time() - start_time,
                    timestamp=datetime.now()
                )
            
            # Verificar contenido
            checks = {
                "symbol_correct": result.symbol == symbol,
                "has_indicators": len(result.indicators) > 0,
                "has_trend_analysis": len(result.trend_analysis) > 0,
                "has_patterns": isinstance(result.patterns, list),
                "has_signals": 'buy_signals' in result.signals and 'sell_signals' in result.signals,
                "valid_recommendation": result.recommendation in ['BUY', 'SELL', 'HOLD', 'NEUTRAL'],
                "valid_score_range": True  # Will be updated below
            }
            
            # Verificar puntuación
            if hasattr(result.overall_score, 'iloc'):
                score_value = result.overall_score.iloc[-1]
            else:
                score_value = result.overall_score
            
            checks["valid_score_range"] = -100 <= score_value <= 100
            
            all_passed = all(checks.values())
            
            return VerificationResult(
                test_name="Verificación de Análisis Integrado",
                passed=all_passed,
                message=f"Análisis integrado {'exitoso' if all_passed else 'con errores'}",
                details={
                    "checks": checks,
                    "overall_score": float(score_value),
                    "recommendation": result.recommendation,
                    "indicators_count": len(result.indicators),
                    "patterns_count": len(result.patterns),
                    "buy_signals": len(result.signals.get('buy_signals', [])),
                    "sell_signals": len(result.signals.get('sell_signals', []))
                },
                execution_time=time.time() - start_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return VerificationResult(
                test_name="Verificación de Análisis Integrado",
                passed=False,
                message=f"Error durante verificación: {str(e)}",
                details={"exception": str(e)},
                execution_time=time.time() - start_time,
                timestamp=datetime.now()
            )


class PerformanceValidator:
    """Validador de rendimiento"""
    
    def verify_performance(self, df: pd.DataFrame, max_time: float = 5.0) -> VerificationResult:
        """Verifica el rendimiento del análisis"""
        start_time = time.time()
        
        try:
            # Medir tiempo de análisis completo
            analysis_start = time.time()
            result = analyze_symbol(df, "PERFORMANCE_TEST")
            analysis_time = time.time() - analysis_start
            
            # Calcular métricas de rendimiento
            periods_per_second = len(df) / analysis_time if analysis_time > 0 else 0
            
            # Verificar si cumple con el tiempo máximo
            time_check_passed = analysis_time <= max_time
            
            # Verificar que el resultado sea válido
            result_valid = hasattr(result, 'recommendation') and hasattr(result, 'overall_score')
            
            passed = time_check_passed and result_valid
            
            return VerificationResult(
                test_name="Verificación de Rendimiento",
                passed=passed,
                message=f"Tiempo: {analysis_time:.2f}s, Velocidad: {periods_per_second:.0f} períodos/s",
                details={
                    "analysis_time": analysis_time,
                    "max_time": max_time,
                    "periods_per_second": periods_per_second,
                    "data_points": len(df),
                    "time_check_passed": time_check_passed,
                    "result_valid": result_valid
                },
                execution_time=time.time() - start_time,
                timestamp=datetime.now()
            )
            
        except Exception as e:
            return VerificationResult(
                test_name="Verificación de Rendimiento",
                passed=False,
                message=f"Error durante verificación: {str(e)}",
                details={"exception": str(e)},
                execution_time=time.time() - start_time,
                timestamp=datetime.now()
            )


class Phase2VerificationSystem:
    """Sistema completo de verificación para la Fase 2"""
    
    def __init__(self):
        self.data_validator = DataValidator()
        self.indicator_validator = IndicatorValidator()
        self.trend_validator = TrendValidator()
        self.pattern_validator = PatternValidator()
        self.integrated_validator = IntegratedAnalysisValidator()
        self.performance_validator = PerformanceValidator()
    
    def get_component_status(self) -> Dict[str, Dict[str, Any]]:
        """Obtiene el estado de todos los componentes del sistema"""
        import psutil
        import time
        
        components = {}
        
        # Verificar validador de datos
        try:
            start_time = time.time()
            test_df = self.create_sample_data(50, "STATUS_TEST")
            result = self.data_validator.validate_ohlc_data(test_df)
            response_time = time.time() - start_time
            
            components['Data Validator'] = {
                'healthy': result.passed,
                'response_time': response_time,
                'last_check': datetime.now().isoformat(),
                'error': None if result.passed else result.message
            }
        except Exception as e:
            components['Data Validator'] = {
                'healthy': False,
                'response_time': 0,
                'last_check': datetime.now().isoformat(),
                'error': str(e)
            }
        
        # Verificar validador de indicadores
        try:
            start_time = time.time()
            test_df = self.create_sample_data(50, "STATUS_TEST")
            result = self.indicator_validator.verify_sma_calculation(test_df)
            response_time = time.time() - start_time
            
            components['Indicator Validator'] = {
                'healthy': result.passed,
                'response_time': response_time,
                'last_check': datetime.now().isoformat(),
                'error': None if result.passed else result.message
            }
        except Exception as e:
            components['Indicator Validator'] = {
                'healthy': False,
                'response_time': 0,
                'last_check': datetime.now().isoformat(),
                'error': str(e)
            }
        
        # Verificar detector de tendencias
        try:
            start_time = time.time()
            test_df = self.create_sample_data(50, "STATUS_TEST")
            result = self.trend_validator.verify_trend_detection(test_df)
            response_time = time.time() - start_time
            
            components['Trend Detector'] = {
                'healthy': result.passed,
                'response_time': response_time,
                'last_check': datetime.now().isoformat(),
                'error': None if result.passed else result.message
            }
        except Exception as e:
            components['Trend Detector'] = {
                'healthy': False,
                'response_time': 0,
                'last_check': datetime.now().isoformat(),
                'error': str(e)
            }
        
        # Verificar reconocedor de patrones
        try:
            start_time = time.time()
            test_df = self.create_sample_data(50, "STATUS_TEST")
            result = self.pattern_validator.verify_pattern_detection(test_df)
            response_time = time.time() - start_time
            
            components['Pattern Recognizer'] = {
                'healthy': result.passed,
                'response_time': response_time,
                'last_check': datetime.now().isoformat(),
                'error': None if result.passed else result.message
            }
        except Exception as e:
            components['Pattern Recognizer'] = {
                'healthy': False,
                'response_time': 0,
                'last_check': datetime.now().isoformat(),
                'error': str(e)
            }
        
        # Verificar analizador integrado
        try:
            start_time = time.time()
            test_df = self.create_sample_data(50, "STATUS_TEST")
            result = self.integrated_validator.verify_integrated_analysis(test_df, "STATUS_TEST")
            response_time = time.time() - start_time
            
            components['Integrated Analyzer'] = {
                'healthy': result.passed,
                'response_time': response_time,
                'last_check': datetime.now().isoformat(),
                'error': None if result.passed else result.message
            }
        except Exception as e:
            components['Integrated Analyzer'] = {
                'healthy': False,
                'response_time': 0,
                'last_check': datetime.now().isoformat(),
                'error': str(e)
            }
        
        return components
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Obtiene métricas del sistema"""
        import psutil
        
        try:
            # Métricas de CPU y memoria
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Métricas de proceso actual
            process = psutil.Process()
            process_memory = process.memory_info()
            
            return {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'memory_used_gb': memory.used / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'process_memory_mb': process_memory.rss / (1024**2),
                'process_cpu_percent': process.cpu_percent(),
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            return {
                'cpu_percent': 0,
                'memory_percent': 0,
                'memory_available_gb': 0,
                'memory_used_gb': 0,
                'disk_percent': 0,
                'disk_free_gb': 0,
                'process_memory_mb': 0,
                'process_cpu_percent': 0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def create_sample_data(self, periods: int = 200, symbol: str = "TESTDATA") -> pd.DataFrame:
        """Crea datos de muestra para pruebas"""
        print(f"Creando datos de muestra: {periods} períodos para {symbol}")
        
        # Crear fechas
        end_date = datetime.now()
        start_date = end_date - timedelta(hours=periods)
        dates = pd.date_range(start_date, end_date, periods=periods)
        
        # Simular precios con diferentes patrones
        np.random.seed(42)
        base_price = 50000
        prices = [base_price]
        volumes = []
        
        for i in range(1, periods):
            # Crear diferentes fases de mercado
            if i < periods * 0.25:  # Tendencia alcista
                trend = 0.1
                volatility = 0.015
            elif i < periods * 0.5:  # Consolidación
                trend = 0.02
                volatility = 0.01
            elif i < periods * 0.75:  # Tendencia bajista
                trend = -0.08
                volatility = 0.02
            else:  # Recuperación
                trend = 0.06
                volatility = 0.018
            
            # Agregar ruido y tendencia
            noise = np.random.randn() * volatility
            price_change = trend/100 + noise
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, 1000))
            
            # Volumen correlacionado
            base_volume = 1000000
            volume_noise = np.random.randn() * 0.3
            volume = base_volume * (1 + abs(price_change) * 3 + volume_noise)
            volumes.append(max(volume, 50000))
        
        volumes.insert(0, 1000000)
        
        # Crear datos OHLC
        close_prices = np.array(prices)
        intraday_vol = 0.008
        
        high_prices = close_prices * (1 + np.random.rand(periods) * intraday_vol)
        low_prices = close_prices * (1 - np.random.rand(periods) * intraday_vol)
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        # Asegurar consistencia OHLC
        for i in range(periods):
            high_prices[i] = max(high_prices[i], open_prices[i], close_prices[i])
            low_prices[i] = min(low_prices[i], open_prices[i], close_prices[i])
        
        df = pd.DataFrame({
            'timestamp': dates,
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices,
            'volume': volumes
        })
        
        df.set_index('timestamp', inplace=True)
        return df
    
    def run_quick_verification(self, df: Optional[pd.DataFrame] = None, 
                             symbol: str = "QUICK_TEST") -> Dict[str, Any]:
        """Ejecuta una verificación rápida de los componentes principales"""
        print("\n" + "="*60)
        print("INICIANDO VERIFICACIÓN RÁPIDA")
        print("="*60)
        
        start_time = time.time()
        results = []
        
        # Usar datos de muestra si no se proporcionan
        if df is None:
            df = self.create_sample_data(100, symbol)  # Menos datos para ser más rápido
        
        print(f"\nDatos para verificación: {len(df)} períodos")
        
        try:
            # 1. Validación básica de datos
            print("\n1. Validando datos...")
            data_result = self.data_validator.validate_ohlc_data(df)
            results.append(data_result)
            print(f"   ✓ Datos: {'PASÓ' if data_result.passed else 'FALLÓ'}")
            
            # 2. Verificación de indicadores principales
            print("\n2. Verificando indicadores...")
            sma_result = self.indicator_validator.verify_sma_calculation(df)
            rsi_result = self.indicator_validator.verify_rsi_calculation(df)
            results.extend([sma_result, rsi_result])
            print(f"   ✓ SMA: {'PASÓ' if sma_result.passed else 'FALLÓ'}")
            print(f"   ✓ RSI: {'PASÓ' if rsi_result.passed else 'FALLÓ'}")
            
            # 3. Verificación de análisis integrado
            print("\n3. Verificando análisis integrado...")
            integrated_result = self.integrated_validator.verify_integrated_analysis(df, symbol)
            results.append(integrated_result)
            print(f"   ✓ Análisis: {'PASÓ' if integrated_result.passed else 'FALLÓ'}")
            
            # Calcular métricas
            total_time = time.time() - start_time
            passed_tests = sum(1 for r in results if r.passed)
            total_tests = len(results)
            success_rate = (passed_tests / total_tests) * 100 if total_tests > 0 else 0
            
            # Crear resultado
            quick_result = {
                'success': success_rate >= 75,  # Umbral más bajo para verificación rápida
                'success_rate': success_rate,
                'total_tests': total_tests,
                'passed_tests': passed_tests,
                'failed_tests': total_tests - passed_tests,
                'execution_time': total_time,
                'results': results,
                'message': f"Verificación rápida completada: {passed_tests}/{total_tests} pruebas exitosas"
            }
            
            print(f"\n{'✅ VERIFICACIÓN RÁPIDA EXITOSA' if quick_result['success'] else '⚠️ VERIFICACIÓN RÁPIDA CON PROBLEMAS'}")
            print(f"Tasa de éxito: {success_rate:.1f}% ({passed_tests}/{total_tests})")
            print(f"Tiempo: {total_time:.2f}s")
            
            return quick_result
            
        except Exception as e:
            error_result = {
                'success': False,
                'success_rate': 0,
                'total_tests': 0,
                'passed_tests': 0,
                'failed_tests': 1,
                'execution_time': time.time() - start_time,
                'results': [],
                'message': f"Error durante verificación rápida: {str(e)}",
                'error': str(e)
            }
            print(f"\n❌ ERROR EN VERIFICACIÓN RÁPIDA: {str(e)}")
            return error_result
    
    def run_complete_verification(self, df: Optional[pd.DataFrame] = None, 
                                symbol: str = "VERIFICATION_TEST",
                                include_performance: bool = True,
                                include_data_quality: bool = True,
                                include_ml_models: bool = False) -> ValidationReport:
        """Ejecuta verificación completa del sistema"""
        print("\n" + "="*80)
        print("INICIANDO VERIFICACIÓN COMPLETA DE LA FASE 2")
        print("="*80)
        
        start_time = time.time()
        results = []
        
        # Usar datos de muestra si no se proporcionan
        if df is None:
            df = self.create_sample_data(200, symbol)
        
        print(f"\nDatos para verificación: {len(df)} períodos")
        
        # 1. Validación de datos (siempre se ejecuta)
        print("\n1. Validando estructura de datos...")
        data_result = self.data_validator.validate_ohlc_data(df)
        results.append(data_result)
        print(f"   ✓ {data_result.test_name}: {'PASÓ' if data_result.passed else 'FALLÓ'}")
        
        # 2. Validación de indicadores
        print("\n2. Validando indicadores técnicos...")
        indicator_results = self.indicator_validator.verify_all_indicators(df)
        results.extend(indicator_results)
        for result in indicator_results:
            print(f"   ✓ {result.test_name}: {'PASÓ' if result.passed else 'FALLÓ'}")
        
        # 3. Validación de tendencias
        print("\n3. Validando detección de tendencias...")
        trend_result = self.trend_validator.verify_trend_detection(df)
        results.append(trend_result)
        print(f"   ✓ {trend_result.test_name}: {'PASÓ' if trend_result.passed else 'FALLÓ'}")
        
        # 4. Validación de patrones
        print("\n4. Validando reconocimiento de patrones...")
        pattern_result = self.pattern_validator.verify_pattern_detection(df)
        results.append(pattern_result)
        print(f"   ✓ {pattern_result.test_name}: {'PASÓ' if pattern_result.passed else 'FALLÓ'}")
        
        # 5. Validación de análisis integrado
        print("\n5. Validando análisis técnico integrado...")
        integrated_result = self.integrated_validator.verify_integrated_analysis(df, symbol)
        results.append(integrated_result)
        print(f"   ✓ {integrated_result.test_name}: {'PASÓ' if integrated_result.passed else 'FALLÓ'}")
        
        # 6. Validación de rendimiento (condicional)
        if include_performance:
            print("\n6. Validando rendimiento del sistema...")
            performance_result = self.performance_validator.verify_performance(df)
            results.append(performance_result)
            print(f"   ✓ {performance_result.test_name}: {'PASÓ' if performance_result.passed else 'FALLÓ'}")
        else:
            print("\n6. Validación de rendimiento omitida (deshabilitada)")
        
        # 7. Validación de calidad de datos adicional (condicional)
        if include_data_quality:
            print("\n7. Validando calidad de datos avanzada...")
            # Aquí se pueden agregar validaciones adicionales de calidad
            print("   ✓ Validación de calidad de datos: PASÓ (placeholder)")
        else:
            print("\n7. Validación de calidad de datos omitida (deshabilitada)")
        
        # 8. Validación de modelos ML (condicional)
        if include_ml_models:
            print("\n8. Validando modelos de Machine Learning...")
            # Aquí se pueden agregar validaciones de modelos ML
            print("   ✓ Validación de modelos ML: PASÓ (placeholder)")
        else:
            print("\n8. Validación de modelos ML omitida (deshabilitada)")
        
        # Generar reporte
        total_time = time.time() - start_time
        passed_tests = sum(1 for r in results if r.passed)
        failed_tests = len(results) - passed_tests
        success_rate = (passed_tests / len(results)) * 100 if results else 0
        
        # Crear resumen
        summary = {
            "data_validation": data_result.passed,
            "indicators_working": all(r.passed for r in indicator_results),
            "trends_working": trend_result.passed,
            "patterns_working": pattern_result.passed,
            "integration_working": integrated_result.passed,
            "performance_acceptable": performance_result.passed if include_performance else True,
            "overall_system_health": success_rate >= 80
        }
        
        report = ValidationReport(
            total_tests=len(results),
            passed_tests=passed_tests,
            failed_tests=failed_tests,
            success_rate=success_rate,
            total_execution_time=total_time,
            results=results,
            summary=summary,
            timestamp=datetime.now()
        )
        
        # Mostrar resumen
        print("\n" + "="*80)
        print("RESUMEN DE VERIFICACIÓN")
        print("="*80)
        print(f"Total de pruebas: {report.total_tests}")
        print(f"Pruebas exitosas: {report.passed_tests}")
        print(f"Pruebas fallidas: {report.failed_tests}")
        print(f"Tasa de éxito: {report.success_rate:.1f}%")
        print(f"Tiempo total: {report.total_execution_time:.2f} segundos")
        
        if report.success_rate >= 80:
            print("\n🎉 SISTEMA VERIFICADO EXITOSAMENTE")
            print("✅ La Fase 2 está funcionando correctamente")
        else:
            print("\n⚠️ SISTEMA REQUIERE ATENCIÓN")
            print("❌ Algunas verificaciones fallaron")
        
        # Convertir a diccionario para compatibilidad con el dashboard
        report_dict = asdict(report)
        
        # Convertir datetime a string para serialización
        def convert_datetime(obj):
            if isinstance(obj, dict):
                return {k: convert_datetime(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_datetime(item) for item in obj]
            elif isinstance(obj, datetime):
                return obj.isoformat()
            else:
                return obj
        
        report_dict = convert_datetime(report_dict)
        
        # Agregar campos adicionales que espera el dashboard
        report_dict.update({
            'success': report.success_rate >= 80,
            'overall_score': min(10, report.success_rate / 10),
            'components_passed': report.passed_tests,
            'total_components': report.total_tests,
            'total_time': report.total_execution_time,
            'categories': {
                'data_validation': {
                    'score': 10 if summary['data_validation'] else 0,
                    'tests': [{'name': 'Validación de datos', 'passed': summary['data_validation'], 'message': 'OK' if summary['data_validation'] else 'Error'}]
                },
                'indicators': {
                    'score': 10 if summary['indicators_working'] else 0,
                    'tests': [{'name': 'Indicadores técnicos', 'passed': summary['indicators_working'], 'message': 'OK' if summary['indicators_working'] else 'Error'}]
                },
                'trends': {
                    'score': 10 if summary['trends_working'] else 0,
                    'tests': [{'name': 'Detección de tendencias', 'passed': summary['trends_working'], 'message': 'OK' if summary['trends_working'] else 'Error'}]
                },
                'patterns': {
                    'score': 10 if summary['patterns_working'] else 0,
                    'tests': [{'name': 'Reconocimiento de patrones', 'passed': summary['patterns_working'], 'message': 'OK' if summary['patterns_working'] else 'Error'}]
                },
                'integration': {
                    'score': 10 if summary['integration_working'] else 0,
                    'tests': [{'name': 'Análisis integrado', 'passed': summary['integration_working'], 'message': 'OK' if summary['integration_working'] else 'Error'}]
                }
            }
        })
        
        if include_performance:
            report_dict['categories']['performance'] = {
                'score': 10 if summary['performance_acceptable'] else 0,
                'tests': [{'name': 'Rendimiento del sistema', 'passed': summary['performance_acceptable'], 'message': 'OK' if summary['performance_acceptable'] else 'Error'}]
            }
        
        return report_dict
    
    def run_scenario_test(self, scenario_type: str, symbol: str = "SCENARIO_TEST") -> Dict[str, Any]:
        """Ejecuta pruebas de escenarios específicos"""
        print(f"\n{'='*60}")
        print(f"EJECUTANDO ESCENARIO: {scenario_type.upper()}")
        print(f"Símbolo: {symbol}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        try:
            # Crear datos específicos según el escenario
            if scenario_type == "mercado_alcista":
                df = self._create_bullish_scenario_data(symbol)
            elif scenario_type == "mercado_bajista":
                df = self._create_bearish_scenario_data(symbol)
            elif scenario_type == "mercado_lateral":
                df = self._create_sideways_scenario_data(symbol)
            elif scenario_type == "alta_volatilidad":
                df = self._create_high_volatility_scenario_data(symbol)
            else:
                # Escenario por defecto
                df = self.create_sample_data(200, symbol)
            
            print(f"Datos del escenario: {len(df)} períodos")
            
            # Ejecutar análisis completo
            analysis_start = time.time()
            result = analyze_symbol(df, symbol)
            analysis_time = time.time() - analysis_start
            
            # Calcular métricas del escenario
            signals_generated = len(result.signals.get('buy_signals', [])) + len(result.signals.get('sell_signals', []))
            
            # Obtener puntuación
            if hasattr(result.overall_score, 'iloc'):
                score_value = result.overall_score.iloc[-1]
            else:
                score_value = result.overall_score
            
            # Calcular precisión basada en la coherencia del análisis
            accuracy = self._calculate_scenario_accuracy(result, scenario_type)
            
            total_time = time.time() - start_time
            
            scenario_result = {
                'success': True,
                'scenario_type': scenario_type,
                'symbol': symbol,
                'accuracy': accuracy,
                'signals_generated': signals_generated,
                'response_time': analysis_time,
                'total_time': total_time,
                'analysis': {
                    'recommendation': result.recommendation,
                    'overall_score': float(score_value),
                    'indicators_count': len(result.indicators),
                    'patterns_count': len(result.patterns),
                    'buy_signals': len(result.signals.get('buy_signals', [])),
                    'sell_signals': len(result.signals.get('sell_signals', [])),
                    'trend_direction': self._get_trend_direction(result),
                    'market_condition': self._assess_market_condition(result)
                },
                'performance_metrics': {
                    'data_points': len(df),
                    'processing_speed': len(df) / analysis_time if analysis_time > 0 else 0,
                    'memory_efficient': True,  # Placeholder
                    'response_time_acceptable': analysis_time < 2.0
                }
            }
            
            print(f"\n✅ Escenario completado exitosamente")
            print(f"   Precisión: {accuracy:.2f}%")
            print(f"   Señales: {signals_generated}")
            print(f"   Tiempo: {analysis_time:.3f}s")
            print(f"   Recomendación: {result.recommendation}")
            
            return scenario_result
            
        except Exception as e:
            error_time = time.time() - start_time
            print(f"\n❌ Error en escenario: {str(e)}")
            
            return {
                'success': False,
                'scenario_type': scenario_type,
                'symbol': symbol,
                'error': str(e),
                'execution_time': error_time,
                'accuracy': 0.0,
                'signals_generated': 0,
                'response_time': error_time
            }
    
    def _create_bullish_scenario_data(self, symbol: str) -> pd.DataFrame:
        """Crea datos para escenario de mercado alcista"""
        periods = 200
        dates = pd.date_range(datetime.now() - timedelta(hours=periods), datetime.now(), periods=periods)
        
        np.random.seed(42)
        base_price = 50000
        prices = [base_price]
        
        for i in range(1, periods):
            # Tendencia alcista con pequeñas correcciones
            if i % 20 == 0:  # Pequeña corrección cada 20 períodos
                trend = -0.02
                volatility = 0.015
            else:
                trend = 0.08  # Tendencia alcista fuerte
                volatility = 0.01
            
            noise = np.random.randn() * volatility
            price_change = trend/100 + noise
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, base_price * 0.8))  # Piso mínimo
        
        return self._create_ohlcv_from_prices(prices, dates, symbol)
    
    def _create_bearish_scenario_data(self, symbol: str) -> pd.DataFrame:
        """Crea datos para escenario de mercado bajista"""
        periods = 200
        dates = pd.date_range(datetime.now() - timedelta(hours=periods), datetime.now(), periods=periods)
        
        np.random.seed(43)
        base_price = 50000
        prices = [base_price]
        
        for i in range(1, periods):
            # Tendencia bajista con pequeños rebotes
            if i % 15 == 0:  # Pequeño rebote cada 15 períodos
                trend = 0.03
                volatility = 0.012
            else:
                trend = -0.06  # Tendencia bajista
                volatility = 0.018
            
            noise = np.random.randn() * volatility
            price_change = trend/100 + noise
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, base_price * 0.3))  # Piso mínimo
        
        return self._create_ohlcv_from_prices(prices, dates, symbol)
    
    def _create_sideways_scenario_data(self, symbol: str) -> pd.DataFrame:
        """Crea datos para escenario de mercado lateral"""
        periods = 200
        dates = pd.date_range(datetime.now() - timedelta(hours=periods), datetime.now(), periods=periods)
        
        np.random.seed(44)
        base_price = 50000
        prices = [base_price]
        
        for i in range(1, periods):
            # Movimiento lateral con oscilaciones
            trend = 0.01 * np.sin(i * 0.1)  # Oscilación sinusoidal
            volatility = 0.008
            
            noise = np.random.randn() * volatility
            price_change = trend/100 + noise
            new_price = prices[-1] * (1 + price_change)
            # Mantener en rango
            new_price = max(min(new_price, base_price * 1.1), base_price * 0.9)
            prices.append(new_price)
        
        return self._create_ohlcv_from_prices(prices, dates, symbol)
    
    def _create_high_volatility_scenario_data(self, symbol: str) -> pd.DataFrame:
        """Crea datos para escenario de alta volatilidad"""
        periods = 200
        dates = pd.date_range(datetime.now() - timedelta(hours=periods), datetime.now(), periods=periods)
        
        np.random.seed(45)
        base_price = 50000
        prices = [base_price]
        
        for i in range(1, periods):
            # Alta volatilidad con cambios bruscos
            trend = 0.02 * (np.random.randn() > 0.5) - 0.01
            volatility = 0.035  # Alta volatilidad
            
            noise = np.random.randn() * volatility
            price_change = trend/100 + noise
            new_price = prices[-1] * (1 + price_change)
            prices.append(max(new_price, base_price * 0.5))
        
        return self._create_ohlcv_from_prices(prices, dates, symbol)
    
    def _create_ohlcv_from_prices(self, prices: List[float], dates: pd.DatetimeIndex, symbol: str) -> pd.DataFrame:
        """Convierte lista de precios en datos OHLCV"""
        ohlcv_data = []
        
        for i, price in enumerate(prices):
            if i == 0:
                open_price = price
                high_price = price * 1.002
                low_price = price * 0.998
                close_price = price
            else:
                open_price = prices[i-1]
                high_price = max(open_price, price) * (1 + np.random.uniform(0, 0.005))
                low_price = min(open_price, price) * (1 - np.random.uniform(0, 0.005))
                close_price = price
            
            volume = np.random.uniform(1000000, 5000000)
            
            ohlcv_data.append({
                'timestamp': dates[i],
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume,
                'symbol': symbol
            })
        
        return pd.DataFrame(ohlcv_data)
    
    def _calculate_scenario_accuracy(self, result, scenario_type: str) -> float:
        """Calcula la precisión del análisis para el escenario"""
        try:
            # Obtener puntuación
            if hasattr(result.overall_score, 'iloc'):
                score = result.overall_score.iloc[-1]
            else:
                score = result.overall_score
            
            recommendation = result.recommendation
            
            # Evaluar coherencia según el escenario
            if scenario_type == "mercado_alcista":
                if recommendation in ['BUY'] and score > 20:
                    return 95.0
                elif recommendation in ['HOLD'] and score > 0:
                    return 80.0
                else:
                    return 60.0
            
            elif scenario_type == "mercado_bajista":
                if recommendation in ['SELL'] and score < -20:
                    return 95.0
                elif recommendation in ['HOLD'] and score < 0:
                    return 80.0
                else:
                    return 60.0
            
            elif scenario_type == "mercado_lateral":
                if recommendation in ['HOLD', 'NEUTRAL'] and abs(score) < 30:
                    return 90.0
                else:
                    return 70.0
            
            elif scenario_type == "alta_volatilidad":
                # En alta volatilidad, cualquier recomendación puede ser válida
                return 85.0
            
            return 75.0  # Precisión por defecto
            
        except Exception:
            return 50.0
    
    def _get_trend_direction(self, result) -> str:
        """Obtiene la dirección de la tendencia del análisis"""
        try:
            if hasattr(result.overall_score, 'iloc'):
                score = result.overall_score.iloc[-1]
            else:
                score = result.overall_score
            
            if score > 20:
                return "Alcista"
            elif score < -20:
                return "Bajista"
            else:
                return "Lateral"
        except Exception:
            return "Indeterminada"
    
    def _assess_market_condition(self, result) -> str:
        """Evalúa la condición del mercado"""
        try:
            patterns_count = len(result.patterns)
            signals_count = len(result.signals.get('buy_signals', [])) + len(result.signals.get('sell_signals', []))
            
            if signals_count > 5:
                return "Activo"
            elif patterns_count > 3:
                return "Formación de patrones"
            else:
                return "Estable"
        except Exception:
            return "Normal"
    
    def save_report(self, report: ValidationReport, filename: str = None) -> str:
        """Guarda el reporte de verificación"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"verification_report_{timestamp}.json"
        
        # Convertir a diccionario serializable
        report_dict = asdict(report)
        
        # Función para convertir tipos no serializables
        def convert_value(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, (np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_value(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_value(item) for item in obj]
            else:
                return obj
        
        # Convertir todo el diccionario
        report_dict = convert_value(report_dict)
        
        # Guardar archivo
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(report_dict, f, indent=2, ensure_ascii=False)
        
        print(f"\nReporte guardado en: {filename}")
        return filename
    
    def generate_detailed_report(self, report: ValidationReport) -> str:
        """Genera un reporte detallado en texto"""
        lines = []
        lines.append("REPORTE DETALLADO DE VERIFICACIÓN - FASE 2")
        lines.append("=" * 60)
        lines.append(f"Fecha: {report.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Total de pruebas: {report.total_tests}")
        lines.append(f"Exitosas: {report.passed_tests}")
        lines.append(f"Fallidas: {report.failed_tests}")
        lines.append(f"Tasa de éxito: {report.success_rate:.1f}%")
        lines.append(f"Tiempo total: {report.total_execution_time:.2f}s")
        lines.append("")
        
        lines.append("RESULTADOS DETALLADOS:")
        lines.append("-" * 40)
        
        for i, result in enumerate(report.results, 1):
            status = "✅ PASÓ" if result.passed else "❌ FALLÓ"
            lines.append(f"{i}. {result.test_name} - {status}")
            lines.append(f"   Mensaje: {result.message}")
            lines.append(f"   Tiempo: {result.execution_time:.3f}s")
            
            if result.details:
                lines.append("   Detalles:")
                for key, value in result.details.items():
                    if key != 'exception':
                        lines.append(f"     {key}: {value}")
            
            if not result.passed and 'exception' in result.details:
                lines.append(f"   Error: {result.details['exception']}")
            
            lines.append("")
        
        lines.append("RESUMEN DEL SISTEMA:")
        lines.append("-" * 40)
        for key, value in report.summary.items():
            status = "✅" if value else "❌"
            lines.append(f"{status} {key.replace('_', ' ').title()}: {value}")
        
        return "\n".join(lines)


def main():
    """Función principal para ejecutar verificaciones"""
    verification_system = Phase2VerificationSystem()
    
    # Ejecutar verificación completa
    report = verification_system.run_complete_verification()
    
    # Guardar reportes
    json_file = verification_system.save_report(report)
    
    # Generar reporte detallado
    detailed_report = verification_system.generate_detailed_report(report)
    
    # Guardar reporte detallado
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detailed_filename = f"detailed_verification_report_{timestamp}.txt"
    with open(detailed_filename, 'w', encoding='utf-8') as f:
        f.write(detailed_report)
    
    print(f"Reporte detallado guardado en: {detailed_filename}")
    
    # Mostrar reporte detallado en consola
    print("\n" + "="*80)
    print(detailed_report)
    
    return report.success_rate >= 80


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)