"""Módulo Integrador de Análisis Técnico

Combina indicadores técnicos, detección de tendencias y reconocimiento de patrones
para proporcionar un análisis técnico completo de los datos de mercado.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime

from technical_indicators import TechnicalIndicators
from trend_detection import TrendDetector
from pattern_recognition import PatternRecognizer, PatternResult


@dataclass
class TechnicalAnalysisResult:
    """Resultado completo del análisis técnico"""
    timestamp: datetime
    symbol: str
    
    # Indicadores técnicos
    indicators: Dict[str, Any]
    
    # Análisis de tendencia
    trend_analysis: Dict[str, Any]
    
    # Patrones detectados
    patterns: List[PatternResult]
    
    # Señales de trading
    signals: Dict[str, Any]
    
    # Puntuación general
    overall_score: float
    
    # Recomendación
    recommendation: str


class TechnicalAnalyzer:
    """Analizador técnico integrado para la Fase 2"""
    
    def __init__(self, 
                 indicator_periods: Optional[Dict[str, int]] = None,
                 trend_sensitivity: float = 0.02,
                 pattern_confidence_threshold: float = 0.6):
        """
        Args:
            indicator_periods: Períodos personalizados para indicadores
            trend_sensitivity: Sensibilidad para detección de tendencias
            pattern_confidence_threshold: Umbral de confianza para patrones
        """
        # Configurar períodos por defecto
        default_periods = {
            'sma_short': 10,
            'sma_long': 20,
            'ema_short': 12,
            'ema_long': 26,
            'rsi_period': 14,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'bb_period': 20,
            'bb_std': 2
        }
        
        self.periods = {**default_periods, **(indicator_periods or {})}
        
        # Inicializar componentes
        self.indicators = TechnicalIndicators()
        self.trend_detector = TrendDetector(significance_threshold=trend_sensitivity)
        self.pattern_recognizer = PatternRecognizer(
            confidence_threshold=pattern_confidence_threshold
        )
    
    def analyze(self, df: pd.DataFrame, symbol: str = "UNKNOWN") -> TechnicalAnalysisResult:
        """
        Realiza análisis técnico completo
        
        Args:
            df: DataFrame con datos OHLC
            symbol: Símbolo del activo
            
        Returns:
            Resultado completo del análisis técnico
        """
        if len(df) < 50:
            raise ValueError("Se necesitan al menos 50 períodos para análisis técnico")
        
        # 1. Calcular indicadores técnicos
        indicators_data = self._calculate_all_indicators(df)
        
        # 2. Análisis de tendencia
        trend_analysis = self._analyze_trends(df)
        
        # 3. Detectar patrones
        patterns = self.pattern_recognizer.detect_all_patterns(df)
        
        # 4. Generar señales de trading
        signals = self._generate_trading_signals(df, indicators_data, trend_analysis, patterns)
        
        # 5. Calcular puntuación general
        overall_score = self._calculate_overall_score(indicators_data, trend_analysis, patterns, signals)
        
        # 6. Generar recomendación
        recommendation = self._generate_recommendation(overall_score, signals)
        
        return TechnicalAnalysisResult(
            timestamp=df.index[-1] if hasattr(df.index, 'to_pydatetime') else datetime.now(),
            symbol=symbol,
            indicators=indicators_data,
            trend_analysis=trend_analysis,
            patterns=patterns,
            signals=signals,
            overall_score=overall_score,
            recommendation=recommendation
        )
    
    def _calculate_all_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calcula todos los indicadores técnicos"""
        # Use the existing calculate_all_indicators method
        df_with_indicators = self.indicators.calculate_all_indicators(df)
        
        # Extract indicators as dictionary
        indicators = {}
        for col in df_with_indicators.columns:
            if col not in ['open', 'high', 'low', 'close', 'volume']:
                indicators[col] = df_with_indicators[col]
        
        return indicators
    
    def _analyze_trends(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analiza las tendencias del mercado"""
        trend_analysis = {}
        
        # Detección de cambios de tendencia
        trend_changes = self.trend_detector.detect_trend_changes(df)
        trend_analysis['trend_changes'] = trend_changes
        
        # Análisis ZigZag usando método zigzag
        zigzag_data = self.trend_detector.detect_trend_changes(df, method='zigzag')
        trend_analysis['zigzag'] = zigzag_data
        
        # Tendencia por regresión lineal
        linear_trend = self.trend_detector.detect_trend_changes(df, method='regression')
        trend_analysis['linear_trend'] = linear_trend
        
        # Niveles de soporte y resistencia
        support_resistance = self.trend_detector.detect_support_resistance_levels(df)
        trend_analysis['support_resistance'] = support_resistance
        
        # Puntuación de fuerza de tendencia
        trend_strength = self.trend_detector.calculate_trend_strength_score(df)
        trend_analysis['strength_score'] = trend_strength
        
        return trend_analysis
    
    def _generate_trading_signals(self, df: pd.DataFrame, indicators: Dict, 
                                trend_analysis: Dict, patterns: List[PatternResult]) -> Dict[str, Any]:
        """Genera señales de trading basadas en el análisis"""
        signals = {
            'buy_signals': [],
            'sell_signals': [],
            'neutral_signals': [],
            'strength': 0.0
        }
        
        current_price = df['close'].iloc[-1]
        
        # Señales basadas en indicadores
        signal_strength = 0
        total_signals = 0
        
        # 1. Señales de cruce de medias móviles
        if 'sma_20' in indicators and 'sma_50' in indicators and len(indicators['sma_20']) > 1 and len(indicators['sma_50']) > 1:
            sma_short_current = indicators['sma_20'].iloc[-1]
            sma_long_current = indicators['sma_50'].iloc[-1]
            sma_short_prev = indicators['sma_20'].iloc[-2]
            sma_long_prev = indicators['sma_50'].iloc[-2]
            
            # Golden Cross
            if sma_short_prev <= sma_long_prev and sma_short_current > sma_long_current:
                signals['buy_signals'].append({
                    'type': 'golden_cross',
                    'description': 'Cruce alcista de medias móviles',
                    'strength': 0.7
                })
                signal_strength += 0.7
            
            # Death Cross
            elif sma_short_prev >= sma_long_prev and sma_short_current < sma_long_current:
                signals['sell_signals'].append({
                    'type': 'death_cross',
                    'description': 'Cruce bajista de medias móviles',
                    'strength': 0.7
                })
                signal_strength -= 0.7
            
            total_signals += 1
        
        # 2. Señales de RSI
        if 'rsi' in indicators and len(indicators['rsi']) > 0:
            rsi_current = indicators['rsi'].iloc[-1]
            
            if rsi_current < 30:
                signals['buy_signals'].append({
                    'type': 'rsi_oversold',
                    'description': f'RSI en sobreventa ({rsi_current:.1f})',
                    'strength': 0.6
                })
                signal_strength += 0.6
            elif rsi_current > 70:
                signals['sell_signals'].append({
                    'type': 'rsi_overbought',
                    'description': f'RSI en sobrecompra ({rsi_current:.1f})',
                    'strength': 0.6
                })
                signal_strength -= 0.6
            
            total_signals += 1
        
        # 3. Señales de MACD
        if 'macd' in indicators and 'macd_signal' in indicators:
            macd_current = indicators['macd'].iloc[-1]
            signal_current = indicators['macd_signal'].iloc[-1]
            
            if len(indicators['macd']) > 1:
                macd_prev = indicators['macd'].iloc[-2]
                signal_prev = indicators['macd_signal'].iloc[-2]
                
                # Cruce alcista MACD
                if macd_prev <= signal_prev and macd_current > signal_current:
                    signals['buy_signals'].append({
                        'type': 'macd_bullish_cross',
                        'description': 'Cruce alcista MACD',
                        'strength': 0.6
                    })
                    signal_strength += 0.6
                
                # Cruce bajista MACD
                elif macd_prev >= signal_prev and macd_current < signal_current:
                    signals['sell_signals'].append({
                        'type': 'macd_bearish_cross',
                        'description': 'Cruce bajista MACD',
                        'strength': 0.6
                    })
                    signal_strength -= 0.6
            
            total_signals += 1
        
        # 4. Señales de Bandas de Bollinger
        if 'bb_upper' in indicators and 'bb_lower' in indicators:
            bb_upper = indicators['bb_upper'].iloc[-1]
            bb_lower = indicators['bb_lower'].iloc[-1]
            
            if current_price <= bb_lower:
                signals['buy_signals'].append({
                    'type': 'bb_oversold',
                    'description': 'Precio en banda inferior de Bollinger',
                    'strength': 0.5
                })
                signal_strength += 0.5
            elif current_price >= bb_upper:
                signals['sell_signals'].append({
                    'type': 'bb_overbought',
                    'description': 'Precio en banda superior de Bollinger',
                    'strength': 0.5
                })
                signal_strength -= 0.5
            
            total_signals += 1
        
        # 5. Señales basadas en patrones
        pattern_signal_strength = 0
        for pattern in patterns:
            if pattern.confidence > 0.7:
                if 'bullish' in pattern.pattern_type or 'hammer' in pattern.pattern_type or 'ascending' in pattern.pattern_type:
                    signals['buy_signals'].append({
                        'type': f'pattern_{pattern.pattern_type}',
                        'description': pattern.description,
                        'strength': pattern.confidence * 0.8
                    })
                    pattern_signal_strength += pattern.confidence * 0.8
                elif 'bearish' in pattern.pattern_type or 'shooting' in pattern.pattern_type or 'descending' in pattern.pattern_type:
                    signals['sell_signals'].append({
                        'type': f'pattern_{pattern.pattern_type}',
                        'description': pattern.description,
                        'strength': pattern.confidence * 0.8
                    })
                    pattern_signal_strength -= pattern.confidence * 0.8
        
        # 6. Señales basadas en tendencia
        if 'strength_score' in trend_analysis:
            trend_strength_series = trend_analysis['strength_score']
            trend_strength = trend_strength_series.iloc[-1] if len(trend_strength_series) > 0 else 0
            if trend_strength > 70:  # Adjusted for 0-100 scale
                signals['buy_signals'].append({
                    'type': 'strong_uptrend',
                    'description': f'Tendencia alcista fuerte ({trend_strength:.2f})',
                    'strength': 0.6
                })
                signal_strength += 0.6
            elif trend_strength < 30:  # Adjusted for 0-100 scale
                signals['sell_signals'].append({
                    'type': 'strong_downtrend',
                    'description': f'Tendencia bajista fuerte ({abs(trend_strength):.2f})',
                    'strength': 0.6
                })
                signal_strength -= 0.6
        
        # Calcular fuerza total de señales
        total_strength = signal_strength + pattern_signal_strength
        if total_signals > 0:
            signals['strength'] = total_strength / max(total_signals, 1)
        else:
            signals['strength'] = 0
        
        return signals
    
    def _calculate_overall_score(self, indicators: Dict, trend_analysis: Dict, 
                               patterns: List[PatternResult], signals: Dict) -> float:
        """Calcula una puntuación general del análisis"""
        score = 0.0
        components = 0
        
        # Puntuación basada en indicadores (30%)
        indicator_score = 0
        if 'rsi' in indicators and len(indicators['rsi']) > 0:
            rsi = indicators['rsi'].iloc[-1]
            if 30 <= rsi <= 70:
                indicator_score += 0.5  # RSI neutral es bueno
            elif rsi < 20 or rsi > 80:
                indicator_score += 0.2  # Extremos
            else:
                indicator_score += 0.3
        
        if 'trend_strength' in indicators:
            trend_strength = abs(indicators['trend_strength'])
            indicator_score += min(trend_strength, 1.0) * 0.5
        
        score += indicator_score * 0.3
        components += 1
        
        # Puntuación basada en tendencia (40%)
        if 'strength_score' in trend_analysis:
            trend_score = abs(trend_analysis['strength_score'])
            score += trend_score * 0.4
            components += 1
        
        # Puntuación basada en patrones (20%)
        if patterns:
            pattern_score = sum(p.confidence for p in patterns[:3]) / min(len(patterns), 3)
            score += pattern_score * 0.2
            components += 1
        
        # Puntuación basada en señales (10%)
        signal_score = abs(signals['strength'])
        score += signal_score * 0.1
        components += 1
        
        return score / max(components, 1)
    
    def _generate_recommendation(self, overall_score: float, signals: Dict) -> str:
        """Genera una recomendación basada en el análisis"""
        signal_strength = signals['strength']
        buy_signals = len(signals['buy_signals'])
        sell_signals = len(signals['sell_signals'])
        
        # Handle Series format for overall_score - ensure it's always a float
        if hasattr(overall_score, 'iloc'):
            score_val = float(overall_score.iloc[-1])
        elif hasattr(overall_score, '__iter__') and not isinstance(overall_score, str):
            # Handle array-like objects
            score_val = float(overall_score[-1]) if len(overall_score) > 0 else 0.0
        else:
            score_val = float(overall_score)
        
        if signal_strength > 0.5 and buy_signals > sell_signals:
            if score_val > 70:  # Adjusted for 0-100 scale
                return "COMPRA FUERTE"
            else:
                return "COMPRA"
        elif signal_strength < -0.5 and sell_signals > buy_signals:
            if score_val > 70:  # Adjusted for 0-100 scale
                return "VENTA FUERTE"
            else:
                return "VENTA"
        elif abs(signal_strength) < 0.2:
            return "MANTENER"
        else:
            return "NEUTRAL"
    
    def get_analysis_summary(self, result: TechnicalAnalysisResult) -> Dict[str, Any]:
        """Obtiene un resumen del análisis técnico"""
        return {
            'symbol': result.symbol,
            'timestamp': result.timestamp,
            'recommendation': result.recommendation,
            'overall_score': result.overall_score,
            'signal_strength': result.signals['strength'],
            'buy_signals_count': len(result.signals['buy_signals']),
            'sell_signals_count': len(result.signals['sell_signals']),
            'patterns_detected': len(result.patterns),
            'trend_strength': result.trend_analysis.get('strength_score', 0),
            'market_regime': result.indicators.get('market_regime', 'Unknown'),
            'key_levels': {
                'support': result.indicators.get('support_level'),
                'resistance': result.indicators.get('resistance_level')
            }
        }


def analyze_symbol(df: pd.DataFrame, symbol: str = "UNKNOWN", 
                  custom_config: Optional[Dict] = None) -> TechnicalAnalysisResult:
    """
    Función de conveniencia para análisis técnico completo
    
    Args:
        df: DataFrame con datos OHLC
        symbol: Símbolo del activo
        custom_config: Configuración personalizada
        
    Returns:
        Resultado completo del análisis técnico
    """
    config = custom_config or {}
    analyzer = TechnicalAnalyzer(**config)
    return analyzer.analyze(df, symbol)


if __name__ == "__main__":
    # Ejemplo de uso
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    
    # Crear datos de ejemplo
    dates = pd.date_range('2023-01-01', periods=100, freq='1H')
    np.random.seed(42)
    
    # Simular datos OHLC realistas
    base_price = 100
    prices = [base_price]
    volumes = []
    
    for i in range(1, 100):
        # Simular movimiento de precio con tendencia y ruido
        trend = 0.1 if i < 50 else -0.05
        noise = np.random.randn() * 2
        price = prices[-1] * (1 + trend/100 + noise/100)
        prices.append(max(price, 1))  # Evitar precios negativos
        volumes.append(np.random.randint(1000, 10000))
    
    volumes.insert(0, 5000)  # Volumen inicial
    
    # Crear OHLC realista
    close_prices = np.array(prices)
    high_prices = close_prices * (1 + np.random.rand(100) * 0.02)
    low_prices = close_prices * (1 - np.random.rand(100) * 0.02)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    
    sample_df = pd.DataFrame({
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }, index=dates)
    
    # Realizar análisis técnico completo
    analyzer = TechnicalAnalyzer()
    result = analyzer.analyze(sample_df, "BTCUSDT")
    
    # Mostrar resultados
    print(f"Análisis Técnico para {result.symbol}")
    print(f"Timestamp: {result.timestamp}")
    print(f"Recomendación: {result.recommendation}")
    # Handle Series format for overall_score in print statement
    score_val = result.overall_score.iloc[-1] if hasattr(result.overall_score, 'iloc') else float(result.overall_score)
    print(f"Puntuación General: {score_val:.2f}")
    print(f"Fuerza de Señales: {result.signals['strength']:.2f}")
    print(f"\nSeñales de Compra: {len(result.signals['buy_signals'])}")
    for signal in result.signals['buy_signals'][:3]:
        print(f"  - {signal['description']} (Fuerza: {signal['strength']:.2f})")
    
    print(f"\nSeñales de Venta: {len(result.signals['sell_signals'])}")
    for signal in result.signals['sell_signals'][:3]:
        print(f"  - {signal['description']} (Fuerza: {signal['strength']:.2f})")
    
    print(f"\nPatrones Detectados: {len(result.patterns)}")
    for pattern in result.patterns[:3]:
        print(f"  - {pattern.description} (Confianza: {pattern.confidence:.2f})")
    
    # Resumen
    summary = analyzer.get_analysis_summary(result)
    print(f"\nResumen del Análisis:")
    for key, value in summary.items():
        if key != 'key_levels':
            print(f"  {key}: {value}")
        else:
            print(f"  Niveles Clave: Soporte={value['support']:.2f}, Resistencia={value['resistance']:.2f}")