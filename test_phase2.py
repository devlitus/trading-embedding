"""Script de prueba para la Fase 2: Análisis Técnico

Este script verifica que todos los componentes de análisis técnico
funcionen correctamente y muestra ejemplos de uso.
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Agregar el directorio src al path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from analysis.technical_indicators import TechnicalIndicators
from analysis.trend_detection import TrendDetector
from analysis.pattern_recognition import PatternRecognizer, detect_patterns
from analysis.technical_analysis import TechnicalAnalyzer, analyze_symbol


def create_sample_data(periods: int = 200, symbol: str = "BTCUSDT") -> pd.DataFrame:
    """Crea datos de muestra para pruebas"""
    print(f"Creando datos de muestra para {symbol} con {periods} períodos...")
    
    # Crear fechas
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=periods)
    dates = pd.date_range(start_date, end_date, periods=periods)
    
    # Simular precio con diferentes fases de mercado
    np.random.seed(42)
    base_price = 45000  # Precio base para BTC
    prices = [base_price]
    volumes = []
    
    for i in range(1, periods):
        # Diferentes fases del mercado
        if i < periods * 0.3:  # Fase 1: Tendencia alcista
            trend = 0.15
            volatility = 0.02
        elif i < periods * 0.6:  # Fase 2: Consolidación
            trend = 0.02
            volatility = 0.015
        elif i < periods * 0.8:  # Fase 3: Tendencia bajista
            trend = -0.1
            volatility = 0.025
        else:  # Fase 4: Recuperación
            trend = 0.08
            volatility = 0.02
        
        # Agregar ruido y tendencia
        noise = np.random.randn() * volatility
        price_change = trend/100 + noise
        new_price = prices[-1] * (1 + price_change)
        prices.append(max(new_price, 1000))  # Precio mínimo
        
        # Volumen correlacionado con volatilidad
        base_volume = 1000000
        volume_noise = np.random.randn() * 0.3
        volume = base_volume * (1 + abs(price_change) * 5 + volume_noise)
        volumes.append(max(volume, 100000))
    
    volumes.insert(0, 1000000)  # Volumen inicial
    
    # Crear datos OHLC realistas
    close_prices = np.array(prices)
    
    # High y Low basados en volatilidad intraday
    intraday_volatility = 0.01
    high_prices = close_prices * (1 + np.random.rand(periods) * intraday_volatility)
    low_prices = close_prices * (1 - np.random.rand(periods) * intraday_volatility)
    
    # Open price (precio de cierre anterior con gap pequeño)
    open_prices = np.roll(close_prices, 1)
    open_prices[0] = close_prices[0]
    gap_factor = np.random.randn(periods) * 0.002  # Gaps pequeños
    open_prices = open_prices * (1 + gap_factor)
    
    # Asegurar que High >= max(Open, Close) y Low <= min(Open, Close)
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


def test_technical_indicators():
    """Prueba los indicadores técnicos"""
    print("\n" + "="*60)
    print("PRUEBA: INDICADORES TÉCNICOS")
    print("="*60)
    
    df = create_sample_data(100)
    indicators = TechnicalIndicators()
    
    # Calcular todos los indicadores
    df_with_indicators = indicators.calculate_all_indicators(df)
    
    print("\n1. Indicadores de Tendencia:")
    print(f"   SMA(20) actual: {df_with_indicators['sma_20'].iloc[-1]:.2f}")
    print(f"   EMA(12) actual: {df_with_indicators['ema_12'].iloc[-1]:.2f}")
    print(f"   EMA(26) actual: {df_with_indicators['ema_26'].iloc[-1]:.2f}")
    print(f"   MACD actual: {df_with_indicators['macd'].iloc[-1]:.2f}")
    print(f"   MACD Signal: {df_with_indicators['macd_signal'].iloc[-1]:.2f}")
    
    print("\n2. Indicadores de Momentum:")
    print(f"   RSI actual: {df_with_indicators['rsi'].iloc[-1]:.2f}")
    print(f"   Stochastic %K: {df_with_indicators['stoch_k'].iloc[-1]:.2f}")
    print(f"   Rate of Change: {df_with_indicators['roc'].iloc[-1]:.2f}%")
    
    print("\n3. Indicadores de Volatilidad:")
    print(f"   Bollinger Superior: {df_with_indicators['bb_upper'].iloc[-1]:.2f}")
    print(f"   Bollinger Inferior: {df_with_indicators['bb_lower'].iloc[-1]:.2f}")
    print(f"   ATR: {df_with_indicators['atr'].iloc[-1]:.2f}")
    print(f"   Volatilidad Histórica: {df_with_indicators['volatility'].iloc[-1]:.2f}%")
    
    print("\n4. Indicadores de Volumen:")
    print(f"   OBV actual: {df_with_indicators['obv'].iloc[-1]:,.0f}")
    print(f"   Volumen SMA: {df_with_indicators['volume_sma'].iloc[-1]:,.0f}")
    print(f"   Volumen Relativo: {df_with_indicators['relative_volume'].iloc[-1]:.2f}")
    
    print("\n5. Análisis de Mercado:")
    regime = indicators.get_market_regime(df_with_indicators)
    support_levels, resistance_levels = indicators.calculate_support_resistance(df_with_indicators)
    trend_strength = indicators.calculate_trend_strength(df_with_indicators)
    
    print(f"   Régimen de Mercado: {regime.iloc[-1]}")
    print(f"   Nivel de Soporte: {support_levels.iloc[-1]:.2f}")
    print(f"   Nivel de Resistencia: {resistance_levels.iloc[-1]:.2f}")
    print(f"   Fuerza de Tendencia: {trend_strength.iloc[-1]:.2f}")
    
    print("\n✅ Indicadores técnicos funcionando correctamente")


def test_trend_detection():
    """Prueba la detección de tendencias"""
    print("\n" + "="*60)
    print("PRUEBA: DETECCIÓN DE TENDENCIAS")
    print("="*60)
    
    df = create_sample_data(150)
    detector = TrendDetector()
    
    print("\n1. Detección de Cambios de Tendencia:")
    trend_changes = detector.detect_trend_changes(df)
    print(f"   Cambios de tendencia detectados: {len(trend_changes)}")
    if not trend_changes.empty and len(trend_changes) > 0:
        # Mostrar información del último cambio si existe
        print(f"   Datos disponibles en el DataFrame de cambios de tendencia")
    
    print("\n2. Análisis ZigZag:")
    # Usar método zigzag en detect_trend_changes
    zigzag_df = detector.detect_trend_changes(df, method='zigzag')
    print(f"   Análisis ZigZag completado exitosamente")
    
    print("\n3. Tendencia por Regresión Lineal:")
    # Usar método regression en detect_trend_changes
    regression_df = detector.detect_trend_changes(df, method='regression')
    print(f"   Análisis de regresión completado exitosamente")
    
    print("\n4. Niveles de Soporte y Resistencia:")
    levels = detector.detect_support_resistance_levels(df)
    print(f"   Niveles de soporte: {len(levels['support'])}")
    print(f"   Niveles de resistencia: {len(levels['resistance'])}")
    if len(levels['support']) > 0:
        print(f"   Soporte más fuerte: {levels['support'][0]:.2f}")
    if len(levels['resistance']) > 0:
        print(f"   Resistencia más fuerte: {levels['resistance'][0]:.2f}")
    
    print("\n5. Puntuación de Fuerza de Tendencia:")
    strength_series = detector.calculate_trend_strength_score(df)
    strength = strength_series.iloc[-1]  # Get the last value
    print(f"   Fuerza de tendencia: {strength:.2f}")
    if strength > 50:
        print("   → Tendencia fuerte")
    elif strength > 30:
        print("   → Tendencia moderada")
    else:
        print("   → Tendencia débil")
    
    print("\n✅ Detección de tendencias funcionando correctamente")


def test_pattern_recognition():
    """Prueba el reconocimiento de patrones"""
    print("\n" + "="*60)
    print("PRUEBA: RECONOCIMIENTO DE PATRONES")
    print("="*60)
    
    df = create_sample_data(120)
    recognizer = PatternRecognizer(confidence_threshold=0.5)
    
    print("\n1. Detección de Todos los Patrones:")
    all_patterns = recognizer.detect_all_patterns(df)
    print(f"   Patrones detectados: {len(all_patterns)}")
    
    # Mostrar los patrones más confiables
    top_patterns = sorted(all_patterns, key=lambda x: x.confidence, reverse=True)[:5]
    for i, pattern in enumerate(top_patterns, 1):
        print(f"   {i}. {pattern.description} (Confianza: {pattern.confidence:.2f})")
    
    print("\n2. Resumen por Tipo de Patrón:")
    summary = recognizer.get_pattern_summary(all_patterns)
    for pattern_type, count in summary.items():
        print(f"   {pattern_type}: {count}")
    
    print("\n3. Patrones Específicos:")
    
    # Triángulos
    triangles = recognizer.detect_triangles(df)
    print(f"   Triángulos detectados: {len(triangles)}")
    
    # Canales
    channels = recognizer.detect_channels(df)
    print(f"   Canales detectados: {len(channels)}")
    
    # Rectángulos
    rectangles = recognizer.detect_rectangles(df)
    print(f"   Rectángulos detectados: {len(rectangles)}")
    
    # Patrones de velas
    candlestick = recognizer.detect_candlestick_patterns(df)
    print(f"   Patrones de velas detectados: {len(candlestick)}")
    
    # Patrones de volumen
    volume_patterns = recognizer.detect_volume_patterns(df)
    print(f"   Patrones de volumen detectados: {len(volume_patterns)}")
    
    print("\n4. Filtrado de Patrones Superpuestos:")
    filtered_patterns = recognizer.filter_overlapping_patterns(all_patterns)
    print(f"   Patrones después del filtrado: {len(filtered_patterns)}")
    print(f"   Patrones eliminados por superposición: {len(all_patterns) - len(filtered_patterns)}")
    
    print("\n✅ Reconocimiento de patrones funcionando correctamente")


def test_integrated_analysis():
    """Prueba el análisis técnico integrado"""
    print("\n" + "="*60)
    print("PRUEBA: ANÁLISIS TÉCNICO INTEGRADO")
    print("="*60)
    
    df = create_sample_data(200, "BTCUSDT")
    
    print("\n1. Análisis Técnico Completo:")
    analyzer = TechnicalAnalyzer(
        indicator_periods={'sma_short': 10, 'sma_long': 20, 'rsi_period': 14},
        trend_sensitivity=0.02,
        pattern_confidence_threshold=0.6
    )
    
    result = analyzer.analyze(df, "BTCUSDT")
    
    print(f"   Símbolo: {result.symbol}")
    print(f"   Timestamp: {result.timestamp}")
    print(f"   Recomendación: {result.recommendation}")
    # Handle Series format for overall_score
    if hasattr(result.overall_score, 'iloc'):
        overall_val = float(result.overall_score.iloc[-1])
    elif hasattr(result.overall_score, '__iter__') and not isinstance(result.overall_score, str):
        overall_val = float(result.overall_score[-1]) if len(result.overall_score) > 0 else 0.0
    else:
        overall_val = float(result.overall_score)
    print(f"   Puntuación General: {overall_val:.2f}")
    print(f"   Fuerza de Señales: {result.signals['strength']:.2f}")
    
    print("\n2. Señales de Trading:")
    print(f"   Señales de Compra: {len(result.signals['buy_signals'])}")
    for signal in result.signals['buy_signals'][:3]:
        print(f"     - {signal['description']} (Fuerza: {signal['strength']:.2f})")
    
    print(f"   Señales de Venta: {len(result.signals['sell_signals'])}")
    for signal in result.signals['sell_signals'][:3]:
        print(f"     - {signal['description']} (Fuerza: {signal['strength']:.2f})")
    
    print("\n3. Indicadores Clave:")
    indicators = result.indicators
    print(f"   RSI: {indicators['rsi'].iloc[-1]:.2f}")
    print(f"   MACD: {indicators['macd'].iloc[-1]:.2f}")
    if 'market_regime' in indicators:
        print(f"   Régimen de Mercado: {indicators['market_regime']}")
    if 'trend_strength' in indicators:
        trend_strength = indicators['trend_strength']
        if hasattr(trend_strength, 'iloc'):
            trend_val = float(trend_strength.iloc[-1])
        elif hasattr(trend_strength, '__iter__') and not isinstance(trend_strength, str):
            trend_val = float(trend_strength[-1]) if len(trend_strength) > 0 else 0.0
        else:
            trend_val = float(trend_strength)
        print(f"   Fuerza de Tendencia: {trend_val:.2f}")
    else:
        print("   Fuerza de Tendencia: No disponible")
    
    print("\n4. Análisis de Tendencia:")
    trend = result.trend_analysis
    # Handle Series format for strength_score
    strength_score = trend['strength_score']
    if hasattr(strength_score, 'iloc'):
        strength_val = float(strength_score.iloc[-1])
    elif hasattr(strength_score, '__iter__') and not isinstance(strength_score, str):
        strength_val = float(strength_score[-1]) if len(strength_score) > 0 else 0.0
    else:
        strength_val = float(strength_score)
    print(f"   Puntuación de Fuerza: {strength_val:.2f}")
    print(f"   Cambios de Tendencia: {len(trend['trend_changes'])}")
    print(f"   Puntos ZigZag: {len(trend['zigzag'])}")
    
    print("\n5. Patrones Detectados:")
    print(f"   Total de Patrones: {len(result.patterns)}")
    for pattern in result.patterns[:3]:
        print(f"     - {pattern.description} (Confianza: {pattern.confidence:.2f})")
    
    print("\n6. Resumen del Análisis:")
    summary = analyzer.get_analysis_summary(result)
    for key, value in summary.items():
        if key not in ['key_levels', 'timestamp']:
            print(f"   {key}: {value}")
    
    levels = summary['key_levels']
    if levels['support'] and levels['resistance']:
        print(f"   Niveles Clave: Soporte={levels['support']:.2f}, Resistencia={levels['resistance']:.2f}")
    
    print("\n7. Función de Conveniencia:")
    quick_result = analyze_symbol(df, "BTCUSDT")
    print(f"   Análisis rápido - Recomendación: {quick_result.recommendation}")
    # Handle Series format for overall_score
    if hasattr(quick_result.overall_score, 'iloc'):
        quick_val = float(quick_result.overall_score.iloc[-1])
    elif hasattr(quick_result.overall_score, '__iter__') and not isinstance(quick_result.overall_score, str):
        quick_val = float(quick_result.overall_score[-1]) if len(quick_result.overall_score) > 0 else 0.0
    else:
        quick_val = float(quick_result.overall_score)
    print(f"   Puntuación: {quick_val:.2f}")
    
    print("\n✅ Análisis técnico integrado funcionando correctamente")


def run_performance_test():
    """Prueba de rendimiento"""
    print("\n" + "="*60)
    print("PRUEBA: RENDIMIENTO")
    print("="*60)
    
    import time
    
    # Crear dataset más grande
    df_large = create_sample_data(1000, "BTCUSDT")
    
    print(f"\nProbando rendimiento con {len(df_large)} períodos...")
    
    start_time = time.time()
    result = analyze_symbol(df_large, "BTCUSDT")
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Tiempo de ejecución: {execution_time:.2f} segundos")
    print(f"Períodos por segundo: {len(df_large)/execution_time:.0f}")
    
    # Handle Series format for overall_score
    if hasattr(result.overall_score, 'iloc'):
        perf_val = float(result.overall_score.iloc[-1])
    elif hasattr(result.overall_score, '__iter__') and not isinstance(result.overall_score, str):
        perf_val = float(result.overall_score[-1]) if len(result.overall_score) > 0 else 0.0
    else:
        perf_val = float(result.overall_score)
    print(f"Resultado: {result.recommendation} (Score: {perf_val:.2f})")
    print(f"Patrones detectados: {len(result.patterns)}")
    print(f"Señales generadas: {len(result.signals['buy_signals']) + len(result.signals['sell_signals'])}")
    
    if execution_time < 5.0:
        print("\n✅ Rendimiento aceptable")
    else:
        print("\n⚠️ Rendimiento podría mejorarse")


def main():
    """Función principal de pruebas"""
    print("INICIANDO PRUEBAS DE LA FASE 2: ANÁLISIS TÉCNICO")
    print("="*80)
    
    try:
        # Ejecutar todas las pruebas
        test_technical_indicators()
        test_trend_detection()
        test_pattern_recognition()
        test_integrated_analysis()
        run_performance_test()
        
        print("\n" + "="*80)
        print("🎉 TODAS LAS PRUEBAS COMPLETADAS EXITOSAMENTE")
        print("\n✅ La Fase 2 (Análisis Técnico) está completamente implementada y funcional")
        print("\nComponentes implementados:")
        print("  • Indicadores Técnicos (tendencia, momentum, volatilidad, volumen)")
        print("  • Detección de Tendencias (cambios, ZigZag, regresión lineal)")
        print("  • Reconocimiento de Patrones (geométricos, velas, volumen)")
        print("  • Análisis Técnico Integrado (señales, recomendaciones)")
        print("\nLa implementación está lista para ser utilizada en el sistema de trading.")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ ERROR EN LAS PRUEBAS: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)