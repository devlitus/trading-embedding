#!/usr/bin/env python3
"""
Demo de Estrategia Híbrida de Datos
Demuestra el uso optimizado de base de datos y CSV según el caso de uso
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
import time

# Agregar src al path
sys.path.append(str(Path(__file__).parent / "src"))

# Importar utilidades comunes
from utils.demo_utils import (
    setup_project_path, setup_logging, print_banner, print_section,
    generate_sample_ohlc_data, format_execution_time, create_demo_summary
)

from src.data.data_access_layer import DataAccessLayer
from src.data.data_strategy import DataStrategy, DataUsagePattern



def print_subsection(title: str):
    """Imprime una subsección con formato."""
    print(f"\n--- {title} ---")

def demo_data_access_layer():
    """Demuestra la capa de acceso a datos híbrida."""
    print_section("DEMO: Data Access Layer - Acceso Híbrido")
    
    # Inicializar capa de acceso
    dal = DataAccessLayer()
    
    print_subsection("1. Resumen del Estado Actual")
    summary = dal.get_data_summary()
    
    print("📊 Base de Datos:")
    for key, info in summary['database'].items():
        print(f"  • {key}: {info['records']} registros ({info['date_range']})")
    
    print("\n📁 Archivos CSV:")
    for key, info in summary['csv_files'].items():
        if isinstance(info, dict):
            print(f"  • {key}: {info['records']} registros, {info['file_size']}")
        else:
            print(f"  • {key}: {info}")
    
    print("\n🔄 Estado de Sincronización:")
    for key, status in summary['sync_status'].items():
        status_icon = "✅" if status == "Sincronizado" else "⚠️"
        print(f"  {status_icon} {key}: {status}")
    
    print("\n💡 Recomendaciones:")
    for rec in summary['recommendations']:
        print(f"  • {rec}")
    
    print_subsection("2. Sincronización de Datos")
    print("Sincronizando fuentes de datos...")
    sync_results = dal.sync_data_sources()
    
    for key, result in sync_results.items():
        result_icon = "✅" if "Sincronizado" in result else "ℹ️"
        print(f"  {result_icon} {key}: {result}")
    
    return dal

def demo_realtime_trading(strategy: DataStrategy):
    """Demuestra estrategia para trading en tiempo real."""
    print_section("DEMO: Trading en Tiempo Real - Máxima Velocidad")
    
    symbol = "BTCUSDT"
    interval = "1h"
    
    print(f"🚀 Obteniendo datos para trading en tiempo real: {symbol} {interval}")
    print("   Prioridad: Cache → Base de Datos → CSV")
    
    start_time = time.time()
    data = strategy.get_data(
        pattern=DataUsagePattern.REALTIME_TRADING,
        symbol=symbol,
        interval=interval,
        limit=100
    )
    end_time = time.time()
    
    if not data.empty:
        print(f"\n📈 Datos obtenidos en {(end_time - start_time)*1000:.1f}ms")
        print(f"   • Registros: {len(data)}")
        print(f"   • Último precio: ${data['close'].iloc[-1]:.2f}")
        print(f"   • Cambio de precio: {data['price_change'].iloc[-1]*100:.2f}%")
        print(f"   • Volumen promedio: {data['volume_ma'].iloc[-1]:,.0f}")
        
        # Mostrar últimos 3 registros
        print("\n📊 Últimos 3 registros:")
        recent_data = data[['datetime', 'close', 'volume', 'price_change']].tail(3)
        for _, row in recent_data.iterrows():
            change_icon = "📈" if row['price_change'] > 0 else "📉" if row['price_change'] < 0 else "➡️"
            print(f"   {change_icon} {row['datetime']}: ${row['close']:.2f} (Vol: {row['volume']:,.0f})")
    else:
        print("❌ No se encontraron datos")

def demo_ml_training(strategy: DataStrategy):
    """Demuestra estrategia para entrenamiento de ML."""
    print_section("DEMO: Machine Learning - Datos Completos y Features")
    
    symbol = "BTCUSDT"
    interval = "1h"
    
    print(f"🤖 Obteniendo datos para entrenamiento ML: {symbol} {interval}")
    print("   Prioridad: CSV → Base de Datos")
    print("   Incluye: Preprocesamiento + Feature Engineering")
    
    start_time = time.time()
    data = strategy.get_data(
        pattern=DataUsagePattern.ML_TRAINING,
        symbol=symbol,
        interval=interval,
        lookback_days=30,
        include_features=True
    )
    end_time = time.time()
    
    if not data.empty:
        print(f"\n🧠 Datos preparados en {(end_time - start_time)*1000:.1f}ms")
        print(f"   • Registros: {len(data)}")
        print(f"   • Features: {len(data.columns)}")
        print(f"   • Rango de fechas: {data['datetime'].min()} a {data['datetime'].max()}")
        
        # Mostrar features disponibles
        print("\n🔧 Features disponibles:")
        feature_cols = [col for col in data.columns if col not in ['datetime', 'timestamp']]
        for i, col in enumerate(feature_cols[:10]):  # Mostrar primeras 10
            print(f"   • {col}")
        if len(feature_cols) > 10:
            print(f"   ... y {len(feature_cols) - 10} más")
        
        # Estadísticas básicas
        print("\n📊 Estadísticas de returns:")
        if 'returns' in data.columns:
            returns = data['returns'].dropna()
            print(f"   • Media: {returns.mean()*100:.4f}%")
            print(f"   • Volatilidad: {returns.std()*100:.4f}%")
            print(f"   • Sharpe ratio: {returns.mean()/returns.std():.4f}")
        
        # Verificar calidad de datos
        if hasattr(data, 'attrs') and 'data_quality' in data.attrs:
            quality = data.attrs['data_quality']
            print(f"\n✅ Calidad de datos: {quality['completeness']:.1f}% completo")
    else:
        print("❌ No se encontraron datos")

def demo_backtesting(strategy: DataStrategy):
    """Demuestra estrategia para backtesting."""
    print_section("DEMO: Backtesting - Datos Históricos Completos")
    
    symbol = "BTCUSDT"
    interval = "1h"
    
    print(f"📈 Obteniendo datos para backtesting: {symbol} {interval}")
    print("   Prioridad: CSV → Base de Datos")
    print("   Garantiza: Continuidad temporal + Datos completos")
    
    # Definir período de backtesting
    end_date = datetime.now()
    start_date = end_date - timedelta(days=15)
    
    start_time = time.time()
    data = strategy.get_data(
        pattern=DataUsagePattern.BACKTESTING,
        symbol=symbol,
        interval=interval,
        start_date=start_date,
        end_date=end_date
    )
    end_time = time.time()
    
    if not data.empty:
        print(f"\n⏱️ Datos cargados en {(end_time - start_time)*1000:.1f}ms")
        print(f"   • Registros: {len(data)}")
        print(f"   • Período: {data['datetime'].min()} a {data['datetime'].max()}")
        
        # Verificar continuidad
        if len(data) > 1:
            time_diffs = data['datetime'].diff().dropna()
            median_diff = time_diffs.median()
            gaps = time_diffs[time_diffs > median_diff * 2]
            
            print(f"   • Intervalo típico: {median_diff}")
            print(f"   • Gaps detectados: {len(gaps)}")
        
        # Simular estrategia simple
        if 'forward_return' in data.columns:
            print("\n🎯 Simulación de estrategia simple (Buy & Hold):")
            total_return = (data['close'].iloc[-1] / data['close'].iloc[0]) - 1
            print(f"   • Retorno total: {total_return*100:.2f}%")
            
            # Calcular métricas básicas
            if 'returns' in data.columns:
                returns = data['returns'].dropna()
                winning_trades = (returns > 0).sum()
                total_trades = len(returns)
                win_rate = winning_trades / total_trades if total_trades > 0 else 0
                
                print(f"   • Win rate: {win_rate*100:.1f}%")
                print(f"   • Mejor día: {returns.max()*100:.2f}%")
                print(f"   • Peor día: {returns.min()*100:.2f}%")
    else:
        print("❌ No se encontraron datos para el período especificado")

def demo_api_serving(strategy: DataStrategy):
    """Demuestra estrategia para servir datos vía API."""
    print_section("DEMO: API Serving - Respuestas Rápidas y Formateadas")
    
    symbol = "ETHUSDT"
    interval = "1h"
    
    print(f"🌐 Obteniendo datos para API: {symbol} {interval}")
    print("   Prioridad: Cache → Base de Datos")
    print("   Formato: JSON-ready con precisión optimizada")
    
    start_time = time.time()
    data = strategy.get_data(
        pattern=DataUsagePattern.API_SERVING,
        symbol=symbol,
        interval=interval,
        limit=50,
        format_for_json=True
    )
    end_time = time.time()
    
    if not data.empty:
        print(f"\n⚡ Respuesta API en {(end_time - start_time)*1000:.1f}ms")
        print(f"   • Registros: {len(data)}")
        print(f"   • Tamaño estimado: {len(data.to_json())//1024:.1f} KB")
        
        # Mostrar formato de respuesta
        print("\n📋 Formato de respuesta (primeros 2 registros):")
        sample_data = data.head(2)
        for _, row in sample_data.iterrows():
            print(f"   {{")
            print(f"     'datetime': '{row['datetime']}',")
            print(f"     'open': {row['open']},")
            print(f"     'high': {row['high']},")
            print(f"     'low': {row['low']},")
            print(f"     'close': {row['close']},")
            print(f"     'volume': {row['volume']}")
            print(f"   }}")
            break  # Solo mostrar uno
        print("   ...")
    else:
        print("❌ No se encontraron datos")

def demo_dashboard(strategy: DataStrategy):
    """Demuestra estrategia para dashboards."""
    print_section("DEMO: Dashboard - Visualización con Agregaciones")
    
    symbol = "BTCUSDT"
    interval = "1h"
    
    print(f"📊 Obteniendo datos para dashboard: {symbol} {interval}")
    print("   Incluye: SMA, Bollinger Bands, Volumen promedio")
    
    start_time = time.time()
    data = strategy.get_data(
        pattern=DataUsagePattern.DASHBOARD,
        symbol=symbol,
        interval=interval,
        include_aggregations=True
    )
    end_time = time.time()
    
    if not data.empty:
        print(f"\n📈 Datos para dashboard en {(end_time - start_time)*1000:.1f}ms")
        print(f"   • Registros: {len(data)}")
        
        # Mostrar indicadores calculados
        latest = data.iloc[-1]
        print("\n📊 Indicadores actuales:")
        print(f"   • Precio: ${latest['close']:.2f}")
        
        if 'sma_20' in data.columns and pd.notna(latest['sma_20']):
            print(f"   • SMA 20: ${latest['sma_20']:.2f}")
            trend = "📈 Alcista" if latest['close'] > latest['sma_20'] else "📉 Bajista"
            print(f"   • Tendencia: {trend}")
        
        if 'bb_upper' in data.columns and pd.notna(latest['bb_upper']):
            bb_position = (latest['close'] - latest['bb_lower']) / (latest['bb_upper'] - latest['bb_lower'])
            print(f"   • Posición en BB: {bb_position*100:.1f}%")
        
        if 'volume_sma' in data.columns and pd.notna(latest['volume_sma']):
            vol_ratio = latest['volume'] / latest['volume_sma']
            vol_status = "🔥 Alto" if vol_ratio > 1.5 else "📊 Normal" if vol_ratio > 0.5 else "💤 Bajo"
            print(f"   • Volumen: {vol_status} ({vol_ratio:.1f}x promedio)")
        
        # Contar indicadores disponibles
        indicators = [col for col in data.columns if col.startswith(('sma_', 'bb_', 'volume_'))]
        print(f"\n🔧 Indicadores disponibles: {len(indicators)}")
        for indicator in indicators[:5]:  # Mostrar primeros 5
            print(f"   • {indicator}")
    else:
        print("❌ No se encontraron datos")

def demo_performance_comparison():
    """Compara rendimiento entre diferentes estrategias."""
    print_section("DEMO: Comparación de Rendimiento")
    
    dal = DataAccessLayer()
    strategy = DataStrategy(dal)
    
    symbol = "BTCUSDT"
    interval = "1h"
    limit = 100
    
    patterns_to_test = [
        DataUsagePattern.REALTIME_TRADING,
        DataUsagePattern.ML_TRAINING,
        DataUsagePattern.API_SERVING
    ]
    
    print(f"⚡ Comparando rendimiento para {symbol} {interval} (límite: {limit})")
    print("\n🏁 Resultados:")
    
    results = []
    for pattern in patterns_to_test:
        start_time = time.time()
        data = strategy.get_data(
            pattern=pattern,
            symbol=symbol,
            interval=interval,
            limit=limit
        )
        end_time = time.time()
        
        duration_ms = (end_time - start_time) * 1000
        results.append({
            'pattern': pattern.value,
            'duration_ms': duration_ms,
            'records': len(data) if not data.empty else 0,
            'columns': len(data.columns) if not data.empty else 0
        })
        
        print(f"   • {pattern.value:20} {duration_ms:6.1f}ms  {len(data):3d} registros  {len(data.columns):2d} columnas")
    
    # Encontrar el más rápido
    fastest = min(results, key=lambda x: x['duration_ms'])
    print(f"\n🏆 Más rápido: {fastest['pattern']} ({fastest['duration_ms']:.1f}ms)")

def main():
    """Función principal del demo."""
    print_banner("DEMO: Estrategia Híbrida de Datos para Trading")
    print("   Optimización inteligente según el caso de uso")
    
    # Configurar logging
    logger = setup_logging('logs/hybrid_strategy_demo.log')
    logger.info("Iniciando demostración de estrategia híbrida")
    
    start_time = time.time()
    
    try:
        # 1. Inicializar sistema
        dal = demo_data_access_layer()
        strategy = DataStrategy(dal)
        
        # 2. Mostrar información de estrategias
        print_section("ESTRATEGIAS DISPONIBLES")
        strategies_info = strategy.get_strategy_info()
        for pattern_name, info in strategies_info.items():
            print(f"\n🎯 {pattern_name.upper()}:")
            print(f"   • {info['description']}")
            print(f"   • Prioridad: {' → '.join(info['data_source_priority'])}")
            print(f"   • Optimizaciones: {', '.join(info['optimizations'])}")
        
        # 3. Demos específicos
        demo_realtime_trading(strategy)
        demo_ml_training(strategy)
        demo_backtesting(strategy)
        demo_api_serving(strategy)
        demo_dashboard(strategy)
        
        # 4. Comparación de rendimiento
        demo_performance_comparison()
        
        # Resumen final
        execution_time = time.time() - start_time
        
        results = {
            "Tiempo de ejecución": format_execution_time(execution_time),
            "Componentes demostrados": "7",
            "Estrategias evaluadas": "5",
            "Estado": "✅ Completado exitosamente"
        }
        
        print(create_demo_summary(results))
        
        print_section("RESUMEN Y RECOMENDACIONES")
        print("✅ Demo completado exitosamente")
        print("\n💡 Recomendaciones de uso:")
        print("   • Trading en tiempo real: Usa REALTIME_TRADING para mínima latencia")
        print("   • Entrenamiento ML: Usa ML_TRAINING para features completas")
        print("   • Backtesting: Usa BACKTESTING para datos históricos completos")
        print("   • APIs: Usa API_SERVING para respuestas optimizadas")
        print("   • Dashboards: Usa DASHBOARD para visualizaciones enriquecidas")
        
        print("\n🔄 Mantenimiento:")
        print("   • Ejecuta sync_data_sources() regularmente")
        print("   • Monitorea el rendimiento de cache")
        print("   • Revisa recomendaciones del sistema")
        
        logger.info(f"Demostración completada en {format_execution_time(execution_time)}")
        
    except Exception as e:
        print(f"\n❌ Error durante el demo: {str(e)}")
        print("\n🔧 Posibles soluciones:")
        print("   • Verifica que la base de datos existe (trading.db)")
        print("   • Asegúrate de que hay datos históricos disponibles")
        print("   • Ejecuta primero: python src/data/data_manager.py")
        raise

if __name__ == "__main__":
    main()