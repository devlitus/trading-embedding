#!/usr/bin/env python3
"""
Demostración de Integración de la Estrategia Híbrida de Datos

Este script muestra cómo integrar la estrategia híbrida de datos
en diferentes componentes del sistema de trading.
"""

import sys
from pathlib import Path
import asyncio
import pandas as pd
from datetime import datetime, timedelta
import time

# Configurar rutas
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'src' / 'data'))
sys.path.insert(0, str(project_root / 'src' / 'analysis'))

from src.data.data_strategy import DataStrategy, DataUsagePattern
from src.data.data_access_layer import DataAccessLayer
from src.analysis.technical_indicators import TechnicalIndicators

class TradingSystemIntegration:
    """Integración de la estrategia híbrida en el sistema de trading"""
    
    def __init__(self):
        self.data_access = DataAccessLayer()
        self.data_strategy = DataStrategy(self.data_access)
        self.technical_indicators = TechnicalIndicators()
        
    def real_time_trading_pipeline(self, symbol: str = "BTCUSDT", interval: str = "1h"):
        """Pipeline completo para trading en tiempo real"""
        print(f"\n🚀 Pipeline de Trading en Tiempo Real - {symbol}")
        print("=" * 50)
        
        try:
            # 1. Obtener datos optimizados para tiempo real
            start_time = time.time()
            data = self.data_strategy.get_data(
                symbol=symbol,
                interval=interval,
                pattern=DataUsagePattern.REALTIME_TRADING,
                limit=100
            )
            data_time = time.time() - start_time
            
            print(f"✅ Datos obtenidos en {data_time:.3f}s")
            print(f"📊 Registros: {len(data)}")
            
            # 2. Calcular indicadores técnicos
            start_time = time.time()
            
            # RSI
            data['rsi'] = self.technical_indicators.rsi(data['close'])
            
            # MACD
            macd_data = self.technical_indicators.macd(data['close'])
            data['macd'] = macd_data['macd']
            data['macd_signal'] = macd_data['signal']
            data['macd_histogram'] = macd_data['histogram']
            
            # Bollinger Bands
            bb_data = self.technical_indicators.bollinger_bands(data['close'])
            data['bb_upper'] = bb_data['upper']
            data['bb_middle'] = bb_data['middle']
            data['bb_lower'] = bb_data['lower']
            
            indicators_time = time.time() - start_time
            print(f"📈 Indicadores calculados en {indicators_time:.3f}s")
            
            # 3. Generar señales de trading
            signals = self._generate_trading_signals(data)
            print(f"🎯 Señales generadas: {len(signals)} señales")
            
            # 4. Mostrar últimas señales
            if not signals.empty:
                print("\n📋 Últimas 3 señales:")
                for _, signal in signals.tail(3).iterrows():
                    print(f"  {signal['datetime']}: {signal['signal']} - {signal['reason']}")
            
            return {
                'data': data,
                'signals': signals,
                'performance': {
                    'data_retrieval_time': data_time,
                    'indicators_time': indicators_time,
                    'total_time': data_time + indicators_time
                }
            }
            
        except Exception as e:
            print(f"❌ Error en pipeline de tiempo real: {e}")
            return None
    
    def ml_training_pipeline(self, symbol: str = "BTCUSDT", interval: str = "1h"):
        """Pipeline para entrenamiento de modelos ML"""
        print(f"\n🤖 Pipeline de Entrenamiento ML - {symbol}")
        print("=" * 50)
        
        try:
            # 1. Obtener datos optimizados para ML
            start_time = time.time()
            data = self.data_strategy.get_data(
                symbol=symbol,
                interval=interval,
                pattern=DataUsagePattern.ML_TRAINING,
                limit=1000  # Más datos para ML
            )
            data_time = time.time() - start_time
            
            print(f"✅ Datos ML obtenidos en {data_time:.3f}s")
            print(f"📊 Registros: {len(data)}")
            
            # 2. Verificar características de los datos ML
            print("\n📈 Características de los datos ML:")
            print(f"  - Columnas: {list(data.columns)}")
            print(f"  - Rango de fechas: {data['datetime'].min()} a {data['datetime'].max()}")
            
            if 'returns' in data.columns:
                print(f"  - Returns promedio: {data['returns'].mean():.6f}")
                print(f"  - Volatilidad: {data['returns'].std():.6f}")
            
            if 'volume_sma' in data.columns:
                print(f"  - Volumen SMA promedio: {data['volume_sma'].mean():.2f}")
            
            # 3. Preparar features para ML
            features = self._prepare_ml_features(data)
            print(f"🔧 Features preparadas: {len(features.columns)} columnas")
            
            return {
                'data': data,
                'features': features,
                'performance': {
                    'data_retrieval_time': data_time
                }
            }
            
        except Exception as e:
            print(f"❌ Error en pipeline ML: {e}")
            return None
    
    def backtesting_pipeline(self, symbol: str = "BTCUSDT", interval: str = "4h"):
        """Pipeline para backtesting de estrategias"""
        print(f"\n📊 Pipeline de Backtesting - {symbol}")
        print("=" * 50)
        
        try:
            # 1. Obtener datos optimizados para backtesting
            start_time = time.time()
            data = self.data_strategy.get_data(
                symbol=symbol,
                interval=interval,
                pattern=DataUsagePattern.BACKTESTING,
                limit=500
            )
            data_time = time.time() - start_time
            
            print(f"✅ Datos backtesting obtenidos en {data_time:.3f}s")
            print(f"📊 Registros: {len(data)}")
            
            # 2. Ejecutar backtesting simple
            results = self._run_simple_backtest(data)
            
            print(f"\n📈 Resultados del Backtesting:")
            print(f"  - Total de operaciones: {results['total_trades']}")
            print(f"  - Operaciones ganadoras: {results['winning_trades']}")
            print(f"  - Tasa de éxito: {results['win_rate']:.2%}")
            print(f"  - Retorno total: {results['total_return']:.2%}")
            
            return {
                'data': data,
                'backtest_results': results,
                'performance': {
                    'data_retrieval_time': data_time
                }
            }
            
        except Exception as e:
            print(f"❌ Error en pipeline backtesting: {e}")
            return None
    
    def api_serving_pipeline(self, symbol: str = "BTCUSDT", interval: str = "1h"):
        """Pipeline para servir datos via API"""
        print(f"\n🌐 Pipeline de API Serving - {symbol}")
        print("=" * 50)
        
        try:
            # 1. Obtener datos optimizados para API
            start_time = time.time()
            data = self.data_strategy.get_data(
                symbol=symbol,
                interval=interval,
                pattern=DataUsagePattern.API_SERVING,
                limit=50
            )
            data_time = time.time() - start_time
            
            print(f"✅ Datos API obtenidos en {data_time:.3f}s")
            print(f"📊 Registros: {len(data)}")
            
            # 2. Formatear para respuesta API
            api_response = {
                'symbol': symbol,
                'interval': interval,
                'timestamp': datetime.now().isoformat(),
                'data_count': len(data),
                'latest_price': float(data['close'].iloc[-1]) if not data.empty else None,
                'data': data.to_dict('records') if len(data) <= 10 else data.tail(10).to_dict('records')
            }
            
            print(f"🔧 Respuesta API preparada")
            print(f"  - Último precio: ${api_response['latest_price']:.2f}")
            print(f"  - Registros en respuesta: {len(api_response['data'])}")
            
            return {
                'api_response': api_response,
                'performance': {
                    'data_retrieval_time': data_time
                }
            }
            
        except Exception as e:
            print(f"❌ Error en pipeline API: {e}")
            return None
    
    def _generate_trading_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """Generar señales de trading simples"""
        signals = []
        
        for i in range(1, len(data)):
            current = data.iloc[i]
            previous = data.iloc[i-1]
            
            # Señal RSI
            if current['rsi'] < 30 and previous['rsi'] >= 30:
                signals.append({
                    'datetime': current['datetime'],
                    'signal': 'BUY',
                    'reason': 'RSI oversold',
                    'price': current['close']
                })
            elif current['rsi'] > 70 and previous['rsi'] <= 70:
                signals.append({
                    'datetime': current['datetime'],
                    'signal': 'SELL',
                    'reason': 'RSI overbought',
                    'price': current['close']
                })
            
            # Señal MACD
            if (current['macd'] > current['macd_signal'] and 
                previous['macd'] <= previous['macd_signal']):
                signals.append({
                    'datetime': current['datetime'],
                    'signal': 'BUY',
                    'reason': 'MACD bullish crossover',
                    'price': current['close']
                })
        
        return pd.DataFrame(signals)
    
    def _prepare_ml_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Preparar features para ML"""
        features = data.copy()
        
        # Agregar más features técnicas
        if 'close' in features.columns:
            # Moving averages
            features['sma_10'] = features['close'].rolling(10).mean()
            features['sma_20'] = features['close'].rolling(20).mean()
            features['ema_12'] = features['close'].ewm(span=12).mean()
            
            # Price ratios
            features['price_sma10_ratio'] = features['close'] / features['sma_10']
            features['price_sma20_ratio'] = features['close'] / features['sma_20']
        
        # Limpiar NaN
        features = features.dropna()
        
        return features
    
    def _run_simple_backtest(self, data: pd.DataFrame) -> dict:
        """Ejecutar backtesting simple"""
        # Estrategia simple: comprar cuando RSI < 30, vender cuando RSI > 70
        signals = self._generate_trading_signals(data)
        
        if signals.empty:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'total_return': 0
            }
        
        # Simular operaciones
        trades = []
        position = None
        
        for _, signal in signals.iterrows():
            if signal['signal'] == 'BUY' and position is None:
                position = {'entry_price': signal['price'], 'entry_date': signal['datetime']}
            elif signal['signal'] == 'SELL' and position is not None:
                exit_price = signal['price']
                return_pct = (exit_price - position['entry_price']) / position['entry_price']
                trades.append({
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'return': return_pct,
                    'winning': return_pct > 0
                })
                position = None
        
        if not trades:
            return {
                'total_trades': 0,
                'winning_trades': 0,
                'win_rate': 0,
                'total_return': 0
            }
        
        winning_trades = sum(1 for t in trades if t['winning'])
        total_return = sum(t['return'] for t in trades)
        
        return {
            'total_trades': len(trades),
            'winning_trades': winning_trades,
            'win_rate': winning_trades / len(trades),
            'total_return': total_return
        }

def main():
    """Función principal de demostración"""
    print("🚀 Demostración de Integración de Estrategia Híbrida de Datos")
    print("=" * 70)
    
    integration = TradingSystemIntegration()
    
    # 1. Pipeline de Trading en Tiempo Real
    rt_results = integration.real_time_trading_pipeline()
    
    # 2. Pipeline de Entrenamiento ML
    ml_results = integration.ml_training_pipeline()
    
    # 3. Pipeline de Backtesting
    bt_results = integration.backtesting_pipeline()
    
    # 4. Pipeline de API Serving
    api_results = integration.api_serving_pipeline()
    
    # Resumen de rendimiento
    print("\n📊 Resumen de Rendimiento")
    print("=" * 30)
    
    if rt_results:
        print(f"⚡ Tiempo Real: {rt_results['performance']['total_time']:.3f}s")
    
    if ml_results:
        print(f"🤖 ML Training: {ml_results['performance']['data_retrieval_time']:.3f}s")
    
    if bt_results:
        print(f"📊 Backtesting: {bt_results['performance']['data_retrieval_time']:.3f}s")
    
    if api_results:
        print(f"🌐 API Serving: {api_results['performance']['data_retrieval_time']:.3f}s")
    
    print("\n✅ Demostración completada exitosamente!")
    print("\n💡 Recomendaciones:")
    print("  - Usar REALTIME_TRADING para operaciones en vivo")
    print("  - Usar ML_TRAINING para entrenar modelos con datos históricos")
    print("  - Usar BACKTESTING para validar estrategias")
    print("  - Usar API_SERVING para endpoints web rápidos")
    print("  - Usar DASHBOARD para visualizaciones interactivas")

if __name__ == "__main__":
    main()