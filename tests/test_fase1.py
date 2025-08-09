#!/usr/bin/env python3
"""
Tests para la Fase 1: Adquisición de Datos

Este archivo contiene tests unitarios para verificar el funcionamiento
correcto de todos los componentes de la Fase 1.
"""

import sys
import os
from pathlib import Path
import unittest
from unittest.mock import Mock, patch, MagicMock
import pandas as pd
from datetime import datetime, timedelta
import tempfile
import sqlite3

# Agregar el directorio src al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / 'src'))

from data.binance_client import BinanceClient
from data.database import TradingDatabase
from data.cache import TradingCache
from data.data_manager import DataManager

class TestBinanceClient(unittest.TestCase):
    """Tests para el cliente de Binance"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.client = BinanceClient()
    
    def test_client_initialization(self):
        """Test de inicialización del cliente"""
        self.assertIsNotNone(self.client)
        self.assertEqual(self.client.base_url, "https://api.binance.com")
    
    def test_to_timestamp(self):
        """Test de conversión de fecha a timestamp"""
        # Test con string de fecha
        timestamp = self.client._to_timestamp("2023-01-01")
        self.assertIsInstance(timestamp, int)
        self.assertGreater(timestamp, 0)
        
        # Test con datetime
        dt = datetime(2023, 1, 1)
        timestamp = self.client._to_timestamp(dt)
        self.assertEqual(timestamp, 1672531200000)  # 2023-01-01 00:00:00 UTC en ms
    
    @patch('data.binance_client.requests.get')
    def test_get_klines_success(self, mock_get):
        """Test de obtención exitosa de klines"""
        # Mock de respuesta exitosa
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = [
            [1672531200000, "100.0", "110.0", "95.0", "105.0", "1000.0", 1672534799999, "105000.0", 100, "500.0", "52500.0", "0"]
        ]
        mock_get.return_value = mock_response
        
        df = self.client.get_klines('BTCUSDT', '1h')
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertFalse(df.empty)
        self.assertIn('open', df.columns)
        self.assertIn('high', df.columns)
        self.assertIn('low', df.columns)
        self.assertIn('close', df.columns)
        self.assertIn('volume', df.columns)
    
    @patch('data.binance_client.requests.get')
    def test_get_klines_error(self, mock_get):
        """Test de manejo de errores en klines"""
        # Mock de respuesta con error
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.json.return_value = {"code": -1121, "msg": "Invalid symbol."}
        mock_get.return_value = mock_response
        
        df = self.client.get_klines('INVALID', '1h')
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)

class TestTradingDatabase(unittest.TestCase):
    """Tests para la base de datos"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        # Crear base de datos temporal
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        self.db = TradingDatabase(self.temp_db.name)
    
    def tearDown(self):
        """Limpieza después de cada test"""
        # Cerrar conexión y eliminar archivo temporal
        if hasattr(self.db, 'conn'):
            self.db.conn.close()
        os.unlink(self.temp_db.name)
    
    def test_database_initialization(self):
        """Test de inicialización de la base de datos"""
        self.assertIsNotNone(self.db)
        self.assertTrue(os.path.exists(self.temp_db.name))
        
        # Verificar que las tablas se crearon
        conn = sqlite3.connect(self.temp_db.name)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        self.assertIn('ohlc_data', tables)
        self.assertIn('symbol_metadata', tables)
        self.assertIn('pattern_labels', tables)
        
        conn.close()
    
    def test_insert_ohlc_data(self):
        """Test de inserción de datos OHLC"""
        # Datos de prueba
        test_data = [
            {
                'timestamp': 1672531200000,
                'open': 100.0,
                'high': 110.0,
                'low': 95.0,
                'close': 105.0,
                'volume': 1000.0,
                'quote_volume': 105000.0,
                'trades_count': 100,
                'taker_buy_base_volume': 500.0,
                'taker_buy_quote_volume': 52500.0
            }
        ]
        
        inserted_count = self.db.insert_ohlc_data(test_data, 'BTCUSDT', '1h')
        self.assertEqual(inserted_count, 1)
        
        # Verificar que se insertó correctamente
        df = self.db.get_ohlc_data('BTCUSDT', '1h')
        self.assertEqual(len(df), 1)
        self.assertEqual(df.iloc[0]['open'], 100.0)
    
    def test_get_available_symbols(self):
        """Test de obtención de símbolos disponibles"""
        # Insertar datos de prueba
        test_data = [{
            'timestamp': 1672531200000,
            'open': 100.0, 'high': 110.0, 'low': 95.0, 'close': 105.0,
            'volume': 1000.0, 'quote_volume': 105000.0, 'trades_count': 100,
            'taker_buy_base_volume': 500.0, 'taker_buy_quote_volume': 52500.0
        }]
        
        self.db.insert_ohlc_data(test_data, 'BTCUSDT', '1h')
        self.db.insert_ohlc_data(test_data, 'ETHUSDT', '1h')
        
        symbols = self.db.get_available_symbols()
        self.assertIn('BTCUSDT', symbols)
        self.assertIn('ETHUSDT', symbols)
        self.assertEqual(len(symbols), 2)

class TestTradingCache(unittest.TestCase):
    """Tests para el sistema de caché"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.cache = TradingCache(use_redis=False, default_ttl=60)
    
    def test_cache_initialization(self):
        """Test de inicialización del caché"""
        self.assertIsNotNone(self.cache)
        self.assertFalse(self.cache.use_redis)
        self.assertEqual(self.cache.default_ttl, 60)
    
    def test_set_and_get(self):
        """Test de almacenamiento y recuperación básica"""
        test_data = {'symbol': 'BTCUSDT', 'price': 50000.0}
        
        # Almacenar datos
        result = self.cache.set('test_key', test_data)
        self.assertTrue(result)
        
        # Recuperar datos
        cached_data = self.cache.get('test_key')
        self.assertEqual(cached_data, test_data)
    
    def test_cache_expiration(self):
        """Test de expiración del caché"""
        test_data = {'symbol': 'BTCUSDT', 'price': 50000.0}
        
        # Almacenar con TTL muy corto
        self.cache.set('test_key', test_data, ttl=1)
        
        # Verificar que existe inmediatamente
        self.assertTrue(self.cache.exists('test_key'))
        
        # Esperar a que expire
        import time
        time.sleep(2)
        
        # Verificar que expiró
        cached_data = self.cache.get('test_key')
        self.assertIsNone(cached_data)
    
    def test_cache_ohlc_data(self):
        """Test de caché específico para datos OHLC"""
        # Crear DataFrame de prueba
        test_df = pd.DataFrame({
            'open': [100.0, 101.0],
            'high': [110.0, 111.0],
            'low': [95.0, 96.0],
            'close': [105.0, 106.0],
            'volume': [1000.0, 1100.0]
        })
        
        # Almacenar en caché
        result = self.cache.cache_ohlc_data('BTCUSDT', '1h', test_df)
        self.assertTrue(result)
        
        # Recuperar del caché
        cached_df = self.cache.get_cached_ohlc_data('BTCUSDT', '1h')
        self.assertIsInstance(cached_df, pd.DataFrame)
        self.assertEqual(len(cached_df), 2)
        pd.testing.assert_frame_equal(cached_df, test_df)
    
    def test_cache_stats(self):
        """Test de estadísticas del caché"""
        # Realizar algunas operaciones
        self.cache.set('key1', 'value1')
        self.cache.set('key2', 'value2')
        self.cache.get('key1')  # Hit
        self.cache.get('nonexistent')  # Miss
        
        stats = self.cache.get_cache_stats()
        
        self.assertIn('total_keys', stats)
        self.assertIn('hits', stats)
        self.assertIn('misses', stats)
        self.assertEqual(stats['total_keys'], 2)
        self.assertEqual(stats['hits'], 1)
        self.assertEqual(stats['misses'], 1)

class TestDataManager(unittest.TestCase):
    """Tests para el gestor de datos integrado"""
    
    def setUp(self):
        """Configuración inicial para cada test"""
        # Crear base de datos temporal
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.data_manager = DataManager(
            db_path=self.temp_db.name,
            use_cache=True,
            cache_ttl=60
        )
    
    def tearDown(self):
        """Limpieza después de cada test"""
        # Eliminar archivo temporal
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_data_manager_initialization(self):
        """Test de inicialización del gestor de datos"""
        self.assertIsNotNone(self.data_manager)
        self.assertIsNotNone(self.data_manager.binance_client)
        self.assertIsNotNone(self.data_manager.database)
        self.assertIsNotNone(self.data_manager.cache)
        self.assertTrue(self.data_manager.use_cache)
    
    def test_health_check(self):
        """Test de health check"""
        health = self.data_manager.health_check()
        
        self.assertIn('timestamp', health)
        self.assertIn('components', health)
        self.assertIn('binance_api', health['components'])
        self.assertIn('database', health['components'])
        self.assertIn('cache', health['components'])
    
    @patch('data.binance_client.BinanceClient.get_historical_data')
    def test_fetch_and_store_data_mock(self, mock_get_historical):
        """Test de obtención y almacenamiento con mock"""
        # Mock de datos de Binance
        mock_df = pd.DataFrame({
            'timestamp': [1672531200000, 1672534800000],
            'open': [100.0, 101.0],
            'high': [110.0, 111.0],
            'low': [95.0, 96.0],
            'close': [105.0, 106.0],
            'volume': [1000.0, 1100.0],
            'quote_volume': [105000.0, 116600.0],
            'trades_count': [100, 110],
            'taker_buy_base_volume': [500.0, 550.0],
            'taker_buy_quote_volume': [52500.0, 58300.0]
        })
        mock_df.index = pd.to_datetime(mock_df['timestamp'], unit='ms')
        mock_get_historical.return_value = mock_df
        
        # Ejecutar fetch_and_store_data
        result = self.data_manager.fetch_and_store_data('BTCUSDT', '1h', days_back=1)
        
        # Verificar resultado
        self.assertEqual(result['symbol'], 'BTCUSDT')
        self.assertEqual(result['source'], 'binance')
        self.assertEqual(result['records_count'], 2)
        self.assertIn('execution_time', result)
    
    def test_get_database_stats(self):
        """Test de estadísticas de base de datos"""
        stats = self.data_manager.get_database_stats()
        
        self.assertIn('total_symbols', stats)
        self.assertIn('symbols', stats)
        self.assertIn('symbol_details', stats)
        self.assertIsInstance(stats['total_symbols'], int)
    
    def test_get_cache_stats(self):
        """Test de estadísticas de caché"""
        stats = self.data_manager.get_cache_stats()
        
        self.assertIn('cache_enabled', stats)
        self.assertTrue(stats['cache_enabled'])

class TestIntegration(unittest.TestCase):
    """Tests de integración entre componentes"""
    
    def setUp(self):
        """Configuración inicial para tests de integración"""
        # Crear base de datos temporal
        self.temp_db = tempfile.NamedTemporaryFile(delete=False, suffix='.db')
        self.temp_db.close()
        
        self.data_manager = DataManager(
            db_path=self.temp_db.name,
            use_cache=True,
            cache_ttl=60
        )
    
    def tearDown(self):
        """Limpieza después de cada test"""
        if os.path.exists(self.temp_db.name):
            os.unlink(self.temp_db.name)
    
    def test_full_workflow_mock(self):
        """Test del flujo completo con datos mock"""
        with patch('data.binance_client.BinanceClient.get_historical_data') as mock_get_historical:
            # Mock de datos
            mock_df = pd.DataFrame({
                'timestamp': [1672531200000],
                'open': [100.0], 'high': [110.0], 'low': [95.0], 'close': [105.0],
                'volume': [1000.0], 'quote_volume': [105000.0], 'trades_count': [100],
                'taker_buy_base_volume': [500.0], 'taker_buy_quote_volume': [52500.0]
            })
            mock_df.index = pd.to_datetime(mock_df['timestamp'], unit='ms')
            mock_get_historical.return_value = mock_df
            
            # 1. Obtener y almacenar datos
            result = self.data_manager.fetch_and_store_data('BTCUSDT', '1h', days_back=1)
            self.assertEqual(result['records_count'], 1)
            
            # 2. Verificar que se almacenó en BD
            df_from_db = self.data_manager.get_data('BTCUSDT', '1h', prefer_cache=False)
            self.assertEqual(len(df_from_db), 1)
            
            # 3. Verificar que se almacenó en caché
            df_from_cache = self.data_manager.get_data('BTCUSDT', '1h', prefer_cache=True)
            self.assertEqual(len(df_from_cache), 1)
            
            # 4. Verificar estadísticas
            db_stats = self.data_manager.get_database_stats()
            self.assertGreaterEqual(db_stats['total_symbols'], 1)
            
            cache_stats = self.data_manager.get_cache_stats()
            self.assertTrue(cache_stats['cache_enabled'])

def run_tests():
    """Ejecuta todos los tests"""
    # Crear suite de tests
    test_suite = unittest.TestSuite()
    
    # Agregar tests de cada clase
    test_classes = [
        TestBinanceClient,
        TestTradingDatabase,
        TestTradingCache,
        TestDataManager,
        TestIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    return result.wasSuccessful()

if __name__ == '__main__':
    print("🧪 EJECUTANDO TESTS DE LA FASE 1")
    print("=" * 50)
    
    success = run_tests()
    
    if success:
        print("\n✅ TODOS LOS TESTS PASARON CORRECTAMENTE")
        print("🎯 La Fase 1 está completamente funcional")
    else:
        print("\n❌ ALGUNOS TESTS FALLARON")
        print("🔧 Revisar los errores reportados")
    
    sys.exit(0 if success else 1)