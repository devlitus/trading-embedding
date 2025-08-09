from binance_client import BinanceClient
from database import TradingDatabase
from cache import TradingCache
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime, timedelta
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataManager:
    """
    Gestor principal de datos que integra Binance API, base de datos y caché
    """
    
    def __init__(self, 
                 db_path: str = "data/trading.db",
                 use_cache: bool = True,
                 cache_ttl: int = 3600):
        """
        Inicializa el gestor de datos
        
        Args:
            db_path: Ruta de la base de datos
            use_cache: Si usar caché
            cache_ttl: TTL del caché en segundos
        """
        self.binance_client = BinanceClient()
        self.database = TradingDatabase(db_path)
        self.cache = TradingCache(use_redis=False, default_ttl=cache_ttl) if use_cache else None
        self.use_cache = use_cache
        
        logger.info("DataManager inicializado correctamente")
    
    def fetch_and_store_data(self, 
                            symbol: str, 
                            interval: str = '1h',
                            days_back: int = 30,
                            force_refresh: bool = False) -> Dict[str, Any]:
        """
        Obtiene datos de Binance y los almacena en BD y caché
        
        Args:
            symbol: Símbolo del par (ej: 'BTCUSDT')
            interval: Intervalo de tiempo
            days_back: Días hacia atrás
            force_refresh: Forzar actualización ignorando caché
            
        Returns:
            Diccionario con estadísticas de la operación
        """
        start_time = datetime.now()
        
        try:
            # Verificar caché primero (si no es refresh forzado)
            cached_data = None
            if self.use_cache and not force_refresh:
                cached_data = self.cache.get_cached_ohlc_data(symbol, interval)
                if cached_data is not None:
                    logger.info(f"Datos encontrados en caché para {symbol}")
                    return {
                        'symbol': symbol,
                        'interval': interval,
                        'source': 'cache',
                        'records_count': len(cached_data),
                        'execution_time': (datetime.now() - start_time).total_seconds()
                    }
            
            # Obtener datos de Binance
            logger.info(f"Obteniendo datos de Binance para {symbol} ({interval})")
            df = self.binance_client.get_historical_data(
                symbol=symbol,
                interval=interval,
                days_back=days_back
            )
            
            if df.empty:
                logger.warning(f"No se obtuvieron datos para {symbol}")
                return {
                    'symbol': symbol,
                    'interval': interval,
                    'source': 'binance',
                    'records_count': 0,
                    'error': 'No data received'
                }
            
            # Convertir DataFrame a lista de diccionarios para la BD
            data_records = []
            for idx, row in df.iterrows():
                data_records.append({
                    'timestamp': int(row['timestamp']),
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': float(row['volume']),
                    'quote_volume': float(row.get('quote_volume', 0)),
                    'trades_count': int(row.get('trades_count', 0)),
                    'taker_buy_base_volume': float(row.get('taker_buy_base_volume', 0)),
                    'taker_buy_quote_volume': float(row.get('taker_buy_quote_volume', 0))
                })
            
            # Almacenar en base de datos
            inserted_count = self.database.insert_ohlc_data(data_records, symbol, interval)
            
            # Almacenar en caché
            if self.use_cache:
                self.cache.cache_ohlc_data(symbol, interval, df)
                logger.info(f"Datos almacenados en caché para {symbol}")
            
            # Obtener y guardar metadatos del símbolo
            try:
                symbol_info = self.binance_client.get_symbol_info(symbol)
                self.database.save_symbol_metadata(symbol, symbol_info)
            except Exception as e:
                logger.warning(f"No se pudieron obtener metadatos para {symbol}: {e}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            result = {
                'symbol': symbol,
                'interval': interval,
                'source': 'binance',
                'records_count': len(data_records),
                'inserted_count': inserted_count,
                'execution_time': execution_time,
                'date_range': {
                    'start': df.index.min().isoformat(),
                    'end': df.index.max().isoformat()
                }
            }
            
            logger.info(f"Procesamiento completado para {symbol}: {inserted_count} registros insertados")
            return result
            
        except Exception as e:
            logger.error(f"Error procesando datos para {symbol}: {e}")
            return {
                'symbol': symbol,
                'interval': interval,
                'source': 'error',
                'records_count': 0,
                'error': str(e),
                'execution_time': (datetime.now() - start_time).total_seconds()
            }
    
    def get_data(self, 
                 symbol: str, 
                 interval: str,
                 start_time: Optional[datetime] = None,
                 end_time: Optional[datetime] = None,
                 limit: Optional[int] = None,
                 prefer_cache: bool = True) -> pd.DataFrame:
        """
        Obtiene datos OHLC priorizando caché o base de datos
        
        Args:
            symbol: Símbolo del par
            interval: Intervalo de tiempo
            start_time: Fecha de inicio
            end_time: Fecha de fin
            limit: Límite de registros
            prefer_cache: Priorizar caché sobre BD
            
        Returns:
            DataFrame con datos OHLC
        """
        # Intentar caché primero si está habilitado
        if self.use_cache and prefer_cache:
            cached_data = self.cache.get_cached_ohlc_data(symbol, interval)
            if cached_data is not None and not cached_data.empty:
                logger.info(f"Datos obtenidos del caché para {symbol}")
                
                # Filtrar por fechas si se especifican
                if start_time or end_time:
                    if start_time:
                        cached_data = cached_data[cached_data.index >= start_time]
                    if end_time:
                        cached_data = cached_data[cached_data.index <= end_time]
                
                # Aplicar límite si se especifica
                if limit:
                    cached_data = cached_data.tail(limit)
                
                return cached_data
        
        # Obtener de base de datos
        start_ts = int(start_time.timestamp() * 1000) if start_time else None
        end_ts = int(end_time.timestamp() * 1000) if end_time else None
        
        df = self.database.get_ohlc_data(
            symbol=symbol,
            interval=interval,
            start_time=start_ts,
            end_time=end_ts,
            limit=limit
        )
        
        logger.info(f"Datos obtenidos de BD para {symbol}: {len(df)} registros")
        return df
    
    def bulk_fetch_symbols(self, 
                          symbols: List[str], 
                          interval: str = '1h',
                          days_back: int = 30,
                          delay_between_requests: float = 0.5) -> List[Dict[str, Any]]:
        """
        Obtiene datos para múltiples símbolos
        
        Args:
            symbols: Lista de símbolos
            interval: Intervalo de tiempo
            days_back: Días hacia atrás
            delay_between_requests: Delay entre requests para rate limiting
            
        Returns:
            Lista con resultados de cada símbolo
        """
        results = []
        
        for i, symbol in enumerate(symbols):
            logger.info(f"Procesando {symbol} ({i+1}/{len(symbols)})")
            
            result = self.fetch_and_store_data(
                symbol=symbol,
                interval=interval,
                days_back=days_back
            )
            results.append(result)
            
            # Rate limiting
            if i < len(symbols) - 1:  # No delay después del último
                time.sleep(delay_between_requests)
        
        return results
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas de la base de datos
        """
        symbols = self.database.get_available_symbols()
        
        stats = {
            'total_symbols': len(symbols),
            'symbols': symbols,
            'symbol_details': {}
        }
        
        # Detalles de todos los símbolos
        for symbol in symbols:
            try:
                range_info = self.database.get_data_range(symbol, '1h')
                # Obtener intervalos disponibles para este símbolo
                intervals = self.database.get_available_intervals(symbol)
                
                # Contar registros totales
                total_records = 0
                for interval in intervals:
                    # Usar context manager para la conexión
                    import sqlite3
                    with sqlite3.connect(self.database.db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute(
                            "SELECT COUNT(*) FROM ohlc_data WHERE symbol = ? AND interval_type = ?",
                            (symbol.upper(), interval)
                        )
                        result = cursor.fetchone()
                        total_records += result[0] if result else 0
                
                stats['symbol_details'][symbol] = {
                    'total_records': total_records,
                    'intervals': intervals,
                    'earliest_timestamp': range_info.get('earliest_timestamp'),
                    'latest_timestamp': range_info.get('latest_timestamp')
                }
            except Exception as e:
                logger.warning(f"Error obteniendo stats para {symbol}: {e}")
                stats['symbol_details'][symbol] = {
                    'total_records': 0,
                    'intervals': [],
                    'earliest_timestamp': None,
                    'latest_timestamp': None
                }
        
        return stats
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Obtiene estadísticas del caché
        """
        if not self.use_cache:
            return {'cache_enabled': False}
        
        return {
            'cache_enabled': True,
            **self.cache.get_cache_stats()
        }
    
    def cleanup_old_data(self, symbol: str, interval: str, keep_days: int = 365) -> int:
        """
        Limpia datos antiguos de la base de datos
        """
        return self.database.cleanup_old_data(symbol, interval, keep_days)
    
    def health_check(self) -> Dict[str, Any]:
        """
        Verifica el estado de todos los componentes
        """
        health = {
            'timestamp': datetime.now().isoformat(),
            'components': {}
        }
        
        # Test Binance API
        try:
            symbols = self.binance_client.get_available_symbols()
            health['components']['binance_api'] = {
                'status': 'healthy',
                'available_symbols': len(symbols)
            }
        except Exception as e:
            health['components']['binance_api'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Test Database
        try:
            db_symbols = self.database.get_available_symbols()
            health['components']['database'] = {
                'status': 'healthy',
                'stored_symbols': len(db_symbols)
            }
        except Exception as e:
            health['components']['database'] = {
                'status': 'error',
                'error': str(e)
            }
        
        # Test Cache
        if self.use_cache:
            try:
                cache_stats = self.cache.get_cache_stats()
                health['components']['cache'] = {
                    'status': 'healthy',
                    **cache_stats
                }
            except Exception as e:
                health['components']['cache'] = {
                    'status': 'error',
                    'error': str(e)
                }
        else:
            health['components']['cache'] = {
                'status': 'disabled'
            }
        
        return health

# Ejemplo de uso
if __name__ == "__main__":
    # Crear gestor de datos
    data_manager = DataManager()
    
    # Health check
    health = data_manager.health_check()
    print("=== HEALTH CHECK ===")
    for component, status in health['components'].items():
        print(f"{component}: {status['status']}")
    
    # Obtener datos de ejemplo
    print("\n=== OBTENIENDO DATOS ===")
    result = data_manager.fetch_and_store_data('BTCUSDT', '1h', days_back=7)
    print(f"Resultado: {result}")
    
    # Estadísticas
    print("\n=== ESTADÍSTICAS ===")
    db_stats = data_manager.get_database_stats()
    print(f"BD: {db_stats['total_symbols']} símbolos")
    
    cache_stats = data_manager.get_cache_stats()
    print(f"Caché: {cache_stats}")