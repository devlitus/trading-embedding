try:
    import redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    redis = None

import json
import pickle
import pandas as pd
from typing import Any, Optional, Union
import logging
from datetime import datetime, timedelta
import hashlib

# Importar gestor de configuración
try:
    from ..config import is_cache_disabled, is_development_mode, get_config
except ImportError:
    # Fallback si no se puede importar el gestor de configuración
    def is_cache_disabled():
        return False
    def is_development_mode():
        return False
    def get_config(key, default=None):
        return default

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingCache:
    """
    Sistema de caché para datos de trading usando Redis o memoria local
    """
    
    def __init__(self, 
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 use_redis: bool = False,
                 default_ttl: int = 3600):
        """
        Inicializa el sistema de caché
        
        Args:
            redis_host: Host de Redis
            redis_port: Puerto de Redis
            redis_db: Base de datos de Redis
            use_redis: Si usar Redis o caché en memoria
            default_ttl: TTL por defecto en segundos
        """
        # Verificar si la caché está deshabilitada por configuración
        self.cache_disabled = is_cache_disabled()
        
        if self.cache_disabled:
            logger.info("Caché deshabilitada por configuración de desarrollo")
            self.use_redis = False
            self.default_ttl = default_ttl
            self.memory_cache = {}
            self.cache_timestamps = {}
            return
        
        self.use_redis = use_redis and REDIS_AVAILABLE
        self.default_ttl = default_ttl
        self.memory_cache = {}
        self.cache_timestamps = {}
        
        if use_redis and not REDIS_AVAILABLE:
            logger.warning("Redis solicitado pero no está disponible. Usando caché en memoria")
            self.use_redis = False
        elif self.use_redis:
            try:
                self.redis_client = redis.Redis(
                    host=redis_host,
                    port=redis_port,
                    db=redis_db,
                    decode_responses=False
                )
                # Test connection
                self.redis_client.ping()
                logger.info("Conectado a Redis exitosamente")
            except Exception as e:
                logger.warning(f"No se pudo conectar a Redis: {e}. Usando caché en memoria")
                self.use_redis = False
    
    def _generate_key(self, prefix: str, **kwargs) -> str:
        """
        Genera una clave única para el caché
        """
        key_data = f"{prefix}:" + ":".join([f"{k}={v}" for k, v in sorted(kwargs.items())])
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Almacena un valor en el caché
        
        Args:
            key: Clave del caché
            value: Valor a almacenar
            ttl: Tiempo de vida en segundos
            
        Returns:
            True si se almacenó correctamente
        """
        # Si la caché está deshabilitada, simular éxito pero no almacenar
        if self.cache_disabled:
            return True
            
        ttl = ttl or self.default_ttl
        
        try:
            if self.use_redis:
                serialized_value = pickle.dumps(value)
                return self.redis_client.setex(key, ttl, serialized_value)
            else:
                # Caché en memoria
                self.memory_cache[key] = value
                self.cache_timestamps[key] = datetime.now() + timedelta(seconds=ttl)
                return True
                
        except Exception as e:
            logger.error(f"Error almacenando en caché: {e}")
            return False
    
    def get(self, key: str) -> Optional[Any]:
        """
        Obtiene un valor del caché
        
        Args:
            key: Clave del caché
            
        Returns:
            Valor almacenado o None si no existe o expiró
        """
        # Si la caché está deshabilitada, siempre retornar None
        if self.cache_disabled:
            return None
            
        try:
            if self.use_redis:
                serialized_value = self.redis_client.get(key)
                if serialized_value:
                    return pickle.loads(serialized_value)
                return None
            else:
                # Caché en memoria
                if key in self.memory_cache:
                    if datetime.now() < self.cache_timestamps[key]:
                        return self.memory_cache[key]
                    else:
                        # Expirado, eliminar
                        del self.memory_cache[key]
                        del self.cache_timestamps[key]
                return None
                
        except Exception as e:
            logger.error(f"Error obteniendo del caché: {e}")
            return None
    
    def delete(self, key: str) -> bool:
        """
        Elimina una clave del caché
        """
        # Si la caché está deshabilitada, simular éxito
        if self.cache_disabled:
            return True
            
        try:
            if self.use_redis:
                return bool(self.redis_client.delete(key))
            else:
                if key in self.memory_cache:
                    del self.memory_cache[key]
                    del self.cache_timestamps[key]
                    return True
                return False
                
        except Exception as e:
            logger.error(f"Error eliminando del caché: {e}")
            return False
    
    def exists(self, key: str) -> bool:
        """
        Verifica si una clave existe en el caché
        """
        # Si la caché está deshabilitada, siempre retornar False
        if self.cache_disabled:
            return False
            
        try:
            if self.use_redis:
                return bool(self.redis_client.exists(key))
            else:
                if key in self.memory_cache:
                    if datetime.now() < self.cache_timestamps[key]:
                        return True
                    else:
                        # Expirado, eliminar
                        del self.memory_cache[key]
                        del self.cache_timestamps[key]
                return False
                
        except Exception as e:
            logger.error(f"Error verificando existencia en caché: {e}")
            return False
    
    def clear_all(self) -> bool:
        """
        Limpia todo el caché
        """
        # Si la caché está deshabilitada, simular éxito
        if self.cache_disabled:
            return True
            
        try:
            if self.use_redis:
                return bool(self.redis_client.flushdb())
            else:
                self.memory_cache.clear()
                self.cache_timestamps.clear()
                return True
                
        except Exception as e:
            logger.error(f"Error limpiando caché: {e}")
            return False
    
    def cache_ohlc_data(self, 
                        symbol: str, 
                        interval: str, 
                        data: Union[pd.DataFrame, list],
                        ttl: Optional[int] = None) -> bool:
        """
        Almacena datos OHLC en caché
        
        Args:
            symbol: Símbolo del par
            interval: Intervalo de tiempo
            data: Datos OHLC
            ttl: Tiempo de vida
            
        Returns:
            True si se almacenó correctamente
        """
        key = self._generate_key('ohlc', symbol=symbol, interval=interval)
        return self.set(key, data, ttl)
    
    def get_cached_ohlc_data(self, 
                            symbol: str, 
                            interval: str) -> Optional[Union[pd.DataFrame, list]]:
        """
        Obtiene datos OHLC del caché
        
        Args:
            symbol: Símbolo del par
            interval: Intervalo de tiempo
            
        Returns:
            Datos OHLC o None si no están en caché
        """
        key = self._generate_key('ohlc', symbol=symbol, interval=interval)
        return self.get(key)
    
    def cache_technical_indicators(self, 
                                  symbol: str, 
                                  interval: str,
                                  indicators: dict,
                                  ttl: Optional[int] = None) -> bool:
        """
        Almacena indicadores técnicos en caché
        """
        key = self._generate_key('indicators', symbol=symbol, interval=interval)
        return self.set(key, indicators, ttl)
    
    def get_cached_indicators(self, 
                             symbol: str, 
                             interval: str) -> Optional[dict]:
        """
        Obtiene indicadores técnicos del caché
        """
        key = self._generate_key('indicators', symbol=symbol, interval=interval)
        return self.get(key)
    
    def cache_pattern_analysis(self, 
                              symbol: str, 
                              interval: str,
                              analysis: dict,
                              ttl: Optional[int] = None) -> bool:
        """
        Almacena análisis de patrones en caché
        """
        key = self._generate_key('patterns', symbol=symbol, interval=interval)
        return self.set(key, analysis, ttl)
    
    def get_cached_pattern_analysis(self, 
                                   symbol: str, 
                                   interval: str) -> Optional[dict]:
        """
        Obtiene análisis de patrones del caché
        """
        key = self._generate_key('patterns', symbol=symbol, interval=interval)
        return self.get(key)
    
    def get_cache_stats(self) -> dict:
        """
        Obtiene estadísticas del caché
        """
        # Si la caché está deshabilitada, retornar estadísticas indicándolo
        if self.cache_disabled:
            return {
                'type': 'disabled',
                'status': 'Cache disabled by development configuration',
                'keys_count': 0,
                'memory_usage': '0 bytes',
                'connected': False,
                'development_mode': is_development_mode()
            }
            
        try:
            if self.use_redis:
                info = self.redis_client.info()
                return {
                    'type': 'redis',
                    'used_memory': info.get('used_memory_human', 'N/A'),
                    'connected_clients': info.get('connected_clients', 0),
                    'total_commands_processed': info.get('total_commands_processed', 0)
                }
            else:
                active_keys = 0
                expired_keys = 0
                now = datetime.now()
                
                for key, expiry in self.cache_timestamps.items():
                    if now < expiry:
                        active_keys += 1
                    else:
                        expired_keys += 1
                
                return {
                    'type': 'memory',
                    'active_keys': active_keys,
                    'expired_keys': expired_keys,
                    'total_keys': len(self.memory_cache)
                }
                
        except Exception as e:
            logger.error(f"Error obteniendo estadísticas: {e}")
            return {'error': str(e)}
    
    def cleanup_expired(self) -> int:
        """
        Limpia claves expiradas del caché en memoria
        
        Returns:
            Número de claves eliminadas
        """
        if self.use_redis:
            return 0  # Redis maneja esto automáticamente
        
        expired_keys = []
        now = datetime.now()
        
        for key, expiry in self.cache_timestamps.items():
            if now >= expiry:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.memory_cache[key]
            del self.cache_timestamps[key]
        
        if expired_keys:
            logger.info(f"Limpiadas {len(expired_keys)} claves expiradas")
        
        return len(expired_keys)

# Instancia global del caché
cache = TradingCache()

# Ejemplo de uso
if __name__ == "__main__":
    # Crear instancia de caché
    trading_cache = TradingCache(use_redis=False)
    
    # Datos de ejemplo
    sample_data = {
        'symbol': 'BTCUSDT',
        'price': 47000.0,
        'timestamp': datetime.now().isoformat()
    }
    
    # Almacenar en caché
    key = "test_data"
    trading_cache.set(key, sample_data, ttl=60)
    
    # Obtener del caché
    cached_data = trading_cache.get(key)
    print(f"Datos en caché: {cached_data}")
    
    # Estadísticas
    stats = trading_cache.get_cache_stats()
    print(f"Estadísticas del caché: {stats}")