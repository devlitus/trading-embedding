import sqlite3
import pandas as pd
from typing import List, Dict, Any, Optional
import logging
from pathlib import Path
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TradingDatabase:
    """
    Clase para manejar la base de datos SQLite del sistema de trading
    """
    
    def __init__(self, db_path: str = "data/trading.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()
        
    def _init_database(self):
        """
        Inicializa las tablas de la base de datos
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Tabla para datos OHLC
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS ohlc_data (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    interval_type TEXT NOT NULL,
                    timestamp INTEGER NOT NULL,
                    open_price REAL NOT NULL,
                    high_price REAL NOT NULL,
                    low_price REAL NOT NULL,
                    close_price REAL NOT NULL,
                    volume REAL NOT NULL,
                    quote_volume REAL,
                    trades_count INTEGER,
                    taker_buy_base_volume REAL,
                    taker_buy_quote_volume REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(symbol, interval_type, timestamp)
                )
            """)
            
            # Tabla para metadatos de símbolos
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS symbol_metadata (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT UNIQUE NOT NULL,
                    base_asset TEXT,
                    quote_asset TEXT,
                    status TEXT,
                    min_price REAL,
                    max_price REAL,
                    tick_size REAL,
                    min_qty REAL,
                    max_qty REAL,
                    step_size REAL,
                    metadata_json TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Tabla para etiquetas de patrones (para futuras fases)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS pattern_labels (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    interval_type TEXT NOT NULL,
                    start_timestamp INTEGER NOT NULL,
                    end_timestamp INTEGER NOT NULL,
                    pattern_type TEXT NOT NULL,
                    confidence_score REAL,
                    is_validated BOOLEAN DEFAULT FALSE,
                    validator_id TEXT,
                    notes TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Índices para optimizar consultas
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_ohlc_symbol_interval_time 
                ON ohlc_data(symbol, interval_type, timestamp)
            """)
            
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_pattern_labels_symbol_time 
                ON pattern_labels(symbol, start_timestamp, end_timestamp)
            """)
            
            conn.commit()
            logger.info("Base de datos inicializada correctamente")
    
    def insert_ohlc_data(self, data: List[Dict[str, Any]], symbol: str, interval: str) -> int:
        """
        Inserta datos OHLC en la base de datos
        
        Args:
            data: Lista de diccionarios con datos OHLC
            symbol: Símbolo del par de trading
            interval: Intervalo de tiempo
            
        Returns:
            Número de registros insertados
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            inserted_count = 0
            
            for record in data:
                try:
                    cursor.execute("""
                        INSERT OR IGNORE INTO ohlc_data (
                            symbol, interval_type, timestamp, open_price, high_price,
                            low_price, close_price, volume, quote_volume, trades_count,
                            taker_buy_base_volume, taker_buy_quote_volume
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol.upper(),
                        interval,
                        record['timestamp'],
                        record['open'],
                        record['high'],
                        record['low'],
                        record['close'],
                        record['volume'],
                        record.get('quote_volume'),
                        record.get('trades_count'),
                        record.get('taker_buy_base_volume'),
                        record.get('taker_buy_quote_volume')
                    ))
                    
                    if cursor.rowcount > 0:
                        inserted_count += 1
                        
                except sqlite3.Error as e:
                    logger.error(f"Error insertando registro: {e}")
                    continue
            
            conn.commit()
            logger.info(f"Insertados {inserted_count} nuevos registros para {symbol}")
            return inserted_count
    
    def get_ohlc_data(self, 
                      symbol: str, 
                      interval: str,
                      start_time: Optional[int] = None,
                      end_time: Optional[int] = None,
                      limit: Optional[int] = None) -> pd.DataFrame:
        """
        Obtiene datos OHLC de la base de datos
        
        Args:
            symbol: Símbolo del par
            interval: Intervalo de tiempo
            start_time: Timestamp de inicio (opcional)
            end_time: Timestamp de fin (opcional)
            limit: Límite de registros (opcional)
            
        Returns:
            DataFrame con datos OHLC
        """
        with sqlite3.connect(self.db_path) as conn:
            query = """
                SELECT timestamp, open_price, high_price, low_price, close_price,
                       volume, quote_volume, trades_count
                FROM ohlc_data 
                WHERE symbol = ? AND interval_type = ?
            """
            
            params = [symbol.upper(), interval]
            
            if start_time:
                query += " AND timestamp >= ?"
                params.append(start_time)
                
            if end_time:
                query += " AND timestamp <= ?"
                params.append(end_time)
                
            query += " ORDER BY timestamp ASC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            df = pd.read_sql_query(query, conn, params=params)
            
            if not df.empty:
                df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('datetime', inplace=True)
                df.rename(columns={
                    'open_price': 'open',
                    'high_price': 'high', 
                    'low_price': 'low',
                    'close_price': 'close'
                }, inplace=True)
                
            return df
    
    def save_symbol_metadata(self, symbol: str, metadata: Dict[str, Any]):
        """
        Guarda metadatos de un símbolo
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT OR REPLACE INTO symbol_metadata (
                    symbol, base_asset, quote_asset, status, min_price, max_price,
                    tick_size, min_qty, max_qty, step_size, metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
            """, (
                symbol.upper(),
                metadata.get('baseAsset'),
                metadata.get('quoteAsset'),
                metadata.get('status'),
                self._extract_filter_value(metadata, 'PRICE_FILTER', 'minPrice'),
                self._extract_filter_value(metadata, 'PRICE_FILTER', 'maxPrice'),
                self._extract_filter_value(metadata, 'PRICE_FILTER', 'tickSize'),
                self._extract_filter_value(metadata, 'LOT_SIZE', 'minQty'),
                self._extract_filter_value(metadata, 'LOT_SIZE', 'maxQty'),
                self._extract_filter_value(metadata, 'LOT_SIZE', 'stepSize'),
                json.dumps(metadata)
            ))
            
            conn.commit()
    
    def _extract_filter_value(self, metadata: Dict, filter_type: str, field: str) -> Optional[float]:
        """
        Extrae valores de filtros de metadatos de Binance
        """
        filters = metadata.get('filters', [])
        for f in filters:
            if f.get('filterType') == filter_type:
                value = f.get(field)
                return float(value) if value and value != '0.00000000' else None
        return None
    
    def get_available_symbols(self) -> List[str]:
        """
        Obtiene lista de símbolos disponibles en la base de datos
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT symbol FROM ohlc_data ORDER BY symbol")
            return [row[0] for row in cursor.fetchall()]
    
    def get_available_intervals(self, symbol: str) -> List[str]:
        """
        Obtiene lista de intervalos disponibles para un símbolo específico
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT DISTINCT interval_type FROM ohlc_data WHERE symbol = ? ORDER BY interval_type", 
                (symbol.upper(),)
            )
            intervals = [row[0] for row in cursor.fetchall()]
            return intervals
    
    def get_data_range(self, symbol: str, interval: str) -> Dict[str, Any]:
        """
        Obtiene el rango de fechas disponible para un símbolo
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT MIN(timestamp) as min_time, MAX(timestamp) as max_time, COUNT(*) as count
                FROM ohlc_data 
                WHERE symbol = ? AND interval_type = ?
            """, (symbol.upper(), interval))
            
            result = cursor.fetchone()
            if result and result[0]:
                return {
                    'min_datetime': datetime.fromtimestamp(result[0] / 1000),
                    'max_datetime': datetime.fromtimestamp(result[1] / 1000),
                    'total_records': result[2]
                }
            return {'total_records': 0}
    
    def cleanup_old_data(self, symbol: str, interval: str, keep_days: int = 365):
        """
        Limpia datos antiguos para mantener solo los últimos N días
        """
        cutoff_time = int((datetime.now().timestamp() - (keep_days * 24 * 3600)) * 1000)
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                DELETE FROM ohlc_data 
                WHERE symbol = ? AND interval_type = ? AND timestamp < ?
            """, (symbol.upper(), interval, cutoff_time))
            
            deleted_count = cursor.rowcount
            conn.commit()
            
            logger.info(f"Eliminados {deleted_count} registros antiguos de {symbol}")
            return deleted_count

# Ejemplo de uso
if __name__ == "__main__":
    db = TradingDatabase()
    
    # Ejemplo de datos ficticios
    sample_data = [
        {
            'timestamp': 1640995200000,
            'open': 47000.0,
            'high': 47500.0,
            'low': 46800.0,
            'close': 47200.0,
            'volume': 1234.56
        }
    ]
    
    # Insertar datos de ejemplo
    db.insert_ohlc_data(sample_data, 'BTCUSDT', '1h')
    
    # Obtener datos
    df = db.get_ohlc_data('BTCUSDT', '1h')
    print(f"Datos en BD: {len(df)} registros")
    print(df.head())