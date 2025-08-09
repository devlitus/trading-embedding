#!/usr/bin/env python3
"""
Data Access Layer - Capa de acceso a datos híbrida
Maneja tanto base de datos SQLite como archivos CSV de manera optimizada
"""

import pandas as pd
import sqlite3
from pathlib import Path
from typing import Optional, List, Dict, Union
from datetime import datetime, timedelta
import logging
from .database import TradingDatabase
from .cache import TradingCache

logger = logging.getLogger(__name__)

class DataAccessLayer:
    """
    Capa de acceso a datos híbrida que optimiza el uso de:
    - Base de datos SQLite: Para operaciones en tiempo real, APIs, consultas complejas
    - Archivos CSV: Para análisis ML, investigación, compatibilidad con data science tools
    """
    
    def __init__(self, db_path: str = "data/trading.db", csv_dir: str = "data/historical"):
        self.db_path = Path(db_path)
        self.csv_dir = Path(csv_dir)
        self.csv_dir.mkdir(parents=True, exist_ok=True)
        
        # Inicializar componentes
        self.database = TradingDatabase(str(self.db_path))
        self.cache = TradingCache()
        
        # Metadatos de sincronización
        self.sync_metadata = {}
        
    def get_data_for_realtime(self, symbol: str, interval: str, 
                             limit: Optional[int] = None,
                             start_time: Optional[datetime] = None,
                             end_time: Optional[datetime] = None) -> pd.DataFrame:
        """
        Obtiene datos optimizados para operaciones en tiempo real.
        Usa cache -> DB -> CSV como fallback.
        """
        logger.info(f"Obteniendo datos en tiempo real para {symbol} {interval}")
        
        # 1. Intentar cache primero (más rápido)
        try:
            cached_data = self.cache.get_cached_ohlc_data(symbol, interval, limit or 1000)
            if cached_data is not None and not cached_data.empty:
                logger.debug(f"Datos encontrados en cache: {len(cached_data)} registros")
                return self._filter_by_time(cached_data, start_time, end_time, limit)
        except Exception as e:
            logger.warning(f"Error accediendo cache: {e}")
        
        # 2. Consultar base de datos (consultas SQL eficientes)
        try:
            db_data = self.database.get_ohlc_data(
                symbol=symbol,
                interval=interval,
                limit=limit,
                start_time=start_time,
                end_time=end_time
            )
            if db_data is not None and not db_data.empty:
                logger.debug(f"Datos encontrados en DB: {len(db_data)} registros")
                # Actualizar cache para próximas consultas
                self._update_cache_async(symbol, interval, db_data)
                return db_data
        except Exception as e:
            logger.warning(f"Error accediendo base de datos: {e}")
        
        # 3. Fallback a CSV (menos eficiente pero disponible)
        return self._get_data_from_csv(symbol, interval, start_time, end_time, limit)
    
    def get_data_for_ml_analysis(self, symbol: str, interval: str,
                                lookback_days: Optional[int] = None) -> pd.DataFrame:
        """
        Obtiene datos optimizados para análisis de Machine Learning.
        Prioriza CSV para compatibilidad con pandas/numpy/scikit-learn.
        """
        logger.info(f"Obteniendo datos para ML: {symbol} {interval}")
        
        # 1. Intentar CSV primero (mejor para ML)
        csv_data = self._get_data_from_csv(symbol, interval, lookback_days=lookback_days)
        if csv_data is not None and not csv_data.empty:
            logger.debug(f"Datos CSV para ML: {len(csv_data)} registros")
            return self._prepare_for_ml(csv_data)
        
        # 2. Fallback a base de datos y exportar a CSV
        logger.info("CSV no disponible, obteniendo de DB y exportando...")
        db_data = self.database.get_ohlc_data(symbol=symbol, interval=interval)
        if db_data is not None and not db_data.empty:
            # Exportar a CSV para futuras consultas ML
            self._export_to_csv(symbol, interval, db_data)
            return self._prepare_for_ml(db_data)
        
        logger.warning(f"No se encontraron datos para {symbol} {interval}")
        return pd.DataFrame()
    
    def sync_data_sources(self, symbols: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Sincroniza datos entre base de datos y archivos CSV.
        Mantiene ambos formatos actualizados.
        """
        logger.info("Iniciando sincronización de fuentes de datos")
        results = {}
        
        if symbols is None:
            # Obtener todos los símbolos de la base de datos
            symbols = self._get_available_symbols()
        
        for symbol in symbols:
            try:
                intervals = self._get_available_intervals(symbol)
                for interval in intervals:
                    # Verificar si necesita sincronización
                    if self._needs_sync(symbol, interval):
                        db_data = self.database.get_ohlc_data(symbol=symbol, interval=interval)
                        if db_data is not None and not db_data.empty:
                            self._export_to_csv(symbol, interval, db_data)
                            results[f"{symbol}_{interval}"] = "Sincronizado"
                        else:
                            results[f"{symbol}_{interval}"] = "Sin datos en DB"
                    else:
                        results[f"{symbol}_{interval}"] = "Ya sincronizado"
            except Exception as e:
                results[symbol] = f"Error: {str(e)}"
                logger.error(f"Error sincronizando {symbol}: {e}")
        
        logger.info(f"Sincronización completada: {len(results)} elementos procesados")
        return results
    
    def get_data_summary(self) -> Dict[str, any]:
        """
        Obtiene resumen del estado de ambas fuentes de datos.
        """
        summary = {
            "database": self._get_db_summary(),
            "csv_files": self._get_csv_summary(),
            "sync_status": self._get_sync_status(),
            "recommendations": self._get_recommendations()
        }
        return summary
    
    # Métodos privados de utilidad
    
    def _filter_by_time(self, df: pd.DataFrame, start_time: Optional[datetime],
                       end_time: Optional[datetime], limit: Optional[int]) -> pd.DataFrame:
        """Filtra DataFrame por tiempo y límite."""
        if df.empty:
            return df
        
        # Asegurar que datetime esté en formato correcto
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.sort_values('datetime')
        
        # Filtrar por tiempo
        if start_time:
            df = df[df['datetime'] >= start_time]
        if end_time:
            df = df[df['datetime'] <= end_time]
        
        # Aplicar límite
        if limit:
            df = df.tail(limit)
        
        return df
    
    def _get_data_from_csv(self, symbol: str, interval: str,
                          start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None,
                          limit: Optional[int] = None,
                          lookback_days: Optional[int] = None) -> pd.DataFrame:
        """Obtiene datos de archivo CSV."""
        csv_file = self.csv_dir / f"{symbol}_{interval}.csv"
        
        if not csv_file.exists():
            logger.debug(f"Archivo CSV no existe: {csv_file}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(csv_file)
            df['datetime'] = pd.to_datetime(df['datetime'])
            
            # Filtrar por lookback_days si se especifica
            if lookback_days:
                cutoff_date = datetime.now() - timedelta(days=lookback_days)
                df = df[df['datetime'] >= cutoff_date]
            
            return self._filter_by_time(df, start_time, end_time, limit)
        except Exception as e:
            logger.error(f"Error leyendo CSV {csv_file}: {e}")
            return pd.DataFrame()
    
    def _export_to_csv(self, symbol: str, interval: str, data: pd.DataFrame):
        """Exporta datos a archivo CSV."""
        if data.empty:
            return
        
        csv_file = self.csv_dir / f"{symbol}_{interval}.csv"
        try:
            data.to_csv(csv_file, index=False)
            logger.debug(f"Datos exportados a {csv_file}: {len(data)} registros")
            
            # Actualizar metadatos de sincronización
            self.sync_metadata[f"{symbol}_{interval}"] = {
                "last_sync": datetime.now(),
                "records": len(data),
                "file_path": str(csv_file)
            }
        except Exception as e:
            logger.error(f"Error exportando a CSV {csv_file}: {e}")
    
    def _prepare_for_ml(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara datos para análisis de Machine Learning."""
        if df.empty:
            return df
        
        # Normalizar nombres de columnas de base de datos a formato estándar
        column_mapping = {
            'timestamp': 'datetime',
            'open_price': 'open',
            'high_price': 'high', 
            'low_price': 'low',
            'close_price': 'close',
            'trades_count': 'trades_count'
        }
        
        # Renombrar columnas si existen y no hay duplicados
        for old_col, new_col in column_mapping.items():
            if old_col in df.columns and new_col not in df.columns:
                df = df.rename(columns={old_col: new_col})
        
        # Asegurar tipos de datos correctos
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Ordenar por fecha
        if 'datetime' in df.columns:
            df = df.sort_values('datetime').reset_index(drop=True)
        
        # Eliminar valores nulos
        df = df.dropna()
        
        return df
    
    def _update_cache_async(self, symbol: str, interval: str, data: pd.DataFrame):
        """Actualiza cache de manera asíncrona (no bloquea)."""
        try:
            # En una implementación real, esto sería asíncrono
            self.cache.cache_ohlc_data(symbol, interval, data)
        except Exception as e:
            logger.warning(f"Error actualizando cache: {e}")
    
    def _get_available_symbols(self) -> List[str]:
        """Obtiene símbolos disponibles en la base de datos."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("SELECT DISTINCT symbol FROM ohlc_data")
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error obteniendo símbolos: {e}")
            return []
    
    def _get_available_intervals(self, symbol: str) -> List[str]:
        """Obtiene intervalos disponibles para un símbolo."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    "SELECT DISTINCT interval FROM ohlc_data WHERE symbol = ?",
                    (symbol,)
                )
                return [row[0] for row in cursor.fetchall()]
        except Exception as e:
            logger.error(f"Error obteniendo intervalos para {symbol}: {e}")
            return []
    
    def _needs_sync(self, symbol: str, interval: str) -> bool:
        """Verifica si necesita sincronización entre DB y CSV."""
        key = f"{symbol}_{interval}"
        csv_file = self.csv_dir / f"{symbol}_{interval}.csv"
        
        # Si no existe CSV, necesita sincronización
        if not csv_file.exists():
            return True
        
        # Verificar metadatos de sincronización
        if key in self.sync_metadata:
            last_sync = self.sync_metadata[key]["last_sync"]
            # Sincronizar si han pasado más de 1 hora
            return (datetime.now() - last_sync) > timedelta(hours=1)
        
        return True
    
    def _get_db_summary(self) -> Dict[str, any]:
        """Obtiene resumen de la base de datos."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT symbol, interval, COUNT(*) as records,
                           MIN(datetime) as earliest, MAX(datetime) as latest
                    FROM ohlc_data 
                    GROUP BY symbol, interval
                """)
                results = cursor.fetchall()
                
                summary = {}
                for symbol, interval, records, earliest, latest in results:
                    key = f"{symbol}_{interval}"
                    summary[key] = {
                        "records": records,
                        "date_range": f"{earliest} to {latest}"
                    }
                return summary
        except Exception as e:
            logger.error(f"Error obteniendo resumen DB: {e}")
            return {}
    
    def _get_csv_summary(self) -> Dict[str, any]:
        """Obtiene resumen de archivos CSV."""
        summary = {}
        for csv_file in self.csv_dir.glob("*.csv"):
            if csv_file.name == "data_summary.txt":
                continue
            
            try:
                df = pd.read_csv(csv_file)
                key = csv_file.stem  # nombre sin extensión
                summary[key] = {
                    "records": len(df),
                    "file_size": f"{csv_file.stat().st_size / 1024:.1f} KB",
                    "last_modified": datetime.fromtimestamp(csv_file.stat().st_mtime)
                }
            except Exception as e:
                summary[csv_file.name] = f"Error: {str(e)}"
        
        return summary
    
    def _get_sync_status(self) -> Dict[str, str]:
        """Obtiene estado de sincronización."""
        status = {}
        symbols = self._get_available_symbols()
        
        for symbol in symbols:
            intervals = self._get_available_intervals(symbol)
            for interval in intervals:
                key = f"{symbol}_{interval}"
                csv_file = self.csv_dir / f"{symbol}_{interval}.csv"
                
                if csv_file.exists():
                    if self._needs_sync(symbol, interval):
                        status[key] = "Necesita sincronización"
                    else:
                        status[key] = "Sincronizado"
                else:
                    status[key] = "CSV faltante"
        
        return status
    
    def _get_recommendations(self) -> List[str]:
        """Obtiene recomendaciones de optimización."""
        recommendations = []
        
        # Verificar archivos CSV faltantes
        symbols = self._get_available_symbols()
        missing_csvs = 0
        for symbol in symbols:
            intervals = self._get_available_intervals(symbol)
            for interval in intervals:
                csv_file = self.csv_dir / f"{symbol}_{interval}.csv"
                if not csv_file.exists():
                    missing_csvs += 1
        
        if missing_csvs > 0:
            recommendations.append(
                f"Ejecutar sync_data_sources() para crear {missing_csvs} archivos CSV faltantes"
            )
        
        # Verificar tamaño de cache
        try:
            cache_size = len(self.cache.ohlc_cache) if hasattr(self.cache, 'ohlc_cache') else 0
            if cache_size == 0:
                recommendations.append("Considerar pre-cargar datos frecuentes en cache")
        except:
            pass
        
        # Verificar archivos CSV grandes
        large_files = []
        for csv_file in self.csv_dir.glob("*.csv"):
            if csv_file.stat().st_size > 1024 * 1024:  # > 1MB
                large_files.append(csv_file.name)
        
        if large_files:
            recommendations.append(
                f"Considerar particionado para archivos grandes: {', '.join(large_files)}"
            )
        
        if not recommendations:
            recommendations.append("Sistema optimizado - no se requieren acciones")
        
        return recommendations