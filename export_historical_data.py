#!/usr/bin/env python3
"""
Script para exportar datos históricos de la base de datos a archivos CSV
"""

import os
import pandas as pd
from src.data.database import TradingDatabase
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def export_historical_data():
    """
    Exporta todos los datos históricos de la base de datos a archivos CSV
    """
    # Inicializar base de datos
    db = TradingDatabase('data/trading.db')
    
    # Crear directorio de datos históricos si no existe
    historical_dir = Path('data/historical')
    historical_dir.mkdir(parents=True, exist_ok=True)
    
    # Obtener símbolos disponibles
    symbols = db.get_available_symbols()
    logger.info(f"Símbolos encontrados: {symbols}")
    
    if not symbols:
        logger.warning("No se encontraron símbolos en la base de datos")
        return
    
    # Exportar datos para cada símbolo
    for symbol in symbols:
        try:
            # Obtener intervalos disponibles para este símbolo
            intervals = db.get_available_intervals(symbol)
            logger.info(f"Procesando {symbol} con intervalos: {intervals}")
            
            for interval in intervals:
                # Obtener datos OHLC
                df = db.get_ohlc_data(symbol, interval)
                
                if not df.empty:
                    # Crear nombre de archivo
                    filename = f"{symbol}_{interval}.csv"
                    filepath = historical_dir / filename
                    
                    # Agregar columna de datetime legible
                    df_export = df.copy()
                    df_export['datetime'] = pd.to_datetime(df_export['timestamp'], unit='ms')
                    
                    # Reordenar columnas
                    columns_order = ['datetime', 'timestamp', 'open_price', 'high_price', 
                                   'low_price', 'close_price', 'volume', 'quote_volume', 'trades_count']
                    df_export = df_export.reindex(columns=[col for col in columns_order if col in df_export.columns])
                    
                    # Exportar a CSV
                    df_export.to_csv(filepath, index=False)
                    logger.info(f"Exportado {len(df_export)} registros de {symbol} ({interval}) a {filepath}")
                    
                    # Mostrar estadísticas
                    date_range = db.get_data_range(symbol, interval)
                    if date_range.get('total_records', 0) > 0:
                        logger.info(f"  Rango: {date_range['min_datetime']} - {date_range['max_datetime']}")
                        logger.info(f"  Total registros: {date_range['total_records']}")
                else:
                    logger.warning(f"No hay datos para {symbol} ({interval})")
                    
        except Exception as e:
            logger.error(f"Error procesando {symbol}: {e}")
    
    # Crear archivo de resumen
    summary_file = historical_dir / 'data_summary.txt'
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DE DATOS HISTÓRICOS\n")
        f.write("=" * 50 + "\n\n")
        
        for symbol in symbols:
            f.write(f"Símbolo: {symbol}\n")
            try:
                intervals = db.get_available_intervals(symbol)
                for interval in intervals:
                    date_range = db.get_data_range(symbol, interval)
                    if date_range.get('total_records', 0) > 0:
                        f.write(f"  {interval}: {date_range['total_records']} registros\n")
                        f.write(f"    Desde: {date_range['min_datetime']}\n")
                        f.write(f"    Hasta: {date_range['max_datetime']}\n")
                f.write("\n")
            except Exception as e:
                f.write(f"  Error: {e}\n\n")
    
    logger.info(f"Resumen guardado en {summary_file}")
    logger.info("Exportación completada")

if __name__ == "__main__":
    export_historical_data()