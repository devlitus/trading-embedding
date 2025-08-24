#!/usr/bin/env python3
"""
Utilidades Comunes para Demos y Scripts

Este módulo consolida funciones comunes utilizadas en múltiples archivos de demostración
para evitar duplicación de código y mantener consistencia.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, Any


def setup_project_path(script_file: str) -> None:
    """
    Configura el path del proyecto para importar módulos src.
    
    Args:
        script_file: __file__ del script que llama a esta función
    """
    project_root = Path(script_file).parent
    src_path = project_root / "src"
    
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Configura el sistema de logging de manera estándar.
    
    Args:
        log_file: Nombre del archivo de log (opcional)
        level: Nivel de logging
        
    Returns:
        Logger configurado
    """
    handlers = [logging.StreamHandler()]
    
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    # Reducir verbosidad de librerías externas
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    
    return logging.getLogger(__name__)


def print_banner(title: str, width: int = 80) -> None:
    """
    Imprime un banner decorativo para títulos principales.
    
    Args:
        title: Título a mostrar
        width: Ancho del banner
    """
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_section(title: str, width: int = 60) -> None:
    """
    Imprime una sección con formato estándar.
    
    Args:
        title: Título de la sección
        width: Ancho de la línea
    """
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_subsection(title: str) -> None:
    """
    Imprime una subsección con formato.
    
    Args:
        title: Título de la subsección
    """
    print(f"\n--- {title} ---")


def generate_sample_ohlc_data(
    symbol: str = "BTCUSDT",
    periods: int = 200,
    interval: str = "1h",
    base_price: float = 45000.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Genera datos OHLC de muestra para pruebas y demostraciones.
    
    Args:
        symbol: Símbolo del activo
        periods: Número de períodos a generar
        interval: Intervalo de tiempo
        base_price: Precio base inicial
        seed: Semilla para reproducibilidad
        
    Returns:
        DataFrame con datos OHLC sintéticos
    """
    np.random.seed(seed)
    
    # Generar fechas según el intervalo
    if interval == "1h":
        freq = "1H"
    elif interval == "4h":
        freq = "4H"
    elif interval == "1d":
        freq = "1D"
    else:
        freq = "1H"  # Default
    
    end_date = datetime.now()
    start_date = end_date - timedelta(hours=periods if "h" in interval else periods * 24)
    dates = pd.date_range(start=start_date, end=end_date, periods=periods)
    
    # Generar datos con diferentes fases de mercado
    data = []
    current_price = base_price
    
    for i, timestamp in enumerate(dates):
        # Diferentes fases del mercado para mayor realismo
        if i < periods * 0.3:  # Fase alcista
            trend = 0.15
            volatility = 0.02
        elif i < periods * 0.6:  # Consolidación
            trend = 0.02
            volatility = 0.015
        elif i < periods * 0.8:  # Fase bajista
            trend = -0.1
            volatility = 0.025
        else:  # Recuperación
            trend = 0.08
            volatility = 0.02
        
        # Calcular cambio de precio
        price_change = np.random.normal(trend / 100, volatility)
        current_price *= (1 + price_change)
        
        # Generar OHLC realista
        daily_volatility = abs(np.random.normal(0, volatility / 2))
        high = current_price * (1 + daily_volatility)
        low = current_price * (1 - daily_volatility)
        
        # Asegurar que open y close estén dentro del rango
        open_price = np.random.uniform(low, high)
        close_price = current_price
        
        # Volumen sintético
        base_volume = 1000000
        volume_multiplier = 1 + abs(price_change) * 10  # Mayor volumen con mayor volatilidad
        volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 2.0))
        
        data.append({
            'timestamp': timestamp,
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close_price, 2),
            'volume': volume,
            'symbol': symbol,
            'interval': interval
        })
    
    df = pd.DataFrame(data)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    return df


def format_execution_time(seconds: float) -> str:
    """
    Formatea el tiempo de ejecución de manera legible.
    
    Args:
        seconds: Tiempo en segundos
        
    Returns:
        Tiempo formateado como string
    """
    if seconds < 1:
        return f"{seconds*1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.1f}s"


def format_file_size(size_bytes: int) -> str:
    """
    Formatea el tamaño de archivo de manera legible.
    
    Args:
        size_bytes: Tamaño en bytes
        
    Returns:
        Tamaño formateado como string
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def create_demo_summary(results: Dict[str, Any]) -> str:
    """
    Crea un resumen formateado de los resultados de una demostración.
    
    Args:
        results: Diccionario con resultados de la demo
        
    Returns:
        Resumen formateado como string
    """
    summary_lines = []
    summary_lines.append("\n📊 RESUMEN DE LA DEMOSTRACIÓN")
    summary_lines.append("=" * 50)
    
    for key, value in results.items():
        if isinstance(value, dict):
            summary_lines.append(f"\n{key.upper()}:")
            for sub_key, sub_value in value.items():
                summary_lines.append(f"  • {sub_key}: {sub_value}")
        else:
            summary_lines.append(f"• {key}: {value}")
    
    return "\n".join(summary_lines)


def validate_data_quality(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Valida la calidad de un DataFrame de datos OHLC.
    
    Args:
        df: DataFrame con datos OHLC
        
    Returns:
        Diccionario con métricas de calidad
    """
    quality_metrics = {
        'total_records': len(df),
        'missing_values': df.isnull().sum().sum(),
        'duplicate_timestamps': df.duplicated(subset=['timestamp']).sum() if 'timestamp' in df.columns else 0,
        'data_gaps': 0,  # Se calcularía comparando timestamps consecutivos
        'price_anomalies': 0  # Se calcularía detectando cambios extremos
    }
    
    # Calcular anomalías de precio si existen columnas OHLC
    if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
        # Detectar velas con high < low (imposible)
        invalid_candles = (df['high'] < df['low']).sum()
        quality_metrics['invalid_candles'] = invalid_candles
        
        # Detectar cambios de precio extremos (>50%)
        if len(df) > 1:
            price_changes = df['close'].pct_change().abs()
            extreme_changes = (price_changes > 0.5).sum()
            quality_metrics['extreme_price_changes'] = extreme_changes
    
    return quality_metrics