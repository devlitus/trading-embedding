"""Servicio de datos para el dashboard."""

import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import streamlit as st

class DataService:
    """Servicio para manejar operaciones de datos del dashboard."""
    
    def __init__(self, data_manager):
        """Inicializa el servicio de datos.
        
        Args:
            data_manager: Instancia del DataManager
        """
        self.data_manager = data_manager
    
    @st.cache_data(ttl=300)  # Cache por 5 minutos
    def get_available_symbols(_self) -> List[str]:
        """Obtiene la lista de símbolos disponibles.
        
        Returns:
            Lista de símbolos disponibles
        """
        try:
            symbols = _self.data_manager.get_available_symbols()
            return symbols if symbols else []
        except Exception as e:
            st.error(f"Error obteniendo símbolos: {e}")
            return []
    
    @st.cache_data(ttl=60)  # Cache por 1 minuto
    def get_market_data(_self, symbol: str, interval: str, limit: int = 1000) -> pd.DataFrame:
        """Obtiene datos de mercado para un símbolo.
        
        Args:
            symbol: Símbolo del activo
            interval: Intervalo de tiempo
            limit: Número máximo de registros
            
        Returns:
            DataFrame con datos OHLCV
        """
        try:
            data = _self.data_manager.get_latest_data(symbol, interval, limit)
            return data if not data.empty else pd.DataFrame()
        except Exception as e:
            st.error(f"Error obteniendo datos de mercado: {e}")
            return pd.DataFrame()
    
    def get_real_time_data(self, symbol: str, interval: str) -> Optional[Dict[str, Any]]:
        """Obtiene datos en tiempo real para un símbolo.
        
        Args:
            symbol: Símbolo del activo
            interval: Intervalo de tiempo
            
        Returns:
            Diccionario con datos en tiempo real o None
        """
        try:
            # Obtener el último registro
            data = self.data_manager.get_latest_data(symbol, interval, limit=1)
            
            if data.empty:
                return None
            
            latest_row = data.iloc[-1]
            
            return {
                'symbol': symbol,
                'timestamp': data.index[-1],
                'open': latest_row['open'],
                'high': latest_row['high'],
                'low': latest_row['low'],
                'close': latest_row['close'],
                'volume': latest_row['volume'] if 'volume' in latest_row else 0
            }
            
        except Exception as e:
            st.error(f"Error obteniendo datos en tiempo real: {e}")
            return None
    
    def get_historical_data(self, symbol: str, interval: str, days: int = 30) -> pd.DataFrame:
        """Obtiene datos históricos para un período específico.
        
        Args:
            symbol: Símbolo del activo
            interval: Intervalo de tiempo
            days: Número de días hacia atrás
            
        Returns:
            DataFrame con datos históricos
        """
        try:
            # Calcular el número de registros aproximado
            intervals_per_day = {
                '1m': 1440, '3m': 480, '5m': 288, '15m': 96,
                '30m': 48, '1h': 24, '2h': 12, '4h': 6,
                '6h': 4, '8h': 3, '12h': 2, '1d': 1
            }
            
            limit = intervals_per_day.get(interval, 24) * days
            limit = min(limit, 1000)  # Limitar a 1000 registros máximo
            
            data = self.data_manager.get_latest_data(symbol, interval, limit)
            
            # Filtrar por fecha si es necesario
            if not data.empty:
                cutoff_date = datetime.now() - timedelta(days=days)
                data = data[data.index >= cutoff_date]
            
            return data
            
        except Exception as e:
            st.error(f"Error obteniendo datos históricos: {e}")
            return pd.DataFrame()
    
    def get_data_quality_report(self, symbol: str, interval: str) -> Dict[str, Any]:
        """Genera un reporte de calidad de datos.
        
        Args:
            symbol: Símbolo del activo
            interval: Intervalo de tiempo
            
        Returns:
            Diccionario con métricas de calidad
        """
        try:
            data = self.get_market_data(symbol, interval, 1000)
            
            if data.empty:
                return {'error': 'No hay datos disponibles'}
            
            # Calcular métricas de calidad
            report = {
                'total_records': len(data),
                'date_range': {
                    'start': data.index[0].strftime('%Y-%m-%d %H:%M:%S'),
                    'end': data.index[-1].strftime('%Y-%m-%d %H:%M:%S')
                },
                'missing_values': {
                    'total': data.isnull().sum().sum(),
                    'by_column': data.isnull().sum().to_dict()
                },
                'duplicates': data.duplicated().sum(),
                'data_completeness': ((len(data) - data.isnull().sum().sum()) / (len(data) * len(data.columns)) * 100)
            }
            
            # Verificar gaps temporales
            if len(data) > 1:
                time_diffs = data.index.to_series().diff().dropna()
                expected_interval = self._get_expected_interval(interval)
                gaps = (time_diffs > expected_interval * 1.5).sum()
                report['temporal_gaps'] = gaps
                
                # Estadísticas de precios
                report['price_stats'] = {
                    'min_price': data['low'].min(),
                    'max_price': data['high'].max(),
                    'avg_price': data['close'].mean(),
                    'price_volatility': data['close'].pct_change().std() * 100
                }
            
            return report
            
        except Exception as e:
            return {'error': f'Error generando reporte: {str(e)}'}
    
    def _get_expected_interval(self, interval: str) -> pd.Timedelta:
        """Convierte el intervalo string a Timedelta.
        
        Args:
            interval: Intervalo como string (ej: '1h', '5m')
            
        Returns:
            Timedelta correspondiente
        """
        interval_map = {
            '1m': pd.Timedelta(minutes=1),
            '3m': pd.Timedelta(minutes=3),
            '5m': pd.Timedelta(minutes=5),
            '15m': pd.Timedelta(minutes=15),
            '30m': pd.Timedelta(minutes=30),
            '1h': pd.Timedelta(hours=1),
            '2h': pd.Timedelta(hours=2),
            '4h': pd.Timedelta(hours=4),
            '6h': pd.Timedelta(hours=6),
            '8h': pd.Timedelta(hours=8),
            '12h': pd.Timedelta(hours=12),
            '1d': pd.Timedelta(days=1)
        }
        
        return interval_map.get(interval, pd.Timedelta(hours=1))
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """Obtiene información detallada de un símbolo.
        
        Args:
            symbol: Símbolo del activo
            
        Returns:
            Diccionario con información del símbolo
        """
        try:
            # Obtener datos recientes para análisis
            data = self.get_market_data(symbol, '1h', 24)  # Últimas 24 horas
            
            if data.empty:
                return {'error': 'No hay datos disponibles para este símbolo'}
            
            current_price = data['close'].iloc[-1]
            price_24h_ago = data['close'].iloc[0] if len(data) > 1 else current_price
            
            info = {
                'symbol': symbol,
                'current_price': current_price,
                'price_change_24h': current_price - price_24h_ago,
                'price_change_pct_24h': ((current_price - price_24h_ago) / price_24h_ago * 100) if price_24h_ago != 0 else 0,
                'high_24h': data['high'].max(),
                'low_24h': data['low'].min(),
                'volume_24h': data['volume'].sum() if 'volume' in data.columns else 0,
                'last_update': data.index[-1].strftime('%Y-%m-%d %H:%M:%S'),
                'data_points': len(data)
            }
            
            return info
            
        except Exception as e:
            return {'error': f'Error obteniendo información del símbolo: {str(e)}'}
    
    def get_market_overview(self, symbols: List[str]) -> pd.DataFrame:
        """Obtiene un resumen del mercado para múltiples símbolos.
        
        Args:
            symbols: Lista de símbolos
            
        Returns:
            DataFrame con resumen del mercado
        """
        overview_data = []
        
        for symbol in symbols[:10]:  # Limitar a 10 símbolos para performance
            try:
                info = self.get_symbol_info(symbol)
                if 'error' not in info:
                    overview_data.append(info)
            except Exception:
                continue  # Saltar símbolos con errores
        
        if overview_data:
            df = pd.DataFrame(overview_data)
            return df.set_index('symbol')
        else:
            return pd.DataFrame()