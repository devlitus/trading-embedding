import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BinanceClient:
    """
    Cliente para obtener datos OHLC de la API pública de Binance
    """
    
    def __init__(self, base_url: str = "https://api.binance.com"):
        self.base_url = base_url
        self.session = requests.Session()
        
    def get_klines(self, 
                   symbol: str, 
                   interval: str = '1h', 
                   start_time: Optional[str] = None,
                   end_time: Optional[str] = None,
                   limit: int = 1000) -> List[Dict[str, Any]]:
        """
        Obtiene datos OHLC (klines) de Binance
        
        Args:
            symbol: Par de trading (ej: 'BTCUSDT')
            interval: Intervalo de tiempo ('1m', '5m', '15m', '1h', '4h', '1d')
            start_time: Timestamp de inicio (opcional)
            end_time: Timestamp de fin (opcional)
            limit: Número máximo de registros (máx 1000)
            
        Returns:
            Lista de diccionarios con datos OHLC
        """
        endpoint = f"{self.base_url}/api/v3/klines"
        
        params = {
            'symbol': symbol.upper(),
            'interval': interval,
            'limit': limit
        }
        
        if start_time:
            params['startTime'] = self._to_timestamp(start_time)
        if end_time:
            params['endTime'] = self._to_timestamp(end_time)
            
        try:
            response = self.session.get(endpoint, params=params)
            response.raise_for_status()
            
            raw_data = response.json()
            
            # Convertir a formato más legible
            formatted_data = []
            for kline in raw_data:
                formatted_data.append({
                    'timestamp': int(kline[0]),
                    'open': float(kline[1]),
                    'high': float(kline[2]),
                    'low': float(kline[3]),
                    'close': float(kline[4]),
                    'volume': float(kline[5]),
                    'close_time': int(kline[6]),
                    'quote_volume': float(kline[7]),
                    'trades_count': int(kline[8]),
                    'taker_buy_base_volume': float(kline[9]),
                    'taker_buy_quote_volume': float(kline[10])
                })
                
            logger.info(f"Obtenidos {len(formatted_data)} registros para {symbol}")
            return formatted_data
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al obtener datos de Binance: {e}")
            raise
            
    def get_historical_data(self, 
                           symbol: str, 
                           interval: str = '1h',
                           days_back: int = 30) -> pd.DataFrame:
        """
        Obtiene datos históricos de los últimos N días
        
        Args:
            symbol: Par de trading
            interval: Intervalo de tiempo
            days_back: Días hacia atrás desde hoy
            
        Returns:
            DataFrame con datos OHLC
        """
        end_time = datetime.now()
        start_time = end_time - timedelta(days=days_back)
        
        all_data = []
        current_start = start_time
        
        # Binance limita a 1000 registros por request
        while current_start < end_time:
            current_end = min(current_start + timedelta(hours=1000), end_time)
            
            data = self.get_klines(
                symbol=symbol,
                interval=interval,
                start_time=current_start.isoformat(),
                end_time=current_end.isoformat()
            )
            
            all_data.extend(data)
            current_start = current_end
            
            # Rate limiting
            time.sleep(0.1)
            
        # Convertir a DataFrame
        df = pd.DataFrame(all_data)
        if not df.empty:
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('datetime', inplace=True)
            df.sort_index(inplace=True)
            
        return df
    
    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        """
        Obtiene información del símbolo
        """
        endpoint = f"{self.base_url}/api/v3/exchangeInfo"
        
        try:
            response = self.session.get(endpoint)
            response.raise_for_status()
            
            data = response.json()
            
            for symbol_info in data['symbols']:
                if symbol_info['symbol'] == symbol.upper():
                    return symbol_info
                    
            raise ValueError(f"Símbolo {symbol} no encontrado")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al obtener info del símbolo: {e}")
            raise
    
    def _to_timestamp(self, date_str: str) -> int:
        """
        Convierte string de fecha a timestamp de milisegundos
        """
        if isinstance(date_str, str):
            dt = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            return int(dt.timestamp() * 1000)
        return date_str
    
    def get_available_symbols(self) -> List[str]:
        """
        Obtiene lista de símbolos disponibles
        """
        endpoint = f"{self.base_url}/api/v3/exchangeInfo"
        
        try:
            response = self.session.get(endpoint)
            response.raise_for_status()
            
            data = response.json()
            symbols = [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING']
            
            return symbols
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Error al obtener símbolos: {e}")
            raise

# Ejemplo de uso
if __name__ == "__main__":
    client = BinanceClient()
    
    # Obtener datos de BTC/USDT
    try:
        df = client.get_historical_data('BTCUSDT', interval='1h', days_back=7)
        print(f"Datos obtenidos: {len(df)} registros")
        print(df.head())
        
    except Exception as e:
        print(f"Error: {e}")