"""Utilidades de validación para el dashboard."""

import pandas as pd
import numpy as np
from typing import Any, List, Dict, Optional, Union, Tuple
from datetime import datetime, timedelta
import re

class DataValidators:
    """Clase con utilidades de validación de datos."""
    
    @staticmethod
    def validate_symbol(symbol: str) -> Tuple[bool, str]:
        """Valida formato de símbolo de trading.
        
        Args:
            symbol: Símbolo a validar (ej: 'BTCUSDT')
            
        Returns:
            Tupla con (es_válido, mensaje_error)
        """
        if not isinstance(symbol, str):
            return False, "El símbolo debe ser una cadena de texto"
        
        symbol = symbol.upper().strip()
        
        if not symbol:
            return False, "El símbolo no puede estar vacío"
        
        if len(symbol) < 3:
            return False, "El símbolo debe tener al menos 3 caracteres"
        
        if len(symbol) > 20:
            return False, "El símbolo no puede tener más de 20 caracteres"
        
        if not re.match(r'^[A-Z0-9]+$', symbol):
            return False, "El símbolo solo puede contener letras mayúsculas y números"
        
        return True, ""
    
    @staticmethod
    def validate_interval(interval: str) -> Tuple[bool, str]:
        """Valida formato de intervalo de tiempo.
        
        Args:
            interval: Intervalo a validar (ej: '1h', '4h', '1d')
            
        Returns:
            Tupla con (es_válido, mensaje_error)
        """
        valid_intervals = {
            '1m', '3m', '5m', '15m', '30m',
            '1h', '2h', '4h', '6h', '8h', '12h',
            '1d', '3d', '1w', '1M'
        }
        
        if not isinstance(interval, str):
            return False, "El intervalo debe ser una cadena de texto"
        
        if interval not in valid_intervals:
            return False, f"Intervalo inválido. Válidos: {', '.join(sorted(valid_intervals))}"
        
        return True, ""
    
    @staticmethod
    def validate_date_range(start_date: Union[str, datetime], 
                           end_date: Union[str, datetime]) -> Tuple[bool, str]:
        """Valida rango de fechas.
        
        Args:
            start_date: Fecha de inicio
            end_date: Fecha de fin
            
        Returns:
            Tupla con (es_válido, mensaje_error)
        """
        try:
            # Convertir a datetime si son strings
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            # Validar que las fechas sean válidas
            if pd.isna(start_date) or pd.isna(end_date):
                return False, "Las fechas no pueden ser nulas"
            
            # Validar que start_date sea anterior a end_date
            if start_date >= end_date:
                return False, "La fecha de inicio debe ser anterior a la fecha de fin"
            
            # Validar que no sean fechas futuras
            now = datetime.now()
            if start_date > now or end_date > now:
                return False, "Las fechas no pueden ser futuras"
            
            # Validar rango máximo (ej: no más de 2 años)
            max_range = timedelta(days=730)  # 2 años
            if end_date - start_date > max_range:
                return False, "El rango de fechas no puede ser mayor a 2 años"
            
            return True, ""
            
        except Exception as e:
            return False, f"Error al validar fechas: {str(e)}"
    
    @staticmethod
    def validate_numeric_range(value: Union[int, float], 
                              min_val: Optional[float] = None,
                              max_val: Optional[float] = None,
                              field_name: str = "valor") -> Tuple[bool, str]:
        """Valida que un valor numérico esté en un rango.
        
        Args:
            value: Valor a validar
            min_val: Valor mínimo permitido
            max_val: Valor máximo permitido
            field_name: Nombre del campo para mensajes de error
            
        Returns:
            Tupla con (es_válido, mensaje_error)
        """
        if pd.isna(value) or value is None:
            return False, f"El {field_name} no puede ser nulo"
        
        try:
            value = float(value)
        except (ValueError, TypeError):
            return False, f"El {field_name} debe ser un número válido"
        
        if not np.isfinite(value):
            return False, f"El {field_name} debe ser un número finito"
        
        if min_val is not None and value < min_val:
            return False, f"El {field_name} debe ser mayor o igual a {min_val}"
        
        if max_val is not None and value > max_val:
            return False, f"El {field_name} debe ser menor o igual a {max_val}"
        
        return True, ""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, 
                          required_columns: Optional[List[str]] = None,
                          min_rows: int = 1) -> Tuple[bool, str]:
        """Valida estructura de DataFrame.
        
        Args:
            df: DataFrame a validar
            required_columns: Columnas requeridas
            min_rows: Número mínimo de filas
            
        Returns:
            Tupla con (es_válido, mensaje_error)
        """
        if not isinstance(df, pd.DataFrame):
            return False, "Los datos deben ser un DataFrame"
        
        if df.empty:
            return False, "El DataFrame no puede estar vacío"
        
        if len(df) < min_rows:
            return False, f"El DataFrame debe tener al menos {min_rows} filas"
        
        if required_columns:
            missing_columns = set(required_columns) - set(df.columns)
            if missing_columns:
                return False, f"Faltan columnas requeridas: {', '.join(missing_columns)}"
        
        return True, ""
    
    @staticmethod
    def validate_ohlc_data(df: pd.DataFrame) -> Tuple[bool, str]:
        """Valida datos OHLC específicamente.
        
        Args:
            df: DataFrame con datos OHLC
            
        Returns:
            Tupla con (es_válido, mensaje_error)
        """
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        # Validar estructura básica
        is_valid, error_msg = DataValidators.validate_dataframe(df, required_columns)
        if not is_valid:
            return False, error_msg
        
        # Validar que high >= low, open, close
        if not (df['high'] >= df[['open', 'low', 'close']].max(axis=1)).all():
            return False, "El precio máximo debe ser mayor o igual a open, low y close"
        
        # Validar que low <= high, open, close
        if not (df['low'] <= df[['open', 'high', 'close']].min(axis=1)).all():
            return False, "El precio mínimo debe ser menor o igual a open, high y close"
        
        # Validar que los precios sean positivos
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (df[col] <= 0).any():
                return False, f"Los precios en {col} deben ser positivos"
        
        # Validar que el volumen sea no negativo
        if (df['volume'] < 0).any():
            return False, "El volumen no puede ser negativo"
        
        return True, ""
    
    @staticmethod
    def validate_technical_indicators(df: pd.DataFrame, 
                                    indicators: List[str]) -> Tuple[bool, str]:
        """Valida que los indicadores técnicos estén presentes y sean válidos.
        
        Args:
            df: DataFrame con indicadores
            indicators: Lista de indicadores a validar
            
        Returns:
            Tupla con (es_válido, mensaje_error)
        """
        # Validar estructura básica
        is_valid, error_msg = DataValidators.validate_dataframe(df)
        if not is_valid:
            return False, error_msg
        
        # Verificar que los indicadores existan
        missing_indicators = set(indicators) - set(df.columns)
        if missing_indicators:
            return False, f"Faltan indicadores: {', '.join(missing_indicators)}"
        
        # Validar rangos específicos de indicadores
        for indicator in indicators:
            if indicator in df.columns:
                values = df[indicator].dropna()
                
                if indicator.upper() == 'RSI':
                    if not values.between(0, 100).all():
                        return False, "RSI debe estar entre 0 y 100"
                
                elif indicator.upper() in ['STOCH_K', 'STOCH_D']:
                    if not values.between(0, 100).all():
                        return False, f"{indicator} debe estar entre 0 y 100"
                
                elif 'VOLUME' in indicator.upper():
                    if (values < 0).any():
                        return False, f"{indicator} no puede ser negativo"
        
        return True, ""
    
    @staticmethod
    def validate_api_response(response: Dict[str, Any], 
                            required_fields: List[str]) -> Tuple[bool, str]:
        """Valida respuesta de API.
        
        Args:
            response: Diccionario con respuesta de API
            required_fields: Campos requeridos en la respuesta
            
        Returns:
            Tupla con (es_válido, mensaje_error)
        """
        if not isinstance(response, dict):
            return False, "La respuesta debe ser un diccionario"
        
        if not response:
            return False, "La respuesta no puede estar vacía"
        
        # Verificar campos requeridos
        missing_fields = set(required_fields) - set(response.keys())
        if missing_fields:
            return False, f"Faltan campos requeridos: {', '.join(missing_fields)}"
        
        # Verificar si hay errores en la respuesta
        if 'error' in response and response['error']:
            return False, f"Error en API: {response.get('error', 'Error desconocido')}"
        
        if 'code' in response and response['code'] != 200:
            return False, f"Código de error: {response['code']}"
        
        return True, ""

class InputSanitizers:
    """Clase con utilidades para sanitizar inputs del usuario."""
    
    @staticmethod
    def sanitize_symbol(symbol: str) -> str:
        """Sanitiza símbolo de trading.
        
        Args:
            symbol: Símbolo a sanitizar
            
        Returns:
            Símbolo sanitizado
        """
        if not isinstance(symbol, str):
            return ""
        
        # Convertir a mayúsculas y remover espacios
        symbol = symbol.upper().strip()
        
        # Remover caracteres no alfanuméricos
        symbol = re.sub(r'[^A-Z0-9]', '', symbol)
        
        return symbol
    
    @staticmethod
    def sanitize_numeric_input(value: Any, default: float = 0.0) -> float:
        """Sanitiza input numérico.
        
        Args:
            value: Valor a sanitizar
            default: Valor por defecto si no es válido
            
        Returns:
            Valor numérico sanitizado
        """
        try:
            if pd.isna(value) or value is None or value == "":
                return default
            
            # Convertir a float
            numeric_value = float(value)
            
            # Verificar que sea finito
            if not np.isfinite(numeric_value):
                return default
            
            return numeric_value
            
        except (ValueError, TypeError):
            return default
    
    @staticmethod
    def sanitize_date_input(date_input: Any) -> Optional[datetime]:
        """Sanitiza input de fecha.
        
        Args:
            date_input: Fecha a sanitizar
            
        Returns:
            Objeto datetime o None si no es válido
        """
        if pd.isna(date_input) or date_input is None or date_input == "":
            return None
        
        try:
            return pd.to_datetime(date_input).to_pydatetime()
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def sanitize_text_input(text: Any, max_length: int = 100) -> str:
        """Sanitiza input de texto.
        
        Args:
            text: Texto a sanitizar
            max_length: Longitud máxima permitida
            
        Returns:
            Texto sanitizado
        """
        if not isinstance(text, str):
            text = str(text) if text is not None else ""
        
        # Remover caracteres de control y espacios extra
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Truncar si es muy largo
        if len(text) > max_length:
            text = text[:max_length]
        
        return text
    
    @staticmethod
    def sanitize_list_input(list_input: Any, item_type: type = str) -> List[Any]:
        """Sanitiza input de lista.
        
        Args:
            list_input: Lista a sanitizar
            item_type: Tipo esperado de los elementos
            
        Returns:
            Lista sanitizada
        """
        if not isinstance(list_input, (list, tuple)):
            if list_input is None or list_input == "":
                return []
            else:
                list_input = [list_input]
        
        sanitized_list = []
        for item in list_input:
            try:
                if item_type == str:
                    sanitized_item = InputSanitizers.sanitize_text_input(item)
                    if sanitized_item:  # Solo agregar si no está vacío
                        sanitized_list.append(sanitized_item)
                elif item_type in (int, float):
                    sanitized_item = InputSanitizers.sanitize_numeric_input(item)
                    sanitized_list.append(sanitized_item)
                else:
                    sanitized_list.append(item_type(item))
            except (ValueError, TypeError):
                continue  # Omitir elementos inválidos
        
        return sanitized_list