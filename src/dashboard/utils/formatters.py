"""Utilidades de formateo para el dashboard."""

import pandas as pd
import numpy as np
from typing import Any, Union, Optional
from datetime import datetime, timedelta

class DataFormatters:
    """Clase con utilidades de formateo de datos."""
    
    @staticmethod
    def format_currency(value: Union[float, int], currency: str = "USD", decimals: int = 2) -> str:
        """Formatea un valor como moneda.
        
        Args:
            value: Valor numérico
            currency: Código de moneda
            decimals: Número de decimales
            
        Returns:
            String formateado como moneda
        """
        if pd.isna(value) or value is None:
            return "N/A"
        
        try:
            if currency == "USD":
                return f"${value:,.{decimals}f}"
            elif currency == "BTC":
                return f"{value:.{decimals}f} BTC"
            elif currency == "ETH":
                return f"{value:.{decimals}f} ETH"
            else:
                return f"{value:,.{decimals}f} {currency}"
        except (ValueError, TypeError):
            return str(value)
    
    @staticmethod
    def format_percentage(value: Union[float, int], decimals: int = 2, show_sign: bool = True) -> str:
        """Formatea un valor como porcentaje.
        
        Args:
            value: Valor numérico
            decimals: Número de decimales
            show_sign: Si mostrar el signo + para valores positivos
            
        Returns:
            String formateado como porcentaje
        """
        if pd.isna(value) or value is None:
            return "N/A"
        
        try:
            sign = "+" if value > 0 and show_sign else ""
            return f"{sign}{value:.{decimals}f}%"
        except (ValueError, TypeError):
            return str(value)
    
    @staticmethod
    def format_volume(value: Union[float, int]) -> str:
        """Formatea un valor de volumen con sufijos K, M, B.
        
        Args:
            value: Valor numérico del volumen
            
        Returns:
            String formateado con sufijos
        """
        if pd.isna(value) or value is None:
            return "N/A"
        
        try:
            value = float(value)
            if value >= 1_000_000_000:
                return f"{value / 1_000_000_000:.2f}B"
            elif value >= 1_000_000:
                return f"{value / 1_000_000:.2f}M"
            elif value >= 1_000:
                return f"{value / 1_000:.2f}K"
            else:
                return f"{value:.2f}"
        except (ValueError, TypeError):
            return str(value)
    
    @staticmethod
    def format_datetime(dt: Union[datetime, pd.Timestamp, str], format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
        """Formatea una fecha/hora.
        
        Args:
            dt: Objeto datetime, Timestamp o string
            format_str: Formato de salida
            
        Returns:
            String formateado de fecha/hora
        """
        if pd.isna(dt) or dt is None:
            return "N/A"
        
        try:
            if isinstance(dt, str):
                dt = pd.to_datetime(dt)
            elif isinstance(dt, pd.Timestamp):
                dt = dt.to_pydatetime()
            
            return dt.strftime(format_str)
        except (ValueError, TypeError, AttributeError):
            return str(dt)
    
    @staticmethod
    def format_time_ago(dt: Union[datetime, pd.Timestamp]) -> str:
        """Formatea una fecha como 'tiempo transcurrido'.
        
        Args:
            dt: Objeto datetime o Timestamp
            
        Returns:
            String con tiempo transcurrido (ej: '2 horas ago')
        """
        if pd.isna(dt) or dt is None:
            return "N/A"
        
        try:
            if isinstance(dt, pd.Timestamp):
                dt = dt.to_pydatetime()
            
            now = datetime.now()
            if dt.tzinfo is not None:
                # Si dt tiene timezone, convertir now al mismo timezone
                now = now.replace(tzinfo=dt.tzinfo)
            
            diff = now - dt
            
            if diff.days > 0:
                return f"{diff.days} día{'s' if diff.days != 1 else ''} ago"
            elif diff.seconds >= 3600:
                hours = diff.seconds // 3600
                return f"{hours} hora{'s' if hours != 1 else ''} ago"
            elif diff.seconds >= 60:
                minutes = diff.seconds // 60
                return f"{minutes} minuto{'s' if minutes != 1 else ''} ago"
            else:
                return "Hace unos segundos"
                
        except (ValueError, TypeError, AttributeError):
            return str(dt)
    
    @staticmethod
    def format_number(value: Union[float, int], decimals: int = 2, thousands_sep: bool = True) -> str:
        """Formatea un número con separadores de miles.
        
        Args:
            value: Valor numérico
            decimals: Número de decimales
            thousands_sep: Si usar separador de miles
            
        Returns:
            String formateado
        """
        if pd.isna(value) or value is None:
            return "N/A"
        
        try:
            if thousands_sep:
                return f"{value:,.{decimals}f}"
            else:
                return f"{value:.{decimals}f}"
        except (ValueError, TypeError):
            return str(value)
    
    @staticmethod
    def format_change_indicator(value: Union[float, int], use_arrows: bool = True) -> tuple[str, str]:
        """Formatea un indicador de cambio con color y símbolo.
        
        Args:
            value: Valor del cambio
            use_arrows: Si usar flechas en lugar de +/-
            
        Returns:
            Tupla con (símbolo, color)
        """
        if pd.isna(value) or value is None:
            return "○", "gray"
        
        try:
            value = float(value)
            if value > 0:
                symbol = "↗" if use_arrows else "+"
                color = "green"
            elif value < 0:
                symbol = "↘" if use_arrows else "-"
                color = "red"
            else:
                symbol = "→" if use_arrows else "="
                color = "gray"
            
            return symbol, color
        except (ValueError, TypeError):
            return "?", "gray"
    
    @staticmethod
    def truncate_text(text: str, max_length: int = 50, suffix: str = "...") -> str:
        """Trunca texto si excede la longitud máxima.
        
        Args:
            text: Texto a truncar
            max_length: Longitud máxima
            suffix: Sufijo para texto truncado
            
        Returns:
            Texto truncado si es necesario
        """
        if not isinstance(text, str):
            text = str(text)
        
        if len(text) <= max_length:
            return text
        
        return text[:max_length - len(suffix)] + suffix
    
    @staticmethod
    def format_dataframe_for_display(df: pd.DataFrame, 
                                   numeric_columns: Optional[list] = None,
                                   percentage_columns: Optional[list] = None,
                                   currency_columns: Optional[list] = None,
                                   datetime_columns: Optional[list] = None) -> pd.DataFrame:
        """Formatea un DataFrame para mostrar en el dashboard.
        
        Args:
            df: DataFrame a formatear
            numeric_columns: Columnas numéricas a formatear
            percentage_columns: Columnas de porcentaje
            currency_columns: Columnas de moneda
            datetime_columns: Columnas de fecha/hora
            
        Returns:
            DataFrame formateado
        """
        if df.empty:
            return df
        
        df_formatted = df.copy()
        
        # Formatear columnas numéricas
        if numeric_columns:
            for col in numeric_columns:
                if col in df_formatted.columns:
                    df_formatted[col] = df_formatted[col].apply(
                        lambda x: DataFormatters.format_number(x)
                    )
        
        # Formatear columnas de porcentaje
        if percentage_columns:
            for col in percentage_columns:
                if col in df_formatted.columns:
                    df_formatted[col] = df_formatted[col].apply(
                        lambda x: DataFormatters.format_percentage(x)
                    )
        
        # Formatear columnas de moneda
        if currency_columns:
            for col in currency_columns:
                if col in df_formatted.columns:
                    df_formatted[col] = df_formatted[col].apply(
                        lambda x: DataFormatters.format_currency(x)
                    )
        
        # Formatear columnas de fecha/hora
        if datetime_columns:
            for col in datetime_columns:
                if col in df_formatted.columns:
                    df_formatted[col] = df_formatted[col].apply(
                        lambda x: DataFormatters.format_datetime(x)
                    )
        
        return df_formatted

class ColorUtils:
    """Utilidades para manejo de colores en el dashboard."""
    
    # Paleta de colores para gráficos
    COLORS = {
        'primary': '#1f77b4',
        'success': '#2ca02c',
        'danger': '#d62728',
        'warning': '#ff7f0e',
        'info': '#17a2b8',
        'light': '#f8f9fa',
        'dark': '#343a40',
        'bullish': '#00c851',
        'bearish': '#ff4444',
        'neutral': '#6c757d'
    }
    
    @staticmethod
    def get_trend_color(value: Union[float, int]) -> str:
        """Obtiene color basado en tendencia (positivo/negativo).
        
        Args:
            value: Valor numérico
            
        Returns:
            Código de color hexadecimal
        """
        if pd.isna(value) or value is None:
            return ColorUtils.COLORS['neutral']
        
        try:
            value = float(value)
            if value > 0:
                return ColorUtils.COLORS['bullish']
            elif value < 0:
                return ColorUtils.COLORS['bearish']
            else:
                return ColorUtils.COLORS['neutral']
        except (ValueError, TypeError):
            return ColorUtils.COLORS['neutral']
    
    @staticmethod
    def get_gradient_color(value: float, min_val: float, max_val: float, 
                          start_color: str = '#ff4444', end_color: str = '#00c851') -> str:
        """Obtiene color en gradiente basado en valor.
        
        Args:
            value: Valor actual
            min_val: Valor mínimo del rango
            max_val: Valor máximo del rango
            start_color: Color para valor mínimo
            end_color: Color para valor máximo
            
        Returns:
            Código de color hexadecimal
        """
        if pd.isna(value) or min_val == max_val:
            return ColorUtils.COLORS['neutral']
        
        # Normalizar valor entre 0 y 1
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0, min(1, normalized))  # Clamp entre 0 y 1
        
        # Convertir colores hex a RGB
        def hex_to_rgb(hex_color):
            hex_color = hex_color.lstrip('#')
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        
        def rgb_to_hex(rgb):
            return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"
        
        start_rgb = hex_to_rgb(start_color)
        end_rgb = hex_to_rgb(end_color)
        
        # Interpolar entre colores
        interpolated_rgb = tuple(
            int(start_rgb[i] + (end_rgb[i] - start_rgb[i]) * normalized)
            for i in range(3)
        )
        
        return rgb_to_hex(interpolated_rgb)