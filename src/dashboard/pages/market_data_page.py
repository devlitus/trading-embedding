"""Página de datos de mercado del dashboard."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from typing import Optional, List
from datetime import datetime, timedelta
from ..components import DashboardLayout, ChartComponents
from ..services import DataService
from ..utils import DataFormatters, DataValidators, InputSanitizers

class MarketDataPage:
    """Clase para la página de datos de mercado."""
    
    def __init__(self, data_service: Optional[DataService] = None):
        """Inicializa la página de datos de mercado.
        
        Args:
            data_service: Servicio de datos
        """
        self.data_service = data_service or DataService()
    
    def render(self) -> None:
        """Renderiza la página de datos de mercado."""
        st.title("📈 Datos de Mercado")
        st.markdown("---")
        
        # Controles de filtrado
        self._render_controls()
        
        # Obtener parámetros de la sesión
        symbol = st.session_state.get('selected_symbol', 'BTCUSDT')
        interval = st.session_state.get('selected_interval', '1h')
        
        # Mostrar datos del mercado
        self._render_market_overview()
        self._render_symbol_data(symbol, interval)
    
    def _render_controls(self) -> None:
        """Renderiza los controles de filtrado."""
        st.subheader("🎛️ Controles")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Selector de símbolo
            try:
                available_symbols = self.data_service.get_available_symbols()
                if not available_symbols:
                    available_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
            except Exception:
                available_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT']
            
            selected_symbol = st.selectbox(
                "📊 Símbolo",
                options=available_symbols,
                index=0 if 'BTCUSDT' in available_symbols else 0,
                key="symbol_selector"
            )
            
            # Sanitizar y validar símbolo
            selected_symbol = InputSanitizers.sanitize_symbol(selected_symbol)
            is_valid, error_msg = DataValidators.validate_symbol(selected_symbol)
            
            if not is_valid:
                st.error(f"Símbolo inválido: {error_msg}")
                selected_symbol = 'BTCUSDT'
            
            st.session_state['selected_symbol'] = selected_symbol
        
        with col2:
            # Selector de intervalo
            intervals = ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
            selected_interval = st.selectbox(
                "⏰ Intervalo",
                options=intervals,
                index=intervals.index('1h'),
                key="interval_selector"
            )
            
            # Validar intervalo
            is_valid, error_msg = DataValidators.validate_interval(selected_interval)
            if not is_valid:
                st.error(f"Intervalo inválido: {error_msg}")
                selected_interval = '1h'
            
            st.session_state['selected_interval'] = selected_interval
        
        with col3:
            # Selector de límite de datos
            limit_options = [50, 100, 200, 500, 1000]
            selected_limit = st.selectbox(
                "📊 Número de Velas",
                options=limit_options,
                index=limit_options.index(200),
                key="limit_selector"
            )
            
            st.session_state['selected_limit'] = selected_limit
        
        # Botón de actualización
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            if st.button("🔄 Actualizar Datos", key="refresh_data"):
                # Limpiar caché para forzar actualización
                st.cache_data.clear()
                st.rerun()
    
    def _render_market_overview(self) -> None:
        """Renderiza resumen general del mercado."""
        st.subheader("🌐 Resumen del Mercado")
        
        try:
            # Obtener datos de múltiples símbolos para overview
            overview_symbols = ['BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT']
            overview_data = self.data_service.get_market_overview(overview_symbols)
            
            if overview_data:
                # Mostrar métricas en columnas
                cols = st.columns(len(overview_data))
                
                for i, (symbol, data) in enumerate(overview_data.items()):
                    with cols[i]:
                        if data and not data.empty:
                            current_price = data['close'].iloc[-1]
                            prev_price = data['close'].iloc[-2] if len(data) > 1 else current_price
                            change = ((current_price - prev_price) / prev_price) * 100
                            
                            # Formatear valores
                            price_str = DataFormatters.format_currency(current_price)
                            change_str = DataFormatters.format_percentage(change)
                            
                            # Determinar color del cambio
                            delta_color = "normal" if change >= 0 else "inverse"
                            
                            st.metric(
                                label=symbol.replace('USDT', ''),
                                value=price_str,
                                delta=change_str,
                                delta_color=delta_color
                            )
                        else:
                            st.metric(
                                label=symbol.replace('USDT', ''),
                                value="N/A",
                                delta="N/A"
                            )
            else:
                st.info("📊 No hay datos de mercado disponibles")
                
        except Exception as e:
            st.error(f"Error al cargar resumen del mercado: {str(e)}")
    
    def _render_symbol_data(self, symbol: str, interval: str) -> None:
        """Renderiza datos específicos del símbolo.
        
        Args:
            symbol: Símbolo a mostrar
            interval: Intervalo de tiempo
        """
        st.subheader(f"📊 Datos de {symbol}")
        
        try:
            # Obtener límite de datos
            limit = st.session_state.get('selected_limit', 200)
            
            # Cargar datos
            with st.spinner(f"Cargando datos de {symbol}..."):
                data = self.data_service.get_market_data(symbol, interval, limit=limit)
            
            if data is None or data.empty:
                st.warning(f"⚠️ No hay datos disponibles para {symbol}")
                return
            
            # Validar datos OHLC
            is_valid, error_msg = DataValidators.validate_ohlc_data(data)
            if not is_valid:
                st.error(f"Datos inválidos: {error_msg}")
                return
            
            # Mostrar información básica
            self._render_data_info(data, symbol, interval)
            
            # Mostrar gráfico de velas
            self._render_candlestick_chart(data, symbol, interval)
            
            # Mostrar gráfico de volumen
            self._render_volume_chart(data, symbol)
            
            # Mostrar tabla de datos
            self._render_data_table(data)
            
        except Exception as e:
            st.error(f"Error al cargar datos de {symbol}: {str(e)}")
    
    def _render_data_info(self, data: pd.DataFrame, symbol: str, interval: str) -> None:
        """Renderiza información básica de los datos.
        
        Args:
            data: DataFrame con datos OHLC
            symbol: Símbolo
            interval: Intervalo
        """
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📊 Total de Registros",
                DataFormatters.format_number(len(data), decimals=0)
            )
        
        with col2:
            if not data.empty:
                latest_time = data.index[-1] if hasattr(data.index, 'to_pydatetime') else data.index[-1]
                time_ago = DataFormatters.format_time_ago(latest_time)
                st.metric("⏰ Última Actualización", time_ago)
            else:
                st.metric("⏰ Última Actualización", "N/A")
        
        with col3:
            if not data.empty:
                price_range = data['high'].max() - data['low'].min()
                st.metric(
                    "📈 Rango de Precios",
                    DataFormatters.format_currency(price_range)
                )
            else:
                st.metric("📈 Rango de Precios", "N/A")
        
        with col4:
            if not data.empty:
                avg_volume = data['volume'].mean()
                st.metric(
                    "📊 Volumen Promedio",
                    DataFormatters.format_volume(avg_volume)
                )
            else:
                st.metric("📊 Volumen Promedio", "N/A")
    
    def _render_candlestick_chart(self, data: pd.DataFrame, symbol: str, interval: str) -> None:
        """Renderiza gráfico de velas.
        
        Args:
            data: DataFrame con datos OHLC
            symbol: Símbolo
            interval: Intervalo
        """
        st.subheader(f"🕯️ Gráfico de Velas - {symbol} ({interval})")
        
        try:
            fig = ChartComponents.create_candlestick_chart(
                data, 
                title=f"{symbol} - {interval}",
                height=600
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No se pudo generar el gráfico de velas")
                
        except Exception as e:
            st.error(f"Error al generar gráfico de velas: {str(e)}")
    
    def _render_volume_chart(self, data: pd.DataFrame, symbol: str) -> None:
        """Renderiza gráfico de volumen.
        
        Args:
            data: DataFrame con datos OHLC
            symbol: Símbolo
        """
        st.subheader(f"📊 Volumen - {symbol}")
        
        try:
            fig = ChartComponents.create_volume_chart(
                data,
                title=f"Volumen - {symbol}",
                height=300
            )
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.error("No se pudo generar el gráfico de volumen")
                
        except Exception as e:
            st.error(f"Error al generar gráfico de volumen: {str(e)}")
    
    def _render_data_table(self, data: pd.DataFrame) -> None:
        """Renderiza tabla de datos.
        
        Args:
            data: DataFrame con datos OHLC
        """
        st.subheader("📋 Tabla de Datos")
        
        # Opciones de visualización
        col1, col2 = st.columns(2)
        
        with col1:
            show_all = st.checkbox("Mostrar todos los datos", value=False)
        
        with col2:
            rows_to_show = st.slider(
                "Filas a mostrar",
                min_value=10,
                max_value=min(100, len(data)),
                value=20
            ) if not show_all else len(data)
        
        # Preparar datos para mostrar
        display_data = data.tail(rows_to_show) if not show_all else data
        
        # Formatear datos para visualización
        formatted_data = DataFormatters.format_dataframe_for_display(
            display_data,
            currency_columns=['open', 'high', 'low', 'close'],
            numeric_columns=['volume']
        )
        
        # Mostrar tabla
        st.dataframe(
            formatted_data,
            use_container_width=True,
            height=400
        )
        
        # Opción de descarga
        if st.button("💾 Descargar Datos CSV"):
            csv = data.to_csv()
            st.download_button(
                label="📥 Descargar CSV",
                data=csv,
                file_name=f"{st.session_state.get('selected_symbol', 'data')}_{st.session_state.get('selected_interval', '1h')}.csv",
                mime="text/csv"
            )
    
    @staticmethod
    def show() -> None:
        """Método estático para mostrar la página."""
        page = MarketDataPage()
        page.render()

# Función de compatibilidad
def show_market_data_page() -> None:
    """Función de compatibilidad para mostrar la página de datos de mercado."""
    MarketDataPage.show()