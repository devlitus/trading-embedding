#!/usr/bin/env python3
"""
Interfaz de usuario para reportes.

Este módulo contiene todas las funciones de UI para mostrar reportes,
métricas y configuraciones de exportación.
"""

import streamlit as st
import pandas as pd
from datetime import datetime
from typing import Dict, Optional

from .reports_config import ReportsConfig
from .reports_logic import ReportGenerator
from .reports_charts import ReportsCharts


class ReportsUI:
    """Interfaz de usuario para reportes"""
    
    def __init__(self, data_manager=None):
        self.config = ReportsConfig()
        self.charts = ReportsCharts()
        self.data_manager = data_manager
        if data_manager:
            self.report_generator = ReportGenerator(data_manager)
    
    def show_page_header(self) -> None:
        """Muestra el encabezado de la página"""
        st.title("📋 Reportes Avanzados")
        st.markdown(
            "Genera reportes detallados y análisis profundos sobre el rendimiento "
            "y comportamiento de los instrumentos financieros."
        )
    
    def show_report_configuration(self, data_manager) -> tuple:
        """Muestra la configuración del reporte y retorna los valores seleccionados"""
        st.subheader("⚙️ Configuración del Reporte")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            report_type = st.selectbox(
                "Tipo de Reporte",
                self.config.REPORT_TYPES,
                help="Selecciona el tipo de análisis que deseas generar"
            )
        
        with col2:
            symbol_for_report = None
            if report_type != "📋 Estado del Sistema":
                try:
                    available_symbols = data_manager.database.get_available_symbols()
                    if not available_symbols:
                        st.warning("⚠️ No hay símbolos disponibles en la base de datos")
                        return None, None, None, None
                    
                    symbol_for_report = st.selectbox(
                        "Símbolo",
                        available_symbols,
                        index=0,
                        help="Selecciona el símbolo para analizar"
                    )
                except Exception as e:
                    st.error(f"❌ Error obteniendo símbolos: {e}")
                    return None, None, None, None
            else:
                st.info("No se requiere símbolo para el reporte del sistema")
        
        with col3:
            date_range = None
            if report_type != "📋 Estado del Sistema":
                date_range = st.selectbox(
                    "Rango de Fechas",
                    self.config.DATE_RANGES,
                    index=1,
                    help="Selecciona el período de análisis"
                )
            else:
                st.info("Análisis en tiempo real")
        
        # Opciones de exportación
        with st.expander("📥 Opciones de Exportación", expanded=False):
            export_format = st.selectbox(
                "Formato de Exportación",
                self.config.EXPORT_FORMATS,
                help="Selecciona cómo deseas exportar los datos"
            )
        
        return report_type, symbol_for_report, date_range, export_format
    
    def show_performance_report(self, report_generator: ReportGenerator, 
                              symbol: str, date_range: str) -> None:
        """Muestra el reporte de rendimiento"""
        st.subheader(f"📈 Reporte de Rendimiento - {symbol}")
        
        # Obtener datos
        df = report_generator.get_data_for_report(symbol, date_range)
        if df is None:
            return
        
        # Calcular métricas
        metrics = report_generator.calculate_performance_metrics(df)
        if not metrics:
            st.error("No se pudieron calcular las métricas")
            return
        
        # Mostrar métricas principales
        self._show_performance_metrics(metrics)
        
        # Mostrar gráfico
        self.charts.create_performance_chart(df, symbol, metrics['initial_price'])
        
        # Mostrar estadísticas detalladas
        self._show_detailed_performance_stats(df, metrics)
    
    def show_technical_analysis_report(self, report_generator: ReportGenerator,
                                     symbol: str, date_range: str) -> None:
        """Muestra el reporte de análisis técnico"""
        st.subheader(f"📊 Análisis Técnico - {symbol}")
        
        # Obtener datos
        df = report_generator.get_data_for_report(symbol, date_range)
        if df is None:
            return
        
        # Calcular indicadores técnicos
        df_with_indicators = report_generator.calculate_technical_indicators(df)
        
        # Mostrar gráfico técnico
        self.charts.create_technical_analysis_chart(df_with_indicators, symbol)
        
        # Analizar señales
        signals = report_generator.analyze_trading_signals(df_with_indicators)
        if signals:
            self._show_trading_signals(signals)
            self.charts.create_signals_distribution_chart(signals)
    
    def show_volatility_report(self, report_generator: ReportGenerator,
                             symbol: str, date_range: str) -> None:
        """Muestra el reporte de volatilidad"""
        st.subheader(f"📉 Análisis de Volatilidad - {symbol}")
        
        # Obtener datos
        df = report_generator.get_data_for_report(symbol, date_range)
        if df is None:
            return
        
        # Calcular volatilidad
        rolling_vol, vol_metrics = report_generator.calculate_volatility_metrics(df)
        
        # Mostrar métricas de volatilidad
        self._show_volatility_metrics(vol_metrics)
        
        # Mostrar gráfico
        self.charts.create_volatility_chart(df, rolling_vol, symbol)
    
    def show_system_report(self, report_generator: ReportGenerator) -> None:
        """Muestra el reporte del sistema"""
        st.subheader("📋 Estado del Sistema")
        
        # Obtener estado del sistema
        health = report_generator.get_system_health()
        
        # Mostrar métricas del sistema
        self._show_system_metrics(health)
        
        # Mostrar gráfico de estado
        self.charts.create_system_metrics_chart(health)
        
        # Mostrar información del entorno
        self._show_environment_info()
    
    def handle_data_export(self, report_generator: ReportGenerator, symbol: str,
                          date_range: str, export_format: str) -> None:
        """Maneja la exportación de datos"""
        if export_format == "Ver en pantalla":
            return
        
        try:
            df = report_generator.get_data_for_report(symbol, date_range)
            if df is None:
                return
            
            st.subheader("📥 Exportar Datos")
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if export_format == "Descargar CSV":
                csv_data = df.to_csv(index=True)
                st.download_button(
                    label="📥 Descargar CSV",
                    data=csv_data,
                    file_name=f"{symbol}_{date_range.replace(' ', '_')}_{timestamp}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            elif export_format == "Descargar JSON":
                json_data = df.to_json(orient='records', date_format='iso', indent=2)
                st.download_button(
                    label="📥 Descargar JSON",
                    data=json_data,
                    file_name=f"{symbol}_{date_range.replace(' ', '_')}_{timestamp}.json",
                    mime="application/json",
                    use_container_width=True
                )
        
        except Exception as e:
            st.error(f"❌ Error en la exportación: {e}")
    
    def _show_performance_metrics(self, metrics: Dict) -> None:
        """Muestra métricas de rendimiento"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Rendimiento Total",
                f"{metrics['total_return']:.2f}%",
                delta=f"{metrics['total_return']:.2f}%"
            )
        
        with col2:
            st.metric(
                "Precio Inicial",
                f"${metrics['initial_price']:.4f}"
            )
        
        with col3:
            st.metric(
                "Precio Final",
                f"${metrics['final_price']:.4f}"
            )
        
        with col4:
            st.metric(
                "Volatilidad",
                f"{metrics['volatility']:.2f}%"
            )
    
    def _show_detailed_performance_stats(self, df: pd.DataFrame, metrics: Dict) -> None:
        """Muestra estadísticas detalladas de rendimiento"""
        with st.expander("📊 Estadísticas Detalladas", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Métricas de Riesgo:**")
                st.write(f"• Máximo Drawdown: {metrics['max_drawdown']:.2f}%")
                st.write(f"• Ratio de Sharpe: {metrics['sharpe_ratio']:.2f}")
                st.write(f"• Volatilidad Anualizada: {metrics['volatility']:.2f}%")
            
            with col2:
                st.write("**Métricas de Precio:**")
                st.write(f"• Precio Máximo: ${metrics['max_price']:.4f}")
                st.write(f"• Precio Mínimo: ${metrics['min_price']:.4f}")
                st.write(f"• Volumen Promedio: {metrics['avg_volume']:,.0f}")
    
    def _show_trading_signals(self, signals: Dict) -> None:
        """Muestra señales de trading"""
        st.subheader("🎯 Señales de Trading")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Señales RSI:**")
            st.write(f"• Sobreventa (RSI < 30): {signals['oversold_signals']}")
            st.write(f"• Sobrecompra (RSI > 70): {signals['overbought_signals']}")
        
        with col2:
            st.write("**Señales MACD:**")
            st.write(f"• Cruces Alcistas: {signals['macd_bullish']}")
            st.write(f"• Cruces Bajistas: {signals['macd_bearish']}")
        
        with col3:
            st.write("**Señales SMA:**")
            st.write(f"• Cruces Alcistas: {signals['sma_bullish']}")
            st.write(f"• Cruces Bajistas: {signals['sma_bearish']}")
    
    def _show_volatility_metrics(self, metrics: Dict) -> None:
        """Muestra métricas de volatilidad"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Volatilidad Actual",
                f"{metrics['current_volatility']*100:.2f}%"
            )
        
        with col2:
            st.metric(
                "Volatilidad Promedio",
                f"{metrics['avg_volatility']*100:.2f}%"
            )
        
        with col3:
            st.metric(
                "Volatilidad Máxima",
                f"{metrics['max_volatility']*100:.2f}%"
            )
        
        with col4:
            st.metric(
                "Tendencia",
                metrics['volatility_trend']
            )
    
    def _show_system_metrics(self, health: Dict) -> None:
        """Muestra métricas del sistema"""
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            status_color = "🟢" if health.get('database') == 'Conectado' else "🔴"
            st.metric("Base de Datos", f"{status_color} {health.get('database', 'Unknown')}")
        
        with col2:
            status_color = "🟢" if health.get('cache') == 'Activo' else "🟡"
            st.metric("Caché", f"{status_color} {health.get('cache', 'Unknown')}")
        
        with col3:
            status_color = "🟢" if health.get('api') == 'Disponible' else "🔴"
            st.metric("API", f"{status_color} {health.get('api', 'Unknown')}")
        
        with col4:
            st.metric("Símbolos Disponibles", health.get('available_symbols', 0))
    
    def _show_environment_info(self) -> None:
        """Muestra información del entorno"""
        with st.expander("🔧 Información del Entorno", expanded=False):
            import sys
            import platform
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Sistema:**")
                st.write(f"• Plataforma: {platform.system()} {platform.release()}")
                st.write(f"• Python: {sys.version.split()[0]}")
                st.write(f"• Streamlit: {st.__version__}")
            
            with col2:
                st.write("**Configuración:**")
                st.write(f"• Última actualización: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                st.write(f"• Zona horaria: {datetime.now().astimezone().tzinfo}")
    
    def render_main_page(self) -> None:
        """Renderiza la página principal de reportes"""
        # Mostrar encabezado
        self.show_page_header()
        
        # Mostrar configuración y obtener valores
        report_type, symbol_for_report, date_range, export_format = self.show_report_configuration(self.data_manager)
        
        if report_type is None:
            return
        
        # Botón para generar reporte
        if st.button("🔄 Generar Reporte", type="primary", use_container_width=True):
            with st.spinner("Generando reporte..."):
                try:
                    # Generar reporte según el tipo seleccionado
                    if report_type == "📈 Performance":
                        self.show_performance_report(self.report_generator, symbol_for_report, date_range)
                        
                    elif report_type == "📊 Análisis Técnico":
                        self.show_technical_analysis_report(self.report_generator, symbol_for_report, date_range)
                        
                    elif report_type == "📉 Análisis de Volatilidad":
                        self.show_volatility_report(self.report_generator, symbol_for_report, date_range)
                        
                    elif report_type == "📋 Estado del Sistema":
                        self.show_system_report(self.report_generator)
                    
                    # Manejar exportación de datos
                    if export_format != "Ver en pantalla" and symbol_for_report:
                        self.handle_data_export(self.report_generator, symbol_for_report, date_range, export_format)
                    
                    st.success("✅ Reporte generado exitosamente")
                    
                except Exception as e:
                    st.error(f"❌ Error generando reporte: {e}")
                    with st.expander("Ver detalles del error"):
                        st.exception(e)