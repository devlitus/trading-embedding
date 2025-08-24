#!/usr/bin/env python3
"""
Componentes de UI para el módulo de monitoreo en tiempo real.

Este módulo contiene todos los componentes de interfaz de usuario
para la visualización de datos en tiempo real.
"""

import streamlit as st
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from .common import get_plotly_config, CUSTOM_CSS
from .realtime_config import RealtimeConfig
from .realtime_logic import RealtimeDataProcessor, PriceMetrics, Alert


class RealtimeUI:
    """Componentes de UI para monitoreo en tiempo real"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.processor = RealtimeDataProcessor(data_manager)
        self.config = RealtimeConfig()
    
    def render_page_header(self):
        """Renderiza el encabezado de la página"""
        st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
        st.title("🎯 Monitoreo en Tiempo Real")
        
        if not self.data_manager:
            st.error("Error: DataManager no disponible")
            st.stop()
    
    def render_monitoring_config(self) -> tuple:
        """Renderiza la configuración del monitoreo"""
        st.subheader("⚙️ Configuración del Monitoreo")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            symbols_to_monitor = st.multiselect(
                "Símbolos a Monitorear",
                self.config.AVAILABLE_SYMBOLS,
                default=self.config.DEFAULT_SYMBOLS
            )
        
        with col2:
            refresh_interval = st.selectbox(
                "Intervalo de Actualización",
                list(self.config.REFRESH_INTERVALS.keys()),
                index=list(self.config.REFRESH_INTERVALS.keys()).index(
                    self.config.UI_CONFIG['default_refresh_interval']
                )
            )
        
        with col3:
            auto_refresh = st.checkbox(
                "Auto-actualizar", 
                value=self.config.UI_CONFIG['auto_refresh_default']
            )
        
        return symbols_to_monitor, refresh_interval, auto_refresh
    
    def render_price_metrics(self, symbols: List[str]) -> bool:
        """Renderiza las métricas de precio en tiempo real"""
        if not symbols:
            st.info("Selecciona al menos un símbolo para monitorear")
            return False
        
        # Botón de actualización manual
        update_triggered = st.button("🔄 Actualizar Datos")
        
        if update_triggered:
            st.subheader("📊 Precios en Tiempo Real")
            
            # Crear columnas dinámicamente
            max_cols = self.config.UI_CONFIG['max_symbols_per_row']
            
            # Dividir símbolos en filas si hay muchos
            for i in range(0, len(symbols), max_cols):
                row_symbols = symbols[i:i + max_cols]
                cols = st.columns(len(row_symbols))
                
                for j, symbol in enumerate(row_symbols):
                    with cols[j]:
                        self._render_symbol_metric(symbol)
        
        return update_triggered
    
    def _render_symbol_metric(self, symbol: str):
        """Renderiza métrica individual de un símbolo"""
        try:
            metrics = self.processor.get_price_metrics(symbol)
            
            if not metrics:
                st.warning(f"No hay datos para {symbol}")
                return
            
            # Mostrar métrica principal
            delta_color = self.config.get_delta_color(metrics.change)
            
            st.metric(
                label=symbol,
                value=self.config.format_price(metrics.current_price),
                delta=self.config.format_percentage(metrics.change_pct),
                delta_color=delta_color
            )
            
            # Información adicional
            if self.config.UI_CONFIG['show_volume_info'] and metrics.volume > 0:
                st.caption(f"Vol: {self._format_volume(metrics.volume)}")
            
            # Mini gráfico si está habilitado
            if self.config.UI_CONFIG['show_mini_charts']:
                self._render_mini_chart(symbol)
                
        except Exception as e:
            st.error(f"Error obteniendo datos para {symbol}: {e}")
    
    def _render_mini_chart(self, symbol: str):
        """Renderiza mini gráfico para un símbolo"""
        try:
            df = self.processor.get_mini_chart_data(symbol)
            
            if df is None or df.empty:
                return
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                name=symbol,
                line=dict(width=2, color='#1f77b4'),
                hovertemplate='<b>%{y:.2f}</b><br>%{x}<extra></extra>'
            ))
            
            fig.update_layout(
                height=self.config.CHART_CONFIG['mini_chart_height'],
                margin=dict(l=0, r=0, t=0, b=0),
                showlegend=False,
                xaxis=dict(showticklabels=False, showgrid=False),
                yaxis=dict(showticklabels=False, showgrid=False),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            st.plotly_chart(
                fig, 
                use_container_width=True, 
                key=f"mini_{symbol}", 
                config=get_plotly_config()
            )
            
        except Exception as e:
            st.caption(f"Error en gráfico: {str(e)[:30]}...")
    
    def render_alerts_section(self, symbols: List[str]):
        """Renderiza la sección de alertas y señales"""
        st.subheader("🚨 Alertas y Señales")
        
        if not symbols:
            st.info("Selecciona símbolos para ver alertas")
            return
        
        # Contenedor para alertas
        alerts_container = st.container()
        
        with alerts_container:
            all_alerts = []
            
            # Generar alertas para todos los símbolos
            for symbol in symbols:
                symbol_alerts = self.processor.generate_alerts(symbol)
                all_alerts.extend(symbol_alerts)
            
            if not all_alerts:
                st.info("No hay alertas activas en este momento")
                return
            
            # Ordenar alertas por confianza
            all_alerts.sort(key=lambda x: x.confidence, reverse=True)
            
            # Mostrar alertas
            for alert in all_alerts:
                self._render_alert(alert)
    
    def _render_alert(self, alert: Alert):
        """Renderiza una alerta individual"""
        message = f"{alert.emoji} {alert.message}"
        
        if alert.type == 'bullish':
            st.success(message)
        elif alert.type == 'bearish':
            st.error(message)
        elif alert.type == 'warning':
            st.warning(message)
        else:
            st.info(message)
    
    def render_market_summary(self, symbols: List[str]):
        """Renderiza resumen del mercado"""
        if not symbols:
            return
        
        st.subheader("📈 Resumen del Mercado")
        
        try:
            summary = self.processor.get_market_summary(symbols)
            
            # Métricas del resumen
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Símbolos Alcistas",
                    summary['bullish_count'],
                    delta=None
                )
            
            with col2:
                st.metric(
                    "Símbolos Bajistas",
                    summary['bearish_count'],
                    delta=None
                )
            
            with col3:
                st.metric(
                    "Símbolos Neutrales",
                    summary['neutral_count'],
                    delta=None
                )
            
            with col4:
                st.metric(
                    "Cambio Promedio",
                    self.config.format_percentage(summary['avg_change']),
                    delta=None
                )
            
            # Top gainers y losers
            if summary['top_gainers'] or summary['top_losers']:
                col_gain, col_loss = st.columns(2)
                
                with col_gain:
                    st.write("**🚀 Mejores Rendimientos:**")
                    for metrics in summary['top_gainers']:
                        st.write(f"• {metrics.symbol}: {self.config.format_percentage(metrics.change_pct)}")
                
                with col_loss:
                    st.write("**📉 Mayores Caídas:**")
                    for metrics in summary['top_losers']:
                        st.write(f"• {metrics.symbol}: {self.config.format_percentage(metrics.change_pct)}")
                        
        except Exception as e:
            st.error(f"Error generando resumen del mercado: {e}")
    
    def _format_volume(self, volume: float) -> str:
        """Formatea volumen de manera compacta"""
        if volume >= 1_000_000_000:
            return f"{volume / 1_000_000_000:.1f}B"
        elif volume >= 1_000_000:
            return f"{volume / 1_000_000:.1f}M"
        elif volume >= 1_000:
            return f"{volume / 1_000:.1f}K"
        else:
            return f"{volume:.0f}"
    
    def render_main_page(self):
        """Renderiza la página principal de monitoreo"""
        # Encabezado
        self.render_page_header()
        
        # Configuración
        symbols, refresh_interval, auto_refresh = self.render_monitoring_config()
        
        # Convertir intervalo a segundos
        interval_seconds = self.config.get_refresh_seconds(refresh_interval)
        
        # Métricas de precio
        update_triggered = self.render_price_metrics(symbols)
        
        # Si se actualizaron los datos, mostrar alertas y resumen
        if update_triggered or auto_refresh:
            # Alertas
            self.render_alerts_section(symbols)
            
            # Resumen del mercado
            self.render_market_summary(symbols)
            
            # Auto-refresh
            if auto_refresh:
                import time
                time.sleep(interval_seconds)
                st.rerun()