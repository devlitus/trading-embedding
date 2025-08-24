#!/usr/bin/env python3
"""
Gráficos y visualizaciones para reportes.

Este módulo contiene todas las funciones de visualización usando Plotly
para los diferentes tipos de reportes.
"""

import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict

from .reports_config import ReportsConfig


class ReportsCharts:
    """Generador de gráficos para reportes"""
    
    def __init__(self):
        self.config = ReportsConfig()
        self.colors = self.config.get_color_scheme()
        self.plotly_config = self.config.get_plotly_config()
    
    def create_performance_chart(self, df: pd.DataFrame, symbol: str, initial_price: float) -> None:
        """Crea gráfico de rendimiento"""
        if df.empty:
            st.warning("No hay datos para mostrar el gráfico")
            return
        
        # Calcular rendimiento acumulado
        df['cumulative_return'] = ((df['close'] / initial_price) - 1) * 100
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(f'Precio de {symbol}', 'Rendimiento Acumulado (%)'),
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3]
        )
        
        # Gráfico de precio
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                name='Precio',
                line=dict(color=self.colors['primary'], width=2)
            ),
            row=1, col=1
        )
        
        # Gráfico de rendimiento
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['cumulative_return'],
                mode='lines',
                name='Rendimiento %',
                line=dict(color=self.colors['success'], width=2),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'Análisis de Rendimiento - {symbol}',
            height=600,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Fecha", row=2, col=1)
        fig.update_yaxes(title_text="Precio (USDT)", row=1, col=1)
        fig.update_yaxes(title_text="Rendimiento (%)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True, config=self.plotly_config)
    
    def create_technical_analysis_chart(self, df: pd.DataFrame, symbol: str) -> None:
        """Crea gráfico de análisis técnico"""
        if df.empty:
            st.warning("No hay datos para mostrar el gráfico")
            return
        
        fig = make_subplots(
            rows=4, cols=1,
            subplot_titles=(
                f'Precio y Medias Móviles - {symbol}',
                'MACD',
                'RSI',
                'Bandas de Bollinger'
            ),
            vertical_spacing=0.08,
            row_heights=[0.4, 0.2, 0.2, 0.2]
        )
        
        # Precio y medias móviles
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['close'],
                mode='lines', name='Precio',
                line=dict(color=self.colors['primary'], width=2)
            ),
            row=1, col=1
        )
        
        if 'sma_20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['sma_20'],
                    mode='lines', name='SMA 20',
                    line=dict(color=self.colors['secondary'], width=1)
                ),
                row=1, col=1
            )
        
        if 'sma_50' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['sma_50'],
                    mode='lines', name='SMA 50',
                    line=dict(color=self.colors['danger'], width=1)
                ),
                row=1, col=1
            )
        
        # MACD
        if 'macd' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['macd'],
                    mode='lines', name='MACD',
                    line=dict(color=self.colors['primary'])
                ),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['macd_signal'],
                    mode='lines', name='Señal',
                    line=dict(color=self.colors['secondary'])
                ),
                row=2, col=1
            )
        
        # RSI
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['rsi'],
                    mode='lines', name='RSI',
                    line=dict(color=self.colors['info'])
                ),
                row=3, col=1
            )
            
            # Líneas de sobrecompra y sobreventa
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # Bandas de Bollinger
        if 'bb_upper' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['bb_upper'],
                    mode='lines', name='BB Superior',
                    line=dict(color=self.colors['warning'], width=1)
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['bb_middle'],
                    mode='lines', name='BB Media',
                    line=dict(color=self.colors['primary'], width=1)
                ),
                row=4, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['bb_lower'],
                    mode='lines', name='BB Inferior',
                    line=dict(color=self.colors['success'], width=1)
                ),
                row=4, col=1
            )
        
        fig.update_layout(
            title=f'Análisis Técnico Completo - {symbol}',
            height=800,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Fecha", row=4, col=1)
        fig.update_yaxes(title_text="Precio (USDT)", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        fig.update_yaxes(title_text="Precio (USDT)", row=4, col=1)
        
        st.plotly_chart(fig, use_container_width=True, config=self.plotly_config)
    
    def create_volatility_chart(self, df: pd.DataFrame, rolling_vol: pd.Series, symbol: str) -> None:
        """Crea gráfico de volatilidad"""
        if df.empty or rolling_vol.empty:
            st.warning("No hay datos para mostrar el gráfico de volatilidad")
            return
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=(
                f'Precio - {symbol}',
                'Volatilidad Rolling (20 períodos)'
            ),
            vertical_spacing=0.1,
            row_heights=[0.6, 0.4]
        )
        
        # Gráfico de precio
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                name='Precio',
                line=dict(color=self.colors['primary'], width=2)
            ),
            row=1, col=1
        )
        
        # Gráfico de volatilidad
        fig.add_trace(
            go.Scatter(
                x=rolling_vol.index,
                y=rolling_vol * 100,  # Convertir a porcentaje
                mode='lines',
                name='Volatilidad %',
                line=dict(color=self.colors['danger'], width=2),
                fill='tonexty'
            ),
            row=2, col=1
        )
        
        # Línea de volatilidad promedio
        avg_vol = rolling_vol.mean() * 100
        fig.add_hline(
            y=avg_vol,
            line_dash="dash",
            line_color=self.colors['secondary'],
            annotation_text=f"Promedio: {avg_vol:.2f}%",
            row=2, col=1
        )
        
        fig.update_layout(
            title=f'Análisis de Volatilidad - {symbol}',
            height=600,
            showlegend=True,
            template='plotly_white'
        )
        
        fig.update_xaxes(title_text="Fecha", row=2, col=1)
        fig.update_yaxes(title_text="Precio (USDT)", row=1, col=1)
        fig.update_yaxes(title_text="Volatilidad (%)", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True, config=self.plotly_config)
    
    def create_system_metrics_chart(self, health: Dict) -> None:
        """Crea gráfico de métricas del sistema"""
        if not health:
            st.warning("No hay datos de salud del sistema")
            return
        
        # Crear gráfico de estado de componentes
        components = ['database', 'cache', 'api']
        statuses = [health.get(comp, 'Unknown') for comp in components]
        
        # Mapear estados a colores
        color_map = {
            'Conectado': self.colors['success'],
            'Activo': self.colors['success'],
            'Disponible': self.colors['success'],
            'Desconectado': self.colors['danger'],
            'Inactivo': self.colors['warning'],
            'Error': self.colors['danger'],
            'Unknown': self.colors['info']
        }
        
        colors = [color_map.get(status, self.colors['info']) for status in statuses]
        
        fig = go.Figure(data=[
            go.Bar(
                x=components,
                y=[1] * len(components),  # Altura uniforme
                marker_color=colors,
                text=statuses,
                textposition='inside',
                name='Estado del Sistema'
            )
        ])
        
        fig.update_layout(
            title='Estado de Componentes del Sistema',
            xaxis_title='Componentes',
            yaxis_title='Estado',
            height=400,
            template='plotly_white',
            showlegend=False
        )
        
        fig.update_yaxes(showticklabels=False)
        
        st.plotly_chart(fig, use_container_width=True, config=self.plotly_config)
    
    def create_signals_distribution_chart(self, signals: Dict) -> None:
        """Crea gráfico de distribución de señales"""
        if not signals:
            st.warning("No hay datos de señales para mostrar")
            return
        
        # Preparar datos para el gráfico
        signal_types = list(signals.keys())
        signal_counts = list(signals.values())
        
        fig = go.Figure(data=[
            go.Bar(
                x=signal_types,
                y=signal_counts,
                marker_color=self.colors['primary'],
                text=signal_counts,
                textposition='outside'
            )
        ])
        
        fig.update_layout(
            title='Distribución de Señales de Trading',
            xaxis_title='Tipo de Señal',
            yaxis_title='Cantidad',
            height=400,
            template='plotly_white'
        )
        
        st.plotly_chart(fig, use_container_width=True, config=self.plotly_config)