"""Componentes de gráficos para el dashboard."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import numpy as np

class ChartComponents:
    """Componente para crear y mostrar gráficos."""
    
    @staticmethod
    def create_candlestick_chart(data: pd.DataFrame, title: str = "Gráfico de Velas") -> go.Figure:
        """Crea un gráfico de velas japonesas.
        
        Args:
            data: DataFrame con columnas OHLCV
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        fig = go.Figure(data=go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close'],
            name="OHLC"
        ))
        
        fig.update_layout(
            title=title,
            yaxis_title="Precio",
            xaxis_title="Tiempo",
            xaxis_rangeslider_visible=False,
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_volume_chart(data: pd.DataFrame, title: str = "Volumen") -> go.Figure:
        """Crea un gráfico de volumen.
        
        Args:
            data: DataFrame con columna volume
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        colors = ['red' if close < open else 'green' 
                 for close, open in zip(data['close'], data['open'])]
        
        fig = go.Figure(data=go.Bar(
            x=data.index,
            y=data['volume'],
            marker_color=colors,
            name="Volumen"
        ))
        
        fig.update_layout(
            title=title,
            yaxis_title="Volumen",
            xaxis_title="Tiempo",
            height=300
        )
        
        return fig
    
    @staticmethod
    def create_technical_indicators_chart(data: pd.DataFrame, indicators: Dict[str, pd.Series]) -> go.Figure:
        """Crea un gráfico con indicadores técnicos.
        
        Args:
            data: DataFrame con datos OHLCV
            indicators: Diccionario con indicadores técnicos
            
        Returns:
            Figura de Plotly con subplots
        """
        # Crear subplots
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Precio y Medias Móviles', 'RSI', 'MACD'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Gráfico principal - Precio
        fig.add_trace(
            go.Candlestick(
                x=data.index,
                open=data['open'],
                high=data['high'],
                low=data['low'],
                close=data['close'],
                name="OHLC"
            ),
            row=1, col=1
        )
        
        # Medias móviles
        if 'sma_20' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['sma_20'],
                    name="SMA 20",
                    line=dict(color='orange', width=1)
                ),
                row=1, col=1
            )
        
        if 'sma_50' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['sma_50'],
                    name="SMA 50",
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
        
        # RSI
        if 'rsi' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['rsi'],
                    name="RSI",
                    line=dict(color='purple')
                ),
                row=2, col=1
            )
            
            # Líneas de sobrecompra y sobreventa
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        
        # MACD
        if 'macd' in indicators and 'macd_signal' in indicators:
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['macd'],
                    name="MACD",
                    line=dict(color='blue')
                ),
                row=3, col=1
            )
            
            fig.add_trace(
                go.Scatter(
                    x=data.index,
                    y=indicators['macd_signal'],
                    name="Signal",
                    line=dict(color='red')
                ),
                row=3, col=1
            )
            
            if 'macd_histogram' in indicators:
                fig.add_trace(
                    go.Bar(
                        x=data.index,
                        y=indicators['macd_histogram'],
                        name="Histogram",
                        marker_color='gray',
                        opacity=0.6
                    ),
                    row=3, col=1
                )
        
        fig.update_layout(
            height=800,
            showlegend=True,
            title="Análisis Técnico Completo"
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        return fig
    
    @staticmethod
    def create_correlation_heatmap(data: pd.DataFrame, title: str = "Matriz de Correlación") -> go.Figure:
        """Crea un mapa de calor de correlaciones.
        
        Args:
            data: DataFrame con datos numéricos
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        # Calcular matriz de correlación
        corr_matrix = data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=title,
            height=500,
            width=500
        )
        
        return fig
    
    @staticmethod
    def create_distribution_chart(data: pd.Series, title: str = "Distribución") -> go.Figure:
        """Crea un gráfico de distribución.
        
        Args:
            data: Serie de datos
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        fig = go.Figure()
        
        # Histograma
        fig.add_trace(go.Histogram(
            x=data,
            nbinsx=50,
            name="Distribución",
            opacity=0.7
        ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Valor",
            yaxis_title="Frecuencia",
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_time_series_comparison(data_dict: Dict[str, pd.Series], title: str = "Comparación de Series") -> go.Figure:
        """Crea un gráfico de comparación de series temporales.
        
        Args:
            data_dict: Diccionario con series de datos
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        
        for i, (name, series) in enumerate(data_dict.items()):
            fig.add_trace(go.Scatter(
                x=series.index,
                y=series.values,
                mode='lines',
                name=name,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Tiempo",
            yaxis_title="Valor",
            hovermode='x unified',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_performance_summary_chart(metrics: Dict[str, float]) -> go.Figure:
        """Crea un gráfico resumen de rendimiento.
        
        Args:
            metrics: Diccionario con métricas de rendimiento
            
        Returns:
            Figura de Plotly
        """
        # Preparar datos para el gráfico de barras
        names = list(metrics.keys())
        values = list(metrics.values())
        
        # Colores basados en valores positivos/negativos
        colors = ['green' if v >= 0 else 'red' for v in values]
        
        fig = go.Figure(data=go.Bar(
            x=names,
            y=values,
            marker_color=colors,
            text=[f"{v:.2f}" for v in values],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Resumen de Métricas de Rendimiento",
            xaxis_title="Métrica",
            yaxis_title="Valor",
            height=400
        )
        
        return fig
    
    @staticmethod
    def display_chart_with_controls(chart_func, data: pd.DataFrame, **kwargs):
        """Muestra un gráfico con controles interactivos.
        
        Args:
            chart_func: Función que crea el gráfico
            data: DataFrame con datos
            **kwargs: Argumentos adicionales para la función
        """
        # Controles en la barra lateral
        with st.sidebar:
            st.subheader("Controles del Gráfico")
            
            # Control de rango de fechas
            if not data.empty:
                date_range = st.date_input(
                    "Rango de fechas",
                    value=(data.index.min().date(), data.index.max().date()),
                    min_value=data.index.min().date(),
                    max_value=data.index.max().date()
                )
                
                # Filtrar datos por rango de fechas
                if len(date_range) == 2:
                    start_date, end_date = date_range
                    filtered_data = data.loc[start_date:end_date]
                else:
                    filtered_data = data
            else:
                filtered_data = data
        
        # Crear y mostrar el gráfico
        if not filtered_data.empty:
            fig = chart_func(filtered_data, **kwargs)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No hay datos para mostrar en el rango seleccionado")