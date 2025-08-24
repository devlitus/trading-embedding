#!/usr/bin/env python3
"""
Utilidades de gráficos y visualizaciones para Fase 3 - Etiquetado Wyckoff
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

def get_price_at_timestamp(df, timestamp):
    """Obtener precio en un timestamp específico."""
    try:
        if timestamp in df.index:
            return df.loc[timestamp, 'close']
        else:
            # Buscar el timestamp más cercano
            closest_idx = df.index.get_indexer([timestamp], method='nearest')[0]
            return df.iloc[closest_idx]['close']
    except:
        return 0.0

def create_enhanced_price_chart(df, signals, symbol):
    """Crear gráfico de precios mejorado con señales detectadas."""
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'Precio {symbol} con Patrones Wyckoff', 'Volumen'),
        row_heights=[0.7, 0.3]
    )
    
    # Gráfico de velas japonesas
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name="Precio",
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # Colores y símbolos para diferentes tipos de patrones
    pattern_config = {
        'accumulation': {'color': '#00ff88', 'symbol': 'triangle-up', 'name': '🟢 Acumulación'},
        'distribution': {'color': '#ff4444', 'symbol': 'triangle-down', 'name': '🔴 Distribución'},
        'reaccumulation': {'color': '#ffaa00', 'symbol': 'circle', 'name': '🟡 Re-acumulación'},
        'redistribution': {'color': '#ff8800', 'symbol': 'circle', 'name': '🟠 Re-distribución'}
    }
    
    # Añadir señales al gráfico
    for signal in signals:
        pattern_type = signal.phase.lower()
        config = pattern_config.get(pattern_type, {'color': 'purple', 'symbol': 'star', 'name': '⚪ Desconocido'})
        
        signal_price = get_price_at_timestamp(df, signal.timestamp)
        
        fig.add_trace(
            go.Scatter(
                x=[signal.timestamp],
                y=[signal_price],
                mode='markers',
                marker=dict(
                    symbol=config['symbol'],
                    size=15,
                    color=config['color'],
                    line=dict(width=2, color='white')
                ),
                name=f"{config['name']} ({signal.confidence:.2f})",
                hovertemplate=f"<b>{signal.phase.title()}</b><br>" +
                             f"Precio: ${signal_price:.2f}<br>" +
                             f"Confianza: {signal.confidence:.2f}<br>" +
                             f"Tiempo: {signal.timestamp}<br>" +
                             "<extra></extra>"
            ),
            row=1, col=1
        )
    
    # Gráfico de volumen
    colors = ['red' if close < open else 'green' for close, open in zip(df['close'], df['open'])]
    fig.add_trace(
        go.Bar(
            x=df.index,
            y=df['volume'],
            name="Volumen",
            marker_color=colors,
            opacity=0.6
        ),
        row=2, col=1
    )
    
    # Configuración del layout
    fig.update_layout(
        title=f"Análisis de Patrones Wyckoff - {symbol}",
        xaxis_title="Tiempo",
        yaxis_title="Precio (USDT)",
        height=800,
        showlegend=True,
        template="plotly_dark",
        hovermode='x unified'
    )
    
    fig.update_xaxes(rangeslider_visible=False)
    
    return fig

def create_pattern_distribution_chart(patterns):
    """Crear gráfico de distribución de tipos de patrones."""
    if not patterns:
        return None
    
    # Contar tipos de patrones
    pattern_counts = {}
    for pattern in patterns:
        pattern_type = pattern.phase.title()
        pattern_counts[pattern_type] = pattern_counts.get(pattern_type, 0) + 1
    
    # Crear gráfico de barras
    fig = px.bar(
        x=list(pattern_counts.keys()),
        y=list(pattern_counts.values()),
        title="Distribución de Tipos de Patrones Detectados",
        labels={'x': 'Tipo de Patrón', 'y': 'Cantidad'},
        color=list(pattern_counts.values()),
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        template="plotly_white"
    )
    
    return fig

def create_confidence_distribution_chart(patterns):
    """Crear gráfico de distribución de confianza de patrones."""
    if not patterns:
        return None
    
    confidences = [pattern.confidence for pattern in patterns]
    
    fig = px.histogram(
        x=confidences,
        nbins=20,
        title="Distribución de Confianza de Patrones",
        labels={'x': 'Nivel de Confianza', 'y': 'Frecuencia'},
        color_discrete_sequence=['#636EFA']
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        template="plotly_white"
    )
    
    return fig

def create_focused_pattern_chart(df, pattern, symbol, window_size=50):
    """Crear gráfico enfocado en un patrón específico."""
    try:
        pattern_time = pattern.timestamp
        pattern_idx = df.index.get_loc(pattern_time)
        
        start_idx = max(0, pattern_idx - window_size)
        end_idx = min(len(df), pattern_idx + window_size)
        
        focused_df = df.iloc[start_idx:end_idx]
        
        fig = go.Figure()
        
        # Gráfico de velas
        fig.add_trace(
            go.Candlestick(
                x=focused_df.index,
                open=focused_df['open'],
                high=focused_df['high'],
                low=focused_df['low'],
                close=focused_df['close'],
                name="Precio"
            )
        )
        
        # Marcar el patrón
        signal_price = get_price_at_timestamp(df, pattern.timestamp)
        
        pattern_config = {
            'accumulation': {'color': '#00ff88', 'symbol': 'triangle-up'},
            'distribution': {'color': '#ff4444', 'symbol': 'triangle-down'},
            'reaccumulation': {'color': '#ffaa00', 'symbol': 'circle'},
            'redistribution': {'color': '#ff8800', 'symbol': 'circle'}
        }
        
        config = pattern_config.get(pattern.phase.lower(), {'color': 'purple', 'symbol': 'star'})
        
        fig.add_trace(
            go.Scatter(
                x=[pattern.timestamp],
                y=[signal_price],
                mode='markers',
                marker=dict(
                    symbol=config['symbol'],
                    size=20,
                    color=config['color'],
                    line=dict(width=3, color='white')
                ),
                name=f"{pattern.phase.title()}",
                hovertemplate=f"<b>{pattern.phase.title()}</b><br>" +
                             f"Precio: ${signal_price:.2f}<br>" +
                             f"Confianza: {pattern.confidence:.2f}<br>" +
                             "<extra></extra>"
            )
        )
        
        fig.update_layout(
            title=f"Vista Detallada: {pattern.phase.title()} - {symbol}",
            xaxis_title="Tiempo",
            yaxis_title="Precio (USDT)",
            height=500,
            template="plotly_dark"
        )
        
        return fig
        
    except Exception as e:
        return None

def create_labeling_progress_chart(labeling_results):
    """Crear gráfico de progreso del etiquetado."""
    if not labeling_results:
        return None
    
    # Contar estados
    status_counts = {'valid': 0, 'invalid': 0, 'uncertain': 0}
    
    for result in labeling_results.values():
        status = result.get('status', 'uncertain')
        status_counts[status] = status_counts.get(status, 0) + 1
    
    # Crear gráfico de dona
    fig = px.pie(
        values=list(status_counts.values()),
        names=['✅ Válidos', '❌ Inválidos', '❓ Inciertos'],
        title="Estado del Etiquetado",
        color_discrete_sequence=['#00ff88', '#ff4444', '#ffaa00']
    )
    
    fig.update_traces(hole=0.4)
    fig.update_layout(
        height=400,
        template="plotly_white"
    )
    
    return fig

def create_quality_score_chart(labeling_results):
    """Crear gráfico de distribución de puntuaciones de calidad."""
    if not labeling_results:
        return None
    
    quality_scores = [result.get('quality_score', 3) for result in labeling_results.values()]
    
    fig = px.histogram(
        x=quality_scores,
        nbins=5,
        title="Distribución de Puntuaciones de Calidad",
        labels={'x': 'Puntuación de Calidad (1-5)', 'y': 'Frecuencia'},
        color_discrete_sequence=['#636EFA']
    )
    
    fig.update_layout(
        showlegend=False,
        height=400,
        template="plotly_white"
    )
    
    return fig