"""Componentes de métricas para el dashboard."""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

class MetricsComponent:
    """Componente para mostrar métricas del sistema."""
    
    @staticmethod
    def show_system_health(data_manager) -> Dict[str, Any]:
        """Muestra métricas de salud del sistema.
        
        Args:
            data_manager: Instancia del DataManager
            
        Returns:
            Diccionario con métricas del sistema
        """
        metrics = {}
        
        try:
            # Verificar conexión a base de datos
            db_status = "✅ Conectado" if data_manager.test_connection() else "❌ Desconectado"
            metrics['database'] = db_status
            
            # Verificar datos disponibles
            symbols = data_manager.get_available_symbols()
            metrics['symbols_count'] = len(symbols) if symbols else 0
            
            # Verificar última actualización
            if symbols:
                latest_data = data_manager.get_latest_data(symbols[0], '1h', limit=1)
                if not latest_data.empty:
                    last_update = latest_data.index[-1]
                    time_diff = datetime.now() - last_update
                    metrics['last_update'] = f"{time_diff.total_seconds() / 3600:.1f} horas"
                else:
                    metrics['last_update'] = "Sin datos"
            else:
                metrics['last_update'] = "Sin símbolos"
                
        except Exception as e:
            st.error(f"Error obteniendo métricas del sistema: {e}")
            metrics = {
                'database': "❌ Error",
                'symbols_count': 0,
                'last_update': "Error"
            }
        
        return metrics
    
    @staticmethod
    def display_metrics_grid(metrics: Dict[str, Any]):
        """Muestra métricas en una grilla.
        
        Args:
            metrics: Diccionario con métricas a mostrar
        """
        cols = st.columns(len(metrics))
        
        for i, (key, value) in enumerate(metrics.items()):
            with cols[i]:
                # Formatear el título
                title = key.replace('_', ' ').title()
                st.metric(label=title, value=str(value))
    
    @staticmethod
    def show_data_quality_metrics(data_manager, symbol: str, interval: str) -> Dict[str, Any]:
        """Muestra métricas de calidad de datos.
        
        Args:
            data_manager: Instancia del DataManager
            symbol: Símbolo a analizar
            interval: Intervalo de tiempo
            
        Returns:
            Diccionario con métricas de calidad
        """
        try:
            # Obtener datos recientes
            df = data_manager.get_latest_data(symbol, interval, limit=1000)
            
            if df.empty:
                return {'error': 'No hay datos disponibles'}
            
            # Calcular métricas de calidad
            metrics = {
                'total_records': len(df),
                'missing_values': df.isnull().sum().sum(),
                'duplicate_records': df.duplicated().sum(),
                'date_range': f"{df.index[0].strftime('%Y-%m-%d')} - {df.index[-1].strftime('%Y-%m-%d')}",
                'data_completeness': f"{((len(df) - df.isnull().sum().sum()) / (len(df) * len(df.columns)) * 100):.1f}%"
            }
            
            # Verificar gaps en los datos
            time_diffs = df.index.to_series().diff().dropna()
            expected_interval = pd.Timedelta(interval)
            gaps = (time_diffs > expected_interval * 1.5).sum()
            metrics['data_gaps'] = gaps
            
            return metrics
            
        except Exception as e:
            return {'error': f'Error calculando métricas: {str(e)}'}
    
    @staticmethod
    def create_performance_chart(data: pd.DataFrame, title: str = "Performance") -> go.Figure:
        """Crea un gráfico de rendimiento.
        
        Args:
            data: DataFrame con datos de rendimiento
            title: Título del gráfico
            
        Returns:
            Figura de Plotly
        """
        fig = go.Figure()
        
        if 'close' in data.columns:
            fig.add_trace(go.Scatter(
                x=data.index,
                y=data['close'],
                mode='lines',
                name='Precio de Cierre',
                line=dict(color='#1f77b4', width=2)
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Tiempo",
            yaxis_title="Precio",
            hovermode='x unified',
            showlegend=True,
            height=400
        )
        
        return fig
    
    @staticmethod
    def show_trading_metrics(data: pd.DataFrame) -> Dict[str, float]:
        """Calcula y muestra métricas de trading.
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            Diccionario con métricas de trading
        """
        if data.empty or 'close' not in data.columns:
            return {}
        
        try:
            # Calcular métricas básicas
            current_price = data['close'].iloc[-1]
            price_change = data['close'].iloc[-1] - data['close'].iloc[0]
            price_change_pct = (price_change / data['close'].iloc[0]) * 100
            
            # Volatilidad
            returns = data['close'].pct_change().dropna()
            volatility = returns.std() * 100
            
            # Rango de precios
            high_24h = data['high'].max()
            low_24h = data['low'].min()
            
            # Volumen promedio
            avg_volume = data['volume'].mean() if 'volume' in data.columns else 0
            
            metrics = {
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'volatility': volatility,
                'high_24h': high_24h,
                'low_24h': low_24h,
                'avg_volume': avg_volume
            }
            
            return metrics
            
        except Exception as e:
            st.error(f"Error calculando métricas de trading: {e}")
            return {}
    
    @staticmethod
    def display_trading_metrics(metrics: Dict[str, float]):
        """Muestra métricas de trading en el dashboard.
        
        Args:
            metrics: Diccionario con métricas de trading
        """
        if not metrics:
            st.warning("No hay métricas disponibles")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Precio Actual",
                f"${metrics.get('current_price', 0):.2f}",
                f"{metrics.get('price_change', 0):.2f} ({metrics.get('price_change_pct', 0):.2f}%)"
            )
        
        with col2:
            st.metric(
                "Volatilidad",
                f"{metrics.get('volatility', 0):.2f}%"
            )
        
        with col3:
            st.metric(
                "Máximo 24h",
                f"${metrics.get('high_24h', 0):.2f}"
            )
        
        with col4:
            st.metric(
                "Mínimo 24h",
                f"${metrics.get('low_24h', 0):.2f}"
            )