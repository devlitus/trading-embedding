import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time
from datetime import datetime
import sys
from pathlib import Path

# Agregar rutas del proyecto
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from data.data_strategy import DataStrategy, DataUsagePattern
from data.data_access_layer import DataAccessLayer

def show_hybrid_data_strategy_page():
    """Página para demostrar la estrategia híbrida de datos."""
    
    st.title("🔄 Estrategia Híbrida de Datos")
    st.markdown("""
    Esta página demuestra el sistema híbrido de acceso a datos que optimiza el rendimiento 
    según el patrón de uso específico.
    """)
    
    # Sidebar para configuración
    st.sidebar.header("⚙️ Configuración")
    
    # Selección de símbolo e intervalo
    symbol = st.sidebar.selectbox(
        "Símbolo",
        ["BTCUSDT", "ETHUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"],
        index=0
    )
    
    interval = st.sidebar.selectbox(
        "Intervalo",
        ["1h", "4h", "1d", "1w"],
        index=0
    )
    
    # Inicializar estrategia
    try:
        data_access = DataAccessLayer()
        strategy = DataStrategy(data_access)
    except Exception as e:
        st.error(f"Error inicializando estrategia: {e}")
        return
    
    # Tabs para diferentes patrones de uso
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🚀 Trading Tiempo Real", 
        "🤖 Entrenamiento ML", 
        "📊 Backtesting", 
        "🌐 API Serving", 
        "📈 Dashboard"
    ])
    
    with tab1:
        show_realtime_trading_demo(strategy, symbol, interval)
    
    with tab2:
        show_ml_training_demo(strategy, symbol, interval)
    
    with tab3:
        show_backtesting_demo(strategy, symbol, interval)
    
    with tab4:
        show_api_serving_demo(strategy, symbol, interval)
    
    with tab5:
        show_dashboard_demo(strategy, symbol, interval)
    
    # Sección de comparación de rendimiento
    st.header("⚡ Comparación de Rendimiento")
    show_performance_comparison(strategy, symbol, interval)

def show_realtime_trading_demo(strategy, symbol, interval):
    """Demo de trading en tiempo real."""
    st.subheader("🚀 Trading en Tiempo Real")
    st.markdown("""
    **Optimizado para:** Baja latencia, datos más recientes, indicadores básicos
    **Fuente de datos:** Cache → Base de datos → CSV
    """)
    
    if st.button("Obtener Datos para Trading", key="realtime"):
        with st.spinner("Obteniendo datos optimizados para trading..."):
            start_time = time.time()
            
            try:
                data = strategy.get_data(
                    pattern=DataUsagePattern.REALTIME_TRADING,
                    symbol=symbol,
                    interval=interval,
                    limit=100
                )
                
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000
                
                if not data.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Registros", len(data))
                    with col2:
                        st.metric("Columnas", len(data.columns))
                    with col3:
                        st.metric("Tiempo (ms)", f"{execution_time:.1f}")
                    with col4:
                        if 'price_change' in data.columns:
                            last_change = data['price_change'].iloc[-1]
                            st.metric("Último Cambio %", f"{last_change*100:.2f}%")
                    
                    # Mostrar gráfico de precio con indicadores
                    fig = go.Figure()
                    
                    if 'datetime' in data.columns:
                        time_col = 'datetime'
                    else:
                        time_col = data.columns[0]
                    
                    # Precio de cierre
                    close_col = 'close' if 'close' in data.columns else 'close_price'
                    fig.add_trace(go.Scatter(
                        x=data[time_col],
                        y=data[close_col],
                        mode='lines',
                        name='Precio',
                        line=dict(color='#1f77b4', width=2)
                    ))
                    
                    # Media móvil de volumen si existe
                    if 'volume_ma' in data.columns:
                        fig.add_trace(go.Scatter(
                            x=data[time_col],
                            y=data['volume_ma'],
                            mode='lines',
                            name='Volumen MA',
                            yaxis='y2',
                            line=dict(color='orange', width=1)
                        ))
                    
                    fig.update_layout(
                        title=f"Datos de Trading - {symbol} ({interval})",
                        xaxis_title="Tiempo",
                        yaxis_title="Precio",
                        yaxis2=dict(overlaying='y', side='right', title='Volumen MA'),
                        height=400
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar muestra de datos
                    st.subheader("Muestra de Datos")
                    st.dataframe(data.tail(10))
                    
                else:
                    st.warning("No se encontraron datos")
                    
            except Exception as e:
                st.error(f"Error obteniendo datos: {e}")

def show_ml_training_demo(strategy, symbol, interval):
    """Demo de entrenamiento ML."""
    st.subheader("🤖 Entrenamiento de Machine Learning")
    st.markdown("""
    **Optimizado para:** Datasets completos, features avanzadas, consistencia
    **Fuente de datos:** CSV → Base de datos
    """)
    
    include_features = st.checkbox("Incluir features avanzadas", value=True, key="ml_features")
    
    if st.button("Obtener Datos para ML", key="ml"):
        with st.spinner("Preparando dataset para ML..."):
            start_time = time.time()
            
            try:
                data = strategy.get_data(
                    pattern=DataUsagePattern.ML_TRAINING,
                    symbol=symbol,
                    interval=interval,
                    include_features=include_features
                )
                
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000
                
                if not data.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Registros", len(data))
                    with col2:
                        st.metric("Features", len(data.columns))
                    with col3:
                        st.metric("Tiempo (ms)", f"{execution_time:.1f}")
                    with col4:
                        completeness = (1 - data.isnull().sum().sum() / (len(data) * len(data.columns))) * 100
                        st.metric("Completitud %", f"{completeness:.1f}")
                    
                    # Mostrar distribución de features
                    if include_features and any(col in data.columns for col in ['returns', 'volatility', 'rsi']):
                        fig = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=['Returns', 'Volatility', 'RSI', 'Correlación']
                        )
                        
                        if 'returns' in data.columns:
                            fig.add_trace(
                                go.Histogram(x=data['returns'].dropna(), name='Returns'),
                                row=1, col=1
                            )
                        
                        if 'volatility' in data.columns:
                            fig.add_trace(
                                go.Scatter(y=data['volatility'].dropna(), mode='lines', name='Volatility'),
                                row=1, col=2
                            )
                        
                        if 'rsi' in data.columns:
                            fig.add_trace(
                                go.Scatter(y=data['rsi'].dropna(), mode='lines', name='RSI'),
                                row=2, col=1
                            )
                        
                        fig.update_layout(height=600, title="Análisis de Features ML")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Información del dataset
                    st.subheader("Información del Dataset")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Columnas disponibles:**")
                        st.write(list(data.columns))
                    
                    with col2:
                        st.write("**Estadísticas básicas:**")
                        if 'close' in data.columns:
                            close_col = 'close'
                        else:
                            close_col = [col for col in data.columns if 'close' in col.lower()][0] if any('close' in col.lower() for col in data.columns) else data.columns[1]
                        
                        st.write(f"Precio mín: {data[close_col].min():.2f}")
                        st.write(f"Precio máx: {data[close_col].max():.2f}")
                        st.write(f"Precio promedio: {data[close_col].mean():.2f}")
                    
                else:
                    st.warning("No se encontraron datos")
                    
            except Exception as e:
                st.error(f"Error preparando datos ML: {e}")

def show_backtesting_demo(strategy, symbol, interval):
    """Demo de backtesting."""
    st.subheader("📊 Backtesting")
    st.markdown("""
    **Optimizado para:** Datos históricos completos, validación OHLC, continuidad temporal
    **Fuente de datos:** CSV → Base de datos
    """)
    
    if st.button("Obtener Datos para Backtesting", key="backtest"):
        with st.spinner("Preparando datos históricos..."):
            start_time = time.time()
            
            try:
                data = strategy.get_data(
                    pattern=DataUsagePattern.BACKTESTING,
                    symbol=symbol,
                    interval=interval
                )
                
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000
                
                if not data.empty:
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Registros", len(data))
                    with col2:
                        st.metric("Período", f"{len(data)} {interval}")
                    with col3:
                        st.metric("Tiempo (ms)", f"{execution_time:.1f}")
                    with col4:
                        if 'forward_return' in data.columns:
                            avg_return = data['forward_return'].mean() * 100
                            st.metric("Return Promedio %", f"{avg_return:.3f}")
                    
                    # Gráfico de velas
                    fig = go.Figure(data=go.Candlestick(
                        x=data['datetime'] if 'datetime' in data.columns else data.index,
                        open=data['open'] if 'open' in data.columns else data['open_price'],
                        high=data['high'] if 'high' in data.columns else data['high_price'],
                        low=data['low'] if 'low' in data.columns else data['low_price'],
                        close=data['close'] if 'close' in data.columns else data['close_price'],
                        name=symbol
                    ))
                    
                    fig.update_layout(
                        title=f"Datos Históricos - {symbol} ({interval})",
                        xaxis_title="Tiempo",
                        yaxis_title="Precio",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.warning("No se encontraron datos históricos")
                    
            except Exception as e:
                st.error(f"Error preparando datos de backtesting: {e}")

def show_api_serving_demo(strategy, symbol, interval):
    """Demo de API serving."""
    st.subheader("🌐 API Serving")
    st.markdown("""
    **Optimizado para:** Respuestas rápidas, formato JSON, datos compactos
    **Fuente de datos:** Cache → Base de datos
    """)
    
    if st.button("Obtener Datos para API", key="api"):
        with st.spinner("Preparando respuesta API..."):
            start_time = time.time()
            
            try:
                data = strategy.get_data(
                    pattern=DataUsagePattern.API_SERVING,
                    symbol=symbol,
                    interval=interval,
                    limit=50,
                    format_for_json=True
                )
                
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000
                
                if not data.empty:
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Registros", len(data))
                    with col2:
                        st.metric("Tiempo (ms)", f"{execution_time:.1f}")
                    with col3:
                        # Estimar tamaño JSON
                        json_size = len(data.to_json()) / 1024
                        st.metric("Tamaño JSON (KB)", f"{json_size:.1f}")
                    
                    # Mostrar formato JSON
                    st.subheader("Formato de Respuesta API")
                    sample_data = data.head(3).to_dict('records')
                    st.json({
                        "status": "success",
                        "symbol": symbol,
                        "interval": interval,
                        "count": len(data),
                        "data": sample_data
                    })
                    
                    # Tabla de datos
                    st.subheader("Datos Completos")
                    st.dataframe(data)
                    
                else:
                    st.warning("No se encontraron datos")
                    
            except Exception as e:
                st.error(f"Error preparando respuesta API: {e}")

def show_dashboard_demo(strategy, symbol, interval):
    """Demo de dashboard."""
    st.subheader("📈 Dashboard")
    st.markdown("""
    **Optimizado para:** Visualizaciones enriquecidas, múltiples métricas, interactividad
    **Fuente de datos:** Híbrido optimizado
    """)
    
    if st.button("Obtener Datos para Dashboard", key="dashboard"):
        with st.spinner("Preparando datos para visualización..."):
            start_time = time.time()
            
            try:
                data = strategy.get_data(
                    pattern=DataUsagePattern.DASHBOARD,
                    symbol=symbol,
                    interval=interval,
                    enrich_data=True
                )
                
                end_time = time.time()
                execution_time = (end_time - start_time) * 1000
                
                if not data.empty:
                    # Métricas principales
                    col1, col2, col3, col4 = st.columns(4)
                    
                    close_col = 'close' if 'close' in data.columns else 'close_price'
                    
                    with col1:
                        st.metric("Registros", len(data))
                    with col2:
                        st.metric("Precio Actual", f"${data[close_col].iloc[-1]:.2f}")
                    with col3:
                        if len(data) > 1:
                            price_change = ((data[close_col].iloc[-1] / data[close_col].iloc[-2]) - 1) * 100
                            st.metric("Cambio %", f"{price_change:.2f}%")
                    with col4:
                        st.metric("Tiempo (ms)", f"{execution_time:.1f}")
                    
                    # Gráfico principal con múltiples indicadores
                    fig = make_subplots(
                        rows=2, cols=1,
                        shared_xaxes=True,
                        vertical_spacing=0.1,
                        subplot_titles=['Precio y Indicadores', 'Volumen'],
                        row_heights=[0.7, 0.3]
                    )
                    
                    time_col = 'datetime' if 'datetime' in data.columns else data.index
                    
                    # Precio
                    fig.add_trace(
                        go.Scatter(
                            x=time_col,
                            y=data[close_col],
                            mode='lines',
                            name='Precio',
                            line=dict(color='#1f77b4', width=2)
                        ),
                        row=1, col=1
                    )
                    
                    # Volumen
                    volume_col = 'volume' if 'volume' in data.columns else 'volume'
                    if volume_col in data.columns:
                        fig.add_trace(
                            go.Bar(
                                x=time_col,
                                y=data[volume_col],
                                name='Volumen',
                                marker_color='lightblue'
                            ),
                            row=2, col=1
                        )
                    
                    fig.update_layout(
                        title=f"Dashboard Completo - {symbol} ({interval})",
                        height=600,
                        showlegend=True
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.warning("No se encontraron datos")
                    
            except Exception as e:
                st.error(f"Error preparando datos de dashboard: {e}")

def show_performance_comparison(strategy, symbol, interval):
    """Muestra comparación de rendimiento entre patrones."""
    
    if st.button("Ejecutar Comparación de Rendimiento", key="performance"):
        with st.spinner("Ejecutando benchmarks..."):
            results = []
            patterns = [
                (DataUsagePattern.REALTIME_TRADING, "Trading Tiempo Real"),
                (DataUsagePattern.ML_TRAINING, "Entrenamiento ML"),
                (DataUsagePattern.BACKTESTING, "Backtesting"),
                (DataUsagePattern.API_SERVING, "API Serving"),
                (DataUsagePattern.DASHBOARD, "Dashboard")
            ]
            
            for pattern, name in patterns:
                try:
                    start_time = time.time()
                    
                    data = strategy.get_data(
                        pattern=pattern,
                        symbol=symbol,
                        interval=interval,
                        limit=100 if pattern in [DataUsagePattern.REALTIME_TRADING, DataUsagePattern.API_SERVING] else None
                    )
                    
                    end_time = time.time()
                    execution_time = (end_time - start_time) * 1000
                    
                    results.append({
                        'Patrón': name,
                        'Tiempo (ms)': execution_time,
                        'Registros': len(data) if not data.empty else 0,
                        'Columnas': len(data.columns) if not data.empty else 0
                    })
                    
                except Exception as e:
                    results.append({
                        'Patrón': name,
                        'Tiempo (ms)': 0,
                        'Registros': 0,
                        'Columnas': 0,
                        'Error': str(e)
                    })
            
            # Mostrar resultados
            df_results = pd.DataFrame(results)
            
            # Tabla de resultados
            st.subheader("Tabla de Resultados")
            st.dataframe(df_results)
            
            # Gráfico de rendimiento debajo de la tabla
            st.subheader("Gráfico de Rendimiento")
            fig = px.bar(
                df_results,
                x='Patrón',
                y='Tiempo (ms)',
                title="Tiempo de Ejecución por Patrón",
                color='Tiempo (ms)',
                color_continuous_scale='viridis'
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recomendaciones
            fastest = df_results.loc[df_results['Tiempo (ms)'].idxmin()]
            st.success(f"🏆 **Patrón más rápido:** {fastest['Patrón']} ({fastest['Tiempo (ms)']:.1f}ms)")
            
            st.info("""
            **💡 Recomendaciones de uso:**
            - **Trading en tiempo real:** Usa para decisiones rápidas con datos recientes
            - **Entrenamiento ML:** Usa para datasets completos con features avanzadas
            - **Backtesting:** Usa para análisis histórico con validación de datos
            - **API Serving:** Usa para respuestas web optimizadas
            - **Dashboard:** Usa para visualizaciones enriquecidas
            """)

if __name__ == "__main__":
    show_hybrid_data_strategy_page()