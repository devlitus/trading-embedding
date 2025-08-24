import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
import os
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple

# Configurar rutas del proyecto
project_root = Path(r"c:\dev\trading-embedding")
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'src' / 'data'))
sys.path.insert(0, str(project_root / 'src' / 'analysis'))

# Importar módulos del proyecto
try:
    import importlib.util
    
    # Importar DataManager
    data_manager_spec = importlib.util.spec_from_file_location(
        "data_manager", project_root / "src" / "data" / "data_manager.py"
    )
    data_manager_module = importlib.util.module_from_spec(data_manager_spec)
    data_manager_spec.loader.exec_module(data_manager_module)
    DataManager = data_manager_module.DataManager
    
    # Importar análisis técnico
    analysis_spec = importlib.util.spec_from_file_location(
        "technical_analysis", project_root / "src" / "analysis" / "technical_analysis.py"
    )
    analysis_module = importlib.util.module_from_spec(analysis_spec)
    analysis_spec.loader.exec_module(analysis_module)
    analyze_symbol = analysis_module.analyze_symbol
    
except ImportError as e:
    st.error(f"❌ Error importando módulos: {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Error cargando módulos: {e}")
    st.stop()


class ReportGenerator:
    """Generador de reportes para análisis de trading"""
    
    def __init__(self, data_manager):
        self.data_manager = data_manager
        self.plotly_config = self._get_plotly_config()
    
    def _get_plotly_config(self) -> Dict:
        """Configuración estándar para gráficos Plotly"""
        return {
            'displayModeBar': True,
            'displaylogo': False,
            'modeBarButtonsToRemove': [
                'pan2d', 'lasso2d', 'select2d', 'autoScale2d',
                'hoverClosestCartesian', 'hoverCompareCartesian',
                'toggleSpikelines'
            ],
            'toImageButtonOptions': {
                'format': 'png',
                'filename': 'trading_report',
                'height': 600,
                'width': 1200,
                'scale': 2
            }
        }
    
    def _calculate_date_range(self, range_option: str) -> Tuple[datetime, datetime]:
        """Calcula el rango de fechas basado en la opción seleccionada"""
        end_date = datetime.now()
        
        range_mapping = {
            "Últimos 7 días": 7,
            "Últimos 30 días": 30,
            "Últimos 90 días": 90,
            "Último año": 365
        }
        
        days = range_mapping.get(range_option, 30)
        start_date = end_date - timedelta(days=days)
        
        return start_date, end_date
    
    def _get_data_for_report(self, symbol: str, date_range: str) -> Optional[pd.DataFrame]:
        """Obtiene datos para el reporte"""
        try:
            start_date, end_date = self._calculate_date_range(date_range)
            
            df = self.data_manager.get_data(
                symbol=symbol,
                interval='1h',
                start_time=start_date,
                end_time=end_date
            )
            
            if df is None or df.empty:
                st.warning(f"⚠️ No hay datos disponibles para {symbol} en el rango seleccionado")
                return None
            
            return df
            
        except Exception as e:
            st.error(f"❌ Error obteniendo datos: {e}")
            return None
    
    def generate_performance_report(self, symbol: str, date_range: str) -> None:
        """Genera reporte de performance"""
        df = self._get_data_for_report(symbol, date_range)
        if df is None:
            return
        
        st.subheader(f"📈 Reporte de Performance - {symbol}")
        
        # Calcular métricas principales
        initial_price = df['close'].iloc[0]
        final_price = df['close'].iloc[-1]
        total_return = ((final_price - initial_price) / initial_price) * 100
        
        # Métricas adicionales
        max_price = df['close'].max()
        min_price = df['close'].min()
        avg_volume = df['volume'].mean()
        volatility = df['close'].pct_change().std() * 100
        
        # Mostrar métricas principales
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            delta_color = "normal" if total_return >= 0 else "inverse"
            st.metric(
                "Retorno Total", 
                f"{total_return:.2f}%",
                delta=f"{total_return:.2f}%",
                delta_color=delta_color
            )
        
        with col2:
            st.metric("Precio Máximo", f"${max_price:.4f}")
        
        with col3:
            st.metric("Precio Mínimo", f"${min_price:.4f}")
        
        with col4:
            st.metric("Volatilidad", f"{volatility:.2f}%")
        
        # Gráfico de performance
        self._plot_performance_chart(df, symbol, initial_price)
        
        # Tabla de estadísticas detalladas
        self._show_detailed_stats(df, initial_price, final_price, total_return, avg_volume)
    
    def _plot_performance_chart(self, df: pd.DataFrame, symbol: str, initial_price: float) -> None:
        """Crea gráfico de performance"""
        df['cumulative_return'] = ((df['close'] / initial_price) - 1) * 100
        
        fig = go.Figure()
        
        # Línea de retorno acumulado
        color = 'green' if df['cumulative_return'].iloc[-1] >= 0 else 'red'
        
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['cumulative_return'],
            mode='lines',
            name='Retorno Acumulado (%)',
            line=dict(color=color, width=2),
            fill='tonexty' if df['cumulative_return'].iloc[-1] >= 0 else 'tozeroy',
            fillcolor=f'rgba({"0,255,0" if color == "green" else "255,0,0"}, 0.1)'
        ))
        
        # Línea de referencia en 0%
        fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
        
        fig.update_layout(
            title=f"📈 Performance de {symbol}",
            xaxis_title="Fecha",
            yaxis_title="Retorno Acumulado (%)",
            height=400,
            showlegend=True,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True, config=self.plotly_config)
    
    def _show_detailed_stats(self, df: pd.DataFrame, initial_price: float, 
                           final_price: float, total_return: float, avg_volume: float) -> None:
        """Muestra estadísticas detalladas"""
        st.subheader("📊 Estadísticas Detalladas")
        
        returns = df['close'].pct_change().dropna()
        
        stats = {
            'Precio Inicial': f"${initial_price:.4f}",
            'Precio Final': f"${final_price:.4f}",
            'Cambio Absoluto': f"${final_price - initial_price:.4f}",
            'Retorno Total': f"{total_return:.2f}%",
            'Retorno Promedio': f"{returns.mean() * 100:.4f}%",
            'Volatilidad (Desv. Est.)': f"{returns.std() * 100:.2f}%",
            'Máximo Drawdown': f"{self._calculate_max_drawdown(df):.2f}%",
            'Sharpe Ratio': f"{self._calculate_sharpe_ratio(returns):.2f}",
            'Volumen Promedio': f"{avg_volume:,.0f}",
            'Volumen Total': f"{df['volume'].sum():,.0f}",
            'Número de Períodos': f"{len(df):,}",
            'Días Positivos': f"{(returns > 0).sum()} ({(returns > 0).mean() * 100:.1f}%)"
        }
        
        # Crear DataFrame para mostrar las estadísticas
        stats_df = pd.DataFrame([
            {'Métrica': k, 'Valor': v} for k, v in stats.items()
        ])
        
        st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    def _calculate_max_drawdown(self, df: pd.DataFrame) -> float:
        """Calcula el máximo drawdown"""
        cumulative = (1 + df['close'].pct_change()).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        return drawdown.min() * 100
    
    def _calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calcula el Sharpe Ratio"""
        if returns.std() == 0:
            return 0
        
        excess_return = returns.mean() - (risk_free_rate / 365)  # Ajustar para frecuencia diaria
        return (excess_return / returns.std()) * (365 ** 0.5)  # Anualizar
    
    def generate_technical_analysis_report(self, symbol: str, date_range: str) -> None:
        """Genera reporte de análisis técnico"""
        df = self._get_data_for_report(symbol, date_range)
        if df is None:
            return
        
        st.subheader(f"📊 Análisis Técnico - {symbol}")
        
        # Calcular indicadores técnicos
        df = self._calculate_technical_indicators(df)
        
        # Gráfico principal con indicadores
        self._plot_technical_chart(df, symbol)
        
        # Análisis de señales
        self._analyze_trading_signals(df)
    
    def _calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula indicadores técnicos"""
        # Medias móviles
        df['SMA_7'] = df['close'].rolling(window=7).mean()
        df['SMA_21'] = df['close'].rolling(window=21).mean()
        df['SMA_50'] = df['close'].rolling(window=50).mean()
        
        # EMA
        df['EMA_12'] = df['close'].ewm(span=12).mean()
        df['EMA_26'] = df['close'].ewm(span=26).mean()
        
        # MACD
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bandas de Bollinger
        df['BB_Middle'] = df['close'].rolling(window=20).mean()
        bb_std = df['close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
        df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
        return df
    
    def _plot_technical_chart(self, df: pd.DataFrame, symbol: str) -> None:
        """Crea gráfico técnico completo"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Precio y Medias Móviles', 'MACD', 'RSI'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Gráfico de precio con medias móviles y Bandas de Bollinger
        fig.add_trace(
            go.Scatter(x=df.index, y=df['close'], name='Precio', line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Bandas de Bollinger
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Upper'], name='BB Superior', 
                      line=dict(color='gray', dash='dash'), opacity=0.5),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['BB_Lower'], name='BB Inferior', 
                      line=dict(color='gray', dash='dash'), opacity=0.5, 
                      fill='tonexty', fillcolor='rgba(128,128,128,0.1)'),
            row=1, col=1
        )
        
        # Medias móviles
        if not df['SMA_7'].isna().all():
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_7'], name='SMA 7', line=dict(color='orange')),
                row=1, col=1
            )
        
        if not df['SMA_21'].isna().all():
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_21'], name='SMA 21', line=dict(color='red')),
                row=1, col=1
            )
        
        if not df['SMA_50'].isna().all():
            fig.add_trace(
                go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='purple', width=2)),
                row=1, col=1
            )
        
        # MACD
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD'], name='MACD', line=dict(color='blue')),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df['MACD_Signal'], name='Señal MACD', line=dict(color='red')),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=df.index, y=df['MACD_Histogram'], name='Histograma MACD', 
                  marker_color='green', opacity=0.6),
            row=2, col=1
        )
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')),
            row=3, col=1
        )
        
        # Líneas de referencia RSI
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=3, col=1)
        
        fig.update_layout(
            title=f"📊 Análisis Técnico - {symbol}",
            height=800,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Precio ($)", row=1, col=1)
        fig.update_yaxes(title_text="MACD", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1, range=[0, 100])
        
        st.plotly_chart(fig, use_container_width=True, config=self.plotly_config)
    
    def _analyze_trading_signals(self, df: pd.DataFrame) -> None:
        """Analiza señales de trading"""
        st.subheader("🎯 Análisis de Señales")
        
        # Señales basadas en cruces de medias móviles
        signals = []
        
        # Cruce SMA 7 y SMA 21
        if not df['SMA_7'].isna().all() and not df['SMA_21'].isna().all():
            sma_cross = (df['SMA_7'] > df['SMA_21']) & (df['SMA_7'].shift(1) <= df['SMA_21'].shift(1))
            if sma_cross.any():
                signals.append("🟢 Señal alcista: SMA 7 cruza por encima de SMA 21")
            
            sma_cross_down = (df['SMA_7'] < df['SMA_21']) & (df['SMA_7'].shift(1) >= df['SMA_21'].shift(1))
            if sma_cross_down.any():
                signals.append("🔴 Señal bajista: SMA 7 cruza por debajo de SMA 21")
        
        # Señales RSI
        current_rsi = df['RSI'].iloc[-1]
        if not pd.isna(current_rsi):
            if current_rsi > 70:
                signals.append(f"⚠️ RSI en sobrecompra: {current_rsi:.1f}")
            elif current_rsi < 30:
                signals.append(f"⚠️ RSI en sobreventa: {current_rsi:.1f}")
            else:
                signals.append(f"ℹ️ RSI neutral: {current_rsi:.1f}")
        
        # Señales MACD
        if not df['MACD'].isna().all() and not df['MACD_Signal'].isna().all():
            macd_cross = (df['MACD'] > df['MACD_Signal']) & (df['MACD'].shift(1) <= df['MACD_Signal'].shift(1))
            if macd_cross.any():
                signals.append("🟢 Señal alcista: MACD cruza por encima de la señal")
            
            macd_cross_down = (df['MACD'] < df['MACD_Signal']) & (df['MACD'].shift(1) >= df['MACD_Signal'].shift(1))
            if macd_cross_down.any():
                signals.append("🔴 Señal bajista: MACD cruza por debajo de la señal")
        
        # Mostrar señales
        if signals:
            for signal in signals:
                st.write(signal)
        else:
            st.info("No se detectaron señales claras en el período analizado")
    
    def generate_volatility_report(self, symbol: str, date_range: str) -> None:
        """Genera reporte de volatilidad"""
        df = self._get_data_for_report(symbol, date_range)
        if df is None:
            return
        
        st.subheader(f"📉 Análisis de Volatilidad - {symbol}")
        
        # Calcular volatilidad
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * (24 ** 0.5)  # Volatilidad diaria
        
        # Volatilidad móvil
        rolling_vol = returns.rolling(window=24).std() * (24 ** 0.5)
        
        # Métricas de volatilidad
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Volatilidad Actual", f"{volatility:.4f}")
        
        with col2:
            st.metric("Vol. Promedio", f"{rolling_vol.mean():.4f}")
        
        with col3:
            vol_percentile = (rolling_vol.iloc[-1] > rolling_vol).mean() * 100
            st.metric("Percentil Actual", f"{vol_percentile:.1f}%")
        
        with col4:
            vol_trend = "📈" if rolling_vol.iloc[-1] > rolling_vol.iloc[-7:].mean() else "📉"
            st.metric("Tendencia", vol_trend)
        
        # Gráfico de volatilidad
        self._plot_volatility_chart(df, rolling_vol, symbol)
        
        # Estadísticas de volatilidad
        self._show_volatility_stats(rolling_vol)
    
    def _plot_volatility_chart(self, df: pd.DataFrame, rolling_vol: pd.Series, symbol: str) -> None:
        """Crea gráfico de volatilidad"""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=('Precio', 'Volatilidad Móvil (24h)'),
            row_heights=[0.7, 0.3],
            vertical_spacing=0.1
        )
        
        # Precio
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                name='Precio',
                line=dict(color='blue', width=1)
            ),
            row=1, col=1
        )
        
        # Volatilidad
        fig.add_trace(
            go.Scatter(
                x=df.index[24:],
                y=rolling_vol.dropna(),
                mode='lines',
                name='Volatilidad',
                line=dict(color='red', width=2),
                fill='tozeroy',
                fillcolor='rgba(255,0,0,0.1)'
            ),
            row=2, col=1
        )
        
        # Línea de volatilidad promedio
        avg_vol = rolling_vol.mean()
        fig.add_hline(
            y=avg_vol, 
            line_dash="dash", 
            line_color="orange", 
            opacity=0.7,
            row=2, col=1,
            annotation_text=f"Promedio: {avg_vol:.4f}"
        )
        
        fig.update_layout(
            title=f"📉 Análisis de Volatilidad - {symbol}",
            height=600,
            showlegend=True,
            hovermode='x unified'
        )
        
        fig.update_yaxes(title_text="Precio ($)", row=1, col=1)
        fig.update_yaxes(title_text="Volatilidad", row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True, config=self.plotly_config)
    
    def _show_volatility_stats(self, rolling_vol: pd.Series) -> None:
        """Muestra estadísticas de volatilidad"""
        st.subheader("📊 Estadísticas de Volatilidad")
        
        vol_stats = {
            'Volatilidad Mínima': f"{rolling_vol.min():.4f}",
            'Volatilidad Máxima': f"{rolling_vol.max():.4f}",
            'Volatilidad Mediana': f"{rolling_vol.median():.4f}",
            'Desviación Estándar': f"{rolling_vol.std():.4f}",
            'Coeficiente de Variación': f"{(rolling_vol.std() / rolling_vol.mean()):.2f}",
            'Percentil 25': f"{rolling_vol.quantile(0.25):.4f}",
            'Percentil 75': f"{rolling_vol.quantile(0.75):.4f}",
            'Rango Intercuartílico': f"{rolling_vol.quantile(0.75) - rolling_vol.quantile(0.25):.4f}"
        }
        
        vol_df = pd.DataFrame([
            {'Métrica': k, 'Valor': v} for k, v in vol_stats.items()
        ])
        
        st.dataframe(vol_df, use_container_width=True, hide_index=True)
    
    def generate_system_report(self) -> None:
        """Genera reporte del estado del sistema"""
        st.subheader("📋 Reporte del Estado del Sistema")
        
        try:
            # Health check del sistema
            health = self.data_manager.health_check()
            
            # Estado general
            all_healthy = all(
                component['status'] == 'healthy' 
                for component in health['components'].values()
            )
            
            if all_healthy:
                st.success("✅ Sistema operando correctamente")
            else:
                st.error("❌ Sistema con problemas detectados")
            
            # Estadísticas del sistema
            self._show_system_metrics()
            
            # Detalles de componentes
            self._show_component_status(health)
            
            # Información del entorno
            self._show_environment_info()
            
        except Exception as e:
            st.error(f"❌ Error generando reporte del sistema: {e}")
    
    def _show_system_metrics(self) -> None:
        """Muestra métricas del sistema"""
        try:
            db_stats = self.data_manager.get_database_stats()
            cache_stats = self.data_manager.get_cache_stats()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Símbolos en BD", db_stats.get('total_symbols', 0))
            
            with col2:
                st.metric("Total Registros", f"{db_stats.get('total_records', 0):,}")
            
            with col3:
                st.metric("Elementos en Caché", cache_stats.get('total_keys', 0))
            
            with col4:
                memory_mb = cache_stats.get('memory_usage', 0)
                st.metric("Memoria Caché", f"{memory_mb:.1f} MB")
            
        except Exception as e:
            st.warning(f"No se pudieron obtener las métricas del sistema: {e}")
    
    def _show_component_status(self, health: Dict) -> None:
        """Muestra estado de componentes"""
        st.subheader("🔧 Estado de Componentes")
        
        for component_name, component_info in health['components'].items():
            status = component_info['status']
            
            if status == 'healthy':
                st.success(f"✅ {component_name}: Operativo")
            else:
                st.error(f"❌ {component_name}: {component_info.get('message', 'Error')}")
    
    def _show_environment_info(self) -> None:
        """Muestra información del entorno"""
        st.subheader("🌐 Información del Entorno")
        
        env_info = {
            'Python Version': sys.version.split()[0],
            'Streamlit Version': st.__version__,
            'Pandas Version': pd.__version__,
            'Plotly Version': go.__version__,
            'Sistema Operativo': os.name,
            'Directorio de Trabajo': os.getcwd(),
            'Ruta del Proyecto': str(project_root),
            'Timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        env_df = pd.DataFrame([
            {'Componente': k, 'Versión/Valor': v} for k, v in env_info.items()
        ])
        
        st.dataframe(env_df, use_container_width=True, hide_index=True)


def show_reports_page(data_manager):
    """Página principal de reportes"""
    st.title("📋 Reportes Avanzados")
    st.markdown(
        "Genera reportes detallados y análisis profundos sobre el rendimiento "
        "y comportamiento de los instrumentos financieros."
    )
    
    if not data_manager:
        st.error("❌ Error: DataManager no disponible")
        return
    
    # Inicializar generador de reportes
    report_generator = ReportGenerator(data_manager)
    
    # Configuración del reporte
    st.subheader("⚙️ Configuración del Reporte")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        report_type = st.selectbox(
            "Tipo de Reporte",
            [
                "📈 Performance",
                "📊 Análisis Técnico", 
                "📉 Análisis de Volatilidad",
                "📋 Estado del Sistema"
            ],
            help="Selecciona el tipo de análisis que deseas generar"
        )
    
    with col2:
        # Obtener símbolos disponibles (solo si no es reporte del sistema)
        symbol_for_report = None
        if report_type != "📋 Estado del Sistema":
            try:
                available_symbols = data_manager.database.get_available_symbols()
                if not available_symbols:
                    st.warning("⚠️ No hay símbolos disponibles en la base de datos")
                    return
                
                symbol_for_report = st.selectbox(
                    "Símbolo",
                    available_symbols,
                    index=0,
                    help="Selecciona el símbolo para analizar"
                )
            except Exception as e:
                st.error(f"❌ Error obteniendo símbolos: {e}")
                return
        else:
            st.info("No se requiere símbolo para el reporte del sistema")
    
    with col3:
        date_range = None
        if report_type != "📋 Estado del Sistema":
            date_range = st.selectbox(
                "Rango de Fechas",
                ["Últimos 7 días", "Últimos 30 días", "Últimos 90 días", "Último año"],
                index=1,
                help="Selecciona el período de análisis"
            )
        else:
            st.info("Análisis en tiempo real")
    
    # Opciones de exportación
    with st.expander("📥 Opciones de Exportación", expanded=False):
        export_format = st.selectbox(
            "Formato de Exportación",
            ["Ver en pantalla", "Descargar CSV", "Descargar JSON"],
            help="Selecciona cómo deseas exportar los datos"
        )
    
    # Botón para generar reporte
    if st.button("🔄 Generar Reporte", type="primary", use_container_width=True):
        with st.spinner("Generando reporte..."):
            try:
                # Generar reporte según el tipo seleccionado
                if report_type == "📈 Performance":
                    report_generator.generate_performance_report(symbol_for_report, date_range)
                    
                elif report_type == "📊 Análisis Técnico":
                    report_generator.generate_technical_analysis_report(symbol_for_report, date_range)
                    
                elif report_type == "📉 Análisis de Volatilidad":
                    report_generator.generate_volatility_report(symbol_for_report, date_range)
                    
                elif report_type == "📋 Estado del Sistema":
                    report_generator.generate_system_report()
                
                # Manejar exportación de datos
                if export_format != "Ver en pantalla" and symbol_for_report:
                    _handle_data_export(report_generator, symbol_for_report, date_range, export_format)
                
                st.success("✅ Reporte generado exitosamente")
                
            except Exception as e:
                st.error(f"❌ Error generando reporte: {e}")
                with st.expander("Ver detalles del error"):
                    st.exception(e)


def _handle_data_export(report_generator: ReportGenerator, symbol: str, 
                       date_range: str, export_format: str) -> None:
    """Maneja la exportación de datos"""
    try:
        df = report_generator._get_data_for_report(symbol, date_range)
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


if __name__ == "__main__":
    # Para testing independiente
    st.set_page_config(
        page_title="Reportes Trading",
        page_icon="📋",
        layout="wide"
    )
    
    try:
        from src.data.data_manager import DataManager
        data_manager = DataManager()
        show_reports_page(data_manager)
    except Exception as e:
        st.error(f"Error inicializando la aplicación: {e}")