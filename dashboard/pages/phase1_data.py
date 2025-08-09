#!/usr/bin/env python3
"""
Página de Fase 1 - Adquisición de Datos
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
from pages.common import get_plotly_config, CUSTOM_CSS

def show_phase1_data_page(data_manager):
    """Mostrar la página de Fase 1 - Adquisición de Datos"""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    
    # Header en una sola columna
    st.title("📊 Fase 1 - Adquisición de Datos")
    
    # Métricas principales en una sola fila
    if data_manager:
        try:
            db_stats = data_manager.get_database_stats()
            cache_stats = data_manager.get_cache_stats()
            
            # Métricas en una sola fila
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            with metric_col1:
                st.metric("Símbolos", db_stats.get('total_symbols', 0), delta=None, delta_color="normal")
            with metric_col2:
                st.metric("Registros", f"{db_stats.get('total_records', 0):,}", delta=None, delta_color="normal")
            with metric_col3:
                st.metric("Caché", cache_stats.get('total_keys', 0), delta=None, delta_color="normal")
        except:
            pass
    
    if not data_manager:
        st.error("Error: DataManager no disponible")
        st.stop()
    
    # Panel de control en 1 columna con 2 filas
    st.markdown("### ⚙️ Panel de Control y Estado")
    
    # Primera fila: Selectores de configuración
    st.markdown("**🔧 Configuración**")
    control_col1, control_col2, control_col3 = st.columns(3)
    
    with control_col1:
        symbol = st.selectbox(
            "Símbolo",
            ["BTCUSDT", "ETHUSDT", "ADAUSDT", "BNBUSDT", "XRPUSDT", "SOLUSDT"],
            index=0,
            key="phase1_symbol"
        )
    
    with control_col2:
        interval = st.selectbox(
            "Intervalo",
            ["1m", "5m", "15m", "1h", "4h", "1d", "1w"],
            index=5,
            key="phase1_interval"
        )
    
    with control_col3:
        days_back = st.number_input("Días", min_value=1, max_value=365, value=30, key="phase1_days")
    
    st.markdown("---")  # Separador visual entre filas
    
    # Segunda fila: Estados automáticos
    st.markdown("**📊 Estado del Sistema**")
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.markdown("**🔄 Estado de Datos**")
        try:
            # Intentar obtener datos automáticamente
            result = data_manager.fetch_and_store_data(
                symbol=symbol,
                interval=interval,
                days_back=days_back
            )
            
            if 'error' in result:
                st.error(f"❌ Error")
                st.caption(f"Error: {result['error'][:30]}...")
            else:
                records_count = result.get('records_count', 0)
                inserted_count = result.get('inserted_count', records_count)
                st.success(f"✅ {records_count:,} registros")
                st.caption(f"({inserted_count:,} nuevos)")
        except Exception as e:
            st.error("❌ Error")
            st.caption(f"Error: {str(e)[:30]}...")
    
    with status_col2:
        st.markdown("**📊 Visualización**")
        try:
            df = data_manager.get_data(symbol, interval, limit=200)
            if not df.empty:
                current_price = df['close'].iloc[-1]
                change_pct = ((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100)
                st.success(f"✅ ${current_price:.4f}")
                color = "green" if change_pct >= 0 else "red"
                st.markdown(f"<span style='color: {color}'>{change_pct:+.2f}%</span>", unsafe_allow_html=True)
            else:
                st.warning("⚠️ Sin datos")
                st.caption("No disponible")
        except Exception as e:
            st.error("❌ Error")
            st.caption(f"Error: {str(e)[:30]}...")
    
    with status_col3:
        st.markdown("**🔍 Health Check**")
        try:
            health = data_manager.health_check()
            all_healthy = all(component['status'] == 'healthy' for component in health['components'].values())
            
            if all_healthy:
                st.success("✅ Sistema OK")
                st.caption("Todo operativo")
            else:
                st.error("❌ Problemas")
                st.caption("Ver detalles abajo")
        except Exception as e:
            st.error("❌ Error")
            st.caption(f"Error: {str(e)[:30]}...")
    
    # Área principal - Visualización automática de datos a pantalla completa
    try:
        df = data_manager.get_data(symbol, interval, limit=200)
        if not df.empty:
            # Gráfico principal a pantalla completa
            st.markdown(f"### 📈 {symbol} - {interval} (Últimos {len(df)} registros)")
            
            # Crear gráfico de velas optimizado para pantalla completa
            fig = go.Figure()
            
            # Candlestick principal
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name=symbol,
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ))
            
            # Agregar volumen como gráfico secundario
            fig.add_trace(go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volumen',
                yaxis='y2',
                opacity=0.3,
                marker_color='#1f77b4'
            ))
            
            # Layout optimizado para pantalla completa
            fig.update_layout(
                title={
                    'text': f"{symbol} - {interval} | Precio: ${df['close'].iloc[-1]:.4f} | Cambio: {((df['close'].iloc[-1] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100):.2f}%",
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 20}
                },
                yaxis=dict(
                    title="Precio (USDT)",
                    side='left',
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)'
                ),
                yaxis2=dict(
                    title="Volumen",
                    overlaying='y',
                    side='right',
                    showgrid=False
                ),
                xaxis=dict(
                    title="Tiempo",
                    showgrid=True,
                    gridcolor='rgba(128,128,128,0.2)',
                    rangeslider=dict(visible=False)
                ),
                height=700,  # Altura aumentada para pantalla completa
                hovermode='x unified',
                template='plotly_white',
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            st.plotly_chart(fig, use_container_width=True, config=get_plotly_config())
            
            # Estadísticas rápidas del dataset actual
            stats_col1, stats_col2, stats_col3, stats_col4, stats_col5 = st.columns(5)
            
            with stats_col1:
                st.metric("Precio Actual", f"${df['close'].iloc[-1]:.4f}")
            with stats_col2:
                change_pct = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0] * 100)
                st.metric("Cambio Total", f"{change_pct:.2f}%", delta=f"{change_pct:.2f}%")
            with stats_col3:
                st.metric("Máximo", f"${df['high'].max():.4f}")
            with stats_col4:
                st.metric("Mínimo", f"${df['low'].min():.4f}")
            with stats_col5:
                st.metric("Volumen Prom.", f"{df['volume'].mean():,.0f}")
            
            # Mostrar detalles del health check si hay problemas
            try:
                health = data_manager.health_check()
                all_healthy = all(component['status'] == 'healthy' for component in health['components'].values())
                
                if not all_healthy:
                    st.warning("⚠️ Se detectaron problemas en el sistema")
                    with st.expander("Detalles del Health Check"):
                        st.json(health)
            except Exception as e:
                st.error(f"Error en health check: {e}")
            
            # Estado del Sistema - En una sola columna
            st.markdown("---")
            st.markdown("### 📊 Estado del Sistema")
            
            try:
                db_stats = data_manager.get_database_stats()
                cache_stats = data_manager.get_cache_stats()
                
                # Panel de métricas del sistema en una sola fila
                st.markdown(f"**Base de Datos:** {db_stats.get('total_symbols', 0)} símbolos, {db_stats.get('total_records', 0):,} registros | **Caché:** {cache_stats.get('total_keys', 0)} elementos, {cache_stats.get('memory_usage', 0)} MB | **Actualizado:** {datetime.now().strftime('%H:%M:%S')}")
                
                if st.button("🔄 Refrescar Estado", key="refresh_system_status"):
                    st.rerun()
                
                # Tabla compacta de símbolos disponibles
                if db_stats.get('symbol_details'):
                    st.markdown("#### 📋 Símbolos Disponibles")
                    
                    symbols_data = []
                    for sym, info in db_stats['symbol_details'].items():
                        symbols_data.append({
                            'Símbolo': sym,
                            'Registros': f"{info['total_records']:,}",
                            'Último': info.get('latest_timestamp', 'N/A')[:10] if info.get('latest_timestamp') != 'N/A' else 'N/A'
                        })
                    
                    symbols_df = pd.DataFrame(symbols_data)
                    st.dataframe(symbols_df, use_container_width=True, height=200)
            
            except Exception as e:
                st.error(f"Error obteniendo estadísticas del sistema: {e}")
            
        else:
            st.warning(f"No hay datos disponibles para {symbol}. Los datos se están obteniendo automáticamente.")
            st.info("💡 Cambia el símbolo o intervalo para actualizar los datos automáticamente.")
            
    except Exception as e:
        st.error(f"Error mostrando datos: {e}")
        st.info("💡 Intenta cambiar el símbolo o intervalo para recargar los datos.")