#!/usr/bin/env python3
"""
Lógica de datos para Fase 1 - Adquisición de Datos
"""

import streamlit as st
import pandas as pd
from datetime import datetime, date
from .phase1_config import DATA_LIMIT, FALLBACK_LIMIT
from src.utils.data_utils import validate_data_quality as validate_data_quality_core

def get_data_manager():
    """Obtener instancia del DataManager."""
    try:
        from data.data_manager import DataManager
        return DataManager()
    except Exception as e:
        st.error(f"❌ Error al inicializar DataManager: {e}")
        return None

def get_current_day_data(data_manager, symbol, interval):
    """Obtener datos del día actual."""
    try:
        # Obtener datos con límite
        df = data_manager.get_data(symbol, interval, limit=DATA_LIMIT)
        
        if df.empty:
            st.warning(f"⚠️ No hay datos disponibles para {symbol} en intervalo {interval}")
            return pd.DataFrame()
        
        # Filtrar solo datos del día actual
        today = date.today()
        df_today = df[df.index.date == today]
        
        if df_today.empty:
            st.info(f"ℹ️ No hay datos del día actual. Mostrando últimos {FALLBACK_LIMIT} registros.")
            return df.tail(FALLBACK_LIMIT)
        
        return df_today
        
    except Exception as e:
        st.error(f"❌ Error al obtener datos: {e}")
        return pd.DataFrame()

def refresh_system_status(data_manager):
    """Refrescar estado del sistema."""
    if st.button("🔄 Refrescar Estado", key="refresh_status"):
        try:
            # Limpiar caché de Streamlit
            st.cache_data.clear()
            
            # Mostrar mensaje de éxito
            st.success("✅ Estado del sistema actualizado")
            
            # Rerun para actualizar la página
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ Error al refrescar: {e}")

def truncate_database(data_manager):
    """Truncar base de datos con confirmación."""
    if st.button("🗑️ Truncar Base de Datos", key="truncate_db", type="secondary"):
        if st.session_state.get('confirm_truncate', False):
            try:
                # Ejecutar truncate
                data_manager.truncate_all_data()
                st.success("✅ Base de datos truncada exitosamente")
                
                # Reset confirmation
                st.session_state['confirm_truncate'] = False
                
                # Limpiar caché
                st.cache_data.clear()
                
                # Rerun para actualizar
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Error al truncar: {e}")
                st.session_state['confirm_truncate'] = False
        else:
            st.session_state['confirm_truncate'] = True
            st.warning("⚠️ Haz clic nuevamente para confirmar el truncado")

def get_system_stats(data_manager):
    """Obtener estadísticas del sistema."""
    try:
        db_stats = data_manager.get_database_stats()
        cache_stats = data_manager.get_cache_stats()
        
        return {
            'database': db_stats,
            'cache': cache_stats,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    except Exception as e:
        st.error(f"❌ Error al obtener estadísticas: {e}")
        return None

def show_system_stats_details(stats):
    """Mostrar detalles de estadísticas del sistema."""
    if not stats:
        return
    
    st.markdown("#### 📊 Estadísticas del Sistema")
    
    # Estadísticas de base de datos
    db_col, cache_col = st.columns(2)
    
    with db_col:
        st.markdown("**🗄️ Base de Datos**")
        db_stats = stats['database']
        st.write(f"• Símbolos: {db_stats.get('total_symbols', 0)}")
        st.write(f"• Registros: {db_stats.get('total_records', 0):,}")
        st.write(f"• Tamaño: {db_stats.get('database_size_mb', 0):.2f} MB")
    
    with cache_col:
        st.markdown("**💾 Caché**")
        cache_stats = stats['cache']
        st.write(f"• Claves: {cache_stats.get('total_keys', 0)}")
        st.write(f"• Hit Rate: {cache_stats.get('hit_rate', 0):.1f}%")
        st.write(f"• Memoria: {cache_stats.get('memory_usage_mb', 0):.2f} MB")
    
    st.caption(f"Última actualización: {stats['timestamp']}")

def validate_data_quality(df, symbol, interval):
    """Validar calidad de los datos usando función consolidada."""
    if df.empty:
        return {'status': 'error', 'message': 'No hay datos disponibles'}
    
    # Usar función consolidada de data_utils
    quality_metrics = validate_data_quality_core(df)
    
    issues = []
    
    # Convertir métricas a formato de issues
    if quality_metrics['missing_values'] > 0:
        issues.append(f"Se encontraron {quality_metrics['missing_values']} valores nulos")
    
    if quality_metrics['duplicate_timestamps'] > 0:
        issues.append(f"Se encontraron {quality_metrics['duplicate_timestamps']} timestamps duplicados")
    
    if quality_metrics.get('invalid_candles', 0) > 0:
        issues.append("Precios máximos menores que mínimos detectados")
    
    if quality_metrics.get('extreme_price_changes', 0) > 0:
        issues.append(f"Se detectaron {quality_metrics['extreme_price_changes']} cambios de precio extremos")
    
    # Verificar volumen negativo si existe la columna
    if 'volume' in df.columns and (df['volume'] < 0).any():
        issues.append("Volúmenes negativos detectados")
    
    if issues:
        return {'status': 'warning', 'issues': issues}
    else:
        return {'status': 'ok', 'message': 'Datos válidos'}

def show_data_quality_report(df, symbol, interval):
    """Mostrar reporte de calidad de datos."""
    quality = validate_data_quality(df, symbol, interval)
    
    if quality['status'] == 'ok':
        st.success(f"✅ Calidad de datos: {quality['message']}")
    elif quality['status'] == 'warning':
        st.warning("⚠️ Problemas de calidad detectados:")
        for issue in quality['issues']:
            st.write(f"• {issue}")
    else:
        st.error(f"❌ Error de calidad: {quality['message']}")