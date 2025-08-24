#!/usr/bin/env python3
"""
Configuración y constantes para la página de inicio
"""

class HomeConfig:
    """Configuración centralizada para la página de inicio"""
    
    # Títulos y textos
    PAGE_TITLE = "🏠 Dashboard de Trading - Inicio"
    WELCOME_TITLE = "### Bienvenido al Sistema de Trading Integrado"
    
    # Secciones de información
    SYSTEM_DESCRIPTION = """
Este dashboard proporciona un análisis completo de datos de trading con las siguientes funcionalidades:

**📊 Fase 1 - Adquisición de Datos:**
- Conexión en tiempo real con Binance
- Almacenamiento en base de datos SQLite
- Sistema de caché para optimización

**🔍 Fase 2 - Análisis Técnico:**
- Indicadores técnicos avanzados
- Análisis de patrones de precios
- Visualizaciones interactivas

**✅ Verificación del Sistema:**
- Monitoreo de salud de componentes
- Validación de datos
- Métricas de rendimiento

**📈 Análisis Técnico:**
- RSI, MACD, Bandas de Bollinger
- Medias móviles y tendencias
- Señales de compra/venta

**🎯 Monitoreo en Tiempo Real:**
- Datos en vivo de mercado
- Alertas automáticas
- Dashboard interactivo

**📋 Reportes:**
- Análisis histórico
- Métricas de rendimiento
- Exportación de datos
"""
    
    # Secciones de la página
    SECTIONS = {
        'system_status': "📊 Estado del Sistema",
        'database_info': "💾 Información de la Base de Datos",
        'system_info': "ℹ️ Información del Sistema"
    }
    
    # Información de tecnologías
    TECHNOLOGIES = """
**Tecnologías utilizadas:**
- Python 3.x
- Streamlit
- SQLite
- Plotly
- Pandas
- Binance API
"""
    
    # Características del sistema
    FEATURES = """
**Características:**
- Datos en tiempo real
- Análisis técnico avanzado
- Interfaz web interactiva
- Sistema de caché optimizado
- Monitoreo de salud
"""
    
    # Mensajes de estado
    MESSAGES = {
        'db_connected': "✅ Base de datos conectada con {count} símbolos",
        'no_data': "⚠️ No hay datos disponibles en la base de datos",
        'db_error': "❌ Error accediendo a la base de datos: {error}",
        'no_connection': "❌ No se pudo conectar al gestor de datos"
    }
    
    # Etiquetas para estadísticas
    STATS_LABELS = {
        'symbols_available': "**Símbolos disponibles:**",
        'general_stats': "**Estadísticas generales:**",
        'total_records': "• Total de registros: {count:,}",
        'unique_symbols': "• Símbolos únicos: {count}",
        'latest_record': "• Último registro: {timestamp}"
    }