# 📈 Dashboard de Trading - Fase 1

Dashboard interactivo desarrollado con Streamlit para monitorear y visualizar el sistema de adquisición de datos de la Fase 1.

## 🚀 Inicio Rápido

### Opción 1: Script de inicio automático
```bash
python run_dashboard.py
```

### Opción 2: Comando directo
```bash
streamlit run dashboard/dashboard.py
```

### Opción 3: Con configuración específica
```bash
python -m streamlit run dashboard/dashboard.py --server.port=8501 --server.address=localhost
```

## 🌐 Acceso

Una vez iniciado, el dashboard estará disponible en:
- **URL:** http://localhost:8501
- **Puerto:** 8501 (configurable)

## 📋 Funcionalidades

### 🏥 Estado del Sistema
- **Health Check:** Verificación del estado de todos los componentes
- **Binance API:** Estado de conexión con la API
- **Base de Datos:** Estado de la base de datos SQLite
- **Caché:** Estado del sistema de caché (memoria/Redis)

### 💾 Base de Datos
- **Estadísticas generales:** Número de símbolos, registros totales
- **Detalle por símbolo:** Información específica de cada par de trading
- **Rangos temporales:** Fechas de primer y último registro
- **Intervalos disponibles:** Timeframes almacenados

### 🚀 Caché
- **Estado del caché:** Activo/Inactivo
- **Estadísticas de rendimiento:** Hits, misses, hit rate
- **Número de claves:** Elementos almacenados en caché

### 📈 Visualización de Datos
- **Gráficos de velas:** Visualización OHLC interactiva
- **Gráficos de volumen:** Análisis de volumen de trading
- **Métricas en tiempo real:** Precio actual, cambios, máximos/mínimos
- **Datos tabulares:** Vista detallada de los datos históricos
- **Selector de símbolos:** Cambio dinámico entre pares de trading
- **Selector de intervalos:** 1h, 4h, 1d

### 📥 Adquisición de Datos
- **Interfaz de descarga:** Obtener datos de nuevos símbolos
- **Configuración flexible:** Símbolo, intervalo, días hacia atrás
- **Feedback en tiempo real:** Progreso y resultados de la descarga
- **Estadísticas de descarga:** Tiempo de ejecución, número de registros

### 🔍 Monitoreo del Sistema
- **Auto-refresh:** Actualización automática cada 30 segundos
- **Refresh manual:** Botón de actualización bajo demanda
- **Monitoreo continuo:** Seguimiento del estado del sistema

## 🎛️ Navegación

El dashboard utiliza una barra lateral para la navegación entre secciones:

1. **🏥 Estado del Sistema** - Health checks y estado general
2. **💾 Base de Datos** - Estadísticas y contenido de la BD
3. **🚀 Caché** - Rendimiento y estadísticas del caché
4. **📈 Visualización** - Gráficos y análisis de datos
5. **📥 Adquisición** - Descarga de nuevos datos
6. **🔍 Monitoreo** - Supervisión continua del sistema

## 🔧 Configuración

### Variables de Entorno
El dashboard utiliza la misma configuración que el resto del sistema:
- `.env` - Variables de entorno
- `config.yaml` - Configuración principal

### Dependencias
```txt
streamlit>=1.28.0
plotly>=5.17.0
pandas>=2.0.0
```

## 📊 Características Técnicas

### Rendimiento
- **Caché de recursos:** Los componentes pesados se cachean automáticamente
- **Lazy loading:** Carga bajo demanda de datos grandes
- **Optimización de consultas:** Límites en las consultas a la BD

### Responsividad
- **Layout adaptativo:** Se ajusta a diferentes tamaños de pantalla
- **Columnas dinámicas:** Distribución inteligente del contenido
- **Gráficos interactivos:** Zoom, pan, selección de rangos

### Seguridad
- **Localhost only:** Por defecto solo accesible localmente
- **No exposición de datos sensibles:** API keys y secretos protegidos
- **Validación de entrada:** Sanitización de inputs del usuario

## 🐛 Solución de Problemas

### Error: Módulo no encontrado
```bash
# Instalar dependencias
pip install -r requirements.txt
```

### Error: Puerto en uso
```bash
# Usar puerto diferente
streamlit run dashboard/dashboard.py --server.port=8502
```

### Error: Base de datos no encontrada
```bash
# Ejecutar primero el demo para crear datos
python examples/fase1_demo.py
```

### Dashboard no carga
1. Verificar que todas las dependencias están instaladas
2. Comprobar que el puerto 8501 está libre
3. Revisar los logs en la terminal
4. Verificar que la estructura de directorios es correcta

## 📝 Logs y Debug

### Logs del Sistema
Los logs se almacenan en:
- `logs/trading_system.log` - Log principal
- `logs/data_acquisition.log` - Log de adquisición de datos

### Debug Mode
Para activar el modo debug:
```bash
streamlit run dashboard/dashboard.py --logger.level=debug
```

## 🔄 Actualizaciones

Para actualizar el dashboard:
1. Detener el servidor (Ctrl+C)
2. Actualizar el código
3. Reiniciar el dashboard

## 📞 Soporte

Si encuentras problemas:
1. Revisar esta documentación
2. Verificar los logs del sistema
3. Comprobar la configuración
4. Ejecutar los tests: `python tests/test_fase1.py`

## 🎯 Próximas Funcionalidades

- [ ] Alertas en tiempo real
- [ ] Exportación de datos
- [ ] Configuración desde la interfaz
- [ ] Múltiples temas visuales
- [ ] Comparación entre símbolos
- [ ] Análisis técnico básico

---

**Desarrollado para el Sistema de Trading AI - Fase 1**  
*Dashboard v1.0.0*