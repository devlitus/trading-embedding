# 📋 REGLAS DEL PROYECTO - TRADING EMBEDDING SYSTEM

## 🏗️ 1. FRAMEWORK Y VERSIONES DE DEPENDENCIAS

### Dependencias Principales
- **Python**: 3.8+ (requerido para compatibilidad con todas las librerías)
- **Streamlit**: >=1.28.0 (Dashboard y UI)
- **Plotly**: >=5.17.0 (Visualizaciones interactivas)
- **Pandas**: 2.1.4 (Manipulación de datos)
- **NumPy**: 1.24.3 (Cálculos numéricos)
- **python-binance**: 1.0.19 (API de Binance)
- **requests**: 2.31.0 (HTTP requests)

### Dependencias de Base de Datos y Caché
- **SQLite3**: Incluido en Python estándar (Base de datos principal)
- **Redis**: 5.0.1 (Caché distribuido opcional)
- **PyYAML**: 6.0.1 (Configuración)
- **python-dotenv**: 1.0.0 (Variables de entorno)

### Dependencias de Análisis Técnico
- **TA**: 0.11.0 (Indicadores técnicos)
- **SciPy**: 1.16.1 (Análisis científico)
- **python-dateutil**: 2.8.2 (Manejo de fechas)
- **pytz**: 2023.3 (Zonas horarias)

### Dependencias de Logging y Monitoreo
- **Loguru**: 0.7.2 (Sistema de logging avanzado)
- **psutil**: 5.9.6 (Métricas del sistema)

## 🧪 2. FRAMEWORK DE TESTING

### Framework Principal
- **pytest**: 7.4.3 (Framework de testing principal)
- **pytest-cov**: 4.1.0 (Cobertura de código)
- **pytest-mock**: 3.12.0 (Mocking para tests)

### Patrones de Testing
- **Estructura de Tests**: Directorio `tests/` en la raíz del proyecto
- **Nomenclatura**: `test_*.py` para archivos de test
- **Clases de Test**: Heredar de `unittest.TestCase`
- **Mocking**: Usar `unittest.mock` para simular APIs externas
- **Cobertura**: Mantener >80% de cobertura de código

### Tipos de Tests Requeridos
- **Tests Unitarios**: Para cada componente individual
- **Tests de Integración**: Para flujos completos
- **Tests de API**: Para endpoints de Binance (con mocking)
- **Tests de Base de Datos**: Con bases de datos temporales

## 🚫 3. APIS Y SERVICIOS RESTRINGIDOS

### APIs Prohibidas
- **APIs de Trading en Vivo**: Prohibido usar APIs que ejecuten trades reales sin autorización explícita
- **APIs de Pago**: Evitar servicios que requieran pagos automáticos
- **APIs sin Rate Limiting**: No usar APIs que no implementen rate limiting

### Servicios Externos Permitidos
- **Binance API**: Solo para datos históricos y en tiempo real (no trading)
- **APIs Gratuitas**: Preferir servicios con tiers gratuitos
- **APIs con Documentación**: Solo usar APIs bien documentadas

### Configuración de Rate Limiting
- **Binance API**: Máximo 1200 requests/minuto
- **Weight Limit**: Máximo 6000 weight/minuto
- **Retry Logic**: Implementar backoff exponencial
- **Timeout**: 30 segundos máximo por request

## 🏛️ 4. ARQUITECTURA Y PATRONES DE DISEÑO

### Estructura de Proyecto
```
src/
├── data/           # Capa de datos
├── analysis/       # Análisis técnico
├── ml/            # Machine Learning
├── api/           # API endpoints
├── config/        # Configuración
└── utils/         # Utilidades
```

### Patrones Obligatorios
- **Singleton**: Para ConfigManager y conexiones de BD
- **Strategy Pattern**: Para diferentes fuentes de datos
- **Factory Pattern**: Para creación de indicadores técnicos
- **Observer Pattern**: Para notificaciones en tiempo real

### Principios de Diseño
- **Single Responsibility**: Cada clase tiene una responsabilidad
- **Dependency Injection**: Inyectar dependencias en constructores
- **Interface Segregation**: Interfaces específicas y pequeñas
- **Open/Closed**: Abierto para extensión, cerrado para modificación

## 📊 5. GESTIÓN DE DATOS

### Base de Datos
- **Tipo**: SQLite para desarrollo, PostgreSQL para producción
- **Migraciones**: Usar scripts SQL versionados
- **Índices**: Obligatorios en (symbol, interval, timestamp)
- **Cleanup**: Retención automática de 365 días

### Caché
- **Tipo**: Memoria para desarrollo, Redis para producción
- **TTL por Tipo**:
  - OHLC Data: 1800 segundos
  - Pattern Analysis: 3600 segundos
  - Symbol Info: 86400 segundos
  - Technical Indicators: 900 segundos

### Validación de Datos
- **Duplicados**: Verificación obligatoria
- **Gaps**: Máximo 2 horas de diferencia
- **Anomalías**: Máximo 50% de cambio de precio
- **Formato**: Validación de tipos y rangos

## 🔧 6. CONFIGURACIÓN Y ENTORNOS

### Archivos de Configuración
- **config.yaml**: Configuración principal
- **.env**: Variables de entorno sensibles
- **.env.example**: Plantilla de variables de entorno

### Modos de Operación
- **Development**: `development.mode: true`
- **Production**: `development.mode: false`
- **Debug**: `development.debug_mode: true`
- **Cache Disabled**: `development.disable_cache: true`

### Variables de Entorno Requeridas
```bash
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
DATABASE_URL=sqlite:///data/trading.db
REDIS_URL=redis://localhost:6379/0
```

## 📝 7. ESTÁNDARES DE CÓDIGO

### Estilo de Código
- **PEP 8**: Seguir estándares de Python
- **Type Hints**: Obligatorio para funciones públicas
- **Docstrings**: Formato Google/NumPy para todas las clases y métodos
- **Imports**: Absolutos preferidos, relativos solo cuando sea necesario

### Nomenclatura
- **Clases**: PascalCase (ej: `DataManager`)
- **Funciones/Variables**: snake_case (ej: `get_data`)
- **Constantes**: UPPER_SNAKE_CASE (ej: `MAX_RETRIES`)
- **Archivos**: snake_case (ej: `data_manager.py`)

### Comentarios y Documentación
- **Comentarios**: Explicar el "por qué", no el "qué"
- **TODO**: Usar formato `# TODO: descripción`
- **FIXME**: Usar formato `# FIXME: descripción`
- **README**: Mantener actualizado con cada cambio mayor

## 🔒 8. SEGURIDAD

### Manejo de Credenciales
- **Nunca hardcodear**: API keys en código fuente
- **Variables de Entorno**: Usar para información sensible
- **.gitignore**: Incluir archivos de configuración sensibles
- **Rotación**: Rotar API keys regularmente

### Validación de Entrada
- **Sanitización**: Limpiar todos los inputs externos
- **Validación de Tipos**: Verificar tipos de datos
- **Rate Limiting**: Implementar en todas las APIs
- **Error Handling**: No exponer información sensible en errores

## 📊 9. LOGGING Y MONITOREO

### Configuración de Logs
- **Nivel**: INFO para producción, DEBUG para desarrollo
- **Formato**: `{time} | {level} | {name}:{function}:{line} | {message}`
- **Rotación**: Diaria con compresión
- **Retención**: 30 días para logs principales, 90 días para errores

### Métricas Obligatorias
- **API Calls**: Contar y medir latencia
- **Database Operations**: Tiempo de respuesta
- **Cache Hit Rate**: Porcentaje de aciertos
- **Memory Usage**: Uso de memoria del proceso

### Health Checks
- **Intervalo**: Cada 5 minutos
- **Componentes**: API, Database, Cache, File System
- **Alertas**: Configurar para fallos críticos

## 🚀 10. DESPLIEGUE Y CI/CD

### Contenedores
- **Docker**: Usar para entornos consistentes
- **docker-compose**: Para desarrollo local
- **Multi-stage builds**: Para optimizar tamaño de imagen

### Pipelines
- **Tests**: Ejecutar en cada commit
- **Linting**: Verificar estilo de código
- **Security Scan**: Escanear dependencias vulnerables
- **Coverage**: Reportar cobertura de tests

### Ambientes
- **Development**: Local con SQLite
- **Staging**: Réplica de producción
- **Production**: PostgreSQL + Redis

## 📚 11. DEPENDENCIAS FUTURAS (COMENTADAS)

### Machine Learning (Fase 4)
```python
# tensorflow==2.13.0
# scikit-learn==1.3.2
# sentence-transformers==2.2.2
```

### API y Interface (Fase 5)
```python
# fastapi==0.104.1
# uvicorn==0.24.0
```

### Análisis Avanzado
```python
# talib-binary==0.4.26  # Solo si se necesita TA-Lib nativo
# sqlalchemy==2.0.23    # Para ORM avanzado
# pydantic==2.5.0       # Para validación de datos
```

## ⚠️ 12. RESTRICCIONES Y LIMITACIONES

### Performance
- **Memoria**: Máximo 2GB por proceso
- **CPU**: No bloquear el hilo principal >1 segundo
- **Disk I/O**: Usar operaciones asíncronas cuando sea posible
- **Network**: Timeout de 30 segundos máximo

### Escalabilidad
- **Concurrent Users**: Diseñar para 10+ usuarios simultáneos
- **Data Volume**: Soportar hasta 1M registros por símbolo
- **API Calls**: Respetar límites de Binance API

### Compatibilidad
- **Python**: 3.8+ (no usar features de 3.9+)
- **OS**: Windows, Linux, macOS
- **Browsers**: Chrome 90+, Firefox 88+, Safari 14+

---

**Última actualización**: 2025-01-09  
**Versión**: 1.0  
**Mantenedor**: DevAgent  

> 💡 **Nota**: Estas reglas deben revisarse y actualizarse con cada fase del proyecto. Cualquier desviación debe ser documentada y justificada.