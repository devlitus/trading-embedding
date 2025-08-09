# 🚀 Aplicación de Trading con IA - Sistema de Embeddings

## 📋 Descripción

Sistema avanzado de trading que utiliza inteligencia artificial y embeddings para el análisis de patrones de Wyckoff en mercados de criptomonedas. El proyecto está diseñado en 5 fases de desarrollo, actualmente implementando la **Fase 1: Adquisición de Datos**.

## 🎯 Objetivos del Proyecto

- **Detección automática de patrones de Wyckoff** usando embeddings y ML
- **Sistema de etiquetado semi-automático** para crear datasets de calidad
- **Análisis técnico avanzado** con indicadores personalizados
- **API REST** para integración con sistemas externos
- **Interface web interactiva** para visualización y análisis

## 🏗️ Arquitectura del Proyecto

### Fases de Desarrollo

1. **✅ Fase 1: Adquisición de Datos** (Actual)
   - Conexión a Binance API
   - Base de datos SQLite
   - Sistema de caché
   - Gestión integrada de datos

2. **🔄 Fase 2: Análisis Técnico** (Próxima)
   - Indicadores técnicos
   - Detección de tendencias
   - Reconocimiento básico de patrones

3. **📊 Fase 3: Sistema de Etiquetado**
   - Reglas heurísticas para Wyckoff
   - Interface de anotación
   - Gestión de datasets

4. **🤖 Fase 4: Machine Learning**
   - Sistema de embeddings
   - Modelos de Wyckoff
   - Entrenamiento y evaluación

5. **🌐 Fase 5: API y Interface**
   - API REST con FastAPI
   - Dashboard con Streamlit
   - Visualizaciones interactivas

## 📁 Estructura del Proyecto

```
trading_embbeding/
├── src/
│   ├── data/                 # Módulo de datos (Fase 1)
│   │   ├── binance_client.py # Cliente de Binance API
│   │   ├── database.py       # Gestión de base de datos
│   │   ├── cache.py          # Sistema de caché
│   │   └── data_manager.py   # Gestor integrado de datos
│   ├── analysis/             # Análisis técnico (Fase 2)
│   ├── ml/                   # Machine Learning (Fase 4)
│   │   └── labeling/         # Sistema de etiquetado (Fase 3)
│   ├── api/                  # API REST (Fase 5)
│   └── utils/                # Utilidades generales
├── data/                     # Datos almacenados
├── models/                   # Modelos entrenados
├── examples/                 # Ejemplos y demos
├── tests/                    # Tests unitarios
├── logs/                     # Archivos de log
└── docs/                     # Documentación
```

## 🚀 Instalación y Configuración

### Prerrequisitos

- Python 3.8+
- Cuenta en Binance (para API keys)
- Git

### Instalación

1. **Clonar el repositorio**
   ```bash
   git clone <repository-url>
   cd trading_embbeding
   ```

2. **Crear entorno virtual**
   ```bash
   python -m venv venv
   
   # Windows
   venv\Scripts\activate
   
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Instalar dependencias**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configurar variables de entorno**
   ```bash
   # Copiar archivo de ejemplo
   cp .env.example .env
   
   # Editar .env con tus API keys de Binance
   # BINANCE_API_KEY=tu_api_key
   # BINANCE_API_SECRET=tu_api_secret
   ```

5. **Crear directorios necesarios**
   ```bash
   mkdir -p data logs
   ```

## 🎮 Uso - Fase 1

### Dashboard Interactivo (Recomendado)

```bash
# Iniciar el dashboard web
python run_dashboard.py
# O alternativamente:
streamlit run dashboard/dashboard.py
```

El dashboard estará disponible en: http://localhost:8501

### Demo Completo

Ejecuta el demo completo de la Fase 1:

```bash
python examples/fase1_demo.py
```

Este demo demuestra:
- ✅ Health check del sistema
- ✅ Obtención de datos de un símbolo
- ✅ Rendimiento del caché
- ✅ Obtención masiva de datos
- ✅ Recuperación de datos almacenados
- ✅ Estadísticas de BD y caché

### Uso Programático

```python
from src.data.data_manager import DataManager

# Inicializar gestor de datos
data_manager = DataManager(
    db_path="data/trading.db",
    use_cache=True,
    cache_ttl=3600
)

# Health check
health = data_manager.health_check()
print(f"Estado del sistema: {health}")

# Obtener datos de un símbolo
result = data_manager.fetch_and_store_data(
    symbol='BTCUSDT',
    interval='1h',
    days_back=30
)
print(f"Datos obtenidos: {result['records_count']} registros")

# Recuperar datos almacenados
df = data_manager.get_data('BTCUSDT', '1h', limit=100)
print(f"Datos en memoria: {len(df)} registros")

# Obtener múltiples símbolos
symbols = ['ETHUSDT', 'ADAUSDT', 'DOTUSDT']
results = data_manager.bulk_fetch_symbols(symbols, '4h', days_back=7)
print(f"Procesados {len(results)} símbolos")
```

### Componentes Individuales

#### Cliente de Binance

```python
from src.data.binance_client import BinanceClient

client = BinanceClient()

# Obtener datos históricos
df = client.get_historical_data('BTCUSDT', '1h', days_back=7)
print(f"Datos obtenidos: {len(df)} registros")

# Información del símbolo
info = client.get_symbol_info('BTCUSDT')
print(f"Símbolo: {info['symbol']}, Estado: {info['status']}")
```

#### Base de Datos

```python
from src.data.database import TradingDatabase

db = TradingDatabase('data/trading.db')

# Obtener símbolos disponibles
symbols = db.get_available_symbols()
print(f"Símbolos en BD: {symbols}")

# Obtener rango de datos
range_info = db.get_data_range('BTCUSDT', '1h')
print(f"Rango de datos: {range_info}")
```

#### Sistema de Caché

```python
from src.data.cache import TradingCache

cache = TradingCache(use_redis=False)

# Estadísticas del caché
stats = cache.get_cache_stats()
print(f"Estadísticas: {stats}")
```

## 📊 Configuración

El sistema utiliza un archivo `config.yaml` para toda la configuración:

- **Binance API**: Configuración de conexión y rate limiting
- **Base de Datos**: Configuración de SQLite y tablas
- **Caché**: Configuración de TTL y tipos de caché
- **Logging**: Niveles y archivos de log
- **Datos**: Configuración de obtención y validación
- **Monitoreo**: Health checks y métricas

## 🧪 Testing

```bash
# Ejecutar todos los tests
pytest

# Tests con cobertura
pytest --cov=src

# Tests específicos
pytest tests/test_data/
```

## 📈 Monitoreo y Logging

### Logs

Los logs se almacenan en el directorio `logs/`:

- `trading_app.log`: Log principal de la aplicación
- `api_calls.log`: Log específico de llamadas a API
- `errors.log`: Log de errores

### Health Checks

```python
# Verificar estado del sistema
health = data_manager.health_check()

# Componentes verificados:
# - Binance API (conectividad)
# - Base de datos (acceso)
# - Caché (funcionamiento)
```

### Métricas

```python
# Estadísticas de base de datos
db_stats = data_manager.get_database_stats()

# Estadísticas de caché
cache_stats = data_manager.get_cache_stats()
```

## 🔧 Desarrollo

### Estructura de Código

- **Modular**: Cada componente es independiente
- **Configurable**: Toda la configuración en archivos externos
- **Testeable**: Diseño que facilita testing
- **Escalable**: Preparado para crecimiento

### Estándares de Código

- **PEP 8**: Estilo de código Python
- **Type Hints**: Tipado estático
- **Docstrings**: Documentación en código
- **Logging**: Trazabilidad completa

### Contribución

1. Fork del repositorio
2. Crear rama de feature (`git checkout -b feature/nueva-funcionalidad`)
3. Commit de cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## 📋 Roadmap

### ✅ Fase 1: Adquisición de Datos (Completada)
- [x] Cliente de Binance API
- [x] Base de datos SQLite
- [x] Sistema de caché
- [x] Gestor integrado de datos
- [x] Configuración y logging
- [x] Tests y documentación

### 🔄 Próximas Fases

- **Fase 2**: Análisis técnico e indicadores
- **Fase 3**: Sistema de etiquetado semi-automático
- **Fase 4**: Machine Learning y embeddings
- **Fase 5**: API REST e interface web

## 🤝 Soporte

- **Documentación**: Ver carpeta `docs/`
- **Ejemplos**: Ver carpeta `examples/`
- **Issues**: Reportar problemas en GitHub
- **Discusiones**: Usar GitHub Discussions

## 📄 Licencia

[Especificar licencia]

## 🙏 Agradecimientos

- Binance por su API pública
- Comunidad de Python por las librerías utilizadas
- Metodología de Wyckoff para el análisis de mercados

---

**Estado del Proyecto**: 🟢 Fase 1 Completada - Lista para Producción

**Próximo Milestone**: Fase 2 - Análisis Técnico