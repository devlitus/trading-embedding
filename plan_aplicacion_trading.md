# Plan de Aplicación de Trading con AI

## Descripción General
Aplicación para obtener datos OHLC de Binance, realizar análisis técnico y usar inteligencia artificial para detectar patrones financieros como Wyckoff.

## Arquitectura Simplificada

### 1. Módulo de Datos (Data Layer)
- **API Binance**: Conexión para obtener datos OHLC
- **Base de Datos**: Almacenamiento local de datos históricos
- **Cache**: Sistema de caché para optimizar consultas

### 2. Módulo de Análisis Técnico
- **Indicadores Básicos**: RSI, MACD, Bollinger Bands
- **Detección de Tendencias**: Identificación de cambios de tendencia
- **Patrones Básicos**: Soporte, resistencia, triángulos

### 3. Módulo de AI/ML
- **Preprocesamiento**: Normalización y feature engineering
- **Sistema de Embeddings**: Representación vectorial de patrones de mercado
- **Modelo de Detección**: Red neuronal para patrones Wyckoff
- **Entrenamiento**: Sistema de entrenamiento continuo

### 4. API/Interface
- **REST API**: Endpoints para consultas
- **Dashboard Simple**: Visualización básica

## Tecnologías Propuestas

### Backend
- **Python**: Lenguaje principal
- **FastAPI**: Framework web ligero
- **SQLite**: Base de datos simple
- **Pandas**: Manipulación de datos
- **TA-Lib**: Indicadores técnicos

### Machine Learning
- **TensorFlow/Keras**: Modelo de AI y embeddings
- **Scikit-learn**: Preprocesamiento
- **NumPy**: Cálculos numéricos
- **Sentence-Transformers**: Embeddings avanzados

### Datos
- **python-binance**: Cliente API de Binance
- **ccxt**: Alternativa para múltiples exchanges

### Etiquetado y Anotación
- **Streamlit**: Dashboard de anotación manual
- **Plotly**: Visualización interactiva de gráficos
- **SQLAlchemy**: ORM para gestión del dataset
- **Pydantic**: Validación de esquemas de etiquetas

## Estructura del Proyecto

```
trading_app/
├── src/
│   ├── data/
│   │   ├── binance_client.py
│   │   ├── database.py
│   │   └── cache.py
│   ├── analysis/
│   │   ├── technical_indicators.py
│   │   ├── trend_detection.py
│   │   └── pattern_recognition.py
│   ├── ml/
│   │   ├── data_preprocessing.py
│   │   ├── embeddings.py
│   │   ├── wyckoff_model.py
│   │   ├── training.py
│   │   └── labeling/
│   │       ├── heuristic_rules.py
│   │       ├── scoring_system.py
│   │       ├── annotation_tool.py
│   │       └── dataset_manager.py
│   ├── api/
│   │   ├── main.py
│   │   └── endpoints.py
│   └── utils/
│       └── helpers.py
├── data/
│   └── historical/
├── models/
│   └── trained_models/
├── tests/
├── requirements.txt
└── README.md
```

## Funcionalidades Principales

### 1. Obtención de Datos
- Conexión a API de Binance
- Descarga de datos OHLC históricos
- Actualización en tiempo real
- Almacenamiento local

### 2. Análisis Técnico
- Cálculo de indicadores técnicos
- Detección de cambios de tendencia
- Identificación de niveles de soporte/resistencia

### 3. Detección de Patrones Wyckoff
- **Acumulación**: Identificar fases de acumulación
- **Distribución**: Detectar fases de distribución
- **Markup/Markdown**: Reconocer movimientos direccionales
- **Re-acumulación/Re-distribución**: Patrones secundarios

### 4. Modelo de AI
- Entrenamiento con datos históricos
- Clasificación de patrones
- Predicción de probabilidades
- Validación cruzada

## Implementación por Fases

### Fase 1: Fundamentos (Semana 1-2)
- [ ] Configurar entorno de desarrollo
- [ ] Implementar cliente de Binance
- [ ] Crear base de datos básica
- [ ] Obtener y almacenar datos OHLC

### Fase 2: Análisis Técnico (Semana 3-4)
- [ ] Implementar indicadores técnicos básicos
- [ ] Crear sistema de detección de tendencias
- [ ] Desarrollar identificación de patrones simples

### Fase 3: Sistema de Etiquetado (Semana 5-6)
- [ ] Implementar reglas heurísticas para detección automática
- [ ] Crear sistema de puntuación y confianza
- [ ] Desarrollar herramienta de anotación manual
- [ ] Generar dataset inicial etiquetado
- [ ] Validar calidad del dataset

### Fase 4: Machine Learning (Semana 7-8)
- [ ] Preparar datos etiquetados para entrenamiento
- [ ] Implementar sistema de embeddings
- [ ] Crear modelo básico de clasificación
- [ ] Entrenar modelo con patrones Wyckoff
- [ ] Validar y optimizar modelo

### Fase 5: API y Interface (Semana 9-10)
- [ ] Crear API REST
- [ ] Implementar endpoints principales
- [ ] Desarrollar dashboard básico
- [ ] Integrar herramienta de anotación
- [ ] Testing y documentación

## Dataset de Etiquetas para Wyckoff

### Estrategia de Etiquetado Semiautomático

Un sistema híbrido que combina reglas heurísticas automáticas con validación y refinamiento manual para crear un dataset de alta calidad.

### Componentes del Sistema de Etiquetado

#### 1. Motor de Reglas Heurísticas

**Reglas para Acumulación:**
```python
class WyckoffAccumulationRules:
    def detect_preliminary_support(self, data):
        # PS: Alto volumen + caída de precio + rebote
        return (
            data['volume'] > data['volume'].rolling(20).mean() * 1.5 and
            data['close'] < data['low'].rolling(10).min() * 1.02 and
            data['close'] > data['open']
        )
    
    def detect_selling_climax(self, data):
        # SC: Volumen extremo + caída dramática + reversión
        return (
            data['volume'] > data['volume'].rolling(50).quantile(0.95) and
            (data['low'] - data['close'].shift(1)) / data['close'].shift(1) < -0.05 and
            data['close'] > data['low'] * 1.02
        )
```

**Reglas para Distribución:**
```python
class WyckoffDistributionRules:
    def detect_preliminary_supply(self, data):
        # PSY: Alto volumen + subida + rechazo
        return (
            data['volume'] > data['volume'].rolling(20).mean() * 1.3 and
            data['high'] > data['high'].rolling(10).max() * 0.98 and
            data['close'] < data['high'] * 0.97
        )
```

#### 2. Sistema de Puntuación y Confianza

```python
class WyckoffScoring:
    def calculate_phase_confidence(self, signals):
        weights = {
            'volume_confirmation': 0.3,
            'price_action': 0.25,
            'technical_indicators': 0.2,
            'market_structure': 0.15,
            'time_duration': 0.1
        }
        
        confidence_score = sum(
            signals[key] * weights[key] 
            for key in weights.keys()
        )
        
        return min(confidence_score, 1.0)
```

#### 3. Interfaz de Validación Manual

**Herramienta de Anotación:**
- Dashboard web con gráficos interactivos
- Visualización de señales automáticas
- Controles para confirmar/rechazar/modificar etiquetas
- Sistema de comentarios para casos especiales

#### 4. Pipeline de Etiquetado

```
1. Detección Automática
   ├── Aplicar reglas heurísticas
   ├── Calcular scores de confianza
   └── Filtrar por umbral mínimo (>0.6)

2. Pre-validación
   ├── Verificar coherencia temporal
   ├── Eliminar solapamientos
   └── Agrupar fases relacionadas

3. Validación Manual
   ├── Revisar casos de alta confianza (>0.8)
   ├── Validar casos de confianza media (0.6-0.8)
   └── Descartar casos de baja confianza (<0.6)

4. Post-procesamiento
   ├── Balancear dataset por fases
   ├── Crear ventanas de contexto
   └── Generar embeddings
```

### Estructura del Dataset

#### Formato de Etiquetas
```json
{
  "timestamp": "2023-01-15T10:30:00Z",
  "symbol": "BTCUSDT",
  "phase": "accumulation_phase_c",
  "sub_phase": "spring",
  "confidence": 0.85,
  "duration": 1440,  // minutos
  "price_range": {
    "start": 16500,
    "end": 16800,
    "low": 16450,
    "high": 16850
  },
  "volume_profile": "high",
  "validation_status": "confirmed",
  "annotator": "expert_1"
}
```

#### Categorías de Etiquetas

**Acumulación (7 fases):**
- `preliminary_support` (PS)
- `selling_climax` (SC)
- `automatic_rally` (AR)
- `secondary_test` (ST)
- `spring` (Spring)
- `sign_of_strength` (SOS)
- `last_point_support` (LPS)

**Distribución (7 fases):**
- `preliminary_supply` (PSY)
- `buying_climax` (BC)
- `automatic_reaction` (AD)
- `secondary_test_dist` (ST)
- `upthrust` (UT)
- `sign_of_weakness` (SOW)
- `last_point_supply` (LPSY)

**Estados Neutrales:**
- `markup` (Tendencia alcista)
- `markdown` (Tendencia bajista)
- `consolidation` (Lateral)

### Métricas de Calidad del Dataset

#### Métricas de Consistencia
- **Inter-annotator Agreement**: Kappa de Cohen entre anotadores
- **Temporal Consistency**: Coherencia en secuencias de fases
- **Volume-Price Alignment**: Correlación entre patrones de volumen y precio

#### Métricas de Cobertura
- **Phase Distribution**: Balance entre diferentes fases
- **Market Conditions**: Cobertura de mercados alcistas/bajistas/laterales
- **Timeframe Coverage**: Distribución temporal del dataset

### Herramientas de Anotación

#### Dashboard de Validación
```python
# Streamlit app para validación manual
import streamlit as st
import plotly.graph_objects as go

class WyckoffAnnotationTool:
    def render_chart(self, data, predictions):
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=data.index,
            open=data['open'],
            high=data['high'],
            low=data['low'],
            close=data['close']
        ))
        
        # Overlay predictions
        for pred in predictions:
            fig.add_annotation(
                x=pred['timestamp'],
                y=pred['price'],
                text=pred['phase'],
                showarrow=True,
                arrowcolor=pred['color']
            )
        
        return fig
```

### Proceso de Mejora Continua

1. **Feedback Loop**: Resultados del modelo → Refinamiento de reglas
2. **Active Learning**: Seleccionar casos inciertos para anotación manual
3. **Rule Evolution**: Actualizar heurísticas basado en nuevos patrones
4. **Quality Assurance**: Auditorías periódicas del dataset

### Consideraciones Prácticas

#### Escalabilidad
- Procesamiento en lotes para datos históricos
- Pipeline en tiempo real para datos nuevos
- Paralelización por símbolos/timeframes

#### Gestión de Versiones
- Versionado del dataset con Git LFS
- Tracking de cambios en reglas heurísticas
- Reproducibilidad de experimentos

## Sistema de Embeddings

### Propósito
Representar patrones de mercado complejos en espacios vectoriales densos para mejorar la detección de patrones Wyckoff y facilitar el aprendizaje de similitudes entre diferentes períodos temporales.

### Tipos de Embeddings

#### 1. Embeddings de Ventanas Temporales
- **Entrada**: Ventanas deslizantes de datos OHLC (ej. 50-100 períodos)
- **Salida**: Vector denso de 128-512 dimensiones
- **Técnica**: Autoencoder o Transformer encoder

#### 2. Embeddings de Indicadores Técnicos
- **Entrada**: Combinación de RSI, MACD, Bollinger Bands, etc.
- **Salida**: Vector de características técnicas comprimidas
- **Técnica**: Red neuronal densa con capas de reducción

#### 3. Embeddings de Patrones
- **Entrada**: Secuencias que representan fases de Wyckoff
- **Salida**: Representación vectorial del patrón completo
- **Técnica**: LSTM o GRU con attention mechanism

### Arquitectura de Embeddings

```python
# Ejemplo conceptual
class TradingEmbeddings:
    def __init__(self):
        self.temporal_encoder = TemporalEncoder(input_dim=4, embed_dim=256)
        self.technical_encoder = TechnicalEncoder(input_dim=10, embed_dim=128)
        self.pattern_encoder = PatternEncoder(sequence_len=100, embed_dim=512)
    
    def encode_market_state(self, ohlc_window, technical_indicators):
        temporal_embed = self.temporal_encoder(ohlc_window)
        technical_embed = self.technical_encoder(technical_indicators)
        return torch.cat([temporal_embed, technical_embed], dim=-1)
```

### Casos de Uso

1. **Búsqueda de Patrones Similares**: Encontrar períodos históricos con embeddings similares
2. **Transferencia entre Instrumentos**: Aplicar conocimiento de BTC/USDT a ETH/USDT
3. **Detección de Anomalías**: Identificar comportamientos de mercado inusuales
4. **Clustering de Regímenes**: Agrupar diferentes estados del mercado

### Métricas de Evaluación
- **Similitud Coseno**: Medir similitud entre embeddings
- **t-SNE/UMAP**: Visualización de clusters de patrones
- **Silhouette Score**: Calidad de clustering
- **Downstream Task Performance**: Mejora en detección de Wyckoff

## Patrones Wyckoff a Detectar

### 1. Esquema de Acumulación
- **PS (Preliminary Support)**: Soporte preliminar
- **SC (Selling Climax)**: Clímax de venta
- **AR (Automatic Rally)**: Rally automático
- **ST (Secondary Test)**: Test secundario
- **SOS (Sign of Strength)**: Señal de fortaleza
- **LPS (Last Point of Support)**: Último punto de soporte

### 2. Esquema de Distribución
- **PSY (Preliminary Supply)**: Oferta preliminar
- **BC (Buying Climax)**: Clímax de compra
- **AD (Automatic Reaction)**: Reacción automática
- **ST (Secondary Test)**: Test secundario
- **SOW (Sign of Weakness)**: Señal de debilidad
- **LPSY (Last Point of Supply)**: Último punto de oferta

## Métricas y Evaluación

### Métricas del Modelo
- **Precisión**: % de patrones correctamente identificados
- **Recall**: % de patrones reales detectados
- **F1-Score**: Balance entre precisión y recall
- **Matriz de Confusión**: Análisis detallado de errores

### Métricas de Trading
- **Sharpe Ratio**: Retorno ajustado por riesgo
- **Maximum Drawdown**: Pérdida máxima
- **Win Rate**: Porcentaje de operaciones ganadoras
- **Profit Factor**: Relación ganancia/pérdida

## Consideraciones Técnicas

### Limitaciones de API
- Rate limits de Binance
- Manejo de errores de conexión
- Backup de datos

### Optimización
- Procesamiento en paralelo
- Cache inteligente
- Compresión de datos

### Seguridad
- API keys seguras
- Validación de datos
- Logging de errores

## Próximos Pasos

1. **Configurar entorno**: Python, dependencias, IDE
2. **Crear repositorio**: Git, estructura de carpetas
3. **Implementar cliente Binance**: Conexión básica
4. **Diseñar base de datos**: Schema para OHLC
5. **Desarrollar análisis técnico**: Indicadores básicos

## Recursos Adicionales

### Documentación
- [Binance API Documentation](https://binance-docs.github.io/apidocs/)
- [TA-Lib Documentation](https://ta-lib.org/)
- [Wyckoff Method Guide](https://school.stockcharts.com/doku.php?id=market_analysis:the_wyckoff_method)

### Librerías Útiles
- `python-binance`: Cliente oficial de Binance
- `pandas-ta`: Indicadores técnicos
- `plotly`: Visualización de gráficos
- `streamlit`: Dashboard rápido
- `scikit-learn`: Métricas de evaluación (Kappa, etc.)
- `sqlalchemy`: Gestión de base de datos
- `pydantic`: Validación de datos

Este plan proporciona una base sólida para desarrollar una aplicación de trading con AI de manera simple y estructurada.