# Modo de Desarrollo - Sistema de Trading

## Descripción

El sistema de trading incluye un modo de desarrollo que permite deshabilitar la caché y habilitar funcionalidades de debug para facilitar el desarrollo y las pruebas.

## Configuración

### Archivo de Configuración

La configuración se encuentra en `config.yaml` en la sección `development`:

```yaml
development:
  mode: true              # Habilita el modo desarrollo
  disable_cache: true     # Deshabilita la caché completamente
  debug_mode: true        # Habilita logs de debug
  mock_api: false         # Usa API simulada (opcional)
```

### Script de Gestión

Se incluye un script utilitario para gestionar fácilmente la configuración:

```bash
# Ver estado actual
python scripts/toggle_development_mode.py --status

# Habilitar modo desarrollo completo
python scripts/toggle_development_mode.py --dev

# Habilitar modo producción
python scripts/toggle_development_mode.py --prod

# Solo deshabilitar caché
python scripts/toggle_development_mode.py --disable-cache

# Solo habilitar caché
python scripts/toggle_development_mode.py --enable-cache
```

## Funcionalidades del Modo Desarrollo

### 1. Caché Deshabilitada

Cuando `disable_cache: true`:
- El `TradingCache` actúa como un no-op
- Todos los métodos de caché retornan valores por defecto
- Los datos siempre se obtienen directamente de la fuente
- Útil para testing y desarrollo donde necesitas datos frescos

### 2. Logs de Debug

Cuando `debug_mode: true`:
- Se habilitan logs adicionales de debug
- Mayor visibilidad del flujo de datos
- Información detallada de operaciones

### 3. Modo Desarrollo General

Cuando `mode: true`:
- Indica que el sistema está en modo desarrollo
- Puede afectar otros comportamientos del sistema
- Se muestra en los logs de inicialización

## Implementación Técnica

### ConfigManager

El sistema utiliza un `ConfigManager` singleton que:
- Carga la configuración desde `config.yaml`
- Proporciona métodos para acceder a configuraciones específicas
- Incluye funciones de conveniencia para verificar estados

```python
from src.config import ConfigManager, is_cache_disabled, is_development_mode

# Verificar si la caché está deshabilitada
if is_cache_disabled():
    print("Caché deshabilitada")

# Verificar modo desarrollo
if is_development_mode():
    print("Modo desarrollo activo")
```

### TradingCache Modificado

La clase `TradingCache` verifica automáticamente la configuración:
- Si `disable_cache` es `true`, se comporta como no-op
- Todos los métodos (`get`, `set`, `delete`, etc.) retornan inmediatamente
- No se inicializa Redis ni caché en memoria

### DataManager Integrado

El `DataManager` respeta la configuración de desarrollo:
- Verifica `is_cache_disabled()` durante la inicialización
- Ajusta `use_cache` basado en la configuración
- Registra el estado en los logs

## Casos de Uso

### Desarrollo Local
```bash
# Habilitar modo desarrollo para trabajar sin caché
python scripts/toggle_development_mode.py --dev
python -m streamlit run dashboard/dashboard.py
```

### Testing
```bash
# Deshabilitar solo la caché para tests
python scripts/toggle_development_mode.py --disable-cache
python -m pytest tests/
```

### Producción
```bash
# Asegurar modo producción antes del deploy
python scripts/toggle_development_mode.py --prod
```

## Logs de Ejemplo

Con modo desarrollo habilitado:
```
INFO:src.config.config_manager:Configuración cargada desde: config.yaml
INFO:src.data.cache:Caché deshabilitada por configuración de desarrollo
INFO:data_manager:DataManager inicializado con caché deshabilitada por configuración de desarrollo
INFO:data_manager:Modo desarrollo: True, Caché habilitada: False
```

Con modo producción:
```
INFO:src.config.config_manager:Configuración cargada desde: config.yaml
INFO:data_manager:DataManager inicializado correctamente
INFO:data_manager:Modo desarrollo: False, Caché habilitada: True
```

## Notas Importantes

1. **Reinicio Requerido**: Después de cambiar la configuración, reinicia la aplicación
2. **Performance**: El modo desarrollo puede ser más lento sin caché
3. **Datos Frescos**: Sin caché, siempre obtienes los datos más recientes
4. **Testing**: Ideal para pruebas que requieren datos no cacheados
5. **Fallback**: Si hay problemas con la configuración, el sistema usa valores por defecto seguros

## Troubleshooting

### La configuración no se aplica
- Verifica que `config.yaml` existe y es válido
- Reinicia la aplicación después de cambios
- Revisa los logs de inicialización

### Imports fallan
- El sistema incluye fallbacks para imports
- Si `ConfigManager` no se puede importar, usa valores por defecto
- Verifica la estructura de directorios

### Script no funciona
- Asegúrate de que `PyYAML` está instalado: `pip install PyYAML`
- Ejecuta desde el directorio raíz del proyecto
- Verifica permisos de escritura en `config.yaml`