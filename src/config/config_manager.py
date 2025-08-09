#!/usr/bin/env python3
"""
Gestor de configuración - Maneja la carga y acceso a configuraciones del sistema
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Gestor centralizado de configuración que carga y proporciona acceso
    a todas las configuraciones del sistema desde config.yaml
    """
    
    _instance = None
    _config = None
    
    def __new__(cls):
        """Implementa patrón Singleton"""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Inicializa el gestor de configuración"""
        if self._config is None:
            self._load_config()
    
    def _load_config(self) -> None:
        """Carga la configuración desde config.yaml"""
        try:
            # Buscar config.yaml en el directorio raíz del proyecto
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"
            
            if not config_path.exists():
                raise FileNotFoundError(f"Archivo de configuración no encontrado: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
            
            logger.info(f"Configuración cargada desde: {config_path}")
            
        except Exception as e:
            logger.error(f"Error cargando configuración: {e}")
            # Configuración por defecto en caso de error
            self._config = self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Configuración por defecto en caso de error"""
        return {
            'development': {
                'mode': False,
                'disable_cache': False,
                'debug_mode': False,
                'mock_api': False
            },
            'cache': {
                'type': 'memory',
                'default_ttl': 3600,
                'max_size': 1000
            },
            'logging': {
                'level': 'INFO'
            }
        }
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Obtiene un valor de configuración usando notación de punto
        
        Args:
            key_path: Ruta de la clave (ej: 'development.disable_cache')
            default: Valor por defecto si no se encuentra la clave
            
        Returns:
            Valor de configuración o default
        """
        try:
            keys = key_path.split('.')
            value = self._config
            
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default
            
            return value
            
        except Exception as e:
            logger.warning(f"Error obteniendo configuración '{key_path}': {e}")
            return default
    
    def is_development_mode(self) -> bool:
        """Verifica si está en modo desarrollo"""
        return self.get('development.mode', False)
    
    def is_cache_disabled(self) -> bool:
        """Verifica si la caché está deshabilitada"""
        return self.get('development.disable_cache', False)
    
    def is_debug_mode(self) -> bool:
        """Verifica si está en modo debug"""
        return self.get('development.debug_mode', False)
    
    def get_cache_config(self) -> Dict[str, Any]:
        """Obtiene configuración completa de caché"""
        return self.get('cache', {})
    
    def get_logging_level(self) -> str:
        """Obtiene el nivel de logging configurado"""
        return self.get('logging.level', 'INFO')
    
    def reload_config(self) -> None:
        """Recarga la configuración desde el archivo"""
        self._config = None
        self._load_config()
        logger.info("Configuración recargada")
    
    def get_all_config(self) -> Dict[str, Any]:
        """Obtiene toda la configuración"""
        return self._config.copy() if self._config else {}
    
    def set_development_mode(self, enabled: bool) -> None:
        """
        Habilita/deshabilita modo desarrollo programáticamente
        NOTA: Este cambio no persiste al reiniciar la aplicación
        """
        if 'development' not in self._config:
            self._config['development'] = {}
        
        self._config['development']['mode'] = enabled
        logger.info(f"Modo desarrollo {'habilitado' if enabled else 'deshabilitado'}")
    
    def set_cache_disabled(self, disabled: bool) -> None:
        """
        Habilita/deshabilita caché programáticamente
        NOTA: Este cambio no persiste al reiniciar la aplicación
        """
        if 'development' not in self._config:
            self._config['development'] = {}
        
        self._config['development']['disable_cache'] = disabled
        logger.info(f"Caché {'deshabilitada' if disabled else 'habilitada'}")


# Instancia global del gestor de configuración
config_manager = ConfigManager()

# Funciones de conveniencia para acceso rápido
def get_config(key_path: str, default: Any = None) -> Any:
    """Función de conveniencia para obtener configuración"""
    return config_manager.get(key_path, default)

def is_development_mode() -> bool:
    """Función de conveniencia para verificar modo desarrollo"""
    return config_manager.is_development_mode()

def is_cache_disabled() -> bool:
    """Función de conveniencia para verificar si caché está deshabilitada"""
    return config_manager.is_cache_disabled()

def is_debug_mode() -> bool:
    """Función de conveniencia para verificar modo debug"""
    return config_manager.is_debug_mode()


if __name__ == "__main__":
    # Ejemplo de uso
    print("=== Configuración del Sistema ===")
    print(f"Modo desarrollo: {is_development_mode()}")
    print(f"Caché deshabilitada: {is_cache_disabled()}")
    print(f"Modo debug: {is_debug_mode()}")
    print(f"Nivel de logging: {config_manager.get_logging_level()}")
    
    # Configuración de caché
    cache_config = config_manager.get_cache_config()
    print(f"Configuración de caché: {cache_config}")