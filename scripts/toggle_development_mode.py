#!/usr/bin/env python3
"""
Script para alternar entre modo desarrollo y producción
"""

import yaml
import sys
from pathlib import Path
import argparse

def load_config():
    """Carga la configuración actual"""
    config_path = Path(__file__).parent.parent / "config.yaml"
    
    if not config_path.exists():
        print(f"❌ Error: No se encontró config.yaml en {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file), config_path

def save_config(config, config_path):
    """Guarda la configuración"""
    with open(config_path, 'w', encoding='utf-8') as file:
        yaml.dump(config, file, default_flow_style=False, allow_unicode=True, indent=2)

def show_current_status(config):
    """Muestra el estado actual de la configuración"""
    dev_config = config.get('development', {})
    
    print("\n📊 Estado Actual de Configuración:")
    print("=" * 40)
    print(f"🔧 Modo desarrollo: {'✅ Habilitado' if dev_config.get('mode', False) else '❌ Deshabilitado'}")
    print(f"💾 Caché deshabilitada: {'✅ Sí' if dev_config.get('disable_cache', False) else '❌ No'}")
    print(f"🐛 Modo debug: {'✅ Habilitado' if dev_config.get('debug_mode', False) else '❌ Deshabilitado'}")
    print(f"🎭 API simulada: {'✅ Habilitado' if dev_config.get('mock_api', False) else '❌ Deshabilitado'}")
    print("=" * 40)

def enable_development_mode(config):
    """Habilita el modo desarrollo completo"""
    if 'development' not in config:
        config['development'] = {}
    
    config['development']['mode'] = True
    config['development']['disable_cache'] = True
    config['development']['debug_mode'] = True
    config['development']['mock_api'] = False  # Mantener API real por defecto
    
    print("\n✅ Modo desarrollo habilitado:")
    print("   • Caché deshabilitada")
    print("   • Logs de debug habilitados")
    print("   • Modo desarrollo activado")

def enable_production_mode(config):
    """Habilita el modo producción"""
    if 'development' not in config:
        config['development'] = {}
    
    config['development']['mode'] = False
    config['development']['disable_cache'] = False
    config['development']['debug_mode'] = False
    config['development']['mock_api'] = False
    
    print("\n🚀 Modo producción habilitado:")
    print("   • Caché habilitada")
    print("   • Logs normales")
    print("   • Modo desarrollo desactivado")

def toggle_cache_only(config, disable):
    """Alterna solo la configuración de caché"""
    if 'development' not in config:
        config['development'] = {}
    
    config['development']['disable_cache'] = disable
    
    if disable:
        print("\n💾 Caché deshabilitada")
    else:
        print("\n💾 Caché habilitada")

def main():
    parser = argparse.ArgumentParser(
        description="Gestiona la configuración de desarrollo del sistema de trading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  python scripts/toggle_development_mode.py --status
  python scripts/toggle_development_mode.py --dev
  python scripts/toggle_development_mode.py --prod
  python scripts/toggle_development_mode.py --disable-cache
  python scripts/toggle_development_mode.py --enable-cache
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--status', '-s', action='store_true',
                      help='Mostrar estado actual de la configuración')
    group.add_argument('--dev', '-d', action='store_true',
                      help='Habilitar modo desarrollo (caché deshabilitada, debug on)')
    group.add_argument('--prod', '-p', action='store_true',
                      help='Habilitar modo producción (caché habilitada, debug off)')
    group.add_argument('--disable-cache', action='store_true',
                      help='Deshabilitar solo la caché')
    group.add_argument('--enable-cache', action='store_true',
                      help='Habilitar solo la caché')
    
    args = parser.parse_args()
    
    # Cargar configuración
    config, config_path = load_config()
    
    if args.status:
        show_current_status(config)
        return
    
    # Mostrar estado actual primero
    show_current_status(config)
    
    # Aplicar cambios
    if args.dev:
        enable_development_mode(config)
    elif args.prod:
        enable_production_mode(config)
    elif args.disable_cache:
        toggle_cache_only(config, True)
    elif args.enable_cache:
        toggle_cache_only(config, False)
    
    # Guardar configuración
    save_config(config, config_path)
    print(f"\n💾 Configuración guardada en: {config_path}")
    
    # Mostrar estado final
    print("\n📊 Nuevo Estado:")
    show_current_status(config)
    
    print("\n⚠️  Nota: Reinicia la aplicación para que los cambios tengan efecto.")

if __name__ == "__main__":
    main()