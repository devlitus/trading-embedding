#!/usr/bin/env python3
"""
Script de inicio para el Dashboard de Trading - Fase 1

Este script facilita la ejecución del dashboard de Streamlit
con la configuración adecuada.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Verificar que las dependencias estén instaladas"""
    required_packages = ['streamlit', 'plotly']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Faltan dependencias: {', '.join(missing_packages)}")
        print("💡 Instala las dependencias con: pip install -r requirements.txt")
        return False
    
    return True

def run_dashboard():
    """Ejecutar el dashboard de Streamlit"""
    print("🚀 INICIANDO DASHBOARD DE TRADING - FASE 1")
    print("=" * 50)
    
    # Verificar dependencias
    if not check_dependencies():
        sys.exit(1)
    
    # Ruta del dashboard
    dashboard_path = Path(__file__).parent / "dashboard" / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ No se encontró el dashboard en: {dashboard_path}")
        sys.exit(1)
    
    print(f"📂 Dashboard ubicado en: {dashboard_path}")
    print("🌐 El dashboard se abrirá en tu navegador automáticamente")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Para detener el dashboard, presiona Ctrl+C")
    print("\n" + "=" * 50)
    
    try:
        # Ejecutar Streamlit
        cmd = [
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port=8501",
            "--server.address=localhost",
            "--browser.gatherUsageStats=false"
        ]
        
        subprocess.run(cmd, check=True)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Dashboard detenido por el usuario")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error ejecutando el dashboard: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error inesperado: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_dashboard()