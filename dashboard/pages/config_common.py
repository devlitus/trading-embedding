#!/usr/bin/env python3
"""
Configuración común para páginas del dashboard
"""

import sys
from pathlib import Path

# Configuración de rutas del proyecto
project_root = Path(r"c:\dev\trading-embedding")

# Agregar los directorios necesarios al path
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / 'src'))
sys.path.insert(0, str(project_root / 'src' / 'data'))
sys.path.insert(0, str(project_root / 'src' / 'analysis'))

# Configuración de Plotly
def get_plotly_config():
    """Configuración estándar para gráficos Plotly"""
    return {
        'displayModeBar': True,
        'displaylogo': False,
        'modeBarButtonsToRemove': [
            'pan2d', 'lasso2d', 'select2d', 'autoScale2d',
            'hoverClosestCartesian', 'hoverCompareCartesian',
            'toggleSpikelines'
        ],
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'trading_chart',
            'height': 500,
            'width': 1200,
            'scale': 1
        }
    }

# CSS personalizado
CUSTOM_CSS = """
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #1f77b4;
}
.success-card {
    background-color: #d4edda;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #28a745;
    color: black;
}
.error-card {
    background-color: #f8d7da;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #dc3545;
}
.warning-card {
    background-color: #fff3cd;
    padding: 1rem;
    border-radius: 0.5rem;
    border-left: 4px solid #ffc107;
}
</style>
"""