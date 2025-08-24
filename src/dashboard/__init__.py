"""Módulo del dashboard modular."""

# Componentes
from .components import (
    DashboardLayout,
    MetricsComponent,
    ChartComponents
)

# Servicios
from .services import (
    DataService,
    AnalysisService
)

# Utilidades
from .utils import (
    DataFormatters,
    ColorUtils,
    DataValidators,
    InputSanitizers
)

# Páginas
from .pages import (
    HomePage,
    show_home_page,
    MarketDataPage,
    show_market_data_page
)

__all__ = [
    # Componentes
    'DashboardLayout',
    'MetricsComponent', 
    'ChartComponents',
    
    # Servicios
    'DataService',
    'AnalysisService',
    
    # Utilidades
    'DataFormatters',
    'ColorUtils',
    'DataValidators',
    'InputSanitizers',
    
    # Páginas
    'HomePage',
    'show_home_page',
    'MarketDataPage',
    'show_market_data_page'
]