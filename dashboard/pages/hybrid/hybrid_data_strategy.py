# Imports de módulos separados
from .hybrid_config import (
    get_data_strategy, create_sidebar_controls, show_page_header, create_tabs
)
from .hybrid_demos_part1 import show_realtime_trading_demo, show_ml_training_demo
from .hybrid_demos_part2 import show_backtesting_demo, show_api_serving_demo
from .hybrid_demos_part3 import show_dashboard_demo, show_performance_comparison

def show_hybrid_data_strategy_page():
    """Página principal de la estrategia híbrida de datos."""
    # Mostrar encabezado de la página
    show_page_header()
    
    # Crear controles del sidebar
    symbol, interval = create_sidebar_controls()
    
    # Inicializar estrategia
    strategy = get_data_strategy()
    if not strategy:
        return
    
    # Crear tabs para diferentes patrones
    tab1, tab2, tab3, tab4, tab5, tab6 = create_tabs()
    
    with tab1:
        show_realtime_trading_demo(strategy, symbol, interval)
    
    with tab2:
        show_ml_training_demo(strategy, symbol, interval)
    
    with tab3:
        show_backtesting_demo(strategy, symbol, interval)
    
    with tab4:
        show_api_serving_demo(strategy, symbol, interval)
    
    with tab5:
        show_dashboard_demo(strategy, symbol, interval)
    
    with tab6:
        show_performance_comparison(strategy, symbol, interval)













if __name__ == "__main__":
    show_hybrid_data_strategy_page()