#!/usr/bin/env python3
"""
Imports y carga de módulos para páginas del dashboard
"""

import streamlit as st
import importlib.util
from .config_common import project_root

# Variables globales para módulos cargados
BinanceClient = None
TradingDatabase = None
TradingCache = None
DataManager = None
Phase2VerificationSystem = None
analyze_symbol = None

def load_modules():
    """Cargar todos los módulos necesarios"""
    global BinanceClient, TradingDatabase, TradingCache, DataManager
    global Phase2VerificationSystem, analyze_symbol
    
    try:
        # Cargar BinanceClient
        binance_spec = importlib.util.spec_from_file_location(
            "binance_client", 
            project_root / "src" / "data" / "binance_client.py"
        )
        binance_module = importlib.util.module_from_spec(binance_spec)
        binance_spec.loader.exec_module(binance_module)
        BinanceClient = binance_module.BinanceClient
        
        # Cargar TradingDatabase
        database_spec = importlib.util.spec_from_file_location(
            "database", 
            project_root / "src" / "data" / "database.py"
        )
        database_module = importlib.util.module_from_spec(database_spec)
        database_spec.loader.exec_module(database_module)
        TradingDatabase = database_module.TradingDatabase
        
        # Cargar TradingCache
        cache_spec = importlib.util.spec_from_file_location(
            "cache", 
            project_root / "src" / "data" / "cache.py"
        )
        cache_module = importlib.util.module_from_spec(cache_spec)
        cache_spec.loader.exec_module(cache_module)
        TradingCache = cache_module.TradingCache
        
        # Cargar DataManager
        data_manager_spec = importlib.util.spec_from_file_location(
            "data_manager", 
            project_root / "src" / "data" / "data_manager.py"
        )
        data_manager_module = importlib.util.module_from_spec(data_manager_spec)
        data_manager_spec.loader.exec_module(data_manager_module)
        DataManager = data_manager_module.DataManager
        
        # Cargar Phase2VerificationSystem
        verification_spec = importlib.util.spec_from_file_location(
            "verification_system", 
            project_root / "verification_system.py"
        )
        verification_module = importlib.util.module_from_spec(verification_spec)
        verification_spec.loader.exec_module(verification_module)
        Phase2VerificationSystem = verification_module.Phase2VerificationSystem
        
        # Cargar análisis técnico
        analysis_spec = importlib.util.spec_from_file_location(
            "technical_analysis", 
            project_root / "src" / "analysis" / "technical_analysis.py"
        )
        analysis_module = importlib.util.module_from_spec(analysis_spec)
        analysis_spec.loader.exec_module(analysis_module)
        analyze_symbol = analysis_module.analyze_symbol
        
        return True
        
    except ImportError as e:
        st.error(f"Error importando módulos: {e}")
        return False
    except Exception as e:
        st.error(f"Error cargando módulos: {e}")
        return False