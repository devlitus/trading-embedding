#!/usr/bin/env python3
"""
Lógica de negocio para la página de inicio
"""

from typing import Dict, List, Optional, Any
from .home_config import HomeConfig

class HomeDataProcessor:
    """Procesador de datos para la página de inicio"""
    
    def __init__(self, data_manager):
        """Inicializar el procesador de datos"""
        self.data_manager = data_manager
        self.config = HomeConfig()
    
    def get_database_statistics(self) -> Optional[Dict[str, Any]]:
        """
        Obtener estadísticas de la base de datos
        
        Returns:
            Dict con estadísticas de la BD o None si hay error
        """
        if not self.data_manager:
            return None
            
        try:
            return self.data_manager.get_database_stats()
        except Exception as e:
            return {'error': str(e)}
    
    def format_symbol_info(self, symbols_data: List[Dict]) -> List[str]:
        """
        Formatear información de símbolos para mostrar
        
        Args:
            symbols_data: Lista de diccionarios con info de símbolos
            
        Returns:
            Lista de strings formateados
        """
        formatted_symbols = []
        for symbol_info in symbols_data:
            symbol = symbol_info['symbol']
            count = symbol_info['count']
            formatted_symbols.append(f"• {symbol}: {count:,} registros")
        return formatted_symbols
    
    def get_general_stats_info(self, db_stats: Dict[str, Any]) -> Dict[str, str]:
        """
        Obtener información de estadísticas generales formateada
        
        Args:
            db_stats: Estadísticas de la base de datos
            
        Returns:
            Dict con estadísticas formateadas
        """
        stats_info = {}
        
        # Total de registros
        total_records = db_stats.get('total_records', 0)
        stats_info['total_records'] = self.config.STATS_LABELS['total_records'].format(
            count=total_records
        )
        
        # Símbolos únicos
        total_symbols = db_stats.get('total_symbols', 0)
        stats_info['unique_symbols'] = self.config.STATS_LABELS['unique_symbols'].format(
            count=total_symbols
        )
        
        # Último registro
        if db_stats.get('latest_timestamp'):
            stats_info['latest_record'] = self.config.STATS_LABELS['latest_record'].format(
                timestamp=db_stats['latest_timestamp']
            )
        
        return stats_info
    
    def get_connection_status(self) -> Dict[str, Any]:
        """
        Obtener estado de conexión del sistema
        
        Returns:
            Dict con información del estado de conexión
        """
        if not self.data_manager:
            return {
                'status': 'error',
                'message': self.config.MESSAGES['no_connection']
            }
        
        db_stats = self.get_database_statistics()
        
        if not db_stats:
            return {
                'status': 'error', 
                'message': self.config.MESSAGES['no_connection']
            }
        
        if 'error' in db_stats:
            return {
                'status': 'error',
                'message': self.config.MESSAGES['db_error'].format(
                    error=db_stats['error']
                )
            }
        
        if not db_stats.get('symbols'):
            return {
                'status': 'warning',
                'message': self.config.MESSAGES['no_data']
            }
        
        return {
            'status': 'success',
            'message': self.config.MESSAGES['db_connected'].format(
                count=db_stats['total_symbols']
            ),
            'data': db_stats
        }
    
    def has_valid_data(self) -> bool:
        """
        Verificar si hay datos válidos disponibles
        
        Returns:
            True si hay datos válidos, False en caso contrario
        """
        connection_status = self.get_connection_status()
        return connection_status['status'] == 'success'