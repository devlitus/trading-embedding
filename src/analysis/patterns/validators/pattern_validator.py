"""Validador de patrones técnicos."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
from ..types import PatternResult


class PatternValidator:
    """Validador para verificar la calidad y confiabilidad de patrones detectados."""
    
    def __init__(self, min_confidence: float = 0.6, min_volume_ratio: float = 1.2):
        """
        Inicializa el validador de patrones.
        
        Args:
            min_confidence: Confianza mínima requerida para validar un patrón
            min_volume_ratio: Ratio mínimo de volumen para validar breakouts
        """
        self.min_confidence = min_confidence
        self.min_volume_ratio = min_volume_ratio
    
    def validate_pattern(self, pattern: Dict[str, Any], data: pd.DataFrame) -> bool:
        """
        Valida un patrón individual.
        
        Args:
            pattern: Diccionario con información del patrón
            data: DataFrame con datos OHLCV
            
        Returns:
            bool: True si el patrón es válido
        """
        try:
            # Validar confianza mínima
            if pattern.get('confidence', 0) < self.min_confidence:
                return False
            
            # Validar que el patrón tenga suficientes puntos de datos
            start_idx = pattern.get('start_index', 0)
            end_idx = pattern.get('end_index', len(data) - 1)
            
            if end_idx - start_idx < 5:  # Mínimo 5 períodos
                return False
            
            # Validar que los índices estén dentro del rango
            if start_idx < 0 or end_idx >= len(data):
                return False
            
            # Validaciones específicas por tipo de patrón
            pattern_type = pattern.get('type', '')
            
            if 'triangle' in pattern_type.lower():
                return self._validate_triangle_pattern(pattern, data)
            elif 'channel' in pattern_type.lower():
                return self._validate_channel_pattern(pattern, data)
            elif 'rectangle' in pattern_type.lower():
                return self._validate_rectangle_pattern(pattern, data)
            elif any(candle in pattern_type.lower() for candle in ['doji', 'hammer', 'engulfing']):
                return self._validate_candlestick_pattern(pattern, data)
            elif 'volume' in pattern_type.lower():
                return self._validate_volume_pattern(pattern, data)
            
            return True
            
        except Exception:
            return False
    
    def _validate_triangle_pattern(self, pattern: Dict[str, Any], data: pd.DataFrame) -> bool:
        """
        Valida patrones de triángulo.
        
        Args:
            pattern: Información del patrón de triángulo
            data: DataFrame con datos OHLCV
            
        Returns:
            bool: True si el triángulo es válido
        """
        # Verificar que tenga líneas de tendencia válidas
        upper_line = pattern.get('upper_trendline')
        lower_line = pattern.get('lower_trendline')
        
        if not upper_line or not lower_line:
            return False
        
        # Verificar R-squared mínimo para las líneas de tendencia
        min_r_squared = 0.7
        if (upper_line.get('r_squared', 0) < min_r_squared or 
            lower_line.get('r_squared', 0) < min_r_squared):
            return False
        
        # Verificar convergencia
        convergence_point = pattern.get('convergence_point')
        if not convergence_point or convergence_point < 0:
            return False
        
        return True
    
    def _validate_channel_pattern(self, pattern: Dict[str, Any], data: pd.DataFrame) -> bool:
        """
        Valida patrones de canal.
        
        Args:
            pattern: Información del patrón de canal
            data: DataFrame con datos OHLCV
            
        Returns:
            bool: True si el canal es válido
        """
        # Verificar líneas de soporte y resistencia
        support_line = pattern.get('support_line')
        resistance_line = pattern.get('resistance_line')
        
        if not support_line or not resistance_line:
            return False
        
        # Verificar paralelismo (diferencia de pendientes < 20%)
        support_slope = support_line.get('slope', 0)
        resistance_slope = resistance_line.get('slope', 0)
        
        if abs(support_slope) > 0:
            slope_diff = abs((resistance_slope - support_slope) / support_slope)
            if slope_diff > 0.2:  # 20% de diferencia máxima
                return False
        
        return True
    
    def _validate_rectangle_pattern(self, pattern: Dict[str, Any], data: pd.DataFrame) -> bool:
        """
        Valida patrones de rectángulo.
        
        Args:
            pattern: Información del patrón de rectángulo
            data: DataFrame con datos OHLCV
            
        Returns:
            bool: True si el rectángulo es válido
        """
        # Verificar niveles de soporte y resistencia
        support_level = pattern.get('support_level')
        resistance_level = pattern.get('resistance_level')
        
        if support_level is None or resistance_level is None:
            return False
        
        # Verificar que haya suficiente separación entre niveles (mínimo 1%)
        price_range = abs(resistance_level - support_level)
        avg_price = (resistance_level + support_level) / 2
        
        if price_range / avg_price < 0.01:  # Mínimo 1% de rango
            return False
        
        return True
    
    def _validate_candlestick_pattern(self, pattern: Dict[str, Any], data: pd.DataFrame) -> bool:
        """
        Valida patrones de velas japonesas.
        
        Args:
            pattern: Información del patrón de vela
            data: DataFrame con datos OHLCV
            
        Returns:
            bool: True si el patrón de vela es válido
        """
        # Verificar que el patrón tenga índice válido
        pattern_index = pattern.get('index')
        if pattern_index is None or pattern_index < 0 or pattern_index >= len(data):
            return False
        
        # Verificar que haya suficiente contexto (al menos 3 velas antes y después)
        if pattern_index < 3 or pattern_index >= len(data) - 3:
            return False
        
        return True
    
    def _validate_volume_pattern(self, pattern: Dict[str, Any], data: pd.DataFrame) -> bool:
        """
        Valida patrones de volumen.
        
        Args:
            pattern: Información del patrón de volumen
            data: DataFrame con datos OHLCV
            
        Returns:
            bool: True si el patrón de volumen es válido
        """
        # Verificar que haya datos de volumen
        if 'volume' not in data.columns:
            return False
        
        # Verificar que el volumen no sea cero o negativo
        pattern_index = pattern.get('index')
        if pattern_index is not None:
            volume = data.iloc[pattern_index]['volume']
            if volume <= 0:
                return False
        
        return True
    
    def validate_patterns(self, patterns: List[Dict[str, Any]], data: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Valida una lista de patrones.
        
        Args:
            patterns: Lista de patrones a validar
            data: DataFrame con datos OHLCV
            
        Returns:
            List[Dict[str, Any]]: Lista de patrones válidos
        """
        valid_patterns = []
        
        for pattern in patterns:
            if self.validate_pattern(pattern, data):
                valid_patterns.append(pattern)
        
        return valid_patterns
    
    def get_validation_score(self, pattern: Dict[str, Any], data: pd.DataFrame) -> float:
        """
        Calcula un score de validación para un patrón.
        
        Args:
            pattern: Información del patrón
            data: DataFrame con datos OHLCV
            
        Returns:
            float: Score de validación entre 0 y 1
        """
        score = 0.0
        
        try:
            # Score base por confianza
            confidence = pattern.get('confidence', 0)
            score += confidence * 0.4
            
            # Score por duración del patrón
            start_idx = pattern.get('start_index', 0)
            end_idx = pattern.get('end_index', len(data) - 1)
            duration = end_idx - start_idx
            
            # Normalizar duración (óptimo entre 10-50 períodos)
            if 10 <= duration <= 50:
                duration_score = 1.0
            elif duration < 10:
                duration_score = duration / 10.0
            else:
                duration_score = max(0.5, 50.0 / duration)
            
            score += duration_score * 0.2
            
            # Score por volumen (si aplica)
            if 'volume' in data.columns and 'index' in pattern:
                pattern_idx = pattern['index']
                if 0 <= pattern_idx < len(data):
                    current_volume = data.iloc[pattern_idx]['volume']
                    avg_volume = data['volume'].rolling(20).mean().iloc[pattern_idx]
                    
                    if avg_volume > 0:
                        volume_ratio = current_volume / avg_volume
                        volume_score = min(1.0, volume_ratio / 2.0)  # Normalizar a 2x el promedio
                        score += volume_score * 0.2
            
            # Score por calidad técnica específica del patrón
            pattern_type = pattern.get('type', '')
            if 'triangle' in pattern_type.lower():
                tech_score = self._get_triangle_technical_score(pattern)
            elif 'channel' in pattern_type.lower():
                tech_score = self._get_channel_technical_score(pattern)
            else:
                tech_score = 0.5  # Score neutro para otros patrones
            
            score += tech_score * 0.2
            
            return min(1.0, score)
            
        except Exception:
            return 0.0
    
    def _get_triangle_technical_score(self, pattern: Dict[str, Any]) -> float:
        """
        Calcula score técnico para patrones de triángulo.
        
        Args:
            pattern: Información del patrón de triángulo
            
        Returns:
            float: Score técnico entre 0 y 1
        """
        upper_line = pattern.get('upper_trendline', {})
        lower_line = pattern.get('lower_trendline', {})
        
        upper_r2 = upper_line.get('r_squared', 0)
        lower_r2 = lower_line.get('r_squared', 0)
        
        return (upper_r2 + lower_r2) / 2.0
    
    def _get_channel_technical_score(self, pattern: Dict[str, Any]) -> float:
        """
        Calcula score técnico para patrones de canal.
        
        Args:
            pattern: Información del patrón de canal
            
        Returns:
            float: Score técnico entre 0 y 1
        """
        support_line = pattern.get('support_line', {})
        resistance_line = pattern.get('resistance_line', {})
        
        support_r2 = support_line.get('r_squared', 0)
        resistance_r2 = resistance_line.get('r_squared', 0)
        
        return (support_r2 + resistance_r2) / 2.0