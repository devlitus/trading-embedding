"""Reconocedor principal de patrones técnicos."""

import pandas as pd
from typing import Dict, List, Optional

from ..types import PatternResult
from .geometric_patterns import GeometricPatternDetector
from .candlestick_patterns import CandlestickPatternDetector
from .volume_patterns import VolumePatternDetector
from ..validators.pattern_validator import PatternValidator
from ..validators.overlap_filter import OverlapFilter


class PatternRecognizer:
    """Clase principal para reconocimiento de patrones técnicos"""
    
    def __init__(self, min_pattern_length: int = 20, confidence_threshold: float = 0.6):
        """
        Args:
            min_pattern_length: Longitud mínima para considerar un patrón válido
            confidence_threshold: Umbral mínimo de confianza para reportar patrones
        """
        self.min_pattern_length = min_pattern_length
        self.confidence_threshold = confidence_threshold
        
        # Inicializar detectores especializados
        self.geometric_detector = GeometricPatternDetector(min_pattern_length)
        self.candlestick_detector = CandlestickPatternDetector()
        self.volume_detector = VolumePatternDetector(min_pattern_length)
        
        # Inicializar validadores
        self.validator = PatternValidator()
        self.overlap_filter = OverlapFilter()
    
    def detect_all_patterns(self, df: pd.DataFrame, pattern_types: Optional[List[str]] = None) -> List[PatternResult]:
        """
        Detecta todos los patrones disponibles
        
        Args:
            df: DataFrame con datos OHLC
            pattern_types: Lista de tipos de patrones a detectar (None = todos)
            
        Returns:
            Lista de patrones detectados
        """
        patterns = []
        
        # Detectar patrones geométricos
        if not pattern_types or any(pt.startswith(('triangle', 'channel', 'rectangle')) for pt in pattern_types):
            patterns.extend(self.geometric_detector.detect_patterns(df))
        
        # Detectar patrones de velas
        if not pattern_types or any(pt in ['doji', 'hammer', 'shooting_star', 'engulfing'] for pt in pattern_types):
            patterns.extend(self.candlestick_detector.detect_patterns(df))
        
        # Detectar patrones de volumen
        if not pattern_types or any(pt.startswith('volume') for pt in pattern_types):
            patterns.extend(self.volume_detector.detect_patterns(df))
        
        # Filtrar por tipo específico si se especifica
        if pattern_types:
            patterns = [p for p in patterns if p.pattern_type in pattern_types]
        
        # Validar patrones
        patterns = self.validator.validate_patterns(patterns, df)
        
        # Filtrar por confianza
        patterns = [p for p in patterns if p.confidence >= self.confidence_threshold]
        
        # Filtrar patrones superpuestos
        patterns = self.overlap_filter.filter_overlapping_patterns(patterns)
        
        # Ordenar por confianza
        patterns.sort(key=lambda x: x.confidence, reverse=True)
        
        return patterns
    
    def get_pattern_summary(self, patterns: List[PatternResult]) -> Dict[str, int]:
        """
        Genera un resumen de los patrones detectados
        
        Args:
            patterns: Lista de patrones detectados
            
        Returns:
            Diccionario con conteo por tipo de patrón
        """
        summary = {}
        for pattern in patterns:
            pattern_type = pattern.pattern_type
            summary[pattern_type] = summary.get(pattern_type, 0) + 1
        
        return summary
    
    def get_strongest_patterns(self, patterns: List[PatternResult], top_n: int = 5) -> List[PatternResult]:
        """
        Obtiene los patrones más fuertes (mayor confianza)
        
        Args:
            patterns: Lista de patrones detectados
            top_n: Número de patrones a retornar
            
        Returns:
            Lista de los patrones más fuertes
        """
        return sorted(patterns, key=lambda x: x.confidence, reverse=True)[:top_n]
    
    def get_recent_patterns(self, patterns: List[PatternResult], recent_bars: int = 50) -> List[PatternResult]:
        """
        Obtiene patrones recientes (que terminan en las últimas barras)
        
        Args:
            patterns: Lista de patrones detectados
            recent_bars: Número de barras recientes a considerar
            
        Returns:
            Lista de patrones recientes
        """
        if not patterns:
            return []
        
        max_idx = max(p.end_idx for p in patterns)
        threshold = max_idx - recent_bars
        
        return [p for p in patterns if p.end_idx >= threshold]