"""Filtro para manejar solapamientos entre patrones detectados."""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from ..types import PatternResult


class OverlapFilter:
    """Filtro para detectar y resolver solapamientos entre patrones."""
    
    def __init__(self, overlap_threshold: float = 0.5, priority_weights: Optional[Dict[str, float]] = None):
        """
        Inicializa el filtro de solapamiento.
        
        Args:
            overlap_threshold: Umbral de solapamiento para considerar patrones como superpuestos
            priority_weights: Pesos de prioridad por tipo de patrón
        """
        self.overlap_threshold = overlap_threshold
        self.priority_weights = priority_weights or {
            'triangle': 0.8,
            'channel': 0.9,
            'rectangle': 0.7,
            'engulfing': 0.9,
            'doji': 0.6,
            'hammer': 0.7,
            'volume_spike': 0.8,
            'volume_breakout': 0.9
        }
    
    def filter_overlapping_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Filtra patrones superpuestos manteniendo los de mayor prioridad.
        
        Args:
            patterns: Lista de patrones detectados
            
        Returns:
            List[Dict[str, Any]]: Lista de patrones filtrados sin solapamientos
        """
        if not patterns:
            return []
        
        # Ordenar patrones por prioridad (score combinado)
        sorted_patterns = sorted(patterns, key=self._get_pattern_priority, reverse=True)
        
        filtered_patterns = []
        
        for pattern in sorted_patterns:
            # Verificar si el patrón se solapa con alguno ya seleccionado
            if not self._has_significant_overlap(pattern, filtered_patterns):
                filtered_patterns.append(pattern)
        
        return filtered_patterns
    
    def _get_pattern_priority(self, pattern: Dict[str, Any]) -> float:
        """
        Calcula la prioridad de un patrón basada en tipo, confianza y otros factores.
        
        Args:
            pattern: Información del patrón
            
        Returns:
            float: Score de prioridad
        """
        pattern_type = pattern.get('type', '').lower()
        confidence = pattern.get('confidence', 0.5)
        
        # Peso base por tipo de patrón
        type_weight = 0.5
        for pattern_name, weight in self.priority_weights.items():
            if pattern_name in pattern_type:
                type_weight = weight
                break
        
        # Score combinado
        priority_score = (confidence * 0.6) + (type_weight * 0.4)
        
        # Bonus por patrones de breakout o confirmación
        if any(keyword in pattern_type for keyword in ['breakout', 'confirmation', 'engulfing']):
            priority_score += 0.1
        
        return priority_score
    
    def _has_significant_overlap(self, pattern: Dict[str, Any], existing_patterns: List[Dict[str, Any]]) -> bool:
        """
        Verifica si un patrón tiene solapamiento significativo con patrones existentes.
        
        Args:
            pattern: Patrón a verificar
            existing_patterns: Lista de patrones ya seleccionados
            
        Returns:
            bool: True si hay solapamiento significativo
        """
        for existing_pattern in existing_patterns:
            overlap_ratio = self._calculate_overlap_ratio(pattern, existing_pattern)
            if overlap_ratio > self.overlap_threshold:
                return True
        
        return False
    
    def _calculate_overlap_ratio(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> float:
        """
        Calcula el ratio de solapamiento entre dos patrones.
        
        Args:
            pattern1: Primer patrón
            pattern2: Segundo patrón
            
        Returns:
            float: Ratio de solapamiento entre 0 y 1
        """
        # Obtener rangos de índices para ambos patrones
        range1 = self._get_pattern_range(pattern1)
        range2 = self._get_pattern_range(pattern2)
        
        if not range1 or not range2:
            return 0.0
        
        start1, end1 = range1
        start2, end2 = range2
        
        # Calcular intersección
        intersection_start = max(start1, start2)
        intersection_end = min(end1, end2)
        
        if intersection_start >= intersection_end:
            return 0.0  # No hay solapamiento
        
        intersection_length = intersection_end - intersection_start
        
        # Calcular longitudes de los patrones
        length1 = end1 - start1
        length2 = end2 - start2
        
        # Ratio de solapamiento respecto al patrón más pequeño
        min_length = min(length1, length2)
        if min_length == 0:
            return 0.0
        
        return intersection_length / min_length
    
    def _get_pattern_range(self, pattern: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """
        Obtiene el rango de índices que abarca un patrón.
        
        Args:
            pattern: Información del patrón
            
        Returns:
            Optional[Tuple[int, int]]: Tupla con (start_index, end_index) o None
        """
        # Para patrones con rango explícito
        if 'start_index' in pattern and 'end_index' in pattern:
            return (pattern['start_index'], pattern['end_index'])
        
        # Para patrones de vela individual
        if 'index' in pattern:
            idx = pattern['index']
            return (idx, idx + 1)
        
        # Para patrones con lista de índices
        if 'indices' in pattern and pattern['indices']:
            indices = pattern['indices']
            return (min(indices), max(indices) + 1)
        
        return None
    
    def group_related_patterns(self, patterns: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """
        Agrupa patrones relacionados o cercanos en el tiempo.
        
        Args:
            patterns: Lista de patrones
            
        Returns:
            List[List[Dict[str, Any]]]: Lista de grupos de patrones relacionados
        """
        if not patterns:
            return []
        
        # Ordenar patrones por posición temporal
        sorted_patterns = sorted(patterns, key=lambda p: self._get_pattern_center(p))
        
        groups = []
        current_group = [sorted_patterns[0]]
        
        for i in range(1, len(sorted_patterns)):
            current_pattern = sorted_patterns[i]
            last_pattern = current_group[-1]
            
            # Verificar si los patrones están cerca temporalmente
            if self._are_patterns_related(current_pattern, last_pattern):
                current_group.append(current_pattern)
            else:
                # Iniciar nuevo grupo
                groups.append(current_group)
                current_group = [current_pattern]
        
        # Agregar el último grupo
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _get_pattern_center(self, pattern: Dict[str, Any]) -> float:
        """
        Obtiene el punto central de un patrón.
        
        Args:
            pattern: Información del patrón
            
        Returns:
            float: Índice central del patrón
        """
        pattern_range = self._get_pattern_range(pattern)
        if pattern_range:
            start, end = pattern_range
            return (start + end) / 2.0
        
        return pattern.get('index', 0)
    
    def _are_patterns_related(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> bool:
        """
        Determina si dos patrones están relacionados temporalmente.
        
        Args:
            pattern1: Primer patrón
            pattern2: Segundo patrón
            
        Returns:
            bool: True si los patrones están relacionados
        """
        center1 = self._get_pattern_center(pattern1)
        center2 = self._get_pattern_center(pattern2)
        
        # Considerar patrones relacionados si están dentro de 20 períodos
        distance = abs(center1 - center2)
        return distance <= 20
    
    def resolve_conflicts(self, pattern_groups: List[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
        """
        Resuelve conflictos dentro de grupos de patrones relacionados.
        
        Args:
            pattern_groups: Grupos de patrones relacionados
            
        Returns:
            List[Dict[str, Any]]: Lista de patrones resueltos
        """
        resolved_patterns = []
        
        for group in pattern_groups:
            if len(group) == 1:
                resolved_patterns.extend(group)
            else:
                # Resolver conflictos dentro del grupo
                resolved_group = self._resolve_group_conflicts(group)
                resolved_patterns.extend(resolved_group)
        
        return resolved_patterns
    
    def _resolve_group_conflicts(self, group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Resuelve conflictos dentro de un grupo de patrones.
        
        Args:
            group: Grupo de patrones en conflicto
            
        Returns:
            List[Dict[str, Any]]: Patrones resueltos del grupo
        """
        # Estrategia simple: mantener el patrón de mayor prioridad
        # y aquellos que no se solapen significativamente con él
        
        if not group:
            return []
        
        # Ordenar por prioridad
        sorted_group = sorted(group, key=self._get_pattern_priority, reverse=True)
        
        resolved = [sorted_group[0]]  # Mantener el de mayor prioridad
        
        for pattern in sorted_group[1:]:
            # Verificar si se solapa significativamente con alguno ya seleccionado
            if not self._has_significant_overlap(pattern, resolved):
                resolved.append(pattern)
        
        return resolved
    
    def get_overlap_statistics(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calcula estadísticas de solapamiento para una lista de patrones.
        
        Args:
            patterns: Lista de patrones
            
        Returns:
            Dict[str, Any]: Estadísticas de solapamiento
        """
        if len(patterns) < 2:
            return {
                'total_patterns': len(patterns),
                'overlapping_pairs': 0,
                'overlap_ratio': 0.0,
                'max_overlap': 0.0,
                'avg_overlap': 0.0
            }
        
        overlapping_pairs = 0
        overlap_ratios = []
        
        for i in range(len(patterns)):
            for j in range(i + 1, len(patterns)):
                overlap_ratio = self._calculate_overlap_ratio(patterns[i], patterns[j])
                overlap_ratios.append(overlap_ratio)
                
                if overlap_ratio > self.overlap_threshold:
                    overlapping_pairs += 1
        
        total_pairs = len(overlap_ratios)
        
        return {
            'total_patterns': len(patterns),
            'total_pairs': total_pairs,
            'overlapping_pairs': overlapping_pairs,
            'overlap_ratio': overlapping_pairs / total_pairs if total_pairs > 0 else 0.0,
            'max_overlap': max(overlap_ratios) if overlap_ratios else 0.0,
            'avg_overlap': np.mean(overlap_ratios) if overlap_ratios else 0.0,
            'overlap_threshold': self.overlap_threshold
        }