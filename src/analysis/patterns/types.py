"""Tipos de datos para el sistema de reconocimiento de patrones."""

from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class PatternResult:
    """Resultado de detección de un patrón técnico."""
    pattern_type: str
    start_idx: int
    end_idx: int
    confidence: float
    parameters: Dict[str, Any]
    description: str
    
    def __post_init__(self):
        """Validación post-inicialización."""
        if not 0 <= self.confidence <= 1:
            raise ValueError(f"Confidence debe estar entre 0 y 1, recibido: {self.confidence}")
        if self.start_idx > self.end_idx:
            raise ValueError(f"start_idx ({self.start_idx}) no puede ser mayor que end_idx ({self.end_idx})")
    
    @property
    def duration(self) -> int:
        """Duración del patrón en períodos."""
        return self.end_idx - self.start_idx + 1
    
    def overlaps_with(self, other: 'PatternResult') -> bool:
        """Verifica si este patrón se solapa con otro."""
        return not (self.end_idx < other.start_idx or other.end_idx < self.start_idx)
    
    def get_overlap_ratio(self, other: 'PatternResult') -> float:
        """Calcula el ratio de solapamiento con otro patrón."""
        if not self.overlaps_with(other):
            return 0.0
        
        overlap_start = max(self.start_idx, other.start_idx)
        overlap_end = min(self.end_idx, other.end_idx)
        overlap_duration = overlap_end - overlap_start + 1
        
        total_duration = max(self.end_idx, other.end_idx) - min(self.start_idx, other.start_idx) + 1
        return overlap_duration / total_duration