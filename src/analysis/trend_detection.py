"""Detector de tendencias - Módulo principal.

Este módulo proporciona la clase TrendDetector que actúa como interfaz principal
para la detección de tendencias usando múltiples algoritmos.
"""

from .trend.core.detector import TrendDetector

# Re-exportar para mantener compatibilidad
__all__ = ['TrendDetector']