"""Reconocimiento de patrones técnicos en datos de trading.

Este módulo actúa como wrapper para la nueva estructura modular de patrones.
"""

# Importar desde la nueva estructura modular
from .patterns import PatternResult, PatternRecognizer, detect_patterns

# Re-exportar para compatibilidad
__all__ = ['PatternResult', 'PatternRecognizer', 'detect_patterns']