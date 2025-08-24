"""Módulo de reconocimiento de patrones técnicos."""

from .types import PatternResult
from .recognizers.pattern_recognizer import PatternRecognizer
from .recognizers.geometric_patterns import GeometricPatternDetector
from .recognizers.candlestick_patterns import CandlestickPatternDetector
from .recognizers.volume_patterns import VolumePatternDetector
from .validators.pattern_validator import PatternValidator
from .validators.overlap_filter import OverlapFilter

# Función de conveniencia
def detect_patterns(df, pattern_types=None):
    """Función de conveniencia para detectar patrones."""
    recognizer = PatternRecognizer()
    return recognizer.detect_all_patterns(df, pattern_types)

__all__ = [
    'PatternResult',
    'PatternRecognizer',
    'GeometricPatternDetector', 
    'CandlestickPatternDetector',
    'VolumePatternDetector',
    'PatternValidator',
    'OverlapFilter',
    'detect_patterns'
]