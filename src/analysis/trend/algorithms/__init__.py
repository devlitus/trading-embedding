"""Algoritmos de detección de tendencias."""

from .moving_average import MovingAverageDetector
from .zigzag import ZigZagDetector
from .regression import RegressionDetector

__all__ = [
    'MovingAverageDetector',
    'ZigZagDetector', 
    'RegressionDetector'
]