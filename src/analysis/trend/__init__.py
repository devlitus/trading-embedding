"""Módulo de detección de tendencias.

Este módulo proporciona diferentes algoritmos y utilidades
para la detección y análisis de tendencias en datos financieros.
"""

from .core.detector import TrendDetector
from .algorithms.moving_average import MovingAverageDetector
from .algorithms.zigzag import ZigZagDetector
from .algorithms.regression import RegressionDetector
from .utils.trend_utils import (
    filter_significant_changes,
    calculate_trend_consistency,
    detect_trend_exhaustion,
    calculate_volatility_adjusted_trend,
    identify_trend_phases,
    calculate_trend_velocity,
    detect_trend_convergence,
    calculate_support_resistance_strength,
    smooth_trend_signal,
    calculate_trend_correlation,
    detect_trend_breakout_confirmation,
    calculate_trend_momentum_divergence
)

__all__ = [
    'TrendDetector',
    'MovingAverageDetector',
    'ZigZagDetector',
    'RegressionDetector',
    'filter_significant_changes',
    'calculate_trend_consistency',
    'detect_trend_exhaustion',
    'calculate_volatility_adjusted_trend',
    'identify_trend_phases',
    'calculate_trend_velocity',
    'detect_trend_convergence',
    'calculate_support_resistance_strength',
    'smooth_trend_signal',
    'calculate_trend_correlation',
    'detect_trend_breakout_confirmation',
    'calculate_trend_momentum_divergence'
]