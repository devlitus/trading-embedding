"""Utilidades para detección de tendencias."""

from .trend_utils import (
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