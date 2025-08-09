import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class WyckoffScoringSystem:
    """
    Sistema de puntuación para patrones Wyckoff.
    Evalúa la calidad y confianza de los patrones identificados.
    """
    
    def __init__(self):
        self.weights = {
            'volume_consistency': 0.3,
            'price_action': 0.25,
            'pattern_duration': 0.2,
            'market_context': 0.15,
            'confirmation_signals': 0.1
        }
    
    def calculate_score(self, data: pd.DataFrame, pattern: str, analysis: Dict) -> float:
        """
        Calcula un score para el patrón identificado.
        
        Args:
            data: DataFrame con datos OHLCV
            pattern: Tipo de patrón identificado
            analysis: Resultado del análisis heurístico
            
        Returns:
            Score entre 0 y 1
        """
        if data.empty or not pattern:
            return 0.0
        
        scores = {
            'volume_consistency': self._score_volume_consistency(data),
            'price_action': self._score_price_action(data, pattern),
            'pattern_duration': self._score_pattern_duration(data),
            'market_context': self._score_market_context(data),
            'confirmation_signals': self._score_confirmation_signals(analysis)
        }
        
        # Calcular score ponderado
        weighted_score = sum(
            scores[metric] * self.weights[metric] 
            for metric in scores
        )
        
        return min(max(weighted_score, 0.0), 1.0)
    
    def _score_volume_consistency(self, data: pd.DataFrame) -> float:
        """Evalúa la consistencia del volumen."""
        if len(data) < 10:
            return 0.5
        
        volume_std = data['volume'].std()
        volume_mean = data['volume'].mean()
        
        if volume_mean == 0:
            return 0.0
        
        cv = volume_std / volume_mean  # Coeficiente de variación
        
        # Score inverso: menor variabilidad = mejor score
        return max(0.0, 1.0 - min(cv, 1.0))
    
    def _score_price_action(self, data: pd.DataFrame, pattern: str) -> float:
        """Evalúa la acción del precio según el patrón."""
        if len(data) < 5:
            return 0.5
        
        price_volatility = data['close'].pct_change().std()
        
        # Diferentes patrones requieren diferentes características de precio
        if pattern in ['accumulation', 'distribution']:
            # Patrones laterales prefieren baja volatilidad
            return max(0.0, 1.0 - min(price_volatility * 10, 1.0))
        elif pattern in ['markup', 'markdown']:
            # Patrones direccionales pueden tener mayor volatilidad
            return min(price_volatility * 5, 1.0)
        else:
            return 0.5
    
    def _score_pattern_duration(self, data: pd.DataFrame) -> float:
        """Evalúa la duración del patrón."""
        duration = len(data)
        
        # Duración óptima entre 20-100 períodos
        if duration < 10:
            return duration / 10.0
        elif duration <= 100:
            return 1.0
        else:
            return max(0.5, 1.0 - (duration - 100) / 200.0)
    
    def _score_market_context(self, data: pd.DataFrame) -> float:
        """Evalúa el contexto del mercado."""
        if len(data) < 20:
            return 0.5
        
        # Evaluar tendencia general
        long_term_trend = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
        
        # Normalizar entre 0 y 1
        return 0.5 + np.tanh(long_term_trend * 5) * 0.5
    
    def _score_confirmation_signals(self, analysis: Dict) -> float:
        """Evalúa las señales de confirmación."""
        if not analysis or 'signals' not in analysis:
            return 0.0
        
        num_signals = len(analysis.get('signals', []))
        confidence = analysis.get('confidence', 0.0)
        
        # Combinar número de señales y confianza
        signal_score = min(num_signals / 5.0, 1.0)  # Máximo 5 señales
        
        return (signal_score + confidence) / 2.0
    
    def get_score_breakdown(self, data: pd.DataFrame, pattern: str, analysis: Dict) -> Dict[str, float]:
        """Retorna el desglose detallado del score."""
        return {
            'volume_consistency': self._score_volume_consistency(data),
            'price_action': self._score_price_action(data, pattern),
            'pattern_duration': self._score_pattern_duration(data),
            'market_context': self._score_market_context(data),
            'confirmation_signals': self._score_confirmation_signals(analysis)
        }