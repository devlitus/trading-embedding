import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime

class Signal:
    """Clase para representar una señal de trading."""
    def __init__(self, pattern: str, confidence: float, phase: str, description: str, timestamp: datetime = None):
        self.pattern = pattern
        self.confidence = confidence
        self.phase = phase
        self.description = description
        self.timestamp = timestamp or datetime.now()

class WyckoffHeuristicEngine:
    """
    Motor heurístico para análisis de patrones Wyckoff.
    Implementa reglas heurísticas para identificar fases del mercado.
    """
    
    def __init__(self):
        self.patterns = {
            'accumulation': 'Acumulación',
            'markup': 'Markup',
            'distribution': 'Distribución',
            'markdown': 'Markdown',
            'neutral': 'Neutral'
        }
    
    def analyze_pattern(self, data: pd.DataFrame) -> List[Signal]:
        """
        Analiza los datos para identificar patrones Wyckoff.
        
        Args:
            data: DataFrame con datos OHLCV
            
        Returns:
            Lista de objetos Signal con el análisis del patrón
        """
        if data.empty:
            return [Signal(
                pattern='neutral',
                confidence=0.0,
                phase='Neutral',
                description='No hay datos suficientes para análisis'
            )]
        
        # Análisis básico de volumen y precio
        volume_trend = self._analyze_volume_trend(data)
        price_trend = self._analyze_price_trend(data)
        
        # Determinar patrón basado en heurísticas
        pattern = self._determine_pattern(volume_trend, price_trend)
        confidence = self._calculate_confidence(data, pattern)
        phase = self.patterns.get(pattern, 'Neutral')
        
        # Generar señales como objetos Signal
        signal_descriptions = self._generate_signals(data, pattern)
        signals = []
        
        for description in signal_descriptions:
            signals.append(Signal(
                pattern=pattern,
                confidence=confidence,
                phase=phase,
                description=description
            ))
        
        return signals
    
    def _analyze_volume_trend(self, data: pd.DataFrame) -> str:
        """Analiza la tendencia del volumen."""
        if len(data) < 10:
            return 'neutral'
        
        recent_volume = data['volume'].tail(5).mean()
        historical_volume = data['volume'].head(5).mean()
        
        if recent_volume > historical_volume * 1.2:
            return 'increasing'
        elif recent_volume < historical_volume * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def _analyze_price_trend(self, data: pd.DataFrame) -> str:
        """Analiza la tendencia del precio."""
        if len(data) < 10:
            return 'neutral'
        
        price_change = (data['close'].iloc[-1] - data['close'].iloc[0]) / data['close'].iloc[0]
        
        if price_change > 0.02:
            return 'bullish'
        elif price_change < -0.02:
            return 'bearish'
        else:
            return 'sideways'
    
    def _determine_pattern(self, volume_trend: str, price_trend: str) -> str:
        """Determina el patrón Wyckoff basado en volumen y precio."""
        if volume_trend == 'increasing' and price_trend == 'sideways':
            return 'accumulation'
        elif volume_trend == 'stable' and price_trend == 'bullish':
            return 'markup'
        elif volume_trend == 'increasing' and price_trend == 'sideways':
            return 'distribution'
        elif volume_trend == 'decreasing' and price_trend == 'bearish':
            return 'markdown'
        else:
            return 'neutral'
    
    def _calculate_confidence(self, data: pd.DataFrame, pattern: str) -> float:
        """Calcula la confianza del patrón identificado."""
        if pattern == 'neutral':
            return 0.5
        
        # Confianza basada en la consistencia de los datos
        if len(data) < 20:
            return 0.6
        elif len(data) < 50:
            return 0.7
        else:
            return 0.8
    
    def _generate_signals(self, data: pd.DataFrame, pattern: str) -> List[str]:
        """Genera señales basadas en el patrón identificado."""
        signals = []
        
        if pattern == 'accumulation':
            signals.append('Posible zona de acumulación detectada')
            signals.append('Volumen alto con precio lateral')
        elif pattern == 'markup':
            signals.append('Fase de markup en progreso')
            signals.append('Tendencia alcista confirmada')
        elif pattern == 'distribution':
            signals.append('Posible zona de distribución')
            signals.append('Atención a posible reversión')
        elif pattern == 'markdown':
            signals.append('Fase de markdown activa')
            signals.append('Tendencia bajista confirmada')
        else:
            signals.append('Mercado en fase neutral')
        
        return signals