"""Detector de patrones de velas japonesas."""

import pandas as pd
import numpy as np
from typing import List

from ..types import PatternResult


class CandlestickPatternDetector:
    """Detector especializado en patrones de velas japonesas"""
    
    def __init__(self):
        pass
    
    def detect_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detecta todos los patrones de velas"""
        patterns = []
        
        for i in range(1, len(df)):
            # Patrones de una sola vela
            single_patterns = self._detect_single_candlestick_patterns(df, i)
            patterns.extend(single_patterns)
            
            # Patrones de múltiples velas
            if i >= 3:
                multi_patterns = self._detect_multi_candlestick_patterns(df, i)
                patterns.extend(multi_patterns)
        
        return patterns
    
    def _detect_single_candlestick_patterns(self, df: pd.DataFrame, idx: int) -> List[PatternResult]:
        """Detecta patrones de una sola vela"""
        patterns = []
        
        current = df.iloc[idx]
        prev = df.iloc[idx-1]
        
        # Calcular características de la vela
        body_size = abs(current['close'] - current['open'])
        total_range = current['high'] - current['low']
        upper_shadow = current['high'] - max(current['open'], current['close'])
        lower_shadow = min(current['open'], current['close']) - current['low']
        
        if total_range == 0:
            return patterns
        
        body_ratio = body_size / total_range
        upper_shadow_ratio = upper_shadow / total_range
        lower_shadow_ratio = lower_shadow / total_range
        
        # Doji
        if body_ratio < 0.1:
            confidence = 1 - body_ratio * 10
            patterns.append(PatternResult(
                pattern_type="doji",
                start_idx=idx,
                end_idx=idx,
                confidence=confidence,
                parameters={'body_ratio': body_ratio},
                description="Doji - Indecisión del mercado"
            ))
        
        # Hammer/Hanging Man
        if lower_shadow_ratio > 0.6 and upper_shadow_ratio < 0.1 and body_ratio < 0.3:
            is_bullish = current['close'] > prev['close']
            pattern_type = "hammer" if is_bullish else "hanging_man"
            confidence = lower_shadow_ratio * (1 - upper_shadow_ratio)
            
            patterns.append(PatternResult(
                pattern_type=pattern_type,
                start_idx=idx,
                end_idx=idx,
                confidence=confidence,
                parameters={
                    'lower_shadow_ratio': lower_shadow_ratio,
                    'body_ratio': body_ratio
                },
                description=f"{pattern_type.title()} - Posible reversión"
            ))
        
        # Shooting Star/Inverted Hammer
        if upper_shadow_ratio > 0.6 and lower_shadow_ratio < 0.1 and body_ratio < 0.3:
            is_bearish = current['close'] < prev['close']
            pattern_type = "shooting_star" if is_bearish else "inverted_hammer"
            confidence = upper_shadow_ratio * (1 - lower_shadow_ratio)
            
            patterns.append(PatternResult(
                pattern_type=pattern_type,
                start_idx=idx,
                end_idx=idx,
                confidence=confidence,
                parameters={
                    'upper_shadow_ratio': upper_shadow_ratio,
                    'body_ratio': body_ratio
                },
                description=f"{pattern_type.replace('_', ' ').title()} - Posible reversión"
            ))
        
        # Marubozu
        if body_ratio > 0.9 and upper_shadow_ratio < 0.05 and lower_shadow_ratio < 0.05:
            is_bullish = current['close'] > current['open']
            pattern_type = "bullish_marubozu" if is_bullish else "bearish_marubozu"
            confidence = body_ratio
            
            patterns.append(PatternResult(
                pattern_type=pattern_type,
                start_idx=idx,
                end_idx=idx,
                confidence=confidence,
                parameters={'body_ratio': body_ratio},
                description=f"Marubozu {'alcista' if is_bullish else 'bajista'} - Fuerte momentum"
            ))
        
        return patterns
    
    def _detect_multi_candlestick_patterns(self, df: pd.DataFrame, idx: int) -> List[PatternResult]:
        """Detecta patrones de múltiples velas"""
        patterns = []
        
        # Engulfing Pattern
        if idx >= 1:
            current = df.iloc[idx]
            prev = df.iloc[idx-1]
            
            # Bullish Engulfing
            if (prev['close'] < prev['open'] and  # Vela anterior bajista
                current['close'] > current['open'] and  # Vela actual alcista
                current['open'] < prev['close'] and  # Abre por debajo del cierre anterior
                current['close'] > prev['open']):  # Cierra por encima de la apertura anterior
                
                confidence = min(1.0, (current['close'] - current['open']) / (prev['open'] - prev['close']))
                
                patterns.append(PatternResult(
                    pattern_type="bullish_engulfing",
                    start_idx=idx-1,
                    end_idx=idx,
                    confidence=confidence,
                    parameters={},
                    description="Envolvente Alcista - Reversión alcista"
                ))
            
            # Bearish Engulfing
            elif (prev['close'] > prev['open'] and  # Vela anterior alcista
                  current['close'] < current['open'] and  # Vela actual bajista
                  current['open'] > prev['close'] and  # Abre por encima del cierre anterior
                  current['close'] < prev['open']):  # Cierra por debajo de la apertura anterior
                
                confidence = min(1.0, (current['open'] - current['close']) / (prev['close'] - prev['open']))
                
                patterns.append(PatternResult(
                    pattern_type="bearish_engulfing",
                    start_idx=idx-1,
                    end_idx=idx,
                    confidence=confidence,
                    parameters={},
                    description="Envolvente Bajista - Reversión bajista"
                ))
        
        # Morning Star / Evening Star
        if idx >= 2:
            first = df.iloc[idx-2]
            middle = df.iloc[idx-1]
            last = df.iloc[idx]
            
            # Morning Star (patrón alcista)
            if (first['close'] < first['open'] and  # Primera vela bajista
                abs(middle['close'] - middle['open']) < (first['high'] - first['low']) * 0.3 and  # Vela pequeña en el medio
                last['close'] > last['open'] and  # Última vela alcista
                last['close'] > (first['open'] + first['close']) / 2):  # Cierra por encima del punto medio de la primera
                
                confidence = 0.8  # Patrón complejo, confianza base
                
                patterns.append(PatternResult(
                    pattern_type="morning_star",
                    start_idx=idx-2,
                    end_idx=idx,
                    confidence=confidence,
                    parameters={},
                    description="Estrella de la Mañana - Reversión alcista"
                ))
            
            # Evening Star (patrón bajista)
            elif (first['close'] > first['open'] and  # Primera vela alcista
                  abs(middle['close'] - middle['open']) < (first['high'] - first['low']) * 0.3 and  # Vela pequeña en el medio
                  last['close'] < last['open'] and  # Última vela bajista
                  last['close'] < (first['open'] + first['close']) / 2):  # Cierra por debajo del punto medio de la primera
                
                confidence = 0.8  # Patrón complejo, confianza base
                
                patterns.append(PatternResult(
                    pattern_type="evening_star",
                    start_idx=idx-2,
                    end_idx=idx,
                    confidence=confidence,
                    parameters={},
                    description="Estrella de la Tarde - Reversión bajista"
                ))
        
        # Three White Soldiers / Three Black Crows
        if idx >= 2:
            candles = [df.iloc[idx-2], df.iloc[idx-1], df.iloc[idx]]
            
            # Three White Soldiers
            if all(c['close'] > c['open'] for c in candles):  # Todas alcistas
                if all(candles[i]['close'] > candles[i-1]['close'] for i in range(1, 3)):  # Cierres ascendentes
                    confidence = 0.75
                    
                    patterns.append(PatternResult(
                        pattern_type="three_white_soldiers",
                        start_idx=idx-2,
                        end_idx=idx,
                        confidence=confidence,
                        parameters={},
                        description="Tres Soldados Blancos - Fuerte tendencia alcista"
                    ))
            
            # Three Black Crows
            elif all(c['close'] < c['open'] for c in candles):  # Todas bajistas
                if all(candles[i]['close'] < candles[i-1]['close'] for i in range(1, 3)):  # Cierres descendentes
                    confidence = 0.75
                    
                    patterns.append(PatternResult(
                        pattern_type="three_black_crows",
                        start_idx=idx-2,
                        end_idx=idx,
                        confidence=confidence,
                        parameters={},
                        description="Tres Cuervos Negros - Fuerte tendencia bajista"
                    ))
        
        return patterns
    
    def _calculate_body_size(self, candle: pd.Series) -> float:
        """Calcula el tamaño del cuerpo de la vela"""
        return abs(candle['close'] - candle['open'])
    
    def _calculate_upper_shadow(self, candle: pd.Series) -> float:
        """Calcula el tamaño de la sombra superior"""
        return candle['high'] - max(candle['open'], candle['close'])
    
    def _calculate_lower_shadow(self, candle: pd.Series) -> float:
        """Calcula el tamaño de la sombra inferior"""
        return min(candle['open'], candle['close']) - candle['low']
    
    def _is_bullish_candle(self, candle: pd.Series) -> bool:
        """Determina si una vela es alcista"""
        return candle['close'] > candle['open']
    
    def _is_bearish_candle(self, candle: pd.Series) -> bool:
        """Determina si una vela es bajista"""
        return candle['close'] < candle['open']