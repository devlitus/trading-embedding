"""Detector de patrones de volumen."""

import pandas as pd
import numpy as np
from typing import List, Optional
from scipy.signal import find_peaks

from ..types import PatternResult


class VolumePatternDetector:
    """Detector especializado en patrones de volumen"""
    
    def __init__(self, min_pattern_length: int = 20):
        self.min_pattern_length = min_pattern_length
    
    def detect_patterns(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detecta todos los patrones de volumen"""
        patterns = []
        
        patterns.extend(self.detect_volume_spikes(df))
        patterns.extend(self.detect_volume_dry_up(df))
        patterns.extend(self.detect_volume_accumulation(df))
        patterns.extend(self.detect_volume_distribution(df))
        patterns.extend(self.detect_volume_breakout_confirmation(df))
        
        return patterns
    
    def detect_volume_spikes(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detecta picos de volumen anómalos"""
        patterns = []
        
        if 'volume' not in df.columns:
            return patterns
        
        # Calcular media móvil del volumen
        volume_ma = df['volume'].rolling(window=20).mean()
        volume_std = df['volume'].rolling(window=20).std()
        
        # Detectar picos (volumen > media + 2*std)
        spike_threshold = volume_ma + 2 * volume_std
        spikes = df['volume'] > spike_threshold
        
        # Encontrar índices de picos
        spike_indices = df.index[spikes].tolist()
        
        for idx in spike_indices:
            if idx < 20:  # Necesitamos datos históricos
                continue
            
            volume_ratio = df.loc[idx, 'volume'] / volume_ma.loc[idx]
            confidence = min(1.0, (volume_ratio - 1) / 2)  # Normalizar confianza
            
            # Determinar si es alcista o bajista basado en el precio
            price_change = (df.loc[idx, 'close'] - df.loc[idx, 'open']) / df.loc[idx, 'open']
            pattern_type = "volume_spike_bullish" if price_change > 0 else "volume_spike_bearish"
            
            patterns.append(PatternResult(
                pattern_type=pattern_type,
                start_idx=idx,
                end_idx=idx,
                confidence=confidence,
                parameters={
                    'volume_ratio': volume_ratio,
                    'price_change': price_change,
                    'volume': df.loc[idx, 'volume'],
                    'volume_ma': volume_ma.loc[idx]
                },
                description=f"Pico de volumen {volume_ratio:.1f}x la media"
            ))
        
        return patterns
    
    def detect_volume_dry_up(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detecta períodos de volumen seco (baja actividad)"""
        patterns = []
        
        if 'volume' not in df.columns:
            return patterns
        
        # Calcular media móvil del volumen
        volume_ma = df['volume'].rolling(window=20).mean()
        volume_std = df['volume'].rolling(window=20).std()
        
        # Detectar volumen bajo (volumen < media - std)
        low_threshold = volume_ma - volume_std
        low_volume = df['volume'] < low_threshold
        
        # Buscar períodos consecutivos de volumen bajo
        for i in range(len(df) - self.min_pattern_length):
            end_idx = i + self.min_pattern_length
            window_low_volume = low_volume.iloc[i:end_idx]
            
            # Al menos 70% del período debe tener volumen bajo
            if window_low_volume.sum() / len(window_low_volume) >= 0.7:
                avg_volume_ratio = (df['volume'].iloc[i:end_idx] / volume_ma.iloc[i:end_idx]).mean()
                confidence = max(0, 1 - avg_volume_ratio)  # Menor volumen = mayor confianza
                
                patterns.append(PatternResult(
                    pattern_type="volume_dry_up",
                    start_idx=i,
                    end_idx=end_idx - 1,
                    confidence=confidence,
                    parameters={
                        'avg_volume_ratio': avg_volume_ratio,
                        'low_volume_percentage': window_low_volume.sum() / len(window_low_volume)
                    },
                    description=f"Volumen seco - {avg_volume_ratio:.1f}x la media"
                ))
        
        return patterns
    
    def detect_volume_accumulation(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detecta patrones de acumulación (volumen alto con precios estables)"""
        patterns = []
        
        if 'volume' not in df.columns:
            return patterns
        
        for i in range(len(df) - self.min_pattern_length):
            end_idx = i + self.min_pattern_length
            window = df.iloc[i:end_idx]
            
            # Calcular volatilidad de precios
            price_volatility = window['close'].std() / window['close'].mean()
            
            # Calcular volumen promedio
            avg_volume = window['volume'].mean()
            historical_avg_volume = df['volume'].iloc[:i+self.min_pattern_length].rolling(window=50).mean().iloc[-1]
            
            if pd.isna(historical_avg_volume):
                continue
            
            volume_ratio = avg_volume / historical_avg_volume
            
            # Acumulación: volumen alto + baja volatilidad de precios
            if volume_ratio > 1.2 and price_volatility < 0.05:
                confidence = min(1.0, volume_ratio - 1) * (1 - price_volatility * 10)
                
                patterns.append(PatternResult(
                    pattern_type="volume_accumulation",
                    start_idx=i,
                    end_idx=end_idx - 1,
                    confidence=confidence,
                    parameters={
                        'volume_ratio': volume_ratio,
                        'price_volatility': price_volatility,
                        'avg_volume': avg_volume
                    },
                    description=f"Acumulación - Volumen {volume_ratio:.1f}x, volatilidad {price_volatility:.3f}"
                ))
        
        return patterns
    
    def detect_volume_distribution(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detecta patrones de distribución (volumen alto con precios en declive)"""
        patterns = []
        
        if 'volume' not in df.columns:
            return patterns
        
        for i in range(len(df) - self.min_pattern_length):
            end_idx = i + self.min_pattern_length
            window = df.iloc[i:end_idx]
            
            # Calcular tendencia de precios
            price_trend = (window['close'].iloc[-1] - window['close'].iloc[0]) / window['close'].iloc[0]
            
            # Calcular volumen promedio
            avg_volume = window['volume'].mean()
            historical_avg_volume = df['volume'].iloc[:i+self.min_pattern_length].rolling(window=50).mean().iloc[-1]
            
            if pd.isna(historical_avg_volume):
                continue
            
            volume_ratio = avg_volume / historical_avg_volume
            
            # Distribución: volumen alto + tendencia bajista
            if volume_ratio > 1.2 and price_trend < -0.02:
                confidence = min(1.0, volume_ratio - 1) * min(1.0, abs(price_trend) * 10)
                
                patterns.append(PatternResult(
                    pattern_type="volume_distribution",
                    start_idx=i,
                    end_idx=end_idx - 1,
                    confidence=confidence,
                    parameters={
                        'volume_ratio': volume_ratio,
                        'price_trend': price_trend,
                        'avg_volume': avg_volume
                    },
                    description=f"Distribución - Volumen {volume_ratio:.1f}x, caída {price_trend:.1%}"
                ))
        
        return patterns
    
    def detect_volume_breakout_confirmation(self, df: pd.DataFrame) -> List[PatternResult]:
        """Detecta confirmaciones de ruptura por volumen"""
        patterns = []
        
        if 'volume' not in df.columns:
            return patterns
        
        # Calcular máximos y mínimos locales de precio
        highs_idx = find_peaks(df['high'].values, distance=10)[0]
        lows_idx = find_peaks(-df['low'].values, distance=10)[0]
        
        # Verificar rupturas con volumen
        for idx in range(20, len(df)):
            current_price = df.iloc[idx]['close']
            current_volume = df.iloc[idx]['volume']
            
            # Volumen promedio de los últimos 20 períodos
            avg_volume = df['volume'].iloc[idx-20:idx].mean()
            volume_ratio = current_volume / avg_volume
            
            # Buscar rupturas de resistencia
            recent_highs = [h for h in highs_idx if h < idx and h >= idx - 50]
            if recent_highs:
                resistance_level = df.iloc[recent_highs]['high'].max()
                
                if current_price > resistance_level and volume_ratio > 1.5:
                    confidence = min(1.0, (volume_ratio - 1) / 2)
                    
                    patterns.append(PatternResult(
                        pattern_type="volume_breakout_bullish",
                        start_idx=idx,
                        end_idx=idx,
                        confidence=confidence,
                        parameters={
                            'volume_ratio': volume_ratio,
                            'resistance_level': resistance_level,
                            'breakout_price': current_price
                        },
                        description=f"Ruptura alcista confirmada por volumen {volume_ratio:.1f}x"
                    ))
            
            # Buscar rupturas de soporte
            recent_lows = [l for l in lows_idx if l < idx and l >= idx - 50]
            if recent_lows:
                support_level = df.iloc[recent_lows]['low'].min()
                
                if current_price < support_level and volume_ratio > 1.5:
                    confidence = min(1.0, (volume_ratio - 1) / 2)
                    
                    patterns.append(PatternResult(
                        pattern_type="volume_breakout_bearish",
                        start_idx=idx,
                        end_idx=idx,
                        confidence=confidence,
                        parameters={
                            'volume_ratio': volume_ratio,
                            'support_level': support_level,
                            'breakout_price': current_price
                        },
                        description=f"Ruptura bajista confirmada por volumen {volume_ratio:.1f}x"
                    ))
        
        return patterns
    
    def _calculate_volume_profile(self, df: pd.DataFrame, window_size: int = 20) -> pd.Series:
        """Calcula el perfil de volumen para una ventana deslizante"""
        volume_profile = pd.Series(index=df.index, dtype=float)
        
        for i in range(window_size, len(df)):
            window_volume = df['volume'].iloc[i-window_size:i]
            volume_profile.iloc[i] = window_volume.mean()
        
        return volume_profile
    
    def _detect_volume_anomalies(self, df: pd.DataFrame, threshold: float = 2.0) -> List[int]:
        """Detecta anomalías en el volumen usando desviación estándar"""
        volume_ma = df['volume'].rolling(window=20).mean()
        volume_std = df['volume'].rolling(window=20).std()
        
        anomaly_threshold = volume_ma + threshold * volume_std
        anomalies = df.index[df['volume'] > anomaly_threshold].tolist()
        
        return [idx for idx in anomalies if idx >= 20]  # Filtrar índices válidos