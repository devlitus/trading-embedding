import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

class WyckoffFeatureExtractor:
    """
    Extractor de características específicas para patrones de Wyckoff.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def extract_price_action_features(self, ohlc_data: pd.DataFrame) -> Dict[str, float]:
        """
        Extrae características de la acción del precio relevantes para Wyckoff.
        
        Args:
            ohlc_data: DataFrame con datos OHLC
            
        Returns:
            Diccionario de características
        """
        features = {}
        
        # Características básicas de precio
        features['price_range'] = (ohlc_data['high'].max() - ohlc_data['low'].min()) / ohlc_data['close'].iloc[-1]
        features['avg_body_size'] = abs(ohlc_data['close'] - ohlc_data['open']).mean() / ohlc_data['close'].mean()
        features['avg_wick_size'] = ((ohlc_data['high'] - ohlc_data[['open', 'close']].max(axis=1)) + 
                                   (ohlc_data[['open', 'close']].min(axis=1) - ohlc_data['low'])).mean() / ohlc_data['close'].mean()
        
        # Características de tendencia
        features['trend_strength'] = (ohlc_data['close'].iloc[-1] - ohlc_data['close'].iloc[0]) / ohlc_data['close'].iloc[0]
        features['volatility'] = ohlc_data['close'].pct_change().std()
        
        # Características de soporte y resistencia
        features['support_tests'] = self._count_support_tests(ohlc_data)
        features['resistance_tests'] = self._count_resistance_tests(ohlc_data)
        
        # Características de gaps
        features['gap_count'] = self._count_gaps(ohlc_data)
        features['avg_gap_size'] = self._avg_gap_size(ohlc_data)
        
        return features
    
    def extract_volume_features(self, ohlc_data: pd.DataFrame) -> Dict[str, float]:
        """
        Extrae características de volumen relevantes para Wyckoff.
        
        Args:
            ohlc_data: DataFrame con datos OHLC incluyendo volumen
            
        Returns:
            Diccionario de características de volumen
        """
        features = {}
        
        if 'volume' not in ohlc_data.columns:
            return features
        
        volume = ohlc_data['volume']
        price_change = ohlc_data['close'].pct_change()
        
        # Características básicas de volumen
        features['avg_volume'] = volume.mean()
        features['volume_volatility'] = volume.std() / volume.mean()
        features['volume_trend'] = (volume.iloc[-10:].mean() - volume.iloc[:10].mean()) / volume.mean()
        
        # Relación precio-volumen
        features['price_volume_correlation'] = price_change.corr(volume)
        features['volume_price_trend'] = self._calculate_vpt(ohlc_data)
        
        # Características de distribución de volumen
        features['high_volume_days'] = (volume > volume.quantile(0.8)).sum() / len(volume)
        features['low_volume_days'] = (volume < volume.quantile(0.2)).sum() / len(volume)
        
        # On-Balance Volume
        features['obv_trend'] = self._calculate_obv_trend(ohlc_data)
        
        return features
    
    def extract_wyckoff_specific_features(self, ohlc_data: pd.DataFrame) -> Dict[str, float]:
        """
        Extrae características específicas de los patrones de Wyckoff.
        
        Args:
            ohlc_data: DataFrame con datos OHLC
            
        Returns:
            Diccionario de características específicas de Wyckoff
        """
        features = {}
        
        # Características de acumulación/distribución
        features['accumulation_score'] = self._calculate_accumulation_score(ohlc_data)
        features['distribution_score'] = self._calculate_distribution_score(ohlc_data)
        
        # Características de fases
        features['stopping_action'] = self._detect_stopping_action(ohlc_data)
        features['preliminary_support'] = self._detect_preliminary_support(ohlc_data)
        features['selling_climax'] = self._detect_selling_climax(ohlc_data)
        features['automatic_rally'] = self._detect_automatic_rally(ohlc_data)
        features['secondary_test'] = self._detect_secondary_test(ohlc_data)
        
        # Características de spring/upthrust
        features['spring_probability'] = self._calculate_spring_probability(ohlc_data)
        features['upthrust_probability'] = self._calculate_upthrust_probability(ohlc_data)
        
        # Características de esfuerzo vs resultado
        features['effort_result_ratio'] = self._calculate_effort_result_ratio(ohlc_data)
        
        return features
    
    def _count_support_tests(self, ohlc_data: pd.DataFrame) -> int:
        """Cuenta el número de tests de soporte."""
        low_level = ohlc_data['low'].min()
        tolerance = (ohlc_data['high'].max() - ohlc_data['low'].min()) * 0.02
        
        support_tests = 0
        for i in range(1, len(ohlc_data)):
            if abs(ohlc_data['low'].iloc[i] - low_level) <= tolerance:
                support_tests += 1
        
        return support_tests
    
    def _count_resistance_tests(self, ohlc_data: pd.DataFrame) -> int:
        """Cuenta el número de tests de resistencia."""
        high_level = ohlc_data['high'].max()
        tolerance = (ohlc_data['high'].max() - ohlc_data['low'].min()) * 0.02
        
        resistance_tests = 0
        for i in range(1, len(ohlc_data)):
            if abs(ohlc_data['high'].iloc[i] - high_level) <= tolerance:
                resistance_tests += 1
        
        return resistance_tests
    
    def _count_gaps(self, ohlc_data: pd.DataFrame) -> int:
        """Cuenta el número de gaps en los datos."""
        gaps = 0
        for i in range(1, len(ohlc_data)):
            prev_high = ohlc_data['high'].iloc[i-1]
            curr_low = ohlc_data['low'].iloc[i]
            prev_low = ohlc_data['low'].iloc[i-1]
            curr_high = ohlc_data['high'].iloc[i]
            
            # Gap up
            if curr_low > prev_high:
                gaps += 1
            # Gap down
            elif curr_high < prev_low:
                gaps += 1
        
        return gaps
    
    def _avg_gap_size(self, ohlc_data: pd.DataFrame) -> float:
        """Calcula el tamaño promedio de los gaps."""
        gap_sizes = []
        
        for i in range(1, len(ohlc_data)):
            prev_high = ohlc_data['high'].iloc[i-1]
            curr_low = ohlc_data['low'].iloc[i]
            prev_low = ohlc_data['low'].iloc[i-1]
            curr_high = ohlc_data['high'].iloc[i]
            
            # Gap up
            if curr_low > prev_high:
                gap_sizes.append(curr_low - prev_high)
            # Gap down
            elif curr_high < prev_low:
                gap_sizes.append(prev_low - curr_high)
        
        return np.mean(gap_sizes) if gap_sizes else 0.0
    
    def _calculate_vpt(self, ohlc_data: pd.DataFrame) -> float:
        """Calcula la tendencia del Volume Price Trend."""
        if 'volume' not in ohlc_data.columns:
            return 0.0
        
        price_change = ohlc_data['close'].pct_change()
        vpt = (price_change * ohlc_data['volume']).cumsum()
        
        # Calcular tendencia de VPT
        if len(vpt) > 10:
            recent_vpt = vpt.iloc[-10:].mean()
            early_vpt = vpt.iloc[:10].mean()
            return (recent_vpt - early_vpt) / abs(early_vpt) if early_vpt != 0 else 0.0
        
        return 0.0
    
    def _calculate_obv_trend(self, ohlc_data: pd.DataFrame) -> float:
        """Calcula la tendencia del On-Balance Volume."""
        if 'volume' not in ohlc_data.columns:
            return 0.0
        
        price_change = ohlc_data['close'].diff()
        obv = np.where(price_change > 0, ohlc_data['volume'], 
                      np.where(price_change < 0, -ohlc_data['volume'], 0)).cumsum()
        
        # Calcular tendencia de OBV
        if len(obv) > 10:
            recent_obv = obv[-10:].mean()
            early_obv = obv[:10].mean()
            return (recent_obv - early_obv) / abs(early_obv) if early_obv != 0 else 0.0
        
        return 0.0
    
    def _calculate_accumulation_score(self, ohlc_data: pd.DataFrame) -> float:
        """Calcula un score de acumulación basado en características de Wyckoff."""
        score = 0.0
        
        # Precio en rango lateral
        price_range = ohlc_data['high'].max() - ohlc_data['low'].min()
        avg_price = ohlc_data['close'].mean()
        range_ratio = price_range / avg_price
        
        if range_ratio < 0.1:  # Rango lateral estrecho
            score += 0.3
        
        # Volumen creciente en bajas
        if 'volume' in ohlc_data.columns:
            low_prices = ohlc_data['close'] < ohlc_data['close'].quantile(0.3)
            high_volume = ohlc_data['volume'] > ohlc_data['volume'].median()
            
            if (low_prices & high_volume).sum() > len(ohlc_data) * 0.2:
                score += 0.4
        
        # Tests de soporte exitosos
        support_tests = self._count_support_tests(ohlc_data)
        if support_tests >= 2:
            score += 0.3
        
        return min(score, 1.0)
    
    def _calculate_distribution_score(self, ohlc_data: pd.DataFrame) -> float:
        """Calcula un score de distribución basado en características de Wyckoff."""
        score = 0.0
        
        # Precio en rango lateral alto
        recent_high = ohlc_data['high'].iloc[-20:].max()
        overall_high = ohlc_data['high'].max()
        
        if recent_high >= overall_high * 0.95:  # Cerca de máximos
            score += 0.3
        
        # Volumen alto en máximos
        if 'volume' in ohlc_data.columns:
            high_prices = ohlc_data['close'] > ohlc_data['close'].quantile(0.7)
            high_volume = ohlc_data['volume'] > ohlc_data['volume'].quantile(0.8)
            
            if (high_prices & high_volume).sum() > len(ohlc_data) * 0.15:
                score += 0.4
        
        # Tests de resistencia
        resistance_tests = self._count_resistance_tests(ohlc_data)
        if resistance_tests >= 2:
            score += 0.3
        
        return min(score, 1.0)
    
    def _detect_stopping_action(self, ohlc_data: pd.DataFrame) -> float:
        """Detecta acción de parada (stopping action)."""
        # Buscar velas con cuerpos pequeños y mechas largas
        body_size = abs(ohlc_data['close'] - ohlc_data['open'])
        total_range = ohlc_data['high'] - ohlc_data['low']
        
        small_body_ratio = (body_size / total_range < 0.3).sum() / len(ohlc_data)
        
        return small_body_ratio
    
    def _detect_preliminary_support(self, ohlc_data: pd.DataFrame) -> float:
        """Detecta soporte preliminar."""
        # Buscar el primer rebote significativo desde mínimos
        min_idx = ohlc_data['low'].idxmin()
        
        if min_idx < len(ohlc_data) - 5:
            subsequent_data = ohlc_data.loc[min_idx:]
            rally_strength = (subsequent_data['close'].max() - subsequent_data['close'].iloc[0]) / subsequent_data['close'].iloc[0]
            
            return min(rally_strength * 2, 1.0)  # Normalizar a [0,1]
        
        return 0.0
    
    def _detect_selling_climax(self, ohlc_data: pd.DataFrame) -> float:
        """Detecta clímax de venta."""
        if 'volume' not in ohlc_data.columns:
            return 0.0
        
        # Buscar caídas fuertes con volumen alto
        price_drops = ohlc_data['close'].pct_change() < -0.03  # Caída > 3%
        high_volume = ohlc_data['volume'] > ohlc_data['volume'].quantile(0.9)
        
        climax_days = (price_drops & high_volume).sum()
        
        return min(climax_days / 5.0, 1.0)  # Normalizar
    
    def _detect_automatic_rally(self, ohlc_data: pd.DataFrame) -> float:
        """Detecta rally automático."""
        # Buscar rebote fuerte después de mínimos
        min_idx = ohlc_data['low'].idxmin()
        
        if min_idx < len(ohlc_data) - 3:
            rally_data = ohlc_data.loc[min_idx:min_idx+3]
            rally_strength = (rally_data['close'].iloc[-1] - rally_data['close'].iloc[0]) / rally_data['close'].iloc[0]
            
            return min(rally_strength * 5, 1.0)  # Normalizar
        
        return 0.0
    
    def _detect_secondary_test(self, ohlc_data: pd.DataFrame) -> float:
        """Detecta test secundario."""
        # Buscar retorno a niveles de soporte con menor volumen
        low_level = ohlc_data['low'].min()
        tolerance = (ohlc_data['high'].max() - ohlc_data['low'].min()) * 0.03
        
        tests = []
        for i in range(len(ohlc_data)):
            if abs(ohlc_data['low'].iloc[i] - low_level) <= tolerance:
                tests.append(i)
        
        if len(tests) >= 2 and 'volume' in ohlc_data.columns:
            first_test_vol = ohlc_data['volume'].iloc[tests[0]]
            second_test_vol = ohlc_data['volume'].iloc[tests[1]]
            
            if second_test_vol < first_test_vol * 0.8:  # Volumen menor en segundo test
                return 1.0
        
        return 0.0
    
    def _calculate_spring_probability(self, ohlc_data: pd.DataFrame) -> float:
        """Calcula probabilidad de spring."""
        # Buscar ruptura falsa de soporte seguida de recuperación
        support_level = ohlc_data['low'].quantile(0.1)
        
        spring_signals = 0
        for i in range(5, len(ohlc_data)):
            # Ruptura de soporte
            if ohlc_data['low'].iloc[i] < support_level:
                # Recuperación rápida
                if i < len(ohlc_data) - 2:
                    recovery = ohlc_data['close'].iloc[i+1:i+3].min() > support_level
                    if recovery:
                        spring_signals += 1
        
        return min(spring_signals / 3.0, 1.0)
    
    def _calculate_upthrust_probability(self, ohlc_data: pd.DataFrame) -> float:
        """Calcula probabilidad de upthrust."""
        # Buscar ruptura falsa de resistencia seguida de caída
        resistance_level = ohlc_data['high'].quantile(0.9)
        
        upthrust_signals = 0
        for i in range(5, len(ohlc_data)):
            # Ruptura de resistencia
            if ohlc_data['high'].iloc[i] > resistance_level:
                # Caída rápida
                if i < len(ohlc_data) - 2:
                    decline = ohlc_data['close'].iloc[i+1:i+3].max() < resistance_level
                    if decline:
                        upthrust_signals += 1
        
        return min(upthrust_signals / 3.0, 1.0)
    
    def _calculate_effort_result_ratio(self, ohlc_data: pd.DataFrame) -> float:
        """Calcula la relación esfuerzo vs resultado."""
        if 'volume' not in ohlc_data.columns:
            return 0.5
        
        # Correlación entre volumen (esfuerzo) y movimiento de precio (resultado)
        price_movement = abs(ohlc_data['close'].pct_change())
        volume_normalized = ohlc_data['volume'] / ohlc_data['volume'].mean()
        
        correlation = price_movement.corr(volume_normalized)
        
        # Convertir correlación a score [0,1]
        return (correlation + 1) / 2 if not np.isnan(correlation) else 0.5