import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime

from ..core.pattern import WyckoffPattern
from ..features.feature_extractor import WyckoffFeatureExtractor
from ..models.classifier import WyckoffClassifier

class WyckoffAnalyzer:
    """
    Analizador principal de patrones de Wyckoff.
    Integra extracción de características, clasificación y análisis de patrones.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Inicializa el analizador de Wyckoff.
        
        Args:
            model_type: Tipo de modelo para clasificación
        """
        self.classifier = WyckoffClassifier(model_type=model_type)
        self.feature_extractor = WyckoffFeatureExtractor()
        self.logger = logging.getLogger(__name__)
        
        # Configuración de análisis
        self.min_pattern_length = 20  # Mínimo de velas para un patrón
        self.confidence_threshold = 0.7  # Umbral mínimo de confianza
    
    def analyze_pattern(self, ohlc_data: pd.DataFrame, 
                       symbol: str = None) -> Optional[WyckoffPattern]:
        """
        Analiza los datos OHLC para identificar patrones de Wyckoff.
        
        Args:
            ohlc_data: DataFrame con datos OHLC
            symbol: Símbolo del activo (opcional)
            
        Returns:
            WyckoffPattern identificado o None si no se encuentra patrón válido
        """
        if len(ohlc_data) < self.min_pattern_length:
            self.logger.warning(f"Datos insuficientes para análisis: {len(ohlc_data)} velas")
            return None
        
        try:
            # Predecir patrón
            pattern_type, confidence = self.classifier.predict(ohlc_data)
            
            if confidence < self.confidence_threshold:
                self.logger.info(f"Confianza insuficiente: {confidence:.3f}")
                return None
            
            # Determinar fase específica
            phase = self._determine_phase(ohlc_data, pattern_type)
            
            # Extraer niveles clave
            key_levels = self._extract_key_levels(ohlc_data)
            
            # Analizar perfil de volumen
            volume_profile = self._analyze_volume_profile(ohlc_data)
            
            # Analizar acción del precio
            price_action = self._analyze_price_action(ohlc_data)
            
            # Crear patrón
            pattern = WyckoffPattern(
                pattern_type=pattern_type,
                phase=phase,
                confidence=confidence,
                start_time=ohlc_data.index[0],
                end_time=ohlc_data.index[-1],
                key_levels=key_levels,
                volume_profile=volume_profile,
                price_action=price_action
            )
            
            self.logger.info(f"Patrón identificado: {pattern_type} (confianza: {confidence:.3f})")
            
            return pattern
            
        except Exception as e:
            self.logger.error(f"Error en análisis de patrón: {e}")
            return None
    
    def _determine_phase(self, ohlc_data: pd.DataFrame, pattern_type: str) -> str:
        """
        Determina la fase específica del patrón de Wyckoff.
        
        Args:
            ohlc_data: Datos OHLC
            pattern_type: Tipo de patrón identificado
            
        Returns:
            Fase específica del patrón
        """
        # Extraer características de fase
        wyckoff_features = self.feature_extractor.extract_wyckoff_specific_features(ohlc_data)
        
        if pattern_type == 'accumulation':
            # Determinar fase de acumulación
            if wyckoff_features.get('selling_climax_prob', 0) > 0.7:
                return 'selling_climax'
            elif wyckoff_features.get('automatic_rally_prob', 0) > 0.7:
                return 'automatic_rally'
            elif wyckoff_features.get('secondary_test_prob', 0) > 0.7:
                return 'secondary_test'
            elif wyckoff_features.get('spring_prob', 0) > 0.7:
                return 'spring'
            else:
                return 'accumulation_general'
                
        elif pattern_type == 'distribution':
            # Determinar fase de distribución
            if wyckoff_features.get('buying_climax_prob', 0) > 0.7:
                return 'buying_climax'
            elif wyckoff_features.get('automatic_reaction_prob', 0) > 0.7:
                return 'automatic_reaction'
            elif wyckoff_features.get('upthrust_prob', 0) > 0.7:
                return 'upthrust'
            else:
                return 'distribution_general'
                
        elif pattern_type in ['markup', 'markdown']:
            return f'{pattern_type}_trend'
        
        return 'unknown_phase'
    
    def _extract_key_levels(self, ohlc_data: pd.DataFrame) -> Dict[str, float]:
        """
        Extrae niveles clave de soporte y resistencia.
        
        Args:
            ohlc_data: Datos OHLC
            
        Returns:
            Diccionario con niveles clave
        """
        high_prices = ohlc_data['high']
        low_prices = ohlc_data['low']
        close_prices = ohlc_data['close']
        
        # Calcular niveles básicos
        resistance_level = high_prices.max()
        support_level = low_prices.min()
        
        # Niveles de Fibonacci
        price_range = resistance_level - support_level
        fib_levels = {
            'fib_23.6': support_level + (price_range * 0.236),
            'fib_38.2': support_level + (price_range * 0.382),
            'fib_50.0': support_level + (price_range * 0.5),
            'fib_61.8': support_level + (price_range * 0.618),
            'fib_78.6': support_level + (price_range * 0.786)
        }
        
        # Niveles de volumen
        volume_weighted_price = (ohlc_data['close'] * ohlc_data['volume']).sum() / ohlc_data['volume'].sum()
        
        return {
            'resistance': resistance_level,
            'support': support_level,
            'vwap': volume_weighted_price,
            'current_price': close_prices.iloc[-1],
            **fib_levels
        }
    
    def _analyze_volume_profile(self, ohlc_data: pd.DataFrame) -> Dict[str, float]:
        """
        Analiza el perfil de volumen.
        
        Args:
            ohlc_data: Datos OHLC
            
        Returns:
            Diccionario con métricas de volumen
        """
        volume = ohlc_data['volume']
        
        return {
            'avg_volume': volume.mean(),
            'volume_std': volume.std(),
            'volume_trend': self._calculate_volume_trend(volume),
            'high_volume_days': (volume > volume.quantile(0.8)).sum(),
            'low_volume_days': (volume < volume.quantile(0.2)).sum(),
            'volume_concentration': volume.max() / volume.mean()
        }
    
    def _calculate_volume_trend(self, volume: pd.Series) -> float:
        """
        Calcula la tendencia del volumen usando regresión lineal simple.
        
        Args:
            volume: Serie de volúmenes
            
        Returns:
            Pendiente de la tendencia (positiva = creciente, negativa = decreciente)
        """
        x = range(len(volume))
        y = volume.values
        
        # Regresión lineal simple
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x[i] * y[i] for i in range(n))
        sum_x2 = sum(x[i] ** 2 for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x ** 2)
        
        return slope
    
    def _analyze_price_action(self, ohlc_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analiza la acción del precio.
        
        Args:
            ohlc_data: Datos OHLC
            
        Returns:
            Diccionario con métricas de acción del precio
        """
        # Extraer características de precio
        price_features = self.feature_extractor.extract_price_action_features(ohlc_data)
        
        # Análisis adicional
        closes = ohlc_data['close']
        
        return {
            **price_features,
            'price_momentum': self._calculate_momentum(closes),
            'volatility_regime': self._classify_volatility_regime(ohlc_data),
            'trend_strength': self._calculate_trend_strength(closes)
        }
    
    def _calculate_momentum(self, prices: pd.Series, period: int = 10) -> float:
        """
        Calcula el momentum del precio.
        
        Args:
            prices: Serie de precios
            period: Período para el cálculo
            
        Returns:
            Valor de momentum
        """
        if len(prices) < period:
            return 0.0
        
        return (prices.iloc[-1] - prices.iloc[-period]) / prices.iloc[-period]
    
    def _classify_volatility_regime(self, ohlc_data: pd.DataFrame) -> str:
        """
        Clasifica el régimen de volatilidad.
        
        Args:
            ohlc_data: Datos OHLC
            
        Returns:
            Clasificación de volatilidad ('low', 'medium', 'high')
        """
        # Calcular volatilidad realizada
        returns = ohlc_data['close'].pct_change().dropna()
        volatility = returns.std() * (252 ** 0.5)  # Anualizada
        
        if volatility < 0.15:
            return 'low'
        elif volatility < 0.30:
            return 'medium'
        else:
            return 'high'
    
    def _calculate_trend_strength(self, prices: pd.Series, period: int = 20) -> float:
        """
        Calcula la fuerza de la tendencia.
        
        Args:
            prices: Serie de precios
            period: Período para el cálculo
            
        Returns:
            Fuerza de la tendencia (0-1)
        """
        if len(prices) < period:
            return 0.0
        
        # Usar ADX simplificado
        recent_prices = prices.tail(period)
        
        # Calcular direccionalidad
        up_moves = 0
        down_moves = 0
        
        for i in range(1, len(recent_prices)):
            if recent_prices.iloc[i] > recent_prices.iloc[i-1]:
                up_moves += 1
            elif recent_prices.iloc[i] < recent_prices.iloc[i-1]:
                down_moves += 1
        
        total_moves = up_moves + down_moves
        if total_moves == 0:
            return 0.0
        
        directional_strength = abs(up_moves - down_moves) / total_moves
        
        return directional_strength
    
    def train_analyzer(self, training_data: List[Tuple[pd.DataFrame, str]], 
                      validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Entrena el analizador con datos etiquetados.
        
        Args:
            training_data: Lista de (ohlc_data, pattern_label)
            validation_split: Proporción de datos para validación
            
        Returns:
            Métricas de entrenamiento
        """
        self.logger.info("Iniciando entrenamiento del analizador")
        
        # Entrenar el clasificador
        metrics = self.classifier.train(training_data, validation_split)
        
        self.logger.info("Entrenamiento del analizador completado")
        
        return metrics
    
    def save_analyzer(self, filepath: str):
        """Guarda el analizador entrenado."""
        self.classifier.save_model(filepath)
        self.logger.info(f"Analizador guardado en {filepath}")
    
    def load_analyzer(self, filepath: str):
        """Carga un analizador previamente entrenado."""
        self.classifier.load_model(filepath)
        self.logger.info(f"Analizador cargado desde {filepath}")