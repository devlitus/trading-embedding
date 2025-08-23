import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import joblib
import logging
from datetime import datetime
from dataclasses import dataclass

@dataclass
class WyckoffPattern:
    """
    Representa un patrón de Wyckoff identificado.
    """
    pattern_type: str  # 'accumulation', 'distribution', 'markup', 'markdown'
    phase: str  # Fase específica del patrón
    confidence: float  # Confianza en la identificación
    start_time: datetime
    end_time: datetime
    key_levels: Dict[str, float]  # Niveles importantes (soporte, resistencia, etc.)
    volume_profile: Dict[str, float]  # Perfil de volumen
    price_action: Dict[str, Any]  # Características de la acción del precio

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

class WyckoffClassifier:
    """
    Clasificador de patrones de Wyckoff usando machine learning.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Inicializa el clasificador.
        
        Args:
            model_type: Tipo de modelo ('random_forest', 'neural_network')
        """
        self.model_type = model_type
        self.model = None
        self.feature_extractor = WyckoffFeatureExtractor()
        self.logger = logging.getLogger(__name__)
        
        # Mapeo de patrones a índices
        self.pattern_mapping = {
            'accumulation': 0,
            'distribution': 1,
            'markup': 2,
            'markdown': 3,
            'no_pattern': 4
        }
        
        self.reverse_pattern_mapping = {v: k for k, v in self.pattern_mapping.items()}
        
        self._initialize_model()
    
    def _initialize_model(self):
        """Inicializa el modelo según el tipo especificado."""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        elif self.model_type == 'neural_network':
            self.model = self._create_neural_network()
        else:
            raise ValueError(f"Tipo de modelo no soportado: {self.model_type}")
    
    def _create_neural_network(self) -> nn.Module:
        """Crea una red neuronal para clasificación de patrones."""
        class WyckoffNN(nn.Module):
            def __init__(self, input_dim: int, num_classes: int = 5):
                super(WyckoffNN, self).__init__()
                
                self.classifier = nn.Sequential(
                    nn.Linear(input_dim, 128),
                    nn.ReLU(),
                    nn.Dropout(0.3),
                    nn.Linear(128, 64),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, num_classes)
                )
            
            def forward(self, x):
                return self.classifier(x)
        
        return WyckoffNN(input_dim=50)  # Se ajustará según las características
    
    def extract_features(self, ohlc_data: pd.DataFrame) -> np.ndarray:
        """
        Extrae todas las características relevantes para clasificación.
        
        Args:
            ohlc_data: DataFrame con datos OHLC
            
        Returns:
            Array de características
        """
        # Extraer diferentes tipos de características
        price_features = self.feature_extractor.extract_price_action_features(ohlc_data)
        volume_features = self.feature_extractor.extract_volume_features(ohlc_data)
        wyckoff_features = self.feature_extractor.extract_wyckoff_specific_features(ohlc_data)
        
        # Combinar todas las características
        all_features = {**price_features, **volume_features, **wyckoff_features}
        
        # Convertir a array
        feature_vector = np.array(list(all_features.values()))
        
        return feature_vector
    
    def train(self, training_data: List[Tuple[pd.DataFrame, str]], 
              validation_split: float = 0.2) -> Dict[str, Any]:
        """
        Entrena el clasificador con datos etiquetados.
        
        Args:
            training_data: Lista de (ohlc_data, pattern_label)
            validation_split: Proporción de datos para validación
            
        Returns:
            Métricas de entrenamiento
        """
        self.logger.info(f"Iniciando entrenamiento con {len(training_data)} muestras")
        
        # Extraer características y etiquetas
        X = []
        y = []
        
        for ohlc_data, pattern_label in training_data:
            try:
                features = self.extract_features(ohlc_data)
                X.append(features)
                y.append(self.pattern_mapping.get(pattern_label, 4))  # 4 = no_pattern
            except Exception as e:
                self.logger.warning(f"Error extrayendo características: {e}")
                continue
        
        X = np.array(X)
        y = np.array(y)
        
        self.logger.info(f"Características extraídas: {X.shape}")
        
        # Dividir datos
        split_idx = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:split_idx], X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]
        
        # Entrenar modelo
        if self.model_type == 'random_forest':
            self.model.fit(X_train, y_train)
            
            # Validación
            y_pred = self.model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(self.model, X_train, y_train, cv=5)
            
            metrics = {
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_val, y_pred),
                'confusion_matrix': confusion_matrix(y_val, y_pred).tolist()
            }
            
        elif self.model_type == 'neural_network':
            # Implementar entrenamiento de red neuronal
            metrics = self._train_neural_network(X_train, y_train, X_val, y_val)
        
        self.logger.info(f"Entrenamiento completado. Accuracy: {metrics['accuracy']:.3f}")
        
        return metrics
    
    def _train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, Any]:
        """Entrena la red neuronal."""
        # Convertir a tensores
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train)
        X_val_tensor = torch.FloatTensor(X_val)
        y_val_tensor = torch.LongTensor(y_val)
        
        # Configurar entrenamiento
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Entrenar
        epochs = 100
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            outputs = self.model(X_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                self.logger.info(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Validación
        self.model.eval()
        with torch.no_grad():
            val_outputs = self.model(X_val_tensor)
            _, y_pred = torch.max(val_outputs, 1)
            accuracy = (y_pred == y_val_tensor).float().mean().item()
        
        return {
            'accuracy': accuracy,
            'final_loss': loss.item()
        }
    
    def predict(self, ohlc_data: pd.DataFrame) -> Tuple[str, float]:
        """
        Predice el patrón de Wyckoff para los datos dados.
        
        Args:
            ohlc_data: DataFrame con datos OHLC
            
        Returns:
            Tupla de (patrón_predicho, confianza)
        """
        if self.model is None:
            raise ValueError("Modelo no entrenado")
        
        # Extraer características
        features = self.extract_features(ohlc_data).reshape(1, -1)
        
        if self.model_type == 'random_forest':
            # Predicción
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            confidence = probabilities.max()
            
        elif self.model_type == 'neural_network':
            self.model.eval()
            with torch.no_grad():
                features_tensor = torch.FloatTensor(features)
                outputs = self.model(features_tensor)
                probabilities = F.softmax(outputs, dim=1)[0]
                prediction = torch.argmax(probabilities).item()
                confidence = probabilities.max().item()
        
        pattern_name = self.reverse_pattern_mapping[prediction]
        
        return pattern_name, confidence
    
    def save_model(self, filepath: str):
        """Guarda el modelo entrenado."""
        if self.model_type == 'random_forest':
            joblib.dump(self.model, filepath)
        elif self.model_type == 'neural_network':
            torch.save(self.model.state_dict(), filepath)
        
        self.logger.info(f"Modelo guardado en {filepath}")
    
    def load_model(self, filepath: str):
        """Carga un modelo previamente entrenado."""
        if self.model_type == 'random_forest':
            self.model = joblib.load(filepath)
        elif self.model_type == 'neural_network':
            self.model.load_state_dict(torch.load(filepath))
            self.model.eval()
        
        self.logger.info(f"Modelo cargado desde {filepath}")

class WyckoffAnalyzer:
    """
    Analizador completo de patrones de Wyckoff.
    Combina extracción de características, clasificación y análisis.
    """
    
    def __init__(self, model_type: str = 'random_forest'):
        """
        Inicializa el analizador.
        
        Args:
            model_type: Tipo de modelo para clasificación
        """
        self.classifier = WyckoffClassifier(model_type)
        self.feature_extractor = WyckoffFeatureExtractor()
        self.logger = logging.getLogger(__name__)
    
    def analyze_pattern(self, ohlc_data: pd.DataFrame) -> WyckoffPattern:
        """
        Analiza completamente un patrón de Wyckoff.
        
        Args:
            ohlc_data: DataFrame con datos OHLC
            
        Returns:
            Objeto WyckoffPattern con el análisis completo
        """
        # Predicción del patrón
        pattern_type, confidence = self.classifier.predict(ohlc_data)
        
        # Extraer características detalladas
        price_features = self.feature_extractor.extract_price_action_features(ohlc_data)
        volume_features = self.feature_extractor.extract_volume_features(ohlc_data)
        wyckoff_features = self.feature_extractor.extract_wyckoff_specific_features(ohlc_data)
        
        # Identificar niveles clave
        key_levels = {
            'support': ohlc_data['low'].min(),
            'resistance': ohlc_data['high'].max(),
            'current_price': ohlc_data['close'].iloc[-1]
        }
        
        # Crear objeto de patrón
        pattern = WyckoffPattern(
            pattern_type=pattern_type,
            phase=self._determine_phase(pattern_type, wyckoff_features),
            confidence=confidence,
            start_time=ohlc_data.index[0] if hasattr(ohlc_data.index, 'to_pydatetime') else datetime.now(),
            end_time=ohlc_data.index[-1] if hasattr(ohlc_data.index, 'to_pydatetime') else datetime.now(),
            key_levels=key_levels,
            volume_profile=volume_features,
            price_action=price_features
        )
        
        return pattern
    
    def _determine_phase(self, pattern_type: str, wyckoff_features: Dict[str, float]) -> str:
        """
        Determina la fase específica del patrón basado en las características.
        
        Args:
            pattern_type: Tipo de patrón identificado
            wyckoff_features: Características específicas de Wyckoff
            
        Returns:
            Fase específica del patrón
        """
        if pattern_type == 'accumulation':
            if wyckoff_features.get('selling_climax', 0) > 0.7:
                return 'selling_climax'
            elif wyckoff_features.get('preliminary_support', 0) > 0.6:
                return 'preliminary_support'
            elif wyckoff_features.get('secondary_test', 0) > 0.6:
                return 'secondary_test'
            elif wyckoff_features.get('spring_probability', 0) > 0.6:
                return 'spring'
            else:
                return 'accumulation_phase'
        
        elif pattern_type == 'distribution':
            if wyckoff_features.get('upthrust_probability', 0) > 0.6:
                return 'upthrust'
            else:
                return 'distribution_phase'
        
        elif pattern_type == 'markup':
            return 'markup_phase'
        
        elif pattern_type == 'markdown':
            return 'markdown_phase'
        
        else:
            return 'undefined'
    
    def train_analyzer(self, training_data: List[Tuple[pd.DataFrame, str]]) -> Dict[str, Any]:
        """
        Entrena el analizador con datos etiquetados.
        
        Args:
            training_data: Lista de (ohlc_data, pattern_label)
            
        Returns:
            Métricas de entrenamiento
        """
        return self.classifier.train(training_data)
    
    def save_analyzer(self, filepath: str):
        """Guarda el analizador entrenado."""
        self.classifier.save_model(filepath)
    
    def load_analyzer(self, filepath: str):
        """Carga un analizador previamente entrenado."""
        self.classifier.load_model(filepath)