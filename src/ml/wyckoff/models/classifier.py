import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
import logging

# Importaciones opcionales para scikit-learn
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    from sklearn.model_selection import cross_val_score
    import joblib
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None
    classification_report = None
    confusion_matrix = None
    accuracy_score = None
    cross_val_score = None
    joblib = None

# Importaciones opcionales para PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    F = None

from ..features.feature_extractor import WyckoffFeatureExtractor

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
            if not SKLEARN_AVAILABLE:
                raise ImportError("scikit-learn no está disponible. Instale sklearn para usar Random Forest.")
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
    
    def _create_neural_network(self):
        """Crea una red neuronal para clasificación de patrones."""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch no está disponible. Instale torch para usar redes neuronales.")
        
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