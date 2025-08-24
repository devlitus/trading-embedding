from dataclasses import dataclass
from typing import Dict, Any
from datetime import datetime

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