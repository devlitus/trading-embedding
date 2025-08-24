"""Sistema de embeddings para datos de trading.

Este módulo proporciona una interfaz unificada para trabajar con embeddings
de datos de trading, incluyendo encoders especializados y utilidades.
"""

# Importar desde la nueva estructura modular
from .encoders import TemporalEncoder, TechnicalEncoder, PatternEncoder
from .core.embeddings_manager import TradingEmbeddings
from .utils.embedding_utils import EmbeddingUtils

# Mantener compatibilidad hacia atrás
__all__ = [
    'TemporalEncoder',
    'TechnicalEncoder', 
    'PatternEncoder',
    'TradingEmbeddings',
    'EmbeddingUtils'
]