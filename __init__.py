# Main package initialization
from .config.settings import settings
from .services.assistant.chat import ChatService
from .visualization.charts import WealthVisualizer

__version__ = "1.0.0"
__all__ = ['settings', 'ChatService', 'WealthVisualizer']