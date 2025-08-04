from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

class BaseAssistant(ABC):
    """Base class for all assistant services"""
    
    @abstractmethod
    def generate_response(self, query: str) -> str:
        """Generate response to user query"""
        pass
    
    @abstractmethod
    def stop_response(self):
        """Stop any ongoing response generation"""
        pass
        
    @property
    @abstractmethod
    def capabilities(self) -> Dict[str, Any]:
        """Get capabilities of this assistant"""
        pass