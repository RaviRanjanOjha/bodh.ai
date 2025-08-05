# API package initialization
from .routes.chat import router as chat_router
from .routes.clients import router as clients_router
from .routes.documents import router as documents_router
from .routes.compliance import router as compliance_router

__all__ = [
    'chat_router',
    'clients_router',
    'documents_router',
    'compliance_router',
]