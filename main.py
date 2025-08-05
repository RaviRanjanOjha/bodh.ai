import os

os.environ["POSTHOG_DISABLED"] = "true"
try:
    import posthog

    posthog.capture = lambda *args, **kwargs: None
    posthog.identify = lambda *args, **kwargs: None
    posthog.flush = lambda: None
except ImportError:
    pass

from config import settings
from fastapi import FastAPI
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from config import settings
import datetime

from api.routes.chat import router as chat_router
from api.routes.clients import router as clients_router
from api.routes.documents import router as documents_router
from api.routes.compliance import router as compliance_router
from config.logging_config import setup_logging
from fastapi.middleware.cors import CORSMiddleware
from api.routes import faq

import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Initialize logging before anything else
setup_logging()
app = FastAPI(
    title="Wealth Management API",
    description="REST API for Wealth Management AI Assistant",
    version="1.0.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Development mode
        "http://127.0.0.1:5173",  # Development mode
        "http://localhost",  # Production mode (port 80)
        "http://127.0.0.1",  # Production mode (port 80)
        "http://localhost:80",  # Production mode explicit
        "http://127.0.0.1:80",  # Production mode explicit
        "https://bodh-ai.onrender.com",  # Render.com production URL
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Include all routers
app.include_router(chat_router)
app.include_router(clients_router)
app.include_router(documents_router)
app.include_router(compliance_router)
app.include_router(faq.router)


@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now(),
        "api_docs": "/docs",
        "version": "1.0.0",
    }


def run_fastapi():
    uvicorn.run(
        app, host=settings.API_HOST, port=settings.API_PORT, reload=settings.DEBUG_MODE
    )


if __name__ == "__main__":
    run_fastapi()
