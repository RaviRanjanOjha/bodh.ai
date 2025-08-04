import os
from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):

    #Diabling SSL verification temporarily:
    posthog_disabled: bool = True  # or False if you want to default to enabled

    class Config:
        extra = "ignore"  # Optional: lets other unknown fields pass through



    # Application settings
    DEBUG_MODE: bool = os.getenv("DEBUG_MODE", False)
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    
    # Security settings
    CORS_ALLOWED_ORIGINS: List[str] = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-secret-key")
    
    # Database settings
    MONGO_URI: str = os.getenv("MONGO_URI", "mongodb://localhost:27017")
    MONGO_DB_NAME: str = os.getenv("MONGO_DB_NAME", "wealth_assistant")
    
    # LLM settings
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/embedding-001")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gemini-1.5-flash")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.2))
    
    # Document processing
    MAX_FILE_SIZE: int = int(os.getenv("MAX_FILE_SIZE", 10 * 1024 * 1024))  # 10MB
    ALLOWED_FILE_TYPES: List[str] = os.getenv("ALLOWED_FILE_TYPES", "pdf,txt,docx").split(",")
    # MongoDB settings for document processing
    MONGO_DOCS_DB_NAME: str = os.getenv("MONGO_DOCS_DB_NAME", f"{MONGO_DB_NAME}_docs")
    MONGO_DOCS_COLLECTION: str = os.getenv("MONGO_DOCS_COLLECTION", "document_embeddings")
    
    # Document processing
    MAX_DOCUMENT_SIZE: int = int(os.getenv("MAX_DOCUMENT_SIZE", 10 * 1024 * 1024))  # 10MB
    DOCUMENT_CHUNK_SIZE: int = int(os.getenv("DOCUMENT_CHUNK_SIZE", 1000))
    DOCUMENT_CHUNK_OVERLAP: int = int(os.getenv("DOCUMENT_CHUNK_OVERLAP", 200))
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()