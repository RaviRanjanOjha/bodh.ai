from fastapi import Depends, HTTPException
from services.assistant.chat import ChatService
from services.assistant.documents import DocumentService
from config import settings

def get_chat_service():
    return ChatService()

def get_document_service():
    return DocumentService()