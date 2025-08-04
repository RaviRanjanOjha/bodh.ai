from fastapi import Depends, HTTPException
from services.assistant.chat import ChatService
from services.assistant.documents import DocumentService
from visualization.charts import WealthVisualizer
from config import settings

def get_chat_service():
    return ChatService()

def get_visualizer():
    return WealthVisualizer()

def get_document_service():
    return DocumentService()