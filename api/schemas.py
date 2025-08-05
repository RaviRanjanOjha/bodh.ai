from pydantic import BaseModel
from typing import Optional, List, Dict, Any
from datetime import datetime

class HealthCheckResponse(BaseModel):
    status: str
    timestamp: str

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None

class ChatResponse(BaseModel):
    response: str
    session_id: Optional[str]
    timestamp: str

class DocumentUploadResponse(BaseModel):
    status: str
    documents: List[Dict[str, str]]
    processed_at: str

class ClientListResponse(BaseModel):
    clients: List[Dict[str, Any]]
    count: int
    retrieved_at: str

class ClientDetailsResponse(BaseModel):
    client_id: str
    details: Dict[str, Any]
    retrieved_at: str

class VisualizationResponse(BaseModel):
    client: str
    visualizations: Dict[str, Any]

class MarketDataResponse(BaseModel):
    dashboard: Dict[str, Any]
    as_of: str

class ComplianceCheckRequest(BaseModel):
    message: str
    client_id: Optional[str] = None

class ComplianceCheckResponse(BaseModel):
    is_compliant: bool
    reasons: Optional[List[str]] = None
    recommendations: Optional[List[str]] = None
    checked_at: str
class ConversationSearchRequest(BaseModel):
    query: str
    limit: int = 10

class VoiceInputRequest(BaseModel):
    audio_data: str  # Base64 encoded audio
    language: str = "en-US"

class UserProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None
# Add to api/schemas.py
class FavoriteResponse(BaseModel):
    session_id: str
    is_favorite: bool
    fav_position: Optional[int]
    status: str

class FavoriteOrderRequest(BaseModel):
    new_order: List[str]
# Add this to api/schemas.py
class FollowUpQuestionsResponse(BaseModel):
    questions: List[str]
    session_id: str
    timestamp: str