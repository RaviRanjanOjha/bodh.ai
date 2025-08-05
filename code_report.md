# Code Structure Report

## Directory Structure

```
Backend/
    .dockerignore
    .DS_Store
    .env
    Dockerfile
    main.py
    requirements.txt
    temp_test_portfolio.pdf
    __init__.py
    api/
        .DS_Store
        dependencies.py
        exceptions.py
        schemas.py
        __init__.py
        routes/
            chat.py
            clients.py
            compliance.py
            documents.py
            faq.py
            market.py
            visualizations.py
            __init__.py
            __pycache__/
                chat.cpython-313.pyc
                clients.cpython-313.pyc
                compliance.cpython-313.pyc
                documents.cpython-313.pyc
                faq.cpython-313.pyc
                market.cpython-313.pyc
                visualizations.cpython-313.pyc
                __init__.cpython-313.pyc
        __pycache__/
            exceptions.cpython-313.pyc
            schemas.cpython-313.pyc
            __init__.cpython-313.pyc
    config/
        .DS_Store
        logging_config.py
        settings.py
        __init__.py
        __pycache__/
            logging_config.cpython-313.pyc
            settings.cpython-313.pyc
            __init__.cpython-313.pyc
    database/
        .DS_Store
        crud.py
        models.py
        simulated_db.py
        __init__.py
        __pycache__/
            crud.cpython-313.pyc
            models.cpython-313.pyc
            simulated_db.cpython-313.pyc
            __init__.cpython-313.pyc
    logs/
        wealth_assistant_2025-07-10_12-01-05.log
    services/
        .DS_Store
        __init__.py
        assistant/
            base.py
            chat.py
            compliance.py
            documents.py
            visualization.py
            __init__.py
            __pycache__/
                base.cpython-313.pyc
                chat.cpython-313.pyc
                compliance.cpython-313.pyc
                documents.cpython-313.pyc
                visualization.cpython-313.pyc
                __init__.cpython-313.pyc
        llm/
            base.py
            document_embeddings.py
            __init__.py
            __pycache__/
                base.cpython-313.pyc
                chains.cpython-313.pyc
                document_embeddings.cpython-313.pyc
                prompts.cpython-313.pyc
                __init__.cpython-313.pyc
        storage/
            conversation.py
            faq_store.py
            __init__.py
            __pycache__/
                conversation.cpython-313.pyc
                faq_store.cpython-313.pyc
                __init__.cpython-313.pyc
        utils/
            __pycache__/
        __pycache__/
            __init__.cpython-313.pyc
    temp_uploads/
    visualization/
        .DS_Store
        charts.py
        dashboards.py
        utils.py
        __init__.py
        __pycache__/
            charts.cpython-313.pyc
            dashboards.cpython-313.pyc
            utils.cpython-313.pyc
            __init__.cpython-313.pyc
    __pycache__/
        main.cpython-313.pyc
```

## File: `__init__.py`

```python
# Main package initialization
from .config.settings import settings
from .services.assistant.chat import ChatService
from .visualization.charts import WealthVisualizer

__version__ = "1.0.0"
__all__ = ['settings', 'ChatService', 'WealthVisualizer']
```

---

## File: `main.py`

```python
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
from api.routes.market import router as market_router
from api.routes.visualizations import router as visualizations_router
from config.logging_config import setup_logging
from fastapi.middleware.cors import CORSMiddleware
from api.routes import faq

import ssl
ssl._create_default_https_context=ssl._create_unverified_context

# Initialize logging before anything else
setup_logging()
app = FastAPI(
    title="Wealth Management API",
    description="REST API for Wealth Management AI Assistant",
    version="1.0.0",
    openapi_url="/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Development mode
        "http://127.0.0.1:5173",  # Development mode  
        "http://localhost",       # Production mode (port 80)
        "http://127.0.0.1",       # Production mode (port 80)
        "http://localhost:80",    # Production mode explicit
        "http://127.0.0.1:80"     # Production mode explicit
    ],
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
    expose_headers=["*"]  
)

# Include all routers
app.include_router(chat_router)
app.include_router(clients_router)
app.include_router(documents_router)
app.include_router(compliance_router)
app.include_router(market_router)
app.include_router(visualizations_router)
app.include_router(faq.router) 

@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.datetime.now(),
        "api_docs": "/docs",
        "version": "1.0.0"
    }

def run_fastapi():
    uvicorn.run(
        app,
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.DEBUG_MODE
    )

if __name__ == "__main__":
    run_fastapi()
```

---

## File: `api\__init__.py`

```python
# API package initialization
from .routes.chat import router as chat_router
from .routes.clients import router as clients_router
from .routes.documents import router as documents_router
from .routes.compliance import router as compliance_router
from .routes.market import router as market_router
from .routes.visualizations import router as visualizations_router

__all__ = [
    'chat_router',
    'clients_router',
    'documents_router',
    'compliance_router',
    'market_router',
    'visualizations_router'
]
```

---

## File: `api\dependencies.py`

```python
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
```

---

## File: `api\exceptions.py`

```python
from fastapi import HTTPException, status

class ClientNotFoundException(HTTPException):
    def __init__(self, client_id: str):
        super().__init__(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Client with ID {client_id} not found"
        )

class DocumentProcessingException(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=detail
        )

class ComplianceException(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Compliance violation: {detail}"
        )

class MarketDataException(HTTPException):
    def __init__(self, detail: str):
        super().__init__(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Market data error: {detail}"
        )
```

---

## File: `api\schemas.py`

```python
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
```

---

## File: `api\routes\__init__.py`

```python
# Routes package initialization
__all__ = [
    'chat_router',
    'clients_router',
    'documents_router', 
    'compliance_router',
    'market_router',
    'visualizations_router'
]
```

---

## File: `api\routes\chat.py`

```python
from fastapi import APIRouter, Cookie, HTTPException, Response,UploadFile
from typing import Optional, List, Dict, Any, Tuple
import json
import datetime
import uuid
from fastapi.responses import StreamingResponse
from services.assistant.chat import ChatService
from services.storage.conversation import ConversationModel
from services.llm.base import LLMService
from api.schemas import (
    ConversationSearchRequest, 
    VoiceInputRequest,
    UserProfileUpdateRequest,
    ChatRequest, ChatResponse, FollowUpQuestionsResponse
)
# Add to imports
from services.storage.conversation import ConversationModel
from api.schemas import FavoriteResponse, FavoriteOrderRequest
from api.exceptions import ComplianceException
import speech_recognition as sr
import base64
import tempfile
import logging
import traceback


logger = logging.getLogger(__name__)

router = APIRouter(prefix="/chat", tags=["Chat"])

@router.post("/search", response_model=List[Dict[str, Any]])
async def search_conversations(request: ConversationSearchRequest):
    """Search conversations by content"""
    try:
        chat_service = ChatService()
        results = chat_service.search_conversations(
            query=request.query,
            limit=request.limit
        )
        
        return [
            {
                "session_id": conv.session_id,
                "summary": conv.summary or "Chat Discussion",
                "timestamp": conv.updated_at.isoformat(),
                "matching_message": msg["content"][:100],
                "message_timestamp": msg.get("timestamp", conv.updated_at.isoformat())
            }
            for conv, msg in results
        ]
    except Exception as e:
        logger.error(f"Error searching conversations: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

@router.post("/voice-input")
async def process_voice_input(request: VoiceInputRequest):
    """Convert voice input to text"""
    try:
        # Create temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav") as temp_audio:
            # Write base64 audio data to file
            audio_data = base64.b64decode(request.audio_data)
            temp_audio.write(audio_data)
            temp_audio.flush()
            
            # Convert to text
            r = sr.Recognizer()
            with sr.AudioFile(temp_audio.name) as source:
                audio = r.record(source)
                text = r.recognize_google(audio, language=request.language)
                
        return {
            "text": text,
            "language": request.language,
            "status": "success"
        }
    except sr.UnknownValueError:
        raise HTTPException(
            status_code=400, 
            detail="Could not understand audio"
        )
    except sr.RequestError as e:
        raise HTTPException(
            status_code=503,
            detail=f"Speech recognition service error: {e}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/profile")
async def update_user_profile(
    request: UserProfileUpdateRequest,
    session_id: str = Cookie(None)
):
    """Update user profile/preferences"""
    try:
        chat_service = ChatService()
        
        # Get current profile
        profile = chat_service.get_user_profile(session_id)
        
        # Update fields that were provided
        updated_profile = {
            **profile,
            **request.dict(exclude_unset=True)
        }
        
        # Save updated profile
        success = chat_service.update_user_profile(
            session_id, 
            updated_profile
        )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Failed to update profile"
            )
            
        return {"status": "success", "profile": updated_profile}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest, session_id: str = Cookie(None)):
    try:
        logger.info(f"Chat endpoint called with message: {request.message[:50]}...")
        logger.info(f"Request session_id: {request.session_id}")
        logger.info(f"Cookie session_id: {session_id}")
        
        chat_service = ChatService()
        if request.session_id and session_id and request.session_id != session_id:
            logger.warning(f"Session ID mismatch - cleaning up old cookie. Request: {request.session_id}, Cookie: {session_id}")
            response_obj = Response(content="", status_code=200)
            response_obj.delete_cookie("session_id")
            session_id = None  # Clear the local variable too
        target_session_id = request.session_id or session_id or str(uuid.uuid4())
        logger.info(f"Target session_id: {target_session_id}")
        
        # Get existing conversation if it exists - use target_session_id instead of session_id
        existing_convo = chat_service.load_conversation(target_session_id) if target_session_id else None
        existing_messages = existing_convo.messages if existing_convo else []
        logger.info(f"Existing conversation found: {existing_convo is not None}")
        logger.info(f"Existing messages count: {len(existing_messages)}")
        
        # Get response and compliance status
        logger.info("Generating response...")
        response, is_compliant = chat_service.generate_response(request.message, target_session_id)
        logger.info(f"Response generated. Is compliant: {is_compliant}")
        logger.info(f"Response length: {len(response) if response else 0}")
        
        set_cookie = session_id is None
        
        # Only save compliant conversations
        if is_compliant:
            logger.info("Saving conversation to database...")
            new_messages = [
                {
                    "role": "user",
                    "content": request.message,
                    "timestamp": datetime.datetime.now().isoformat()
                },
                {
                    "role": "assistant",
                    "content": response,
                    "timestamp": datetime.datetime.now().isoformat()
                }
            ]
            
            save_result = chat_service.save_conversation(
                session_id=target_session_id,
                messages=new_messages  
            )
            
            if save_result:
                logger.info(f"✅ Conversation saved successfully: {save_result}")
            else:
                logger.error("❌ Failed to save conversation to MongoDB")
        else:
            logger.warning(f"❌ Conversation not saved - not compliant")
        
        resp_data = {
            "response": response,
            "session_id": target_session_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "is_compliant": is_compliant  
        }
        
        response_obj = Response(
            content=json.dumps(resp_data),
            media_type="application/json"
        )
        
        if not session_id:  # Only set cookie if no existing session cookie
                    logger.info(f"Setting session cookie: {target_session_id}")
                    response_obj.set_cookie(
                        key="session_id",
                        value=target_session_id,
                        httponly=True,
                        max_age=30*24*60*60,
                        secure=False,  # For development (HTTP)
                        samesite='Lax'
                    )
        
        logger.info("Chat endpoint completed successfully")
        return response_obj
    except ComplianceException as e:
        raise
    except Exception as e:
        logger.exception("Error in chat endpoint")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stream")
async def chat_stream_endpoint(request: ChatRequest, session_id: str = Cookie(None)):
    """Streaming chat endpoint with proper persistence"""
    try:
        chat_service = ChatService()
        set_cookie = session_id is None
        session_id = session_id or str(uuid.uuid4())
        
        # First save user message immediately
        chat_service.save_conversation(
            session_id=session_id,
            messages=[{
                "role": "user", 
                "content": request.message,
                "timestamp": datetime.datetime.now().isoformat()
            }]
        )
        
        # Generator for streaming response
        async def response_generator():
            full_response = []
            async for chunk, is_compliant in chat_service.generate_response_stream(request.message, session_id):
                if not is_compliant:
                    yield "data: [COMPLIANCE REJECTION]\n\n"
                    break
                    
                chunk_data = json.dumps({
                    "response": chunk,
                    "session_id": session_id,
                    "timestamp": str(datetime.datetime.now()),
                    "is_compliant": is_compliant
                })
                full_response.append(chunk)
                yield f"data: {chunk_data}\n\n"
            
            # Save complete assistant response after streaming finishes
            if full_response:
                chat_service.save_conversation(
                    session_id=session_id,
                    messages=[{
                        "role": "assistant",
                        "content": "".join(full_response),
                        "timestamp": datetime.datetime.now().isoformat()
                    }]
                )

        response_obj = StreamingResponse(
            response_generator(),
            media_type="text/event-stream"
        )
        
        if set_cookie:
            response_obj.set_cookie(
                key="session_id",
                value=session_id,
                httponly=True,
                max_age=30*24*60*60  
            )
            
        return response_obj
        
    except ComplianceException as e:
        raise
    except Exception as e:
        logger.error(f"Error in streaming endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/stop")
async def stop_chat(session_id: str = Cookie(None)):
    try:
        ChatService().stop_response()
        return {
            "status": "success",
            "message": "Response generation stopped",
            "session_id": session_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities")
def chat_capabilities():
    return {
        "streaming_supported": True,
        "endpoints": [
            {"path": "/chat", "type": "standard"},
            {"path": "/chat/stream", "type": "server-sent-events"}
        ]
    }
@router.get("/history", response_model=List[Dict[str, Any]])
async def get_conversation_history(session_id: str = Cookie(None)):
    """Get list of all conversations with summaries"""
    try:
        chat_service = ChatService()
        # This would query all conversations from the database
        conversations = chat_service.conversation_store.get_all_conversations()
        
        return [
            {
                "session_id": conv.session_id,
                "summary": conv.summary or "Chat Discussion",
                "timestamp": conv.updated_at.isoformat(),
                "preview": conv.messages[-1]["content"][:50] if conv.messages else ""
            }
            for conv in conversations
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/history/{session_id}", response_model=Dict[str, Any])
async def get_conversation_details(session_id: str):
    """Get full conversation details by session_id"""
    try:
        chat_service = ChatService()
        conversation = chat_service.load_conversation(session_id)
        
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
            
        return {
            "session_id": conversation.session_id,
            "summary": conversation.summary or "Chat Discussion",
            "messages": conversation.messages,
            "timestamp": conversation.updated_at.isoformat(),
            "client_id": conversation.client_id
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/summary")
async def update_conversation_summary(request: Dict[str, Any]):
    """Update or generate a conversation summary"""
    try:
        chat_service = ChatService()
        session_id = request.get("session_id")
        custom_summary = request.get("summary")
        
        conversation = chat_service.load_conversation(session_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Use custom summary if provided, otherwise generate one
        summary = custom_summary or chat_service._generate_conversation_summary(conversation.messages)
        
        # Update conversation with new summary
        chat_service.conversation_store.update_summary(session_id, summary)
        
        return {"status": "success", "summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/{session_id}")
async def delete_conversation(session_id: str):
    """Delete a conversation by session ID"""
    try:
        chat_service = ChatService()
        
        # Check if conversation exists first
        conversation = chat_service.load_conversation(session_id)
        if not conversation:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        # Delete the conversation
        success = chat_service.conversation_store.delete_conversation(session_id)
        
        if success:
            logger.info(f"Successfully deleted conversation: {session_id}")
            return {
                "status": "success", 
                "message": "Conversation deleted successfully",
                "session_id": session_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete conversation")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting conversation {session_id}: {e}")
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))
# Add these endpoints to chat.py

@router.post("/favorite/{session_id}", response_model=FavoriteResponse)
async def toggle_favorite_conversation(session_id: str):
    """Toggle favorite status for a conversation"""
    try:
        chat_service = ChatService()
        success = chat_service.conversation_store.toggle_favorite(session_id)
        
        if not success:
            raise HTTPException(
                status_code=404,
                detail="Conversation not found or update failed"
            )
            
        # Get updated conversation
        conversation = chat_service.conversation_store.load_conversation(session_id)
        
        return {
            "session_id": session_id,
            "is_favorite": conversation.is_favorite,
            "fav_position": conversation.fav_position,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/favorites", response_model=List[Dict[str, Any]])
async def get_favorite_conversations():
    """Get list of all favorited conversations"""
    try:
        chat_service = ChatService()
        favorites = chat_service.conversation_store.get_favorites()
        
        return [
            {
                "session_id": conv.session_id,
                "summary": conv.summary or "Favorite Conversation",
                "timestamp": conv.updated_at.isoformat(),
                "is_favorite": True,
                "fav_position": conv.fav_position
            }
            for conv in favorites
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/favorites/reorder", response_model=Dict[str, str])
async def reorder_favorites(request: FavoriteOrderRequest):
    """Update the order of favorite conversations"""
    try:
        chat_service = ChatService()
        success = chat_service.conversation_store.reorder_favorites(request.new_order)
        
        if not success:
            raise HTTPException(
                status_code=400,
                detail="Failed to update favorite order"
            )
            
        return {"status": "success", "message": "Favorites reordered"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
@router.get("/followup-questions", response_model=FollowUpQuestionsResponse)
async def get_followup_questions(session_id: str = Cookie(None)):
    """Generate 3 follow-up questions based on the latest 2 messages in the conversation"""
    try:
        logger.info(f"Received request for follow-up questions for session: {session_id}")
        
        if not session_id:
            logger.warning("No session ID provided in request")
            raise HTTPException(status_code=400, detail="No session ID provided")

        chat_service = ChatService()
        
        try:
            conversation = chat_service.load_conversation(session_id)
            logger.info(f"Loaded conversation with {len(conversation.messages) if conversation else 0} messages")
        except Exception as load_error:
            logger.error(f"Error loading conversation: {str(load_error)}", exc_info=True)
            conversation = None
        
        if not conversation or len(conversation.messages) < 2:
            logger.warning("Insufficient messages for generating follow-ups, using defaults")
            default_questions = [
                "Could you tell me more about your investment goals?",
                "What is your current risk tolerance?",
                "Would you like me to analyze any specific part of your portfolio?"
            ]
            return FollowUpQuestionsResponse(
                questions=default_questions,
                session_id=session_id,
                timestamp=datetime.datetime.now().isoformat()
            )
        
        # Get the last 2 messages (user and assistant pair)
        last_messages = conversation.messages[-2:]
        logger.debug(f"Last messages: {last_messages}")
        
        # Build context for prompt
        context = "\n".join([
            f"{msg['role']}: {msg['content']}" 
            for msg in last_messages
        ])
        
        logger.info("Generating follow-up questions using LLM...")
        
        # Generate questions using LLM
        prompt = f"""
        Based on this conversation excerpt, generate exactly 3 professional follow-up questions 
        that a wealth advisor would ask to continue the discussion. The questions should be:
        - Relevant to wealth management
        - Natural follow-ups to the discussion
        - Open-ended to encourage detailed answers
        - Formatted as a JSON list of strings like ["question1", "question2", "question3"]

        Return ONLY the JSON array, nothing else.

        Conversation:
        {context}

        Example response:
        ["What is your investment time horizon?", "Are you considering diversification?", "How can I help with your retirement planning?"]
        """
        
        llm_service = LLMService()
        response = ""
        questions = []
        
        try:
            response = llm_service.generate_response(prompt)
            logger.debug(f"LLM raw response: {response}")
            
            # Clean the response to extract just the JSON array
            json_start = response.find('[')
            json_end = response.rfind(']') + 1
            if json_start != -1 and json_end != -1:
                response = response[json_start:json_end]
            
            questions = json.loads(response)
            
            # Validate we got exactly 3 questions
            if not isinstance(questions, list) or len(questions) != 3:
                logger.warning(f"Invalid question count in response, got {len(questions) if isinstance(questions, list) else 0}")
                raise ValueError("Invalid question format")
                
            logger.info(f"Generated questions: {questions}")
            
        except Exception as parse_error:
            logger.error(f"Error parsing LLM response: {str(parse_error)}", exc_info=True)
            logger.debug(f"Problematic response: {response}")
            
            # Fallback questions
            questions = [
                "Could you elaborate on your investment strategy?",
                "Are you considering any changes to your portfolio?",
                "How can I help you with your financial planning?"
            ]
        
        return FollowUpQuestionsResponse(
            questions=questions,
            session_id=session_id,
            timestamp=datetime.datetime.now().isoformat()
        )
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions
    except Exception as e:
        logger.error(f"Unexpected error generating follow-up questions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to generate follow-up questions. Please try again later."
        )
```

---

## File: `api\routes\clients.py`

```python
from fastapi import APIRouter, HTTPException, Body
from typing import List
from datetime import datetime

from api.schemas import ClientListResponse, ClientDetailsResponse
from database.crud import get_client_list, get_client_details, wealth_db
from api.exceptions import ClientNotFoundException
from config import settings

router = APIRouter(prefix="/clients", tags=["Clients"])

@router.get("", response_model=ClientListResponse)
def get_all_clients():
    try:
        clients = get_client_list()
        return {
            "clients": clients,
            "count": len(clients),
            "retrieved_at": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/{client_id}", response_model=ClientDetailsResponse)
def get_client(client_id: str):
    try:
        client_data = get_client_details(client_id)
        if not client_data:
            raise ClientNotFoundException(client_id)
        
        return {
            "client_id": client_id,
            "details": client_data,
            "retrieved_at": datetime.now().isoformat()
        }
    except ClientNotFoundException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/simulated-db")
def get_client_context_from_prompt(payload: dict = Body(...)):
    prompt = payload.get("prompt", "").lower()

    for client_id, client_data in wealth_db["clients"].items():
        name = client_data.get("name", "").lower()
        if name and name in prompt:
            return get_client_details(client_id)

    return {}  

```

---

## File: `api\routes\compliance.py`

```python
from fastapi import APIRouter, HTTPException
from datetime import datetime
from api.schemas import ComplianceCheckRequest, ComplianceCheckResponse
from services.assistant.compliance import ComplianceChecker
from services.llm.base import LLMService
from api.exceptions import ComplianceException

router = APIRouter(prefix="/compliance", tags=["Compliance"])

@router.post("/check", response_model=ComplianceCheckResponse)
def check_compliance(request: ComplianceCheckRequest):
    try:
        llm_service = LLMService()  # Create LLM instance
        checker = ComplianceChecker(llm=llm_service)  # Pass it to checker
        result = checker.check_message(request.message, request.client_id)
        return {
            "is_compliant": result["is_compliant"],
            "reasons": result.get("reasons"),
            "recommendations": result.get("recommendations"),
            "checked_at": datetime.now().isoformat()
        }
    except ComplianceException as e:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## File: `api\routes\documents.py`

```python
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from typing import List
from services.assistant.documents import DocumentService
from pydantic import BaseModel
import logging
import datetime
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/documents", tags=["Document Management"])



@router.post("/upload")
async def upload_files(files: List[UploadFile] = File(...)):
    try:
        document_service = DocumentService()
        
        # Process files using the correct method name
        status, doc_list = await document_service.process_uploaded_files(files)
        print(document_service.debug_vector_store())
        formatted_documents = [
            {"file_name": name, "status": status} 
            for name, status in doc_list
        ]
        
        return {
            "status": status,
            "documents": formatted_documents,
            "processed_at": datetime.datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Upload failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Upload failed: {str(e)}"
        )
```

---

## File: `api\routes\faq.py`

```python
from fastapi import APIRouter, HTTPException
from services.llm.base import LLMService
from config import settings
from pymongo import MongoClient
from collections import defaultdict
import re
import logging
import json
from typing import List, Dict

router = APIRouter(tags=["FAQ"])
logger = logging.getLogger(__name__)
client = MongoClient(settings.MONGO_URI)
db = client[settings.MONGO_DB_NAME]
conversations_collection = db["conversations"]

class FAQGenerator:
    def __init__(self):
        self.llm = LLMService()
        self.min_occurrences = 2
        self.max_faqs = 6
        self.stop_words = {
            'what', 'how', 'can', 'you', 'me', 'the', 'a', 'an', 
            'is', 'are', 'do', 'does', 'could', 'would', 'please'
        }

    def normalize_question(self, text: str) -> str:
        """Improved text normalization"""
        if not text:
            return ""
        text = re.sub(r"[^\w\s]", "", text.lower())
        text = re.sub(r"\s+", " ", text).strip()
        words = [w for w in text.split() if w not in self.stop_words]
        return " ".join(words)

    def group_similar_questions(self, questions: List[str]) -> Dict[str, List[str]]:
        """Group similar questions using LLM with better error handling"""
        if not questions or len(questions) < 2:
            return {q: [q] for q in questions}
            
        prompt = f"""
        Analyze these wealth management questions and group similar ones together:
        {questions[:50]}
        
        Return ONLY a valid JSON dictionary where:
        - Key is the most representative question from each group
        - Value is list of all similar questions in that group
        
        Example output format:
        {{
            "What is my portfolio performance?": [
                "how is my portfolio doing",
                "show me my portfolio performance"
            ],
            "What are my investment options?": [
                "what can I invest in",
                "show investment choices"
            ]
        }}
        """
        
        try:
            response = self.llm.generate_response(prompt)
            if not response:
                raise ValueError("Empty LLM response")
                
            response = response.strip()
            if not response.startswith("{"):
                response = "{" + response.split("{", 1)[-1]
            if not response.endswith("}"):
                response = response.split("}", 1)[0] + "}"
                
            return json.loads(response)
        except Exception as e:
            logger.warning(f"LLM grouping failed, using fallback method: {str(e)}")
            groups = defaultdict(list)
            for q in questions:
                key = " ".join(q.split()[:3]).lower()
                groups[key].append(q)
            return groups

    def get_representative_question(self, group: List[str]) -> str:
        """Select the best phrased question from a group"""
        if not group:
            return ""
        for q in sorted(group, key=lambda x: -len(x), reverse=True):
            if len(q.split()) >= 4 and q.strip().endswith('?'):
                return q
        return group[0]

@router.get("/faq", summary="Get frequently asked questions")
async def get_faqs():
    """Get intelligent FAQs grouped by semantic similarity"""
    try:
        generator = FAQGenerator()
        question_counter = defaultdict(int)
        raw_questions = []

        cursor = conversations_collection.find(
            {"messages.role": "user"},
            {"messages": {"$elemMatch": {"role": "user"}}}
        ).limit(1000)
        
        for convo in cursor:
            for msg in convo.get("messages", []):
                question = msg.get("content", "").strip()
                if question and len(question.split()) >= 3:
                    raw_questions.append(question)
                    norm_question = generator.normalize_question(question)
                    question_counter[norm_question] += 1

        frequent_questions = [
            q for q in question_counter 
            if question_counter[q] >= generator.min_occurrences
        ]

        if not frequent_questions:
            return {
                "faqs": [
                    "Portfolio performance influencing factors?",
                    "Best options for low-risk investing?",
                    "What is Priya risk analysis?",
                    "Tell me any client name stored in you?"
                ],
                "message": "Insufficient question data",
                "stats": {
                    "total_questions_analyzed": len(raw_questions),
                    "unique_questions": len(question_counter),
                    "frequent_question_groups": 0
                }
            }

        question_groups = generator.group_similar_questions(frequent_questions)

        faqs = []
        for representative, group in question_groups.items():
            total_occurrences = sum(question_counter.get(q, 0) for q in group)
            if total_occurrences >= generator.min_occurrences:
                candidate_questions = [
                    q for q in raw_questions 
                    if generator.normalize_question(q) in group
                ]
                best_question = generator.get_representative_question(candidate_questions)
                if best_question:
                    faqs.append({
                        "question": best_question,
                        "occurrences": total_occurrences,
                        "variations": len(group)
                    })

        faqs.sort(key=lambda x: (-x["occurrences"], -x["variations"]))
        top_faqs = [f["question"] for f in faqs[:generator.max_faqs]]

        default_faqs = [
        "Portfolio performance influencing factors?",
        "Best options for low-risk investing?",
        "What is Priya risk analysis?",
        "Tell me any client name stored in you?"
        ]


        if len(top_faqs) < 4:
            needed = 4 - len(top_faqs)
            top_faqs.extend(default_faqs[:needed])

        return {
            "faqs": top_faqs,
            "stats": {
                "total_questions_analyzed": len(raw_questions),
                "unique_questions": len(question_counter),
                "frequent_question_groups": len(faqs)
            }
        }

    except Exception as e:
        logger.error(f"FAQ generation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500, 
            detail="Could not generate FAQs. Please try again later."
        )

```

---

## File: `api\routes\market.py`

```python
from fastapi import APIRouter, HTTPException
from datetime import datetime

from api.schemas import MarketDataResponse
from visualization.dashboards import create_market_comparison_dashboard as create_market_dashboard
from api.exceptions import MarketDataException
from visualization.utils import plot_to_json  

router = APIRouter(prefix="/market", tags=["Market"])

@router.get("", response_model=MarketDataResponse)
def get_market_data():
    try:
        dashboard = create_market_dashboard()
        dashboard_data=plot_to_json(dashboard)
        return {
            "dashboard": dashboard_data,
            "as_of": datetime.now().isoformat()
        }
    except MarketDataException as e:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## File: `api\routes\visualizations.py`

```python
from fastapi import APIRouter, HTTPException
from datetime import datetime
from visualization.utils import plot_to_json  # Add this import
import json

from api.schemas import VisualizationResponse
from visualization.charts import WealthVisualizer
from database.crud import get_client_details
from api.exceptions import ClientNotFoundException

router = APIRouter(prefix="/visualizations", tags=["Visualizations"])

@router.get("/client/{client_id}", response_model=VisualizationResponse)
def get_client_visualizations(client_id: str):
    try:
        client_data = get_client_details(client_id)
        if not client_data:
            raise ClientNotFoundException(client_id)
            
        visualizer = WealthVisualizer()
        visualizations = visualizer.get_client_visualizations(client_id)
        
        if not visualizations:
            raise HTTPException(
                status_code=404,
                detail=f"No visualization data found for client {client_id}"
            )
            
        # Convert all figures to JSON
        visualization_data = {
            key: plot_to_json(fig) if fig is not None else None
            for key, fig in visualizations.items()
        }
        
        return {
            "client": client_data["name"],
            "visualizations": visualization_data
        }
        
    except ClientNotFoundException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate visualizations: {str(e)}"
        )
```

---

## File: `config\__init__.py`

```python
from .settings import settings
from .logging_config import setup_logging

__all__ = ['settings', 'setup_logging']
```

---

## File: `config\logging_config.py`

```python
import logging
import logging.config
import datetime
import os
from typing import Dict, Any
from pathlib import Path
from config import settings

def setup_logging() -> None:
    """Setup logging configuration"""
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_file = logs_dir / f"wealth_assistant_{current_time}.log"
    
    logging_config: Dict[str, Any] = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "verbose": {
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "simple": {
                "format": "%(levelname)s %(message)s"
            },
        },
        "handlers": {
            "console": {
                "level": "INFO",
                "class": "logging.StreamHandler",
                "formatter": "simple"
            },
            "file": {
                "level": "DEBUG",
                "class": "logging.FileHandler",
                "filename": log_file,
                "formatter": "verbose"
            },
        },
        "loggers": {
            "": {
                "handlers": ["console", "file"],
                "level": "DEBUG" if settings.DEBUG_MODE else "INFO",
                "propagate": True
            },
            "uvicorn": {
                "handlers": ["console", "file"],
                "level": "INFO",
                "propagate": False
            },
            "uvicorn.error": {
                "level": "INFO",
                "handlers": ["console", "file"],
                "propagate": False,
            },
        }
    }
    
    logging.config.dictConfig(logging_config)
    logging.info("Logging configured successfully")
```

---

## File: `config\settings.py`

```python
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
```

---

## File: `database\__init__.py`

```python
from .simulated_db import create_simulated_database
from .crud import get_client_list, get_client_details
from .models import ClientModel

__all__ = [
    'create_simulated_database',
    'get_client_list',
    'get_client_details',
    'ClientModel'
]
```

---

## File: `database\crud.py`

```python
from typing import List, Dict, Any, Optional
from datetime import datetime
from database.models import ClientModel
from database.simulated_db import create_simulated_database

wealth_db = create_simulated_database()

def get_client_list() -> List[Dict[str, Any]]:
    """Get list of all clients with summary information"""
    clients = []
    
    for client_id, client_data in wealth_db["clients"].items():
        clients.append({
            "client_id": client_id,
            "name": client_data["name"],
            "net_worth": client_data["net_worth"],
            "risk_tolerance": client_data["risk_tolerance"],
            "advisor_count": len(client_data.get("advisors", [])),
            "last_review_date": client_data.get("last_review_date"),
            "portfolio_summary": ", ".join([
                f"{k}: {v}%" for k, v in client_data["portfolio"].items()
            ])
        })
        
    return clients

def get_client_details(client_id: str) -> Optional[Dict[str, Any]]:
    """Get comprehensive details for a specific client"""
    if client_id not in wealth_db["clients"]:
        return None
        
    client_data = wealth_db["clients"][client_id]
    
    formatted_data = {
        "name": client_data["name"],
        "net_worth": f"${client_data['net_worth']:,}",
        "risk_tolerance": client_data["risk_tolerance"],
        "goals": client_data["goals"],
        "portfolio_allocation": client_data["portfolio"],
        "accounts": client_data["accounts"],
        "advisors": client_data["advisors"],
        "last_review_date": client_data["last_review_date"],
        "transactions": client_data.get("transactions", []),
        "sector_allocation": client_data.get("sector_allocation", {})
    }
    
    if client_id in wealth_db["risk_metrics"]:
        formatted_data["risk_metrics"] = wealth_db["risk_metrics"][client_id]
        
    return formatted_data

def get_transactions(client_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent transactions for a client"""
    if client_id not in wealth_db["clients"]:
        return []
    
    transactions = wealth_db["clients"][client_id].get("transactions", [])
    return sorted(transactions, key=lambda x: x["date"], reverse=True)[:limit]
```

---

## File: `database\models.py`

```python
from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any, Optional
from datetime import datetime

class Account(BaseModel):
    id: str
    type: str
    balance: float

class Transaction(BaseModel):
    date: str
    amount: float
    type: str
    security: str

class ClientModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str
    name: str
    net_worth: float
    risk_tolerance: str
    goals: List[str]
    portfolio: Dict[str, float]
    accounts: List[Account]
    advisors: List[str]
    last_review_date: datetime  # Change type hint
    historical_performance: Dict[str, Any]
    sector_allocation: Dict[str, float]
    transactions: Optional[List[Transaction]] = None

    def dict(self, *args, **kwargs):
        data = super().dict(*args, **kwargs)
        data["net_worth_formatted"] = f"${self.net_worth:,.2f}"
        return data
```

---

## File: `database\simulated_db.py`

```python
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional
from datetime import datetime, timedelta
from database.models import ClientModel
from pymongo import MongoClient
from bson import ObjectId
import logging
from config import settings

logger = logging.getLogger(__name__)

class HybridDatabase:
    def __init__(self):
        self.mongo_connected = False
        self.mongo_db = None
        try:
            self.mongo_client = MongoClient(settings.MONGO_URI)
            self.mongo_client.admin.command('ping')
            self.mongo_db = self.mongo_client[settings.MONGO_DB_NAME]
            self.mongo_connected = True
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.warning(f"MongoDB connection failed, using simulated data only: {str(e)}")
            self.mongo_connected = False

    def get_clients(self) -> Dict[str, Any]:
        """Get clients from MongoDB or fallback to simulated data"""
        if self.mongo_connected:
            try:
                clients = {}
                for client in self.mongo_db.clients.find():
                    client_data = client.copy()
                    client_data['id'] = str(client_data.pop('_id'))
                    clients[client_data['id']] = ClientModel(**client_data).dict()
                
                if clients:
                    logger.info("Successfully fetched clients from MongoDB")
                    return {"clients": clients}
                
            except Exception as e:
                logger.error(f"Error fetching from MongoDB: {e}")
        
        return self._create_simulated_data()

    def _create_simulated_data(self) -> Dict[str, Any]:
        """Create comprehensive simulated wealth management database"""
        logger.info("Using simulated database as fallback")
        
        dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='ME')

        clients = {
            "client_001": ClientModel(
                id="client_001",
                name="Ravi Mehta",
                net_worth=4500000,
                risk_tolerance="Moderate",
                goals=["Retirement at 60", "College fund for grandchildren"],
                portfolio={
                    "stocks": 55,
                    "bonds": 30,
                    "alternatives": 10,
                    "cash": 5
                },
                accounts=[
                    {"id": "acct_1", "type": "Brokerage", "balance": 2800000},
                    {"id": "acct_2", "type": "IRA", "balance": 1200000},
                    {"id": "acct_3", "type": "Trust", "balance": 500000}
                ],
                advisors=["Sarah Johnson", "Michael Chen"],
                last_review_date=datetime.now() - timedelta(days=30),
                historical_performance={
                    "dates": [d.strftime('%Y-%m-%d') for d in dates],
                    "portfolio_values": [4200000 + i*25000 + np.random.normal(0, 50000) for i in range(len(dates))],
                    "benchmark_values": [4200000 + i*20000 + np.random.normal(0, 40000) for i in range(len(dates))]
                },
                sector_allocation={
                    "Technology": 25,
                    "Healthcare": 15,
                    "Financial Services": 12,
                    "Consumer Discretionary": 10,
                    "Energy": 8,
                    "Utilities": 7,
                    "Real Estate": 6,
                    "Materials": 5,
                    "Other": 12
                },
                transactions=[
                    {
                        "date": (datetime.now() - timedelta(days=i*7)).strftime('%Y-%m-%d'),
                        "amount": np.random.randint(10000, 50000),
                        "type": "buy" if i % 2 == 0 else "sell",
                        "security": f"Stock_{chr(65 + (i % 10))}"
                    } for i in range(1, 13)
                ]
            ),
            "client_002": ClientModel(
                id="client_002",
                name="Pratik Patil",
                net_worth=2200000,
                risk_tolerance="Conservative",
                goals=["Preserve capital", "Charitable giving"],
                portfolio={
                    "stocks": 30,
                    "bonds": 50,
                    "alternatives": 15,
                    "cash": 5
                },
                accounts=[
                    {"id": "acct_4", "type": "Brokerage", "balance": 1500000},
                    {"id": "acct_5", "type": "IRA", "balance": 700000}
                ],
                advisors=["Michael Chen"],
                last_review_date=datetime.now() - timedelta(days=45),
                historical_performance={
                    "dates": [d.strftime('%Y-%m-%d') for d in dates],
                    "portfolio_values": [2100000 + i*8000 + np.random.normal(0, 20000) for i in range(len(dates))],
                    "benchmark_values": [2100000 + i*7000 + np.random.normal(0, 18000) for i in range(len(dates))]
                },
                sector_allocation={
                    "Government Bonds": 30,
                    "Corporate Bonds": 20,
                    "Utilities": 12,
                    "Consumer Staples": 10,
                    "Healthcare": 8,
                    "Dividend Stocks": 8,
                    "REITs": 7,
                    "Cash Equivalents": 5
                },
                transactions=[
                    {
                        "date": (datetime.now() - timedelta(days=i*14)).strftime('%Y-%m-%d'),
                        "amount": np.random.randint(5000, 25000),
                        "type": "buy" if i % 3 == 0 else "sell",
                        "security": f"Bond_{i+1}"
                    } for i in range(1, 7)
                ]
            )
        }

        market_data = {
            "SP500": {
                "ytd_return": 12.4,
                "pe_ratio": 22.3,
                "volatility": 16.2,
                "last_updated": datetime.now().isoformat()
            },
            "Bonds": {
                "ytd_return": 3.2,
                "duration": 6.5,
                "yield": 4.1,
                "last_updated": datetime.now().isoformat()
            },
            "RealEstate": {
                "ytd_return": 5.7,
                "cap_rate": 4.2,
                "occupancy": 92.5,
                "last_updated": datetime.now().isoformat()
            },
            "Commodities": {
                "ytd_return": -2.1,
                "volatility": 24.8,
                "last_updated": datetime.now().isoformat()
            }
        }

        risk_metrics = {
            "client_001": {
                "sharpe_ratio": 1.45,
                "max_drawdown": -8.2,
                "beta": 1.15,
                "alpha": 2.3,
                "var_95": -125000,
                "volatility": 15.7,
                "calculated_on": datetime.now().isoformat()
            },
            "client_002": {
                "sharpe_ratio": 0.98,
                "max_drawdown": -4.1,
                "beta": 0.65,
                "alpha": 1.1,
                "var_95": -45000,
                "volatility": 9.2,
                "calculated_on": datetime.now().isoformat()
            }
        }

        market_events = [
            {
                "date": (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d'),
                "title": f"Market Event {i}",
                "impact": ["High", "Medium", "Low"][i % 3],
                "description": f"This is a simulated market event number {i}",
                "affected_sectors": [
                    ["Technology", "Healthcare"][i % 2],
                    ["Financial", "Energy"][(i+1) % 2]
                ]
            } for i in range(10, 0, -1)
        ]

        return {
            "clients": {k: v.dict() for k, v in clients.items()},
            "market_data": market_data,
            "risk_metrics": risk_metrics,
            "market_events": market_events,
            "last_updated": datetime.now().isoformat()
        }

hybrid_db = HybridDatabase()

def create_simulated_database() -> Dict[str, Any]:
    """Legacy function that now uses the hybrid approach"""
    return hybrid_db.get_clients()
```

---

## File: `services\__init__.py`

```python
from .assistant.chat import ChatService
from .assistant.documents import DocumentService
from .assistant.compliance import ComplianceChecker
from .llm.base import LLMService
from .storage.conversation import ConversationStore

__all__ = [
    'ChatService',
    'DocumentService',
    'ComplianceChecker',
    'LLMService',
    'ConversationStore',

]
```

---

## File: `services\assistant\__init__.py`

```python
from .chat import ChatService
from .documents import DocumentService
from .compliance import ComplianceChecker
from .visualization import VisualizationService

__all__ = [
    'ChatService',
    'DocumentService',
    'ComplianceChecker',
    'VisualizationService'
]
```

---

## File: `services\assistant\base.py`

```python
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
```

---

## File: `services\assistant\chat.py`

```python
from typing import Optional, Generator, List, Dict, Any
import time
import json
from datetime import datetime
from typing import Tuple  
from services.storage.conversation import ConversationModel  
from services.llm.base import LLMService
from services.storage.conversation import ConversationStore
from database.crud import get_client_details
from config import settings
from typing import AsyncGenerator
import asyncio
from services.assistant.base import BaseAssistant
from services.assistant.compliance import ComplianceChecker
from services.assistant.documents import DocumentService
import json  
from datetime import datetime, timedelta 
import plotly.express as px  
import logging
from services.llm.base import LLMService
from database.simulated_db import create_simulated_database
logger = logging.getLogger(__name__)

class ChatService(BaseAssistant):
    def __init__(self):
        super().__init__()
        self.llm_service = LLMService()
        self.document_service = DocumentService()
        self.compliance_checker = ComplianceChecker(self.llm_service)
        self.conversation_store = ConversationStore(
            mongo_uri=settings.MONGO_URI,
            db_name=settings.MONGO_DB_NAME
        )
        self.stop_response_flag = False

    def stop_response(self):
        """Stop any ongoing response generation"""
        self.stop_response_flag = True

    def generate_response(self, query: str, session_id: str = None) -> Tuple[str, bool]:
        """Generate response to user query with conversation context"""
        self.stop_response_flag = False
        
        logger.info(f"Processing query: {query[:50]}...")
        
        # Compliance check 
        compliance_result = self.compliance_checker.check_message(query)
        if not compliance_result["is_compliant"]:
            reasons = ", ".join(compliance_result.get("reasons", []))
            logger.warning(f"Compliance check failed: {reasons}")
            return ("This prompt is out of scope.", False)
        
        logger.info("Compliance check passed")
        
        # Get conversation history for context
        conversation_history = []
        if session_id:
            existing_convo = self.load_conversation(session_id)
            if existing_convo:
                conversation_history = existing_convo.messages
                logger.info(f"Loaded {len(conversation_history)} messages from conversation history")
        
        try:
            if self._is_client_specific_query(query):
                response = self._handle_client_query(query, conversation_history)
            elif self._is_market_query(query):
                response = self._handle_market_query(query, conversation_history)
            else:
                response = self._handle_general_query(query, conversation_history)
            
            logger.info(f"Generated response: {len(response)} characters")
            return (response, True)
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return ("I apologize, but I encountered an error processing your request.", True)

    async def generate_response_stream(self, query: str, session_id: str = None) -> AsyncGenerator[Tuple[str, bool], None]:
        """Async generator for streaming responses"""
        self.stop_response_flag = False
        
        # Compliance check first
        compliance_result = self.compliance_checker.check_message(query)
        if not compliance_result["is_compliant"]:
            reasons = ", ".join(compliance_result.get("reasons", []))
            yield (f"COMPLIANCE REJECTION: {reasons}", False)
            return

        # Get conversation history for context
        conversation_history = []
        if session_id:
            existing_convo = self.load_conversation(session_id)
            if existing_convo:
                conversation_history = existing_convo.messages

        # Simulate streaming chunks (replace with actual LLM streaming)
        mock_response = "This is a simulated streaming response."
        if self._is_client_specific_query(query):
            mock_response = self._handle_client_query(query, conversation_history)
        elif self._is_market_query(query):
            mock_response = self._handle_market_query(query, conversation_history)
        else:
            mock_response = self._handle_general_query(query, conversation_history)

        for chunk in [mock_response[i:i+20] for i in range(0, len(mock_response), 20)]:
            if self.stop_response_flag:
                yield ("(Stopped)", True)
                break
            yield (chunk, True)
            await asyncio.sleep(0.05)  # Simulate processing delay
    

    def _is_client_specific_query(self, query: str) -> bool:
        """Check if query is about a specific client"""
        # Get client names from database instead of hardcoding
        client_data = create_simulated_database().get("clients", {})
        client_names = [client["name"] for client in client_data.values()]
        return any(name.lower() in query.lower() for name in client_names)

    def _is_market_query(self, query: str) -> bool:
        """Check if query is about market conditions"""
        market_terms = ["market", "S&P", "index", "economic", "trend"]
        return any(term.lower() in query.lower() for term in market_terms)

    def _handle_client_query(self, query: str, conversation_history: List[Dict] = None) -> str:
        """Handle queries about specific clients"""
        logger.info(f"Handling client query with LLM: {query[:50]}...")
        try:
            # Use the LLM service to generate a proper response with client context
            response = self.llm_service.generate_response(query, conversation_history)
            logger.info(f"Client query LLM response generated: {len(response) if response else 0} characters")
            return response or "I'm unable to find information about that client. Please check the client name and try again."
        except Exception as e:
            logger.error(f"Error in client query LLM generation: {e}")
            return "I apologize, but I encountered an error processing your client query. Please try again."

    def _handle_market_query(self, query: str, conversation_history: List[Dict] = None) -> str:
        """Handle market-related queries"""
        # Build context from conversation history
        context_prompt = ""
        if conversation_history:
            recent_messages = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
            context_lines = []
            for msg in recent_messages:
                if msg.get("role") == "user":
                    context_lines.append(f"User: {msg.get('content', '')}")
                elif msg.get("role") == "assistant":
                    context_lines.append(f"Assistant: {msg.get('content', '')}")
            
            if context_lines:
                context_prompt = f"""
CONVERSATION CONTEXT:
{chr(10).join(context_lines)}

CURRENT QUESTION:
{query}

Please respond to the current question considering the conversation context above.
"""
        
        prompt = f"""
        You are an expert financial analyst. Answer this market query:
        {context_prompt if context_prompt else query}
        
        Include:
        1. Current market conditions
        2. Historical context when relevant
        3. Potential impact on client portfolios
        4. Recommended actions if appropriate
        
        Keep the response professional but understandable for wealth advisors.
        """
        return self.llm_service.llm.invoke(prompt).content or "Unable to generate market analysis"

    def _handle_general_query(self, query: str, conversation_history: List[Dict] = None) -> str:
        """Handle general wealth management queries"""
        logger.info(f"Handling general query with LLM: {query[:50]}...")
        try:
            # Use the LLM service to generate a proper response
            response = self.llm_service.generate_response(query, conversation_history)
            logger.info(f"LLM response generated: {len(response) if response else 0} characters")
            return response or "I'm unable to process that request at the moment. Please try rephrasing your question."
        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")
            return "I apologize, but I encountered an error processing your request. Please try again."


    def save_conversation(self, session_id: str, messages: List[Dict]) -> str:
        """Save conversation to MongoDB"""
        logger.info(f"Saving conversation for session {session_id} with {len(messages)} messages")
        try:
            # Use the conversation store to save
            result = self.conversation_store.save_conversation(
                session_id=session_id,
                messages=messages
            )
            if result:
                messages=self.conversation_store.conversations.find_one({"session_id": session_id}).get("messages", [])
               
                if len(messages) >= 4:  # At least 2 user-assistant exchanges
                    summary = self._generate_conversation_summary(messages)
                    self.conversation_store.update_summary(session_id, summary)
        
            if not result:  # If empty string returned
                logger.error("Failed to save conversation to MongoDB")
                return session_id  # Still return session_id even if save failed
                
            logger.info(f"Successfully saved conversation for session {session_id}")
            return session_id
        except Exception as e:
            logger.error(f"Error saving conversation: {e}")
            return session_id
    def load_conversation(self, session_id: str) -> Optional[dict]:
        """Load conversation from persistent storage"""
        return self.conversation_store.load_conversation(session_id)
# In services/assistant/chat.py
    def _generate_conversation_summary(self, messages: List[Dict]) -> str:
        """Generate a meaningful summary of the conversation"""
        try:
            # Filter out empty/system messages and get last 10 messages for context
            recent_messages = [
                msg for msg in messages[-10:] 
                if msg.get("content") and msg.get("role") in ["user", "assistant"]
            ]
            
            if not recent_messages:
                return "New Conversation"
                
            # Build prompt for summary generation
            prompt = f"""
            Analyze this conversation and generate a concise 5-7 word summary 
            that captures the main topic. Focus on the key subject being discussed.
            
            Guidelines:
            - Be specific about financial topics if present
            - Don't include greetings/small talk
            - Focus on the core discussion
            
            Conversation Excerpt:
            {recent_messages}
            
            Respond ONLY with the summary, nothing else.
            """
            
            # Get summary from LLM
            summary = self.llm_service.generate_response(prompt).strip()
            
            # Fallback if no good summary
            if not summary or len(summary.split()) > 10:
                if len(recent_messages) >= 2 :
                    user_message = next(
                        (msg["content"] for msg in reversed(recent_messages) 
                        if msg["role"] == "user"), 
                        ""
                    )[:50]
                    return f"Chat: {user_message}..." if user_message else "Discussion"
                return "Chat Discussion"
                
            return summary
            
        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")
            # Fallback based on last user message if available
            last_user_message = next(
                (msg["content"] for msg in reversed(messages) 
                if msg.get("role") == "user"),
                None
            )
            return last_user_message[:50] + "..." if last_user_message else "Chat Discussion"
    @property
    def capabilities(self) -> Dict[str, Any]:
        """Get capabilities of this assistant"""
        return {
            "query_types": ["client_specific", "market", "general"],
            "streaming": True,
            "document_processing": True,
            "compliance_checks": True
        }
    def search_conversations(self, query: str, limit: int = 10) -> List[Tuple[ConversationModel, Dict]]:
        """Search all conversations for matching messages"""
        try:
            all_convos = self.conversation_store.get_all_conversations()
            results = []
            
            for conv in all_convos:
                for msg in conv.messages:
                    if "content" in msg and query.lower() in msg["content"].lower():
                        results.append((conv, msg))
                        break  # Only include one match per conversation
                        
                if len(results) >= limit:
                    break
                    
            return results[:limit]
        except Exception as e:
            logger.error(f"Error searching conversations: {e}")
            return []

    def get_user_profile(self, session_id: str) -> Dict[str, Any]:
        """Get user profile data"""
        try:
            # This would come from your database in a real implementation
            # For now returning a mock profile
            return {
                "name": "User",
                "email": "user@example.com",
                "preferences": {
                    "language": "en",
                    "voice_input": False,
                    "theme": "light"
                }
            }
        except Exception as e:
            logger.error(f"Error getting user profile: {e}")
            return {}

    def update_user_profile(self, session_id: str, profile: Dict[str, Any]) -> bool:
        """Update user profile in database"""
        try:
            # In a real app, this would update the database
            # For now just return success
            return True
        except Exception as e:
            logger.error(f"Error updating user profile: {e}")
            return False
    # In services/assistant/chat.py - Add these methods to the ChatService class:

    def get_conversation_context(self, session_id: str, limit: int = 4) -> str:
        """Get recent conversation messages as context"""
        if not session_id:
            return ""
        
        conversation = self.load_conversation(session_id)
        if not conversation or not conversation.messages:
            return ""
        
        # Get last 'limit' message exchanges (user + assistant pairs)
        messages = conversation.messages[-(limit*2):] if len(conversation.messages) > limit*2 else conversation.messages
        
        context = []
        for msg in messages:
            if msg.get("role") in ["user", "assistant"] and msg.get("content"):
                role = "User" if msg["role"] == "user" else "Assistant"
                context.append(f"{role}: {msg['content']}")
        
        return "\n".join(context)

    def _prepare_chat_context(self, session_id: str, new_message: str) -> str:
        """Prepare full chat context including past messages"""
        past_context = self.get_conversation_context(session_id)
        
        prompt = f"""
        You are a professional wealth management assistant. Continue this conversation:
        
        {past_context}
        
        User: {new_message}
        
        Assistant: """
        
        return prompt
```

---

## File: `services\assistant\compliance.py`

```python
from typing import Dict, List, Optional
import re
from config import settings
from typing import Any
import json

class ComplianceChecker:
    def __init__(self, llm=None):  
        self.blocked_terms = [
            "insider trading", "inside information", "material nonpublic information",
            "manipulate the market", "front running", "ponzi scheme", "pyramid scheme",
            "guaranteed 100% returns", "absolutely risk-free", "guaranteed profit"
        ]
        self.blocked_patterns = [
            r"\bguarante[ed]\s+\d+%\s+return\b",  
            r"\babsolutely\s+risk[\-\s]*free\b",  
            r"\bponzi\s+scheme\b",
            r"\bpyramid\s+scheme\b"
        ]
        self.llm = llm 


    def check_message(self, message: str, client_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Comprehensive compliance check with detailed results
        Returns dict with:
        - is_compliant: bool
        - reasons: list of strings if non-compliant
        - recommendations: list of strings for fixing issues
        """
        result = {
            "is_compliant": True,
            "reasons": [],
            "recommendations": []
        }
        
        # Phase 1: Rule-based checks
        failed_terms = [term for term in self.blocked_terms if term in message.lower()]
        failed_patterns = []
        
        for pattern in self.blocked_patterns:
            if re.search(pattern, message.lower()):
                failed_patterns.append(pattern)
                
        if failed_terms or failed_patterns:
            result["is_compliant"] = False
            result["reasons"].extend(
                f"Blocked term found: {term}" for term in failed_terms
            )
            result["reasons"].extend(
                f"Blocked pattern found: {pattern}" for pattern in failed_patterns
            )
            result["recommendations"].append(
                "Remove or rephrase prohibited terms/phrases"
            )
            
        # Phase 2: LLM check - DISABLED FOR NOW due to API issues
        # if result["is_compliant"]:  # Only do LLM check if passed phase 1
        #     llm_result = self._detailed_llm_check(message, client_id)
        #     if not llm_result.get("is_compliant", True):
        #         result["is_compliant"] = False
        #         result["reasons"].extend(llm_result.get("reasons", []))
        #         result["recommendations"].extend(llm_result.get("recommendations", []))
                
        return result

    def _detailed_llm_check(self, message: str, client_id: Optional[str]) -> Dict[str, Any]:
        """Detailed LLM compliance check with explanation"""
        if not self.llm:  
            return {"is_compliant": True, "reasons": [], "recommendations": []}
            
        prompt = f"""
        Perform comprehensive compliance check on this message:
        Message: {message}
        
        {f"Client ID: {client_id}" if client_id else ""}
        
        Analyze for:
        1. Regulatory compliance (SEC, FINRA)
        2. Fiduciary duty considerations
        3. Suitability for client risk profile
        4. Proper disclosures
        5. Conflicts of interest
        
        Return response as JSON with:
        - is_compliant (boolean)
        - reasons (list of strings if non-compliant)
        - recommendations (list of strings)
        this type of result is required bro: {"is_compliant": True, "reasons": [], "recommendations": []}
        JSON ONLY, no other text.
        """
        
        response = self.llm.generate_response(prompt) 
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            return {"is_compliant": True, "reasons": [], "recommendations": []}
```

---

## File: `services\assistant\documents.py`

```python
from typing import List, Tuple, Dict, Optional
from fastapi import UploadFile
import tempfile
from pathlib import Path
import logging
from ..llm.document_embeddings import DocumentEmbeddingService
from config import settings
from langchain_google_genai import ChatGoogleGenerativeAI
logger = logging.getLogger(__name__)


class DocumentService:
    """Optimized document processing and query service with enhanced error handling"""
    
    def __init__(self):
        # Validate critical settings before initialization
        if not all([settings.MONGO_URI, settings.MONGO_DOCS_DB_NAME, settings.MONGO_DOCS_COLLECTION]):
            raise ValueError("Incomplete MongoDB configuration in settings")
            
        if not settings.GOOGLE_API_KEY:
            raise ValueError("Google API key missing in settings")

        self.embedding_service = DocumentEmbeddingService(
            mongo_uri=settings.MONGO_URI,
            db_name=settings.MONGO_DOCS_DB_NAME,
            collection_name=settings.MONGO_DOCS_COLLECTION,
            google_api_key=settings.GOOGLE_API_KEY
            
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=settings.GOOGLE_API_KEY
        )
    async def process_uploaded_files(self, files: List[UploadFile]) -> Tuple[str, List[Tuple[str, str]]]:
        """Secure PDF processing pipeline with validation and cleanup"""
        if not files:
            logger.warning("Empty file list received")
            return "No files received", []

        processed_results = []
        temp_files = []

        try:
            for file in files:
                try:
                    # Validate file type
                    if not file.filename.lower().endswith('.pdf'):
                        processed_results.append((file.filename, "Invalid file type"))
                        continue

                    # Secure temp file handling
                    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                        content = await file.read()
                        temp_file.write(content)
                        temp_files.append(temp_file.name)

                    # Process document
                    self.embedding_service.process_pdf(temp_files[-1])
                    processed_results.append((file.filename, "Processed successfully"))

                except Exception as e:
                    logger.error(f"Processing failed for {file.filename}: {str(e)}")
                    processed_results.append((file.filename, f"Processing error: {str(e)}"))

            return "Document processing completed", processed_results

        finally:
            # Guaranteed cleanup
            for temp_path in temp_files:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.error(f"Temp file cleanup failed: {str(e)}")

    def query_documents(self, query: str, k: int = 3) -> Dict[str, Optional[List[Dict]]]:
        """Search documents and synthesize an answer"""
        response = {
            "answer": None,
            "documents": None,
            "error": None
        }

        try:
            if not query or not query.strip():
                raise ValueError("Empty query string")

            # Search for relevant document chunks
            results = self.embedding_service.search_documents(query, k=k)
            
            if not results:
                response["answer"] = "No relevant documents found"
                return response

            # Format documents for response
            formatted_docs = []
            for doc in results:
                formatted_docs.append({
                    "content": doc['content'],
                    "source": doc['metadata'].get('source', 'Unknown'),
                    "page": doc['metadata'].get('page', 'N/A'),
                    "score": doc.get('score', 0)
                })

            # Generate synthesized answer using self.llm
            context = "\n\n".join([f"Document {i+1} (Source: {doc['metadata'].get('source', 'Unknown')}, Page: {doc['metadata'].get('page', 'N/A')}):\n{doc['content']}" 
                                 for i, doc in enumerate(results)])
            
            prompt = f"""You are a helpful assistant that answers questions based on provided documents.
            Question: {query}

            Please provide a concise and accurate answer using the following documents.
            Combine information from multiple documents when relevant.
            Dont cite sources in the format [Source: Document X] when referencing specific information.

            Documents:
            {context}

            Your response:
            """
            
            # Generate the answer using self.llm.invoke()
            synthesized_answer = self.llm.invoke(prompt).content
            
            response.update({
                "answer": synthesized_answer.strip(),
                "documents": formatted_docs
            })

        except Exception as e:
            logger.error(f"Query failed: {str(e)}", exc_info=True)
            response["error"] = str(e)
            response["answer"] = "Failed to process query"

        return response

    def debug_vector_store(self) -> Dict[str, any]:
        """Diagnostic method to verify vector store health"""
        diagnostics = {
            "embedding_count": 0,
            "sample_document": None,
            "index_status": None
        }

        try:
            # Basic collection stats
            coll = self.embedding_service.collection
            diagnostics["embedding_count"] = coll.count_documents({"embedding": {"$exists": True}})
            
            # Get a sample embedded document
            sample = coll.find_one({"embedding": {"$exists": True}})
            if sample:
                diagnostics["sample_document"] = {
                    "source": sample.get("metadata", {}).get("source"),
                    "embedding_length": len(sample["embedding"]) if "embedding" in sample else 0
                }
            
            # Check index existence (crude check)
            indexes = list(coll.list_indexes())
            diagnostics["index_status"] = any(
                idx.get("name") == self.embedding_service.INDEX_NAME 
                for idx in indexes
            )

        except Exception as e:
            diagnostics["error"] = str(e)

        return diagnostics
```

---

## File: `services\assistant\visualization.py`

```python
from typing import Dict, Tuple, Optional, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from database.crud import get_client_details
import logging
import plotly.express as px
logger = logging.getLogger(__name__)

class VisualizationService:
    def __init__(self):
        self.client_colors = {
            "client_001": "#1f77b4",  # Blue
            "client_002": "#ff7f0e"   # Orange
        }
        self.chart_themes = {
            "primary": "plotly_white",
            "secondary": "ggplot2"
        }

    def get_client_visualizations(self, client_id: str) -> Dict[str, Optional[go.Figure]]:
        """
        Generate all standard visualizations for a client
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dictionary of visualization figures with keys:
            - portfolio_allocation
            - sector_allocation
            - performance_comparison
            - risk_profile
            - account_breakdown
        """
        client_data = get_client_details(client_id)
        if not client_data:
            logger.error(f"No data found for client {client_id}")
            return {
                "portfolio_allocation": None,
                "sector_allocation": None,
                "performance_comparison": None,
                "risk_profile": None,
                "account_breakdown": None
            }

        try:
            figures = {}
            
            # Portfolio Allocation Pie Chart
            figures["portfolio_allocation"] = self._create_portfolio_allocation_chart(
                client_data["portfolio_allocation"],
                client_data["name"]
            )
            
            # Sector Allocation Bar Chart
            if "sector_allocation" in client_data:
                figures["sector_allocation"] = self._create_sector_allocation_chart(
                    client_data["sector_allocation"],
                    client_data["name"]
                )
            
            # Performance Comparison Line Chart
            if "historical_performance" in client_data:
                figures["performance_comparison"] = self._create_performance_chart(
                    client_data["historical_performance"],
                    client_data["name"]
                )
            
            # Risk Profile Radar Chart
            # Note: Requires risk metrics data not in basic client_data
            
            # Account Breakdown Chart
            figures["account_breakdown"] = self._create_account_breakdown_chart(
                client_data["accounts"],
                client_data["name"]
            )
            
            logger.info(f"Generated visualizations for client {client_id}")
            return figures
            
        except Exception as e:
            logger.error(f"Error generating visualizations for client {client_id}: {e}")
            return {
                "portfolio_allocation": None,
                "sector_allocation": None,
                "performance_comparison": None,
                "risk_profile": None,
                "account_breakdown": None
            }

    def create_custom_visualization(self, data: Dict, visualization_type: str, 
                                  client_context: Optional[Dict] = None) -> Optional[go.Figure]:
        """
        Create custom visualization based on request
        
        Args:
            data: Data payload for visualization
            visualization_type: One of ['pie', 'bar', 'line', 'radar', 'scatter']
            client_context: Optional client attributes for theming
            
        Returns:
            Configured Plotly figure or None on error
        """
        try:
            client_color = self._get_client_color(client_context)
            
            if visualization_type == "pie":
                return self._create_pie_chart(
                    data=data,
                    title=data.get("title", "Custom Pie Chart"),
                    color=client_color
                )
            elif visualization_type == "bar":
                return self._create_bar_chart(
                    data=data,
                    title=data.get("title", "Custom Bar Chart"),
                    color=client_color
                )
            elif visualization_type == "line":
                return self._create_line_chart(
                    data=data,
                    title=data.get("title", "Custom Line Chart"),
                    color=client_color
                )
            elif visualization_type == "radar":
                return self._create_radar_chart(
                    data=data,
                    title=data.get("title", "Custom Radar Chart")
                )
            elif visualization_type == "scatter":
                return self._create_scatter_plot(
                    data=data,
                    title=data.get("title", "Custom Scatter Plot"),
                    color=client_color
                )
            else:
                logger.warning(f"Unknown visualization type: {visualization_type}")
                return None
        except Exception as e:
            logger.error(f"Error creating custom visualization: {e}")
            return None

    def _create_portfolio_allocation_chart(self, portfolio_data: Dict, client_name: str) -> go.Figure:
        """Create pie chart for portfolio allocation"""
        labels = list(portfolio_data.keys())
        values = list(portfolio_data.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker_colors=px.colors.qualitative.Pastel
        )])
        
        fig.update_layout(
            title=f"{client_name}'s Portfolio Allocation",
            height=400
        )
        return fig

    def _create_sector_allocation_chart(self, sector_data: Dict, client_name: str) -> go.Figure:
        """Create bar chart for sector allocation"""
        sectors = list(sector_data.keys())
        allocations = list(sector_data.values())
        
        fig = go.Figure(data=[go.Bar(
            x=allocations,
            y=sectors,
            orientation='h',
            marker_color=px.colors.sequential.Viridis
        )])
        
        fig.update_layout(
            title=f"{client_name}'s Sector Allocation",
            xaxis_title="Allocation (%)",
            height=500
        )
        return fig

    def _create_performance_chart(self, performance_data: Dict, client_name: str) -> go.Figure:
        """Create line chart for performance comparison"""
        dates = pd.to_datetime(performance_data["dates"])
        portfolio_values = performance_data["portfolio_values"]
        benchmark_values = performance_data["benchmark_values"]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio',
            line=dict(width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark_values,
            mode='lines',
            name='Benchmark',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title=f"{client_name}'s Portfolio Performance",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified'
        )
        return fig

    def _create_account_breakdown_chart(self, accounts: List[Dict], client_name: str) -> go.Figure:
        """Create bar chart for account breakdown"""
        account_labels = [f"{acc['type']} (${acc['balance']:,})" for acc in accounts]
        balances = [acc["balance"] for acc in accounts]
        
        fig = go.Figure(data=[go.Bar(
            x=account_labels,
            y=balances,
            marker_color=px.colors.sequential.Blues
        )])
        
        fig.update_layout(
            title=f"{client_name}'s Account Breakdown",
            yaxis_title="Balance ($)",
            yaxis_tickprefix='$',
            yaxis_tickformat=',.0f'
        )
        return fig

    def _get_client_color(self, client_context: Optional[Dict]) -> str:
        """Get theme color for client visualizations"""
        if client_context and "id" in client_context:
            return self.client_colors.get(client_context["id"], "#636EFA")
        return "#636EFA"  # Default Plotly blue
```

---

## File: `services\llm\__init__.py`

```python
from .base import LLMService
from .document_embeddings import DocumentEmbeddingService

__all__ = [
    'LLMService',
    'DocumentEmbeddingService'
    ]
```

---

## File: `services\llm\base.py`

```python
import os
from typing import Optional, Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config import settings
import logging
import json
from services.assistant.documents import DocumentService
logger = logging.getLogger(__name__)

from services.assistant.compliance import ComplianceChecker

from langchain_google_genai import ChatGoogleGenerativeAI
from database.simulated_db import create_simulated_database
wealth_db = create_simulated_database()


class LLMService:
    """Core LLM service for interacting with language models"""
    
    def __init__(self, model: str = None, temperature: float = None):
        """Initialize LLM service with configuration"""
        self.model_name = model or settings.LLM_MODEL
        self.temperature = temperature or settings.LLM_TEMPERATURE
        self.llm = ChatGoogleGenerativeAI(
            model=self.model_name,
            temperature=self.temperature,
            google_api_key=settings.GOOGLE_API_KEY
        )

        self.compliance_checked = False
        self.document_service=DocumentService()
        self.scope_description = """
        This assistant specializes in wealth management topics including:
        - Client portfolio analysis
        - Risk assessment
        - Investment strategy
        - Market comparisons
        - Account-specific queries
        - Financial planning
        - Retirement planning
        - Tax strategies
        - Estate planning
        """
        
    
    

    def is_in_scope(self, query: str) -> bool:
        """Check if the query falls within wealth management scope"""
        # Make this much more lenient - only reject obviously off-topic queries
        out_of_scope_terms = [
            "weather", "sports", "entertainment", "politics", "cooking", "travel",
            "movies", "music", "games", "celebrities", "fashion", "technology news"
        ]
        
        query_lower = query.lower()
        
        # If it contains obvious out-of-scope terms, reject
        if any(term in query_lower for term in out_of_scope_terms):
            return False
            
        # If it contains wealth/finance terms, accept
        finance_terms = [
            "portfolio", "investment", "client", "money", "fund", "stock", "bond",
            "retirement", "wealth", "financial", "asset", "risk", "return", "market",
            "account", "balance", "allocation"
        ]
        
        if any(term in query_lower for term in finance_terms):
            return True
            
        # For everything else (including general greetings), accept
        return True



    def query_wealth_db(self, query: str) -> str:
        db_context = create_simulated_database()
        def minimal_clients():
            return {
                k: {
                "id": v["id"],
                "name": v["name"],
                "client_id": v["id"],
                "risk_tolerance": v.get("risk_tolerance")
            } for k, v in wealth_db["clients"].items()
        }

        is_risk_question = "risk" in query.lower() or "tolerance" in query.lower()
        if is_risk_question:
            db_context = {
                "clients": minimal_clients()
            }
        else:
            db_context = wealth_db  # full database for normal use

        prompt = f"""
        You are a regulated wealth assistant. Answer the question **based only on the data below.**
        
        Rules:
        - If asked about risk level, **only** respond with the risk tolerance. Do NOT include portfolio or allocations.
        - Be concise. 
        - You can reference source structure or extra financial metrics.

        USER QUESTION:
        {query}

        DATA:
        {json.dumps(db_context, indent=2, default=str)}

        FINAL ANSWER:
        """
        return self.llm.invoke(prompt).content.strip()



    # def query_documents(self, query: str) -> str:
    #     if not self.vector_db:
    #       return "No vector db found"
    #     docs=self.vector_db.similarity_search(query,k=3)
    #     if not docs:
    #       return "No similar documents found for query"
    #     context="\n".join([doc.page_content for doc in docs])
    #     prompt=f"""You are an AI assisstant. Answer the following question **strictly** using the text excerpts below.
    #     Resolve the duplicate values only if there are two or more clients with the same full name, by asking the user for full client name or client ID.
    #     DOCUMENT EXCERPTS:
    #     {context}
    #     QUERY:
    #     {query}
    #     If the answer is not found in the excerpts, say "The answer is not available in the provided documents." Keep the response concise and focused."""
    #     answer= self.llm.invoke(prompt).content
    #     return  f"**Matched document chunks**:\\n\n{context}\n\n **Answer**:\n{answer}"

# In services/llm/base.py:

    def generate_response(self, query: str, conversation_history: List[Dict] = None) -> str:
        """Generate response using conversation context"""
        if not self.is_in_scope(query):
            return "This prompt is out of scope."

        # Build context prompt
        context_prompt = ""
        if conversation_history:
            # Limit conversation history to last 6 messages (3 user-assistant exchanges)
            recent_messages = conversation_history[-6:] if len(conversation_history) > 6 else conversation_history
            
            context_lines = [
                f"{msg['role'].capitalize()}: {msg['content']}" 
                for msg in recent_messages 
                if msg.get("content") and msg.get("role") in ["user", "assistant"]
            ]
            
            if context_lines:
                context_prompt = f"""Conversation Context:
    {'\n'.join(context_lines)}

    Current Question: {query}

    Please respond considering the context above."""
        
        # Get data responses
        db_response = self.query_wealth_db(query)
        doc_response = self.document_service.query_documents(query)
        
        if context_prompt:
            prompt = f"""Your role: Professional wealth management assistant
    {context_prompt}

    Available Information:
    - Database: {db_response}
    - Documents: {doc_response}

    Please provide a concise, professional response answering the user's question based on all available context:"""
        else:
            prompt = f"""Your role: Professional wealth management assistant
    Question: {query}

    Available Information:
    - Database: {db_response}
    - Documents: {doc_response}

    Please provide a concise, professional response:"""
        
        return self.llm.invoke(prompt).content.strip()
```

---

## File: `services\llm\document_embeddings.py`

```python
from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from typing import List
import os
from pathlib import Path
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DocumentEmbeddingService:
    def __init__(self, mongo_uri: str, db_name: str, collection_name: str, google_api_key: str):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=google_api_key
        )
        
        # Add LLM for synthesizing answers
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0,
            google_api_key=google_api_key
        )
        
        self.INDEX_NAME = "vertex_2"
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using the LLM"""
        response = self.llm.invoke(prompt).content
        return response.strip()
        
    def _ensure_vector_index(self):
        # Check if vector index exists, create if not
        if self.INDEX_NAME not in self.collection.list_indexes():
            self.collection.create_index(
                [("embedding", "knnVector")],
                knnVector={
                    "dimensions": 768,
                    "similarity": "cosine"
                },
                name=self.INDEX_NAME
            )
    
    def process_pdf(self, file_path: str):
        """Process a PDF file into embeddings"""
        try:
            abs_path = Path(file_path).absolute()
            if not abs_path.exists():
                raise FileNotFoundError(f"PDF file not found at: {abs_path}")
                
            # Load and split document
            loader = PyPDFLoader(str(abs_path))
            pages = loader.load()
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = text_splitter.split_documents(pages)
            
            # Create embeddings
            vector_store = MongoDBAtlasVectorSearch.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                collection=self.collection,
                index_name=self.INDEX_NAME
            )
            
            return vector_store
            
        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise
            
    def search_documents(self, query: str, k: int = 3) -> List[dict]:
        """Search documents using vector similarity"""
        vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embedding_model,
            index_name=self.INDEX_NAME
        )
        
        results = vector_store.similarity_search(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.score if hasattr(doc, 'score') else 0
            }
            for doc in results
        ]
    
    def clear_documents(self):
        """Remove all documents from the collection"""
        self.collection.delete_many({})

    def __del__(self):
        self.client.close()
```

---

## File: `services\storage\__init__.py`

```python
from .conversation import ConversationStore

__all__ = ['ConversationStore']
```

---

## File: `services\storage\conversation.py`

```python
from pymongo import MongoClient
from typing import Dict, List, Optional
from datetime import datetime
from bson import ObjectId
from pydantic import BaseModel, Field
import os
from config import settings
import logging
from typing import Any
import pymongo
logger = logging.getLogger(__name__)

class ConversationModel(BaseModel):
    """Pydantic model for conversation storage"""
    session_id: str
    messages: List[Dict[str, str]]
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    client_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    summary: Optional[str] = None  # Add summary field
    is_favorite: bool = Field(default=False)  # Add this line
    fav_position: Optional[int] = Field(default=None)  # And this line for ordering

class ConversationStore:
    def __init__(self, mongo_uri: str = None, db_name: str = None):
        try:
            self.client = MongoClient(mongo_uri or settings.MONGO_URI)
            self.db = self.client[db_name or settings.MONGO_DB_NAME]
            self.conversations = self.db["conversations"]
            self.conversations.create_index("session_id", unique=True)

        except Exception as e:
            print("FATAL ERROR in MongoDB setup:", str(e))
            raise
    # In Backend/services/storage/conversation.py
    def save_conversation(self, session_id: str, messages: List[Dict], 
                        client_id: Optional[str] = None, 
                        metadata: Optional[Dict] = None) -> str:
        """Save conversation with strict session enforcement"""
        # Input validation
        if not session_id:
            logger.error("No session_id provided")
            return ""
            
        if not isinstance(messages, list) or len(messages) == 0:
            logger.error("Invalid messages format")
            return ""
            
        if len(messages) % 2 != 0:
            logger.error("Messages should be in user-assistant pairs")
            return ""

        logger.info(f"Saving conversation for session {session_id} with {len(messages)} messages")

        try:
            # Check existing conversation
            existing = self.conversations.find_one({"session_id": session_id})
            
            if existing:
                logger.info(f"Found existing conversation for session {session_id}, appending messages")
                # Prevent duplicate message pairs
                existing_messages = existing.get("messages", [])
                
                if existing_messages:
                    last_pair = existing_messages[-2:] if len(existing_messages) >= 2 else []
                    new_pair = messages[:2]
                    
                    if (len(last_pair) == 2 and len(new_pair) == 2 and
                        last_pair[0]["content"] == new_pair[0]["content"] and
                        last_pair[1]["content"] == new_pair[1]["content"]):
                        logger.debug("Skipping duplicate message pair")
                        return session_id
                
                # Update operation
                update_data = {
                    "$push": {"messages": {"$each": messages}},
                    "$set": {"updated_at": datetime.utcnow()}
                }
                
                if client_id:
                    update_data["$set"]["client_id"] = client_id
                if metadata:
                    update_data["$set"]["metadata"] = {
                        **(existing.get("metadata", {})),
                        **(metadata or {})
                    }
                    
                result = self.conversations.update_one(
                    {"session_id": session_id},
                    update_data
                )
                
                return session_id if result.modified_count > 0 else ""
            else:
                # Create new conversation document
                new_convo = {
                    "session_id": session_id,
                    "messages": messages,
                    "created_at": datetime.utcnow(),
                    "updated_at": datetime.utcnow(),
                    "client_id": client_id,
                    "metadata": metadata or {},
                    "version": "1.0"
                }
                
                result = self.conversations.insert_one(new_convo)
                return session_id if result.inserted_id else ""
                
        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")
            return ""
    def load_conversation(self, session_id: str) -> Optional[ConversationModel]:
        """Load conversation from MongoDB"""
        try:
            result = self.conversations.find_one({"session_id": session_id})
            if not result:
                return None
                
            # Convert MongoDB ObjectId to string        
            result["id"] = str(result.pop("_id"))
            return ConversationModel(**result)
            
        except Exception as e:
            logger.error(f"Error loading conversation: {e}")
            return None

    def delete_conversation(self, session_id: str) -> bool:
        """Delete conversation from MongoDB"""
        try:
            result = self.conversations.delete_one({"session_id": session_id})
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting conversation: {e}")
            return False

    def update_metadata(self, session_id: str, metadata: Dict) -> bool:
        """Update conversation metadata"""
        try:
            result = self.conversations.update_one(
                {"session_id": session_id},
                {"$set": {"metadata": metadata, "updated_at": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating conversation metadata: {e}")
            return False
    def get_all_conversations(self) -> List[ConversationModel]:
        """Get latest versions of all conversations"""
        try:
            results = self.conversations.find().sort("updated_at", -1)
            return [
                ConversationModel(**{**conv, "id": str(conv["_id"])}) 
                for conv in results
                if conv
            ]
        except Exception as e:
            logger.error(f"Error fetching all conversations: {e}")
            return []

    def update_summary(self, session_id: str, summary: str) -> bool:
        """Update conversation summary"""
        try:
            result = self.conversations.update_one(
                {"session_id": session_id},
                {"$set": {"summary": summary, "updated_at": datetime.utcnow()}}
            )
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating conversation summary: {e}")
            return False
    def close(self):
        """Close MongoDB connection"""
        try:
            self.client.close()
        except Exception as e:
            logger.error(f"Error closing MongoDB connection: {e}")
    # In ConversationStore class in conversation.py

    def toggle_favorite(self, session_id: str) -> bool:
        """Toggle favorite status of a conversation"""
        try:
            conversation = self.conversations.find_one({"session_id": session_id})
            if not conversation:
                return False
                
            new_status = not conversation.get("is_favorite", False)
            
            # If marking as favorite, set position to last in favorites
            if new_status:
                max_position = self.conversations.find_one(
                    {"is_favorite": True},
                    sort=[("fav_position", -1)]
                )
                new_position = (max_position["fav_position"] + 1) if max_position else 0
            else:
                new_position = None
            
            result = self.conversations.update_one(
                {"session_id": session_id},
                {
                    "$set": {
                        "is_favorite": new_status,
                        "fav_position": new_position,
                        "updated_at": datetime.utcnow()
                    }
                }
            )
            
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error toggling favorite: {e}")
            return False

    def get_favorites(self, limit: int = 20) -> List[ConversationModel]:
        """Get all favorited conversations"""
        try:
            results = self.conversations.find(
                {"is_favorite": True}
            ).sort("fav_position", 1).limit(limit)
            
            return [
                ConversationModel(**{**conv, "id": str(conv["_id"])})
                for conv in results
                if conv
            ]
        except Exception as e:
            logger.error(f"Error fetching favorites: {e}")
            return []

    def reorder_favorites(self, new_order: List[str]) -> bool:
        """Update favorite positions based on new order list"""
        try:
            with self.client.start_session() as session:
                with session.start_transaction():
                    # First reset all positions to None
                    self.conversations.update_many(
                        {"is_favorite": True},
                        {"$set": {"fav_position": None}},
                        session=session
                    )
                    
                    # Then set new positions
                    updates = []
                    for index, session_id in enumerate(new_order):
                        updates.append((
                            {"session_id": session_id},
                            {"$set": {"fav_position": index}}
                        ))
                    
                    # Perform updates in bulk
                    bulk_ops = [
                        pymongo.UpdateOne(query, update)
                        for query, update in updates
                    ]
                    
                    if bulk_ops:
                        self.conversations.bulk_write(bulk_ops, session=session)
                        
                    return True
        except Exception as e:
            logger.error(f"Error reordering favorites: {e}")
            return False
```

---

## File: `services\storage\faq_store.py`

```python
from pymongo import MongoClient
from config import settings
 
class FAQStore:
    def __init__(self, db_name):
        client=MongoClient(settings.MONGO_URI)
        db=client[db_name]
        self.collection=db["faqs"]
   
    def get_all_faqs(self):
        return list(self.collection.find({}, {"_id":0, "question":1}))
 
```

---

## File: `visualization\__init__.py`

```python
from .charts import (
    create_portfolio_pie_chart,
    create_sector_allocation_chart,
    create_performance_comparison,
    create_risk_metrics_radar,
    create_account_breakdown_chart,
    WealthVisualizer
)
from .dashboards import (
    create_market_comparison_dashboard,
    create_client_dashboard
)
from .utils import (
    plot_to_json,
    json_to_plot
)

__all__ = [
    'create_portfolio_pie_chart',
    'create_sector_allocation_chart',
    'create_performance_comparison',
    'create_risk_metrics_radar',
    'create_account_breakdown_chart',
    'WealthVisualizer',
    'create_market_comparison_dashboard',
    'create_client_dashboard',
    'plot_to_json',
    'json_to_plot'
]
```

---

## File: `visualization\charts.py`

```python
import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List
from database.crud import get_client_details
class WealthVisualizer:
    """Wrapper class for all visualization functions"""
    
    @staticmethod
    def get_client_visualizations(client_id: str) -> dict:
        """Get all visualizations for a client"""
        client_data = get_client_details(client_id)
        if not client_data:
            return {}
            
        return {
            "portfolio_allocation": create_portfolio_pie_chart(client_id),
            "sector_allocation": create_sector_allocation_chart(client_id),
            "performance_comparison": create_performance_comparison(client_id),
            "risk_profile": create_risk_metrics_radar(client_id),
            "account_breakdown": create_account_breakdown_chart(client_id)
        }
def create_portfolio_pie_chart(client_id: str) -> Optional[go.Figure]:
    """Create portfolio allocation pie chart for client"""
    client = get_client_details(client_id)
    if not client or "portfolio_allocation" not in client:
        return None
        
    portfolio = client["portfolio_allocation"]
    
    fig = px.pie(
        values=list(portfolio.values()),
        names=list(portfolio.keys()),
        title=f"{client['name']} - Portfolio Allocation",
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    fig.update_layout(
        height=400,
        margin=dict(t=50, b=20)
    )
    
    return fig

def create_sector_allocation_chart(client_id: str) -> Optional[go.Figure]:
    """Create sector allocation bar chart for client"""
    client = get_client_details(client_id)
    if not client or "sector_allocation" not in client:
        return None
        
    sectors = client["sector_allocation"]
    
    fig = px.bar(
        x=list(sectors.values()),
        y=list(sectors.keys()),
        orientation='h',
        title=f"{client['name']} - Sector Allocation",
        color=list(sectors.values()),
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title="Allocation (%)",
        yaxis_title="Sectors",
        height=500,
        showlegend=False,
        margin=dict(l=150)
    )
    
    return fig

def create_performance_comparison(client_id: str) -> Optional[go.Figure]:
    """Create performance comparison line chart"""
    client = get_client_details(client_id)
    if not client or "historical_performance" not in client:
        return None
        
    perf_data = client["historical_performance"]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=perf_data["dates"],
        y=perf_data["portfolio_values"],
        name='Portfolio',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=perf_data["dates"],
        y=perf_data["benchmark_values"],
        name='Benchmark',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f"{client['name']} - Performance Comparison",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_risk_metrics_radar(client_id: str) -> Optional[go.Figure]:
    """Create risk metrics radar chart"""
    client = get_client_details(client_id)
    if not client or "risk_metrics" not in client:
        return None
        
    metrics = client["risk_metrics"]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(metrics.values()),
        theta=list(metrics.keys()),
        fill='toself',
        name="Risk Metrics"
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title=f"{client['name']} - Risk Profile",
        height=400
    )
    
    return fig

def create_account_breakdown_chart(client_id: str) -> Optional[go.Figure]:
    """Create account breakdown bar chart"""
    client = get_client_details(client_id)
    if not client or "accounts" not in client:
        return None
        
    accounts = client["accounts"]
    
    fig = px.bar(
        x=[acc["type"] for acc in accounts],
        y=[acc["balance"] for acc in accounts],
        title=f"{client['name']} - Account Breakdown",
        color=[acc["balance"] for acc in accounts],
        color_continuous_scale='blues'
    )
    
    fig.update_layout(
        xaxis_title="Account Type",
        yaxis_title="Balance ($)",
        yaxis_tickprefix='$',
        showlegend=False,
        height=400
    )
    
    return fig
```

---

## File: `visualization\dashboards.py`

```python
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from database.simulated_db import create_simulated_database
from visualization.charts import create_portfolio_pie_chart,create_performance_comparison,create_risk_metrics_radar,create_sector_allocation_chart
def create_market_comparison_dashboard() -> go.Figure:
    """Create comprehensive market comparison dashboard"""
    db = create_simulated_database()
    market_data = db["market_data"]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'YTD Returns', 
            'Volatility',
            'Risk-Return Profile',
            'Valuation Metrics'
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}]
        ]
    )
    
    # YTD Returns
    fig.add_trace(
        go.Bar(
            x=list(market_data.keys()),
            y=[m["ytd_return"] for m in market_data.values()],
            name="YTD Return (%)",
            marker_color='#1f77b4'
        ),
        row=1, col=1
    )
    
    # Volatility
    fig.add_trace(
        go.Bar(
            x=list(market_data.keys()),
            y=[m.get("volatility", 0) for m in market_data.values()],
            name="Volatility (%)",
            marker_color='#ff7f0e'
        ),
        row=1, col=2
    )
    
    # Risk-Return Profile
    fig.add_trace(
        go.Scatter(
            x=[m.get("volatility", 0) for m in market_data.values()],
            y=[m["ytd_return"] for m in market_data.values()],
            mode='markers+text',
            text=list(market_data.keys()),
            textposition="top center",
            marker=dict(size=16, color='#2ca02c'),
            name="Risk-Return"
        ),
        row=2, col=1
    )
    
    # Valuation Metrics
    fig.add_trace(
        go.Bar(
            x=list(market_data.keys()),
            y=[m.get("pe_ratio", m.get("yield", 0)) for m in market_data.values()],
            name="Valuation",
            marker_color='#9467bd'
        ),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text="Market Overview Dashboard",
        showlegend=False
    )
    
    return fig

def create_client_dashboard(client_id: str) -> go.Figure:
    """Create comprehensive dashboard for a client"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Portfolio Allocation',
            'Performance',
            'Risk Profile',
            'Sector Allocation'
        ),
        specs=[
            [{"type": "pie"}, {"type": "scatter"}],
            [{"type": "scatterpolar"}, {"type": "bar"}]
        ]
    )
    
    # Portfolio Allocation
    portfolio = create_portfolio_pie_chart(client_id)
    if portfolio:
        fig.add_trace(
            portfolio.data[0],
            row=1, col=1
        )
    
    # Performance
    performance = create_performance_comparison(client_id)
    if performance:
        for trace in performance.data:
            fig.add_trace(
                trace,
                row=1, col=2
            )
    
    # Risk Profile
    risk = create_risk_metrics_radar(client_id)
    if risk:
        fig.add_trace(
            risk.data[0],
            row=2, col=1
        )
    
    # Sector Allocation
    sector = create_sector_allocation_chart(client_id)
    if sector:
        fig.add_trace(
            sector.data[0],
            row=2, col=2
        )
    
    fig.update_layout(
        height=1000,
        title_text=f"Client {client_id} Dashboard",
        showlegend=True
    )
    
    return fig
```

---

## File: `visualization\utils.py`

```python
import json
import plotly
from typing import Optional
import plotly.graph_objects as go

def plot_to_json(fig: go.Figure) -> Optional[dict]:
    """Convert Plotly figure to JSON"""
    if fig is None:
        return None
    return json.loads(plotly.io.to_json(fig))

def json_to_plot(plot_json: dict) -> Optional[go.Figure]:
    """Convert JSON back to Plotly figure"""
    if not plot_json:
        return None
    
    try:
        return plotly.io.from_json(json.dumps(plot_json))
    except Exception:
        return None

def create_empty_figure(message: str = "No data available") -> go.Figure:
    """Create plot with message when data is missing"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16)
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig

def style_figure(fig: go.Figure, theme: str = "plotly_white") -> go.Figure:
    """Apply consistent styling to figure"""
    if not fig:
        return fig
    
    fig.update_layout(
        template=theme,
        margin=dict(l=50, r=50, b=50, t=50, pad=10),
        font=dict(family="Arial", size=12),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12
        )
    )
    return fig
```

---



> Total files processed: 38
