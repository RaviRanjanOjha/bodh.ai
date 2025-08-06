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
        that a general assistant would ask to continue the discussion. The questions should be:
        - Relevant to the user's needs
        - Summarization of the data
        - Formatted as a JSON list of strings like ["question1", "question2", "question3"]

        Return ONLY the JSON array, nothing else.

        Conversation:
        {context}

        Example response:
        ["What is total revenue of FY25 quarter 1?", "can you generate a report for FY25?", "Can you show me project wise revenue for FY25?"]
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
                "Could you please tell me project wise revenue?",
                "Are you considering any changes to next financial year based on previous yaer revenue records?",
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