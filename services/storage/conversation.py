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