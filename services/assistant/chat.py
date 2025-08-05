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
            mongo_uri=settings.MONGO_URI, db_name=settings.MONGO_DB_NAME
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
                logger.info(
                    f"Loaded {len(conversation_history)} messages from conversation history"
                )

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
            return (
                "I apologize, but I encountered an error processing your request.",
                True,
            )

    async def generate_response_stream(
        self, query: str, session_id: str = None
    ) -> AsyncGenerator[Tuple[str, bool], None]:
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

        for chunk in [
            mock_response[i : i + 20] for i in range(0, len(mock_response), 20)
        ]:
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

    def _handle_client_query(
        self, query: str, conversation_history: List[Dict] = None
    ) -> str:
        """Handle queries about specific clients"""
        logger.info(f"Handling client query with LLM: {query[:50]}...")
        try:
            # Use the LLM service to generate a proper response with client context
            response = self.llm_service.generate_response(query, conversation_history)
            logger.info(
                f"Client query LLM response generated: {len(response) if response else 0} characters"
            )
            return (
                response
                or "I'm unable to find information about that client. Please check the client name and try again."
            )
        except Exception as e:
            logger.error(f"Error in client query LLM generation: {e}")
            return "I apologize, but I encountered an error processing your client query. Please try again."

    def _handle_market_query(
        self, query: str, conversation_history: List[Dict] = None
    ) -> str:
        """Handle market-related queries"""
        # Build context from conversation history
        context_prompt = ""
        if conversation_history:
            recent_messages = (
                conversation_history[-6:]
                if len(conversation_history) > 6
                else conversation_history
            )
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
        return (
            self.llm_service.llm.invoke(prompt).content
            or "Unable to generate market analysis"
        )

    def _handle_general_query(
        self, query: str, conversation_history: List[Dict] = None
    ) -> str:
        """Handle general wealth management queries"""
        logger.info(f"Handling general query with LLM: {query[:50]}...")
        try:
            # Use the LLM service to generate a proper response
            response = self.llm_service.generate_response(query, conversation_history)
            logger.info(
                f"LLM response generated: {len(response) if response else 0} characters"
            )
            return (
                response
                or "I'm unable to process that request at the moment. Please try rephrasing your question."
            )
        except Exception as e:
            logger.error(f"Error in LLM generation: {e}")
            return "I apologize, but I encountered an error processing your request. Please try again."

    def save_conversation(self, session_id: str, messages: List[Dict]) -> str:
        """Save conversation to MongoDB"""
        logger.info(
            f"Saving conversation for session {session_id} with {len(messages)} messages"
        )
        try:
            # Use the conversation store to save
            result = self.conversation_store.save_conversation(
                session_id=session_id, messages=messages
            )
            if result:
                messages = self.conversation_store.conversations.find_one(
                    {"session_id": session_id}
                ).get("messages", [])

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
                msg
                for msg in messages[-10:]
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
                if len(recent_messages) >= 2:
                    user_message = next(
                        (
                            msg["content"]
                            for msg in reversed(recent_messages)
                            if msg["role"] == "user"
                        ),
                        "",
                    )[:50]
                    return f"Chat: {user_message}..." if user_message else "Discussion"
                return "Chat Discussion"

            return summary

        except Exception as e:
            logger.error(f"Error generating conversation summary: {e}")
            # Fallback based on last user message if available
            last_user_message = next(
                (
                    msg["content"]
                    for msg in reversed(messages)
                    if msg.get("role") == "user"
                ),
                None,
            )
            return (
                last_user_message[:50] + "..."
                if last_user_message
                else "Chat Discussion"
            )

    @property
    def capabilities(self) -> Dict[str, Any]:
        """Get capabilities of this assistant"""
        return {
            "query_types": ["client_specific", "market", "general"],
            "streaming": True,
            "document_processing": True,
            "compliance_checks": True,
        }

    def search_conversations(
        self, query: str, limit: int = 10
    ) -> List[Tuple[ConversationModel, Dict]]:
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
                    "theme": "light",
                },
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
        messages = (
            conversation.messages[-(limit * 2) :]
            if len(conversation.messages) > limit * 2
            else conversation.messages
        )

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
