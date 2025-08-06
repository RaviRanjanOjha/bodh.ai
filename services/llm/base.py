import os
from typing import Optional, Dict, Any, List
from langchain_google_genai import ChatGoogleGenerativeAI
from config import settings
import logging
import json
from services.assistant.documents import DocumentService

logger = logging.getLogger(__name__)

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
            google_api_key=settings.GOOGLE_API_KEY,
        )

        self.compliance_checked = False
        self.document_service = DocumentService()
        self.scope_description = """
        This assistant specializes in general topics topics including:
        - Total revenue
        - Quertly result
        - summerized report
        - sum of numbers in csv columns
        """

    def is_in_scope(self, query: str) -> bool:
        """Check if the query falls within general assistance scope"""
        # Make this much more lenient - only reject obviously off-topic queries
        out_of_scope_terms = [
            "weather",
            "sports",
            "entertainment",
            "politics",
            "cooking",
            "travel",
            "movies",
            "music",
            "games",
            "celebrities",
            "fashion",
            "technology news",
        ]

        query_lower = query.lower()

        # If it contains obvious out-of-scope terms, reject
        if any(term in query_lower for term in out_of_scope_terms):
            return False

        # If it contains any terms (like news, technology), accept
        finance_terms = [
            "revenue",
            "sum",
            "total",
        ]

        if any(term in query_lower for term in finance_terms):
            return True

        # For everything else (including general greetings), accept
        return True

    def query_wealth_db(self, query: str) -> str:
        db_context = create_simulated_database()

        # Simplified client data extraction
        client_data = {
            k: {
                "id": v["id"],
                "name": v["name"],
                **(
                    {"risk_tolerance": v.get("risk_tolerance")}
                    if "risk_tolerance" in v
                    else {}
                ),
                **({"portfolio": v.get("portfolio")} if "portfolio" in v else {}),
            }
            for k, v in wealth_db["clients"].items()
        }

        prompt = f"""
        You are a general professional assistant that can add any number and generate correct result. Answer the question based on this data:
        
        CLIENT DATA:
        {json.dumps(client_data, indent=2)}
        
        QUESTION: {query}
        
        Rules:
        1. Be concise but complete
        2. Add all numbers in the response
        3. Don't make up information not in the data
        
        Answer:
        """

        return self.llm.invoke(prompt).content.strip()

    def generate_response(
        self, query: str, conversation_history: List[Dict] = None
    ) -> str:
        """Generate response using conversation context"""
        if not self.is_in_scope(query):
            return "This prompt is out of scope."

        # Build context prompt
        context_prompt = ""
        if conversation_history:
            # Limit conversation history to last 6 messages (3 user-assistant exchanges)
            recent_messages = (
                conversation_history[-6:]
                if len(conversation_history) > 6
                else conversation_history
            )

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
        # db_response = self.query_wealth_db(query)
        doc_response = self.document_service.query_documents(query)
        sum_instruction = """
            Special Rule:
            - When the user asks to perform a "simple sum", extract all numeric values from the given column (or input list).
            - Add them **one-by-one**, treating each as a float or number.
            - Ignore invalid or non-numeric entries.
            - Use Excel formulas or libraries for this.
            - Show step-by-step if user asks, otherwise just show final result.
            """

        if context_prompt:
            prompt = f"""Your role: Professional general assistant
            {context_prompt}

    Available Information:
    - Documents: {doc_response}

    {sum_instruction}

    Please provide a concise, professional response answering the user's question based on all available context:"""
        else:
            prompt = f"""Your role: Professional general assistant
    Question: {query}

    Available Information:
    - Documents: {doc_response}

    {sum_instruction}

    Please provide a concise, professional response:"""

        return self.llm.invoke(prompt).content.strip()
