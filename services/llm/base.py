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

        # If it contains wealth/finance terms, accept
        finance_terms = [
            "portfolio",
            "investment",
            "client",
            "money",
            "fund",
            "stock",
            "bond",
            "retirement",
            "wealth",
            "financial",
            "asset",
            "risk",
            "return",
            "market",
            "account",
            "balance",
            "allocation",
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
        You are a UXM RFP response assistant. Answer the question based on this data:
        
        CLIENT DATA:
        {json.dumps(client_data, indent=2)}
        
        QUESTION: {query}
        
        Rules:
        1. Be concise but complete
        2. Include client names when relevant
        3. Don't make up information not in the data
        
        Answer:
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

        if context_prompt:
            prompt = f"""Your role: Professional wealth management assistant
    {context_prompt}

    Available Information:
    - Documents: {doc_response}

    Please provide a concise, professional response answering the user's question based on all available context:"""
        else:
            prompt = f"""Your role: Professional wealth management assistant
    Question: {query}

    Available Information:
    - Documents: {doc_response}

    Please provide a concise, professional response:"""

        return self.llm.invoke(prompt).content.strip()
