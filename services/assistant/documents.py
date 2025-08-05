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
        if not all(
            [
                settings.MONGO_URI,
                settings.MONGO_DOCS_DB_NAME,
                settings.MONGO_DOCS_COLLECTION,
            ]
        ):
            raise ValueError("Incomplete MongoDB configuration in settings")

        if not settings.GOOGLE_API_KEY:
            raise ValueError("Google API key missing in settings")

        self.embedding_service = DocumentEmbeddingService(
            mongo_uri=settings.MONGO_URI,
            db_name=settings.MONGO_DOCS_DB_NAME,
            collection_name=settings.MONGO_DOCS_COLLECTION,
            google_api_key=settings.GOOGLE_API_KEY,
        )
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
            google_api_key=settings.GOOGLE_API_KEY,
        )

    async def process_uploaded_files(
        self, files: List[UploadFile]
    ) -> Tuple[str, List[Tuple[str, str]]]:
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
                    if file.filename.lower().endswith(".pdf"):
                        suffix = ".pdf"
                        processor = self.embedding_service.process_pdf

                    elif file.filename.lower().endswith(".xlsx"):
                        suffix = ".xlsx"
                        processor = (
                            self.embedding_service.process_excel
                        )  # <-- We'll define this

                    else:
                        processed_results.append((file.filename, "Invalid file type"))
                        continue

                    # Secure temp file handling
                    with tempfile.NamedTemporaryFile(
                        suffix=".pdf", delete=False
                    ) as temp_file:
                        content = await file.read()
                        temp_file.write(content)
                        temp_files.append(temp_file.name)

                    # Process document
                    processor(temp_files[-1])
                    processed_results.append((file.filename, "Processed successfully"))

                except Exception as e:
                    logger.error(f"Processing failed for {file.filename}: {str(e)}")
                    processed_results.append(
                        (file.filename, f"Processing error: {str(e)}")
                    )

            return "Document processing completed", processed_results

        finally:
            # Guaranteed cleanup
            for temp_path in temp_files:
                try:
                    Path(temp_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.error(f"Temp file cleanup failed: {str(e)}")

    def get_all_excel_text(self):
        cursor = self.collection.find(
            {"type": "excel"},
            {"_id": 0, "metadata": 1, "page_content": 1},
        )
        return "\n".join(doc["page_content"] for doc in cursor)

    def query_documents(
        self, query: str, k: int = 1000
    ) -> Dict[str, Optional[List[Dict]]]:
        """Search documents and synthesize an answer"""
        response = {"answer": None, "documents": None, "error": None}

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
                formatted_docs.append(
                    {
                        "content": doc["content"],
                        "source": doc["metadata"].get("source", "Unknown"),
                        "page": doc["metadata"].get("page", "N/A"),
                        "score": doc.get("score", 0),
                    }
                )

            # Generate synthesized answer using self.llm
            context = "\n\n".join(
                [
                    f"Document {i+1} (Source: {doc['metadata'].get('source', 'Unknown')}, Page: {doc['metadata'].get('page', 'N/A')}):\n{doc['content']}"
                    for i, doc in enumerate(results)
                ]
            )

            prompt = f"""
                You are a financial data assistant.
                Answer the following question using the provided document excerpts.
                
                1. Give a clear, concise answer.
                2. Show a few top contributors or examples.
                3. Mention if negative or outlier values exist.
                4. Ask the user if they’d like to filter the data or download it.
                
                Documents:
                {context}
                
                Question: {query}
                
                Answer:
                """

            # Generate the answer using self.llm.invoke()
            synthesized_answer = self.llm.invoke(prompt).content

            response.update(
                {"answer": synthesized_answer.strip(), "documents": formatted_docs}
            )

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
            "index_status": None,
        }

        try:
            # Basic collection stats
            coll = self.embedding_service.collection
            diagnostics["embedding_count"] = coll.count_documents(
                {"embedding": {"$exists": True}}
            )

            # Get a sample embedded document
            sample = coll.find_one({"embedding": {"$exists": True}})
            if sample:
                diagnostics["sample_document"] = {
                    "source": sample.get("metadata", {}).get("source"),
                    "embedding_length": (
                        len(sample["embedding"]) if "embedding" in sample else 0
                    ),
                }

            # Check index existence (crude check)
            indexes = list(coll.list_indexes())
            diagnostics["index_status"] = any(
                idx.get("name") == self.embedding_service.INDEX_NAME for idx in indexes
            )

        except Exception as e:
            diagnostics["error"] = str(e)

        return diagnostics
