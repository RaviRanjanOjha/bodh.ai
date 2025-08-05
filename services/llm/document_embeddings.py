from pymongo import MongoClient
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_mongodb import MongoDBAtlasVectorSearch
from typing import List
from pathlib import Path
import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


class DocumentEmbeddingService:
    def __init__(
        self, mongo_uri: str, db_name: str, collection_name: str, google_api_key: str
    ):
        self.client = MongoClient(mongo_uri)
        self.db = self.client[db_name]
        self.collection = self.db[collection_name]
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001", google_api_key=google_api_key
        )

        # Add LLM for synthesizing answers
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", temperature=0.3, google_api_key=google_api_key
        )

        self.INDEX_NAME = "vertex_2"

    def generate_response_with_full_context(
        self, prompt: str, use_full_excel: bool = False
    ) -> str:
        if use_full_excel:
            full_text = self.get_all_excel_text()
            full_prompt = f"{full_text}\n\n{prompt}"
            response = self.llm.invoke(full_prompt).content
            return response.strip()
        else:
            return self.generate_response(prompt)

    def _ensure_vector_index(self):
        # Check if vector index exists, create if not
        if self.INDEX_NAME not in self.collection.list_indexes():
            self.collection.create_index(
                [("embedding", "knnVector")],
                knnVector={"dimensions": 768, "similarity": "cosine"},
                name=self.INDEX_NAME,
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
                chunk_size=1000, chunk_overlap=200
            )
            chunks = text_splitter.split_documents(pages)

            # Create embeddings
            vector_store = MongoDBAtlasVectorSearch.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                collection=self.collection,
                index_name=self.INDEX_NAME,
            )

            return vector_store

        except Exception as e:
            logger.error(f"Error processing PDF: {e}")
            raise

    def process_excel(self, filepath: str):
        import pandas as pd

        try:
            df = pd.read_excel(
                filepath, header=1
            )  # ensures first row is treated as header
            text = df.to_csv(index=False, float_format="%.2f")

            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000, chunk_overlap=0
            )
            documents = text_splitter.create_documents([text])

            # Add metadata
            for doc in documents:
                doc.metadata = {"source": filepath, "type": "excel"}

            # Create embeddings and store
            vector_store = MongoDBAtlasVectorSearch.from_documents(
                documents=documents,
                embedding=self.embedding_model,
                collection=self.collection,
                index_name=self.INDEX_NAME,
            )

            return vector_store

        except Exception as e:
            raise RuntimeError(f"Failed to process Excel file {filepath}: {str(e)}")

    def get_all_excel_text(self) -> str:
        """Fetch all Excel-type documents from MongoDB and merge text"""
        docs = self.collection.find({"type": "excel"})
        combined = "\n".join(
            doc.get("text", "") or doc.get("page_content", "") for doc in docs
        )
        return combined

    def search_documents(self, query: str, k: int = 1000) -> List[dict]:
        # need to update all rows from uploaded file/table
        """Search documents using vector similarity"""
        vector_store = MongoDBAtlasVectorSearch(
            collection=self.collection,
            embedding=self.embedding_model,
            index_name=self.INDEX_NAME,
        )

        results = vector_store.similarity_search(query, k=k)
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": doc.score if hasattr(doc, "score") else 0,
            }
            for doc in results
        ]

    def clear_documents(self):
        """Remove all documents from the collection"""
        self.collection.delete_many({})

    def __del__(self):
        self.client.close()
