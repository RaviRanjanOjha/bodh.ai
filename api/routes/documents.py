from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
from services.assistant.documents import DocumentService
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