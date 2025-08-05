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