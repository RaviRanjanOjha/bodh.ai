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
