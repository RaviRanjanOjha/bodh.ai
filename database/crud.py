from typing import List, Dict, Any, Optional
from datetime import datetime
from database.models import ClientModel
from database.simulated_db import create_simulated_database

wealth_db = create_simulated_database()

def get_client_list() -> List[Dict[str, Any]]:
    """Get list of all clients with summary information"""
    clients = []
    
    for client_id, client_data in wealth_db["clients"].items():
        clients.append({
            "client_id": client_id,
            "name": client_data["name"],
            "net_worth": client_data["net_worth"],
            "risk_tolerance": client_data["risk_tolerance"],
            "advisor_count": len(client_data.get("advisors", [])),
            "last_review_date": client_data.get("last_review_date"),
            "portfolio_summary": ", ".join([
                f"{k}: {v}%" for k, v in client_data["portfolio"].items()
            ])
        })
        
    return clients

def get_client_details(client_id: str) -> Optional[Dict[str, Any]]:
    """Get comprehensive details for a specific client"""
    if client_id not in wealth_db["clients"]:
        return None
        
    client_data = wealth_db["clients"][client_id]
    
    formatted_data = {
        "name": client_data["name"],
        "net_worth": f"${client_data['net_worth']:,}",
        "risk_tolerance": client_data["risk_tolerance"],
        "goals": client_data["goals"],
        "portfolio_allocation": client_data["portfolio"],
        "accounts": client_data["accounts"],
        "advisors": client_data["advisors"],
        "last_review_date": client_data["last_review_date"],
        "transactions": client_data.get("transactions", []),
        "sector_allocation": client_data.get("sector_allocation", {})
    }
    
    if client_id in wealth_db["risk_metrics"]:
        formatted_data["risk_metrics"] = wealth_db["risk_metrics"][client_id]
        
    return formatted_data

def get_transactions(client_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent transactions for a client"""
    if client_id not in wealth_db["clients"]:
        return []
    
    transactions = wealth_db["clients"][client_id].get("transactions", [])
    return sorted(transactions, key=lambda x: x["date"], reverse=True)[:limit]