from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any, Optional
from datetime import datetime

class Account(BaseModel):
    id: str
    type: str
    balance: float

class Transaction(BaseModel):
    date: str
    amount: float
    type: str
    security: str

class ClientModel(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    id: str
    name: str
    net_worth: float
    risk_tolerance: str
    goals: List[str]
    portfolio: Dict[str, float]
    accounts: List[Account]
    advisors: List[str]
    last_review_date: datetime  # Change type hint
    historical_performance: Dict[str, Any]
    sector_allocation: Dict[str, float]
    transactions: Optional[List[Transaction]] = None

    def dict(self, *args, **kwargs):
        data = super().dict(*args, **kwargs)
        data["net_worth_formatted"] = f"${self.net_worth:,.2f}"
        return data