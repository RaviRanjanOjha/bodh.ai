from fastapi import APIRouter, HTTPException
from datetime import datetime

from api.schemas import MarketDataResponse
from visualization.dashboards import create_market_comparison_dashboard as create_market_dashboard
from api.exceptions import MarketDataException
from visualization.utils import plot_to_json  

router = APIRouter(prefix="/market", tags=["Market"])

@router.get("", response_model=MarketDataResponse)
def get_market_data():
    try:
        dashboard = create_market_dashboard()
        dashboard_data=plot_to_json(dashboard)
        return {
            "dashboard": dashboard_data,
            "as_of": datetime.now().isoformat()
        }
    except MarketDataException as e:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))