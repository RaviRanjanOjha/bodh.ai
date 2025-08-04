from fastapi import APIRouter, HTTPException
from datetime import datetime
from visualization.utils import plot_to_json  # Add this import
import json

from api.schemas import VisualizationResponse
from visualization.charts import WealthVisualizer
from database.crud import get_client_details
from api.exceptions import ClientNotFoundException

router = APIRouter(prefix="/visualizations", tags=["Visualizations"])

@router.get("/client/{client_id}", response_model=VisualizationResponse)
def get_client_visualizations(client_id: str):
    try:
        client_data = get_client_details(client_id)
        if not client_data:
            raise ClientNotFoundException(client_id)
            
        visualizer = WealthVisualizer()
        visualizations = visualizer.get_client_visualizations(client_id)
        
        if not visualizations:
            raise HTTPException(
                status_code=404,
                detail=f"No visualization data found for client {client_id}"
            )
            
        # Convert all figures to JSON
        visualization_data = {
            key: plot_to_json(fig) if fig is not None else None
            for key, fig in visualizations.items()
        }
        
        return {
            "client": client_data["name"],
            "visualizations": visualization_data
        }
        
    except ClientNotFoundException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate visualizations: {str(e)}"
        )