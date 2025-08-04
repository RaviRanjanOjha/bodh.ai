import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional
from datetime import datetime, timedelta
from database.models import ClientModel
from pymongo import MongoClient
from bson import ObjectId
import logging
from config import settings

logger = logging.getLogger(__name__)


class HybridDatabase:
    def __init__(self):
        self.mongo_connected = False
        self.mongo_db = None
        try:
            self.mongo_client = MongoClient(settings.MONGO_URI)
            self.mongo_client.admin.command("ping")
            self.mongo_db = self.mongo_client[settings.MONGO_DB_NAME]
            self.mongo_connected = True
            logger.info("Successfully connected to MongoDB")
        except Exception as e:
            logger.warning(
                f"MongoDB connection failed, using simulated data only: {str(e)}"
            )
            self.mongo_connected = False

    def get_clients(self) -> Dict[str, Any]:
        """Get clients from MongoDB or fallback to simulated data"""
        if self.mongo_connected:
            try:
                clients = {}
                for client in self.mongo_db.clients.find():
                    client_data = client.copy()
                    client_data["id"] = str(client_data.pop("_id"))
                    clients[client_data["id"]] = ClientModel(**client_data).dict()

                if clients:
                    logger.info("Successfully fetched clients from MongoDB")
                    return {"clients": clients}

            except Exception as e:
                logger.error(f"Error fetching from MongoDB: {e}")

        return self._create_simulated_data()

    def _create_simulated_data(self) -> Dict[str, Any]:
        """Create comprehensive simulated wealth management database"""
        logger.info("Using simulated database as fallback")

        dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="ME")

        clients = {
            "client_001": ClientModel(
                id="client_001",
                name="client",
                net_worth=4500000,
                risk_tolerance="Moderate",
                goals=["Retirement at 60", "College fund for grandchildren"],
                portfolio={"stocks": 55, "bonds": 30, "alternatives": 10, "cash": 5},
                accounts=[
                    {"id": "acct_1", "type": "Brokerage", "balance": 2800000},
                    {"id": "acct_2", "type": "IRA", "balance": 1200000},
                    {"id": "acct_3", "type": "Trust", "balance": 500000},
                ],
                advisors=["Sarah Johnson", "Michael Chen"],
                last_review_date=datetime.now() - timedelta(days=30),
                historical_performance={
                    "dates": [d.strftime("%Y-%m-%d") for d in dates],
                    "portfolio_values": [
                        4200000 + i * 25000 + np.random.normal(0, 50000)
                        for i in range(len(dates))
                    ],
                    "benchmark_values": [
                        4200000 + i * 20000 + np.random.normal(0, 40000)
                        for i in range(len(dates))
                    ],
                },
                sector_allocation={
                    "Technology": 25,
                    "Healthcare": 15,
                    "Financial Services": 12,
                    "Consumer Discretionary": 10,
                    "Energy": 8,
                    "Utilities": 7,
                    "Real Estate": 6,
                    "Materials": 5,
                    "Other": 12,
                },
                transactions=[
                    {
                        "date": (datetime.now() - timedelta(days=i * 7)).strftime(
                            "%Y-%m-%d"
                        ),
                        "amount": np.random.randint(10000, 50000),
                        "type": "buy" if i % 2 == 0 else "sell",
                        "security": f"Stock_{chr(65 + (i % 10))}",
                    }
                    for i in range(1, 13)
                ],
            ),
        }

        market_data = {
            "SP500": {
                "ytd_return": 12.4,
                "pe_ratio": 22.3,
                "volatility": 16.2,
                "last_updated": datetime.now().isoformat(),
            },
            "Bonds": {
                "ytd_return": 3.2,
                "duration": 6.5,
                "yield": 4.1,
                "last_updated": datetime.now().isoformat(),
            },
            "RealEstate": {
                "ytd_return": 5.7,
                "cap_rate": 4.2,
                "occupancy": 92.5,
                "last_updated": datetime.now().isoformat(),
            },
            "Commodities": {
                "ytd_return": -2.1,
                "volatility": 24.8,
                "last_updated": datetime.now().isoformat(),
            },
        }

        risk_metrics = {
            "client_001": {
                "sharpe_ratio": 1.45,
                "max_drawdown": -8.2,
                "beta": 1.15,
                "alpha": 2.3,
                "var_95": -125000,
                "volatility": 15.7,
                "calculated_on": datetime.now().isoformat(),
            },
            "client_002": {
                "sharpe_ratio": 0.98,
                "max_drawdown": -4.1,
                "beta": 0.65,
                "alpha": 1.1,
                "var_95": -45000,
                "volatility": 9.2,
                "calculated_on": datetime.now().isoformat(),
            },
        }

        market_events = [
            {
                "date": (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d"),
                "title": f"Market Event {i}",
                "impact": ["High", "Medium", "Low"][i % 3],
                "description": f"This is a simulated market event number {i}",
                "affected_sectors": [
                    ["Technology", "Healthcare"][i % 2],
                    ["Financial", "Energy"][(i + 1) % 2],
                ],
            }
            for i in range(10, 0, -1)
        ]

        return {
            "clients": {k: v.dict() for k, v in clients.items()},
            "market_data": market_data,
            "risk_metrics": risk_metrics,
            "market_events": market_events,
            "last_updated": datetime.now().isoformat(),
        }


hybrid_db = HybridDatabase()


def create_simulated_database() -> Dict[str, Any]:
    """Legacy function that now uses the hybrid approach"""
    return hybrid_db.get_clients()
