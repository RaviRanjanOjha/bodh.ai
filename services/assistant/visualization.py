from typing import Dict, Tuple, Optional, List
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from database.crud import get_client_details
import logging
import plotly.express as px
logger = logging.getLogger(__name__)

class VisualizationService:
    def __init__(self):
        self.client_colors = {
            "client_001": "#1f77b4",  # Blue
            "client_002": "#ff7f0e"   # Orange
        }
        self.chart_themes = {
            "primary": "plotly_white",
            "secondary": "ggplot2"
        }

    def get_client_visualizations(self, client_id: str) -> Dict[str, Optional[go.Figure]]:
        """
        Generate all standard visualizations for a client
        
        Args:
            client_id: Client identifier
            
        Returns:
            Dictionary of visualization figures with keys:
            - portfolio_allocation
            - sector_allocation
            - performance_comparison
            - risk_profile
            - account_breakdown
        """
        client_data = get_client_details(client_id)
        if not client_data:
            logger.error(f"No data found for client {client_id}")
            return {
                "portfolio_allocation": None,
                "sector_allocation": None,
                "performance_comparison": None,
                "risk_profile": None,
                "account_breakdown": None
            }

        try:
            figures = {}
            
            # Portfolio Allocation Pie Chart
            figures["portfolio_allocation"] = self._create_portfolio_allocation_chart(
                client_data["portfolio_allocation"],
                client_data["name"]
            )
            
            # Sector Allocation Bar Chart
            if "sector_allocation" in client_data:
                figures["sector_allocation"] = self._create_sector_allocation_chart(
                    client_data["sector_allocation"],
                    client_data["name"]
                )
            
            # Performance Comparison Line Chart
            if "historical_performance" in client_data:
                figures["performance_comparison"] = self._create_performance_chart(
                    client_data["historical_performance"],
                    client_data["name"]
                )
            
            # Risk Profile Radar Chart
            # Note: Requires risk metrics data not in basic client_data
            
            # Account Breakdown Chart
            figures["account_breakdown"] = self._create_account_breakdown_chart(
                client_data["accounts"],
                client_data["name"]
            )
            
            logger.info(f"Generated visualizations for client {client_id}")
            return figures
            
        except Exception as e:
            logger.error(f"Error generating visualizations for client {client_id}: {e}")
            return {
                "portfolio_allocation": None,
                "sector_allocation": None,
                "performance_comparison": None,
                "risk_profile": None,
                "account_breakdown": None
            }

    def create_custom_visualization(self, data: Dict, visualization_type: str, 
                                  client_context: Optional[Dict] = None) -> Optional[go.Figure]:
        """
        Create custom visualization based on request
        
        Args:
            data: Data payload for visualization
            visualization_type: One of ['pie', 'bar', 'line', 'radar', 'scatter']
            client_context: Optional client attributes for theming
            
        Returns:
            Configured Plotly figure or None on error
        """
        try:
            client_color = self._get_client_color(client_context)
            
            if visualization_type == "pie":
                return self._create_pie_chart(
                    data=data,
                    title=data.get("title", "Custom Pie Chart"),
                    color=client_color
                )
            elif visualization_type == "bar":
                return self._create_bar_chart(
                    data=data,
                    title=data.get("title", "Custom Bar Chart"),
                    color=client_color
                )
            elif visualization_type == "line":
                return self._create_line_chart(
                    data=data,
                    title=data.get("title", "Custom Line Chart"),
                    color=client_color
                )
            elif visualization_type == "radar":
                return self._create_radar_chart(
                    data=data,
                    title=data.get("title", "Custom Radar Chart")
                )
            elif visualization_type == "scatter":
                return self._create_scatter_plot(
                    data=data,
                    title=data.get("title", "Custom Scatter Plot"),
                    color=client_color
                )
            else:
                logger.warning(f"Unknown visualization type: {visualization_type}")
                return None
        except Exception as e:
            logger.error(f"Error creating custom visualization: {e}")
            return None

    def _create_portfolio_allocation_chart(self, portfolio_data: Dict, client_name: str) -> go.Figure:
        """Create pie chart for portfolio allocation"""
        labels = list(portfolio_data.keys())
        values = list(portfolio_data.values())
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.3,
            marker_colors=px.colors.qualitative.Pastel
        )])
        
        fig.update_layout(
            title=f"{client_name}'s Portfolio Allocation",
            height=400
        )
        return fig

    def _create_sector_allocation_chart(self, sector_data: Dict, client_name: str) -> go.Figure:
        """Create bar chart for sector allocation"""
        sectors = list(sector_data.keys())
        allocations = list(sector_data.values())
        
        fig = go.Figure(data=[go.Bar(
            x=allocations,
            y=sectors,
            orientation='h',
            marker_color=px.colors.sequential.Viridis
        )])
        
        fig.update_layout(
            title=f"{client_name}'s Sector Allocation",
            xaxis_title="Allocation (%)",
            height=500
        )
        return fig

    def _create_performance_chart(self, performance_data: Dict, client_name: str) -> go.Figure:
        """Create line chart for performance comparison"""
        dates = pd.to_datetime(performance_data["dates"])
        portfolio_values = performance_data["portfolio_values"]
        benchmark_values = performance_data["benchmark_values"]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=portfolio_values,
            mode='lines',
            name='Portfolio',
            line=dict(width=3)
        ))
        
        fig.add_trace(go.Scatter(
            x=dates,
            y=benchmark_values,
            mode='lines',
            name='Benchmark',
            line=dict(dash='dash')
        ))
        
        fig.update_layout(
            title=f"{client_name}'s Portfolio Performance",
            xaxis_title="Date",
            yaxis_title="Portfolio Value ($)",
            hovermode='x unified'
        )
        return fig

    def _create_account_breakdown_chart(self, accounts: List[Dict], client_name: str) -> go.Figure:
        """Create bar chart for account breakdown"""
        account_labels = [f"{acc['type']} (${acc['balance']:,})" for acc in accounts]
        balances = [acc["balance"] for acc in accounts]
        
        fig = go.Figure(data=[go.Bar(
            x=account_labels,
            y=balances,
            marker_color=px.colors.sequential.Blues
        )])
        
        fig.update_layout(
            title=f"{client_name}'s Account Breakdown",
            yaxis_title="Balance ($)",
            yaxis_tickprefix='$',
            yaxis_tickformat=',.0f'
        )
        return fig

    def _get_client_color(self, client_context: Optional[Dict]) -> str:
        """Get theme color for client visualizations"""
        if client_context and "id" in client_context:
            return self.client_colors.get(client_context["id"], "#636EFA")
        return "#636EFA"  # Default Plotly blue