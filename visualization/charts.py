import plotly.express as px
import plotly.graph_objects as go
from typing import Optional, List
from database.crud import get_client_details
class WealthVisualizer:
    """Wrapper class for all visualization functions"""
    
    @staticmethod
    def get_client_visualizations(client_id: str) -> dict:
        """Get all visualizations for a client"""
        client_data = get_client_details(client_id)
        if not client_data:
            return {}
            
        return {
            "portfolio_allocation": create_portfolio_pie_chart(client_id),
            "sector_allocation": create_sector_allocation_chart(client_id),
            "performance_comparison": create_performance_comparison(client_id),
            "risk_profile": create_risk_metrics_radar(client_id),
            "account_breakdown": create_account_breakdown_chart(client_id)
        }
def create_portfolio_pie_chart(client_id: str) -> Optional[go.Figure]:
    """Create portfolio allocation pie chart for client"""
    client = get_client_details(client_id)
    if not client or "portfolio_allocation" not in client:
        return None
        
    portfolio = client["portfolio_allocation"]
    
    fig = px.pie(
        values=list(portfolio.values()),
        names=list(portfolio.keys()),
        title=f"{client['name']} - Portfolio Allocation",
        hole=0.3,
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    fig.update_layout(
        height=400,
        margin=dict(t=50, b=20)
    )
    
    return fig

def create_sector_allocation_chart(client_id: str) -> Optional[go.Figure]:
    """Create sector allocation bar chart for client"""
    client = get_client_details(client_id)
    if not client or "sector_allocation" not in client:
        return None
        
    sectors = client["sector_allocation"]
    
    fig = px.bar(
        x=list(sectors.values()),
        y=list(sectors.keys()),
        orientation='h',
        title=f"{client['name']} - Sector Allocation",
        color=list(sectors.values()),
        color_continuous_scale='viridis'
    )
    
    fig.update_layout(
        xaxis_title="Allocation (%)",
        yaxis_title="Sectors",
        height=500,
        showlegend=False,
        margin=dict(l=150)
    )
    
    return fig

def create_performance_comparison(client_id: str) -> Optional[go.Figure]:
    """Create performance comparison line chart"""
    client = get_client_details(client_id)
    if not client or "historical_performance" not in client:
        return None
        
    perf_data = client["historical_performance"]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=perf_data["dates"],
        y=perf_data["portfolio_values"],
        name='Portfolio',
        line=dict(color='blue', width=3)
    ))
    
    fig.add_trace(go.Scatter(
        x=perf_data["dates"],
        y=perf_data["benchmark_values"],
        name='Benchmark',
        line=dict(color='red', dash='dash')
    ))
    
    fig.update_layout(
        title=f"{client['name']} - Performance Comparison",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        hovermode='x unified',
        height=400
    )
    
    return fig

def create_risk_metrics_radar(client_id: str) -> Optional[go.Figure]:
    """Create risk metrics radar chart"""
    client = get_client_details(client_id)
    if not client or "risk_metrics" not in client:
        return None
        
    metrics = client["risk_metrics"]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=list(metrics.values()),
        theta=list(metrics.keys()),
        fill='toself',
        name="Risk Metrics"
    ))
    
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title=f"{client['name']} - Risk Profile",
        height=400
    )
    
    return fig

def create_account_breakdown_chart(client_id: str) -> Optional[go.Figure]:
    """Create account breakdown bar chart"""
    client = get_client_details(client_id)
    if not client or "accounts" not in client:
        return None
        
    accounts = client["accounts"]
    
    fig = px.bar(
        x=[acc["type"] for acc in accounts],
        y=[acc["balance"] for acc in accounts],
        title=f"{client['name']} - Account Breakdown",
        color=[acc["balance"] for acc in accounts],
        color_continuous_scale='blues'
    )
    
    fig.update_layout(
        xaxis_title="Account Type",
        yaxis_title="Balance ($)",
        yaxis_tickprefix='$',
        showlegend=False,
        height=400
    )
    
    return fig