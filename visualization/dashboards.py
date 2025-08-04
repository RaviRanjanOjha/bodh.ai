from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
from database.simulated_db import create_simulated_database
from visualization.charts import (
    create_portfolio_pie_chart,
    create_performance_comparison,
    create_risk_metrics_radar,
    create_sector_allocation_chart,
)


def create_market_comparison_dashboard() -> go.Figure:
    """Create comprehensive market comparison dashboard"""
    db = create_simulated_database()
    market_data = db["market_data"]

    # Create subplots
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "YTD Returns",
            "Volatility",
            "Risk-Return Profile",
            "Valuation Metrics",
        ),
        specs=[
            [{"type": "bar"}, {"type": "bar"}],
            [{"type": "scatter"}, {"type": "bar"}],
        ],
    )

    # YTD Returns
    fig.add_trace(
        go.Bar(
            x=list(market_data.keys()),
            y=[m["ytd_return"] for m in market_data.values()],
            name="YTD Return (%)",
            marker_color="#1f77b4",
        ),
        row=1,
        col=1,
    )

    # Volatility
    fig.add_trace(
        go.Bar(
            x=list(market_data.keys()),
            y=[m.get("volatility", 0) for m in market_data.values()],
            name="Volatility (%)",
            marker_color="#ff7f0e",
        ),
        row=1,
        col=2,
    )

    # Risk-Return Profile
    fig.add_trace(
        go.Scatter(
            x=[m.get("volatility", 0) for m in market_data.values()],
            y=[m["ytd_return"] for m in market_data.values()],
            mode="markers+text",
            text=list(market_data.keys()),
            textposition="top center",
            marker=dict(size=16, color="#2ca02c"),
            name="Risk-Return",
        ),
        row=2,
        col=1,
    )

    # Valuation Metrics
    fig.add_trace(
        go.Bar(
            x=list(market_data.keys()),
            y=[m.get("pe_ratio", m.get("yield", 0)) for m in market_data.values()],
            name="Valuation",
            marker_color="#9467bd",
        ),
        row=2,
        col=2,
    )

    fig.update_layout(
        height=800, title_text="Market Overview Dashboard", showlegend=False
    )

    return fig


def create_client_dashboard(client_id: str) -> go.Figure:
    """Create comprehensive dashboard for a client"""
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Portfolio Allocation",
            "Performance",
            "Risk Profile",
            "Sector Allocation",
        ),
        specs=[
            [{"type": "pie"}, {"type": "scatter"}],
            [{"type": "scatterpolar"}, {"type": "bar"}],
        ],
    )

    # Portfolio Allocation
    portfolio = create_portfolio_pie_chart(client_id)
    if portfolio:
        fig.add_trace(portfolio.data[0], row=1, col=1)

    # Performance
    performance = create_performance_comparison(client_id)
    if performance:
        for trace in performance.data:
            fig.add_trace(trace, row=1, col=2)

    # Risk Profile
    risk = create_risk_metrics_radar(client_id)
    if risk:
        fig.add_trace(risk.data[0], row=2, col=1)

    # Sector Allocation
    sector = create_sector_allocation_chart(client_id)
    if sector:
        fig.add_trace(sector.data[0], row=2, col=2)

    fig.update_layout(
        height=1000, title_text=f"Client {client_id} Dashboard", showlegend=True
    )

    return fig
