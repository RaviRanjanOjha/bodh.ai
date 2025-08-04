import json
import plotly
from typing import Optional
import plotly.graph_objects as go

def plot_to_json(fig: go.Figure) -> Optional[dict]:
    """Convert Plotly figure to JSON"""
    if fig is None:
        return None
    return json.loads(plotly.io.to_json(fig))

def json_to_plot(plot_json: dict) -> Optional[go.Figure]:
    """Convert JSON back to Plotly figure"""
    if not plot_json:
        return None
    
    try:
        return plotly.io.from_json(json.dumps(plot_json))
    except Exception:
        return None

def create_empty_figure(message: str = "No data available") -> go.Figure:
    """Create plot with message when data is missing"""
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper", yref="paper",
        x=0.5, y=0.5,
        showarrow=False,
        font=dict(size=16)
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    return fig

def style_figure(fig: go.Figure, theme: str = "plotly_white") -> go.Figure:
    """Apply consistent styling to figure"""
    if not fig:
        return fig
    
    fig.update_layout(
        template=theme,
        margin=dict(l=50, r=50, b=50, t=50, pad=10),
        font=dict(family="Arial", size=12),
        hoverlabel=dict(
            bgcolor="white",
            font_size=12
        )
    )
    return fig