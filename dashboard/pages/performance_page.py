"""
Page 1: Live Performance Overview

Real-time performance monitoring with key metrics, charts, and alerts.
Displays current positions, P&L, Sharpe ratio, drawdown, and regime status.
"""

from dash import html, dcc, callback, Input, Output
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.performance import (
    calculate_comprehensive_metrics,
    calculate_cumulative_returns,
    calculate_max_drawdown,
    calculate_rolling_sharpe
)


def create_metric_card(label, value, subtitle=None, color_class=""):
    """Create a metric display card"""
    return dbc.Card(
        dbc.CardBody([
            html.Div(label, className="metric-label"),
            html.Div(value, className=f"metric-value {color_class}"),
            html.Div(subtitle, style={'color': '#8a9ba8', 'font-size': '0.8rem'}) if subtitle else None
        ]),
        className="metric-card"
    )


def create_performance_chart(returns_data):
    """Create cumulative returns and drawdown chart"""

    if returns_data is None or len(returns_data) == 0:
        # Return empty chart if no data
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='#0a0e27',
            plot_bgcolor='#1a1f3a',
            height=400,
            title="No data available"
        )
        return fig

    # Calculate cumulative returns and drawdown
    cum_returns = calculate_cumulative_returns(returns_data)
    _, drawdown = calculate_max_drawdown(returns_data)

    # Create subplots
    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.7, 0.3],
        subplot_titles=("Cumulative Returns", "Drawdown")
    )

    # Cumulative returns
    fig.add_trace(
        go.Scatter(
            x=cum_returns.index,
            y=cum_returns.values * 100,
            name="Cumulative Return",
            line=dict(color='#00d4ff', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.1)'
        ),
        row=1, col=1
    )

    # Drawdown
    fig.add_trace(
        go.Scatter(
            x=drawdown.index,
            y=drawdown.values * 100,
            name="Drawdown",
            line=dict(color='#ff4444', width=2),
            fill='tozeroy',
            fillcolor='rgba(255, 68, 68, 0.2)'
        ),
        row=2, col=1
    )

    # Add drawdown warning zones
    fig.add_hline(
        y=-10, line_dash="dash", line_color="#ffaa00", opacity=0.5,
        annotation_text="Warning (-10%)", annotation_position="right",
        row=2, col=1
    )
    fig.add_hline(
        y=-20, line_dash="dash", line_color="#ff4444", opacity=0.5,
        annotation_text="Critical (-20%)", annotation_position="right",
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0a0e27',
        plot_bgcolor='#1a1f3a',
        height=500,
        showlegend=False,
        hovermode='x unified',
        margin=dict(l=60, r=40, t=40, b=40)
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2a3f5f')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2a3f5f')

    fig.update_yaxes(title_text="Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    return fig


def create_rolling_sharpe_chart(returns_data, window=252):
    """Create rolling Sharpe ratio chart"""

    if returns_data is None or len(returns_data) < window:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='#0a0e27',
            plot_bgcolor='#1a1f3a',
            height=300,
            title="Insufficient data for rolling Sharpe"
        )
        return fig

    rolling_sharpe = calculate_rolling_sharpe(returns_data, window=window)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=rolling_sharpe.index,
        y=rolling_sharpe.values,
        name="Rolling Sharpe",
        line=dict(color='#00ff88', width=2),
        fill='tozeroy',
        fillcolor='rgba(0, 255, 136, 0.1)'
    ))

    # Add target line
    fig.add_hline(
        y=1.5, line_dash="dash", line_color="#00d4ff", opacity=0.7,
        annotation_text="Target (1.5)", annotation_position="right"
    )

    # Add warning line
    fig.add_hline(
        y=1.0, line_dash="dash", line_color="#ffaa00", opacity=0.5,
        annotation_text="Warning (1.0)", annotation_position="right"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0a0e27',
        plot_bgcolor='#1a1f3a',
        height=300,
        title="Rolling Sharpe Ratio (252-day)",
        showlegend=False,
        hovermode='x unified',
        margin=dict(l=60, r=40, t=60, b=40)
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2a3f5f')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2a3f5f', title_text="Sharpe Ratio")

    return fig


# Generate sample data for demonstration
def get_sample_data():
    """Generate sample returns data for demonstration"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=1000, freq='D')
    returns = pd.Series(
        np.random.normal(0.0008, 0.012, 1000),  # Slightly positive drift
        index=dates
    )
    return returns


# Layout
layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H2("📊 Live Performance Overview", style={'color': '#00d4ff', 'margin': '20px 0'}),
            html.P("Real-time monitoring of strategy performance and risk metrics",
                   style={'color': '#8a9ba8'})
        ])
    ]),

    # Status Bar with Regime and Alerts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div([
                        html.Span("Current Regime: ", style={'color': '#8a9ba8'}),
                        html.Span("BULL MARKET", className="regime-bull", id="current-regime"),
                    ], style={'display': 'inline-block', 'marginRight': '30px'}),
                    html.Div([
                        html.Span("Last Updated: ", style={'color': '#8a9ba8'}),
                        html.Span(datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                id="last-updated", style={'color': '#00d4ff'}),
                    ], style={'display': 'inline-block', 'marginRight': '30px'}),
                    html.Div([
                        html.Span("Active Alerts: ", style={'color': '#8a9ba8'}),
                        html.Span("0", className="alert-info", id="alert-count", style={'marginLeft': '10px'}),
                    ], style={'display': 'inline-block'}),
                ])
            ], className="mb-3")
        ])
    ]),

    # Key Metrics Row
    dbc.Row([
        dbc.Col(create_metric_card(
            "Sharpe Ratio",
            "1.95",
            "vs Target: 1.50",
            "positive"
        ), md=3),
        dbc.Col(create_metric_card(
            "Annual Return",
            "+18.7%",
            "YTD: +15.2%",
            "positive"
        ), md=3),
        dbc.Col(create_metric_card(
            "Max Drawdown",
            "-7.8%",
            "Limit: -10%",
            ""
        ), md=3),
        dbc.Col(create_metric_card(
            "Current Exposure",
            "85.0%",
            "15 positions",
            ""
        ), md=3),
    ], className="mb-4"),

    # Charts Row
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Performance Chart"),
                dbc.CardBody([
                    dcc.Graph(
                        id='performance-chart',
                        figure=create_performance_chart(get_sample_data()),
                        config={'displayModeBar': False}
                    )
                ])
            ])
        ], md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Rolling Sharpe"),
                dbc.CardBody([
                    dcc.Graph(
                        id='rolling-sharpe-chart',
                        figure=create_rolling_sharpe_chart(get_sample_data()),
                        config={'displayModeBar': False}
                    )
                ])
            ]),
            dbc.Card([
                dbc.CardHeader("Quick Stats"),
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.Span("Win Rate: ", style={'color': '#8a9ba8'}),
                            html.Span("54.3%", style={'color': '#00ff88', 'fontWeight': 'bold'})
                        ], className="mb-2"),
                        html.Div([
                            html.Span("Payoff Ratio: ", style={'color': '#8a9ba8'}),
                            html.Span("1.18", style={'color': '#00d4ff', 'fontWeight': 'bold'})
                        ], className="mb-2"),
                        html.Div([
                            html.Span("Volatility: ", style={'color': '#8a9ba8'}),
                            html.Span("15.5%", style={'color': '#ffaa00', 'fontWeight': 'bold'})
                        ], className="mb-2"),
                        html.Div([
                            html.Span("Calmar Ratio: ", style={'color': '#8a9ba8'}),
                            html.Span("2.40", style={'color': '#00ff88', 'fontWeight': 'bold'})
                        ]),
                    ])
                ])
            ], className="mt-3")
        ], md=4),
    ], className="mb-4"),

    # Positions Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Current Positions (Top 10)"),
                dbc.CardBody([
                    html.Div(id='positions-table', children=[
                        dbc.Table([
                            html.Thead(html.Tr([
                                html.Th("Symbol"),
                                html.Th("Weight"),
                                html.Th("Price"),
                                html.Th("P&L"),
                                html.Th("1D Return"),
                            ])),
                            html.Tbody([
                                html.Tr([
                                    html.Td("AAPL"),
                                    html.Td("8.5%"),
                                    html.Td("$185.42"),
                                    html.Td("+2.3%", style={'color': '#00ff88'}),
                                    html.Td("+1.2%", style={'color': '#00ff88'}),
                                ]),
                                html.Tr([
                                    html.Td("MSFT"),
                                    html.Td("7.2%"),
                                    html.Td("$378.91"),
                                    html.Td("+1.8%", style={'color': '#00ff88'}),
                                    html.Td("+0.8%", style={'color': '#00ff88'}),
                                ]),
                                html.Tr([
                                    html.Td("NVDA"),
                                    html.Td("6.8%"),
                                    html.Td("$495.22"),
                                    html.Td("-0.5%", style={'color': '#ff4444'}),
                                    html.Td("-0.3%", style={'color': '#ff4444'}),
                                ]),
                                # Add more sample rows...
                            ])
                        ], bordered=True, hover=True, responsive=True, color="dark")
                    ])
                ])
            ])
        ])
    ]),

    # Auto-refresh interval (every 5 seconds for demo)
    dcc.Interval(
        id='interval-component',
        interval=5*1000,  # in milliseconds
        n_intervals=0
    )

], fluid=True)


# Callbacks for real-time updates would go here
# @callback(...)
# def update_metrics(n_intervals):
#     # Fetch latest data from database
#     # Update metric cards, charts, positions table
#     pass
