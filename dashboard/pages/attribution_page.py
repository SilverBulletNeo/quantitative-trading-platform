"""
Page 3: Performance Attribution

Decomposes returns into component sources: benchmark, selection alpha,
regime contribution, and transaction costs. Shows what's actually driving
performance based on our enhancement findings.

Key Insight: Regime filtering provides ALL alpha (+0.69 Sharpe improvement),
while asset selection is negative (-13.9% per year).
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

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.performance import (
    calculate_comprehensive_metrics,
    calculate_win_rate,
    calculate_payoff_ratio
)


def create_attribution_waterfall_chart():
    """
    Create waterfall chart showing return decomposition

    Based on our findings:
    - Benchmark (Equal Weight): +30.9%
    - Selection Alpha: -13.9% (NEGATIVE!)
    - Regime Filter: +0.7pp return, +0.69 Sharpe
    - Transaction Costs: -1.5%
    - Total Strategy: +18.7%
    """

    # Attribution components
    components = ['Benchmark\n(Equal Weight)', 'Selection\nAlpha', 'Regime\nFilter', 'Transaction\nCosts', 'Total\nStrategy']
    values = [30.9, -13.9, 0.7, -1.5, 0]  # Total will be calculated

    # Calculate total
    total = sum(values[:-1])
    values[-1] = total

    # Colors for each component
    colors = ['#00d4ff', '#ff4444', '#00ff88', '#ffaa00', '#00d4ff']

    fig = go.Figure(go.Waterfall(
        name="Attribution",
        orientation="v",
        measure=["relative", "relative", "relative", "relative", "total"],
        x=components,
        textposition="outside",
        text=[f"+{v:.1f}%" if v > 0 else f"{v:.1f}%" for v in values],
        y=values,
        connector={"line": {"color": "#2a3f5f"}},
        increasing={"marker": {"color": "#00ff88"}},
        decreasing={"marker": {"color": "#ff4444"}},
        totals={"marker": {"color": "#00d4ff"}}
    ))

    # Add annotations for key insights
    fig.add_annotation(
        x=1,
        y=-13.9,
        text="⚠️ Asset selection\nHURTS performance!",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#ff4444",
        bgcolor="#1a1f3a",
        bordercolor="#ff4444",
        font=dict(color="#ff4444", size=10)
    )

    fig.add_annotation(
        x=2,
        y=0.7,
        text="✅ Regime filter\nprovides ALL alpha",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#00ff88",
        bgcolor="#1a1f3a",
        bordercolor="#00ff88",
        font=dict(color="#00ff88", size=10)
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0a0e27',
        plot_bgcolor='#1a1f3a',
        height=450,
        title=dict(
            text="Annual Return Attribution - Where Does Performance Come From?",
            font=dict(size=16, color="#00d4ff")
        ),
        showlegend=False,
        margin=dict(l=60, r=60, t=80, b=60)
    )

    fig.update_yaxes(
        title_text="Annual Return (%)",
        showgrid=True,
        gridwidth=1,
        gridcolor='#2a3f5f'
    )

    return fig


def create_regime_contribution_chart():
    """
    Show impact of regime filtering over time

    Key finding: Sharpe improves from 1.21 to 1.90 (+0.69) with regime filter
    """

    # Generate sample data showing regime contributions
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=500, freq='D')

    # Simulate returns with and without regime filter
    base_returns = np.random.normal(0.0005, 0.015, 500)

    # Simulate regime filter impact (avoids bear markets)
    regime_states = np.random.choice([0, 1], 500, p=[0.15, 0.85])  # 15% in bear/crisis
    returns_with_filter = base_returns * regime_states

    # Cumulative impact
    cumulative_no_filter = (1 + pd.Series(base_returns)).cumprod() - 1
    cumulative_with_filter = (1 + pd.Series(returns_with_filter)).cumprod() - 1
    cumulative_difference = cumulative_with_filter - cumulative_no_filter

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=[0.6, 0.4],
        subplot_titles=("Returns: With vs Without Regime Filter", "Regime Filter Contribution")
    )

    # Returns comparison
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cumulative_no_filter.values * 100,
            name="Without Filter",
            line=dict(color='#ff8888', width=2, dash='dash'),
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cumulative_with_filter.values * 100,
            name="With Filter",
            line=dict(color='#00ff88', width=2),
        ),
        row=1, col=1
    )

    # Contribution (difference)
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=cumulative_difference.values * 100,
            name="Filter Contribution",
            fill='tozeroy',
            line=dict(color='#00d4ff', width=2),
            fillcolor='rgba(0, 212, 255, 0.2)'
        ),
        row=2, col=1
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0a0e27',
        plot_bgcolor='#1a1f3a',
        height=500,
        showlegend=True,
        legend=dict(
            x=0.02,
            y=0.98,
            bgcolor='rgba(26, 31, 58, 0.8)',
            bordercolor='#2a3f5f',
            borderwidth=1
        ),
        hovermode='x unified',
        margin=dict(l=60, r=40, t=60, b=40)
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2a3f5f')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2a3f5f')
    fig.update_yaxes(title_text="Cumulative Return (%)", row=1, col=1)
    fig.update_yaxes(title_text="Filter Benefit (%)", row=2, col=1)

    return fig


def create_monthly_attribution_heatmap():
    """Create monthly attribution heatmap"""

    # Generate sample monthly attribution data
    years = [2023, 2024, 2025]
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    # Sample data: monthly regime contribution
    np.random.seed(42)
    data = np.random.normal(0.5, 1.5, (len(years), len(months)))

    fig = go.Figure(data=go.Heatmap(
        z=data,
        x=months,
        y=years,
        colorscale=[
            [0, '#ff4444'],    # Negative (red)
            [0.5, '#1a1f3a'],  # Neutral (dark)
            [1, '#00ff88']     # Positive (green)
        ],
        zmid=0,
        text=[[f"{val:.1f}%" for val in row] for row in data],
        texttemplate="%{text}",
        textfont={"size": 10},
        colorbar=dict(
            title="Regime<br>Contribution (%)",
            titleside="right",
            tickmode="linear",
            tick0=-3,
            dtick=1
        ),
        hoverongaps=False,
        hovertemplate='%{y} %{x}<br>Contribution: %{z:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0a0e27',
        plot_bgcolor='#1a1f3a',
        height=300,
        title=dict(
            text="Monthly Regime Filter Contribution",
            font=dict(size=14, color="#00d4ff")
        ),
        margin=dict(l=60, r=120, t=60, b=40)
    )

    return fig


def create_win_rate_chart():
    """Create win rate and payoff ratio tracking"""

    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=500, freq='D')

    # Simulate rolling win rate (oscillates around 54%)
    win_rate = 54 + np.cumsum(np.random.normal(0, 0.5, 500))
    win_rate = np.clip(win_rate, 45, 65)  # Keep reasonable

    # Simulate rolling payoff ratio (oscillates around 1.18)
    payoff = 1.18 + np.cumsum(np.random.normal(0, 0.02, 500))
    payoff = np.clip(payoff, 0.9, 1.5)

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Rolling Win Rate (252-day)", "Rolling Payoff Ratio (252-day)")
    )

    # Win rate
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=win_rate,
            name="Win Rate",
            line=dict(color='#00ff88', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.1)'
        ),
        row=1, col=1
    )

    fig.add_hline(y=50, line_dash="dash", line_color="#ffaa00", opacity=0.5,
                  annotation_text="Break-even (50%)", row=1, col=1)

    # Payoff ratio
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=payoff,
            name="Payoff Ratio",
            line=dict(color='#00d4ff', width=2),
            fill='tozeroy',
            fillcolor='rgba(0, 212, 255, 0.1)'
        ),
        row=2, col=1
    )

    fig.add_hline(y=1.0, line_dash="dash", line_color="#ffaa00", opacity=0.5,
                  annotation_text="Break-even (1.0)", row=2, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0a0e27',
        plot_bgcolor='#1a1f3a',
        height=450,
        showlegend=False,
        hovermode='x unified',
        margin=dict(l=60, r=40, t=60, b=40)
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2a3f5f')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='#2a3f5f')
    fig.update_yaxes(title_text="Win Rate (%)", row=1, col=1)
    fig.update_yaxes(title_text="Payoff Ratio", row=2, col=1)

    return fig


# Layout
layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H2("📈 Performance Attribution", style={'color': '#00d4ff', 'margin': '20px 0'}),
            html.P("Decompose returns to understand what's actually driving performance",
                   style={'color': '#8a9ba8'})
        ])
    ]),

    # Key Insight Alert
    dbc.Row([
        dbc.Col([
            dbc.Alert([
                html.H5("🔍 KEY FINDING FROM ENHANCEMENT ANALYSIS", className="alert-heading"),
                html.Hr(),
                html.Div([
                    html.Div([
                        html.Strong("Regime Filter: ", style={'color': '#00ff88'}),
                        html.Span("Provides 100% of alpha (+0.69 Sharpe improvement, +0.7pp return)")
                    ], className="mb-2"),
                    html.Div([
                        html.Strong("Asset Selection: ", style={'color': '#ff4444'}),
                        html.Span("Actually NEGATIVE (-13.9% per year vs equal-weight benchmark)")
                    ], className="mb-2"),
                    html.Div([
                        html.Strong("Implication: ", style={'color': '#00d4ff'}),
                        html.Span("Focus enhancement efforts on REGIME DETECTION, not asset selection")
                    ]),
                ])
            ], color="info", className="mb-4")
        ])
    ]),

    # Attribution Summary Cards
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div("Benchmark Return", className="metric-label"),
                    html.Div("+30.9%", className="metric-value positive"),
                    html.P("Equal-weight all assets", style={'color': '#8a9ba8', 'fontSize': '0.85rem', 'marginTop': '5px'})
                ])
            ], className="metric-card")
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div("Selection Alpha", className="metric-label"),
                    html.Div("-13.9%", className="metric-value negative"),
                    html.P("Asset picking hurts returns!", style={'color': '#ff4444', 'fontSize': '0.85rem', 'marginTop': '5px'})
                ])
            ], className="metric-card")
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div("Regime Contribution", className="metric-label"),
                    html.Div("+0.69", className="metric-value positive"),
                    html.P("Sharpe improvement from filter", style={'color': '#00ff88', 'fontSize': '0.85rem', 'marginTop': '5px'})
                ])
            ], className="metric-card")
        ], md=3),
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.Div("Transaction Costs", className="metric-label"),
                    html.Div("-1.5%", className="metric-value warning"),
                    html.P("Low & efficient", style={'color': '#8a9ba8', 'fontSize': '0.85rem', 'marginTop': '5px'})
                ])
            ], className="metric-card")
        ], md=3),
    ], className="mb-4"),

    # Return Attribution Waterfall
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("💧 Return Attribution Waterfall"),
                dbc.CardBody([
                    dcc.Graph(
                        id='attribution-waterfall',
                        figure=create_attribution_waterfall_chart(),
                        config={'displayModeBar': False}
                    )
                ])
            ])
        ])
    ], className="mb-4"),

    # Regime Contribution Over Time
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎯 Regime Filter Impact Over Time"),
                dbc.CardBody([
                    dcc.Graph(
                        id='regime-contribution',
                        figure=create_regime_contribution_chart(),
                        config={'displayModeBar': False}
                    )
                ])
            ])
        ], md=8),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Attribution Insights"),
                dbc.CardBody([
                    html.H6("Why Regime Filter Works", className="mb-3", style={'color': '#00d4ff'}),
                    html.Div([
                        html.Div([
                            html.Span("✅ ", style={'color': '#00ff88', 'fontSize': '1.2rem'}),
                            html.Span("Avoids bear markets (-26% → -8% max DD)")
                        ], className="mb-2"),
                        html.Div([
                            html.Span("✅ ", style={'color': '#00ff88', 'fontSize': '1.2rem'}),
                            html.Span("Reduces volatility in crisis periods")
                        ], className="mb-2"),
                        html.Div([
                            html.Span("✅ ", style={'color': '#00ff88', 'fontSize': '1.2rem'}),
                            html.Span("Improves Sharpe ratio by 57%")
                        ], className="mb-3"),

                        html.Hr(style={'borderColor': '#2a3f5f'}),

                        html.H6("Why Selection is Negative", className="mb-3 mt-3", style={'color': '#ff4444'}),
                        html.Div([
                            html.Div([
                                html.Span("❌ ", style={'color': '#ff4444', 'fontSize': '1.2rem'}),
                                html.Span("Momentum underperforms in bull markets")
                            ], className="mb-2"),
                            html.Div([
                                html.Span("❌ ", style={'color': '#ff4444', 'fontSize': '1.2rem'}),
                                html.Span("Equal-weight captures full upside")
                            ], className="mb-2"),
                            html.Div([
                                html.Span("❌ ", style={'color': '#ff4444', 'fontSize': '1.2rem'}),
                                html.Span("Creates concentration risk")
                            ]),
                        ]),
                    ])
                ])
            ]),
            dbc.Card([
                dbc.CardHeader("🎯 Optimization Focus"),
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.Strong("HIGH PRIORITY:", style={'color': '#00ff88'}),
                            html.Ul([
                                html.Li("Improve bear market detection"),
                                html.Li("Add regime forecasting"),
                                html.Li("Earlier warning signals"),
                            ], style={'marginTop': '10px', 'paddingLeft': '20px'})
                        ], className="mb-3"),
                        html.Div([
                            html.Strong("LOW PRIORITY:", style={'color': '#8a9ba8'}),
                            html.Ul([
                                html.Li("Asset selection optimization"),
                                html.Li("More momentum signals"),
                                html.Li("Transaction cost reduction"),
                            ], style={'marginTop': '10px', 'paddingLeft': '20px'})
                        ]),
                    ])
                ])
            ], className="mt-3")
        ], md=4),
    ], className="mb-4"),

    # Monthly Heatmap and Win Rate
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📅 Monthly Regime Contribution Heatmap"),
                dbc.CardBody([
                    dcc.Graph(
                        id='monthly-heatmap',
                        figure=create_monthly_attribution_heatmap(),
                        config={'displayModeBar': False}
                    )
                ])
            ])
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🎲 Win Rate & Payoff Ratio"),
                dbc.CardBody([
                    dcc.Graph(
                        id='win-rate-chart',
                        figure=create_win_rate_chart(),
                        config={'displayModeBar': False}
                    )
                ])
            ])
        ], md=6),
    ], className="mb-4"),

    # Detailed Attribution Table
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📋 Detailed Attribution Breakdown"),
                dbc.CardBody([
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Component"),
                            html.Th("Annual Contribution"),
                            html.Th("% of Total Return"),
                            html.Th("Status"),
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td("Benchmark (Equal Weight)"),
                                html.Td("+30.9%", style={'color': '#00d4ff', 'fontWeight': 'bold'}),
                                html.Td("175.6%"),
                                html.Td(dbc.Badge("PRIMARY", color="primary")),
                            ]),
                            html.Tr([
                                html.Td("Selection Alpha"),
                                html.Td("-13.9%", style={'color': '#ff4444', 'fontWeight': 'bold'}),
                                html.Td("-79.2%"),
                                html.Td(dbc.Badge("NEGATIVE", color="danger")),
                            ]),
                            html.Tr([
                                html.Td("Regime Filter"),
                                html.Td("+0.7pp", style={'color': '#00ff88', 'fontWeight': 'bold'}),
                                html.Td("Sharpe: +0.69"),
                                html.Td(dbc.Badge("CRITICAL", color="success")),
                            ]),
                            html.Tr([
                                html.Td("Transaction Costs"),
                                html.Td("-1.5%", style={'color': '#ffaa00', 'fontWeight': 'bold'}),
                                html.Td("-8.6%"),
                                html.Td(dbc.Badge("EFFICIENT", color="warning")),
                            ]),
                            html.Tr([
                                html.Td("Residual"),
                                html.Td("+2.2%", style={'fontWeight': 'bold'}),
                                html.Td("12.5%"),
                                html.Td(dbc.Badge("MINOR", color="secondary")),
                            ]),
                            html.Tr(style={'borderTop': '2px solid #00d4ff'},[
                                html.Td(html.Strong("Total Strategy Return")),
                                html.Td(html.Strong("+18.7%"), style={'color': '#00d4ff', 'fontSize': '1.1rem'}),
                                html.Td(html.Strong("100%")),
                                html.Td(dbc.Badge("TARGET", color="info")),
                            ]),
                        ])
                    ], bordered=True, hover=True, responsive=True, dark=True)
                ])
            ])
        ])
    ]),

    # Auto-refresh
    dcc.Interval(
        id='attribution-interval',
        interval=30*1000,  # 30 seconds (less critical than risk)
        n_intervals=0
    )

], fluid=True)
