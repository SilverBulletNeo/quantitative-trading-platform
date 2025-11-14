"""
Page 2: Risk Monitoring

Comprehensive risk tracking with drawdown monitoring, VaR/CVaR analysis,
alert management, and circuit breaker status.
"""

from dash import html, dcc, callback, Input, Output, State
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
    calculate_max_drawdown,
    calculate_var,
    calculate_cvar,
    calculate_volatility,
    calculate_underwater
)


def create_risk_metric_card(label, value, status, subtitle=None):
    """Create a risk metric display card with status indicator"""

    # Determine color based on status
    if status == "good":
        color_class = "positive"
        badge_color = "success"
    elif status == "warning":
        color_class = "warning"
        badge_color = "warning"
    else:  # critical
        color_class = "negative"
        badge_color = "danger"

    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Span(label, className="metric-label"),
                dbc.Badge(status.upper(), color=badge_color, className="ms-2")
            ], style={'marginBottom': '10px'}),
            html.Div(value, className=f"metric-value {color_class}"),
            html.Div(subtitle, style={'color': '#8a9ba8', 'font-size': '0.8rem', 'marginTop': '5px'}) if subtitle else None
        ]),
        className="metric-card"
    )


def create_drawdown_tracker_chart(returns_data):
    """Create comprehensive drawdown tracking chart"""

    if returns_data is None or len(returns_data) == 0:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='#0a0e27',
            plot_bgcolor='#1a1f3a',
            height=400,
            title="No data available"
        )
        return fig

    # Calculate drawdown
    underwater = calculate_underwater(returns_data)

    # Create color array based on drawdown severity
    colors = []
    for dd in underwater.values:
        if dd > -0.10:
            colors.append('rgba(0, 255, 136, 0.6)')  # Green - safe
        elif dd > -0.15:
            colors.append('rgba(255, 170, 0, 0.6)')  # Amber - warning
        else:
            colors.append('rgba(255, 68, 68, 0.8)')  # Red - critical

    fig = go.Figure()

    # Drawdown area with color gradient
    fig.add_trace(go.Scatter(
        x=underwater.index,
        y=underwater.values * 100,
        name="Drawdown",
        fill='tozeroy',
        line=dict(color='#00d4ff', width=2),
        fillcolor='rgba(0, 212, 255, 0.2)',
        hovertemplate='Date: %{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
    ))

    # Warning zones
    fig.add_hline(
        y=-10, line_dash="dash", line_color="#ffaa00", opacity=0.7,
        annotation_text="⚠️ WARNING ZONE (-10%)",
        annotation_position="left",
        annotation=dict(font=dict(size=10, color="#ffaa00"))
    )

    fig.add_hline(
        y=-15, line_dash="dash", line_color="#ff8800", opacity=0.7,
        annotation_text="⚠️ ELEVATED RISK (-15%)",
        annotation_position="left",
        annotation=dict(font=dict(size=10, color="#ff8800"))
    )

    fig.add_hline(
        y=-20, line_dash="dash", line_color="#ff4444", opacity=0.9,
        annotation_text="🛑 CIRCUIT BREAKER (-20%)",
        annotation_position="left",
        annotation=dict(font=dict(size=10, color="#ff4444", weight='bold'))
    )

    # Add current drawdown annotation
    current_dd = underwater.iloc[-1] * 100
    fig.add_annotation(
        x=underwater.index[-1],
        y=current_dd,
        text=f"Current: {current_dd:.2f}%",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#00d4ff",
        bgcolor="#1a1f3a",
        bordercolor="#00d4ff",
        font=dict(color="#00d4ff", size=12)
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0a0e27',
        plot_bgcolor='#1a1f3a',
        height=450,
        title=dict(
            text="Drawdown Tracker - Real-Time Monitoring",
            font=dict(size=16, color="#00d4ff")
        ),
        showlegend=False,
        hovermode='x unified',
        margin=dict(l=60, r=40, t=60, b=40)
    )

    fig.update_xaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#2a3f5f',
        title_text="Date"
    )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#2a3f5f',
        title_text="Drawdown (%)",
        range=[-25, 2]  # Fixed range for consistency
    )

    return fig


def create_var_cvar_chart(returns_data):
    """Create VaR and CVaR tracking chart"""

    if returns_data is None or len(returns_data) < 252:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='#0a0e27',
            plot_bgcolor='#1a1f3a',
            height=350,
            title="Insufficient data for VaR/CVaR"
        )
        return fig

    # Calculate rolling VaR and CVaR
    window = 252  # 1 year
    rolling_var_95 = []
    rolling_cvar_95 = []
    rolling_var_99 = []

    for i in range(window, len(returns_data) + 1):
        window_returns = returns_data.iloc[i-window:i]
        rolling_var_95.append(calculate_var(window_returns, 0.95))
        rolling_cvar_95.append(calculate_cvar(window_returns, 0.95))
        rolling_var_99.append(calculate_var(window_returns, 0.99))

    dates = returns_data.index[window-1:]

    fig = go.Figure()

    # VaR 95%
    fig.add_trace(go.Scatter(
        x=dates,
        y=np.array(rolling_var_95) * 100,
        name="VaR 95%",
        line=dict(color='#ffaa00', width=2),
        hovertemplate='VaR 95%: %{y:.2f}%<extra></extra>'
    ))

    # VaR 99%
    fig.add_trace(go.Scatter(
        x=dates,
        y=np.array(rolling_var_99) * 100,
        name="VaR 99%",
        line=dict(color='#ff4444', width=2, dash='dash'),
        hovertemplate='VaR 99%: %{y:.2f}%<extra></extra>'
    ))

    # CVaR 95%
    fig.add_trace(go.Scatter(
        x=dates,
        y=np.array(rolling_cvar_95) * 100,
        name="CVaR 95%",
        line=dict(color='#ff8888', width=2, dash='dot'),
        fill='tonexty',
        fillcolor='rgba(255, 68, 68, 0.1)',
        hovertemplate='CVaR 95%: %{y:.2f}%<extra></extra>'
    ))

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0a0e27',
        plot_bgcolor='#1a1f3a',
        height=350,
        title=dict(
            text="Value at Risk (VaR) & Expected Shortfall (CVaR)",
            font=dict(size=14, color="#00d4ff")
        ),
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
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#2a3f5f',
        title_text="Loss (%)"
    )

    return fig


def create_volatility_chart(returns_data):
    """Create rolling volatility chart"""

    if returns_data is None or len(returns_data) < 60:
        fig = go.Figure()
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor='#0a0e27',
            plot_bgcolor='#1a1f3a',
            height=250,
            title="Insufficient data"
        )
        return fig

    # Calculate rolling volatility at different windows
    vol_30d = returns_data.rolling(30).std() * np.sqrt(252) * 100
    vol_60d = returns_data.rolling(60).std() * np.sqrt(252) * 100
    vol_252d = returns_data.rolling(252).std() * np.sqrt(252) * 100

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=vol_30d.index,
        y=vol_30d.values,
        name="30-day",
        line=dict(color='#00d4ff', width=1.5),
        opacity=0.7
    ))

    fig.add_trace(go.Scatter(
        x=vol_252d.index,
        y=vol_252d.values,
        name="252-day",
        line=dict(color='#00ff88', width=2)
    ))

    # Target volatility line
    fig.add_hline(
        y=15, line_dash="dash", line_color="#ffaa00", opacity=0.5,
        annotation_text="Target (15%)", annotation_position="right"
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor='#0a0e27',
        plot_bgcolor='#1a1f3a',
        height=250,
        title=dict(
            text="Rolling Volatility",
            font=dict(size=14, color="#00d4ff")
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            x=0.5,
            xanchor="center",
            y=1.15,
            bgcolor='rgba(26, 31, 58, 0.8)'
        ),
        hovermode='x unified',
        margin=dict(l=60, r=40, t=40, b=40)
    )

    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='#2a3f5f')
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor='#2a3f5f',
        title_text="Volatility (%)"
    )

    return fig


def create_alert_card(severity, category, message, timestamp, alert_id):
    """Create an alert display card"""

    # Severity colors
    severity_colors = {
        'CRITICAL': 'danger',
        'WARNING': 'warning',
        'INFO': 'info'
    }

    severity_icons = {
        'CRITICAL': '🛑',
        'WARNING': '⚠️',
        'INFO': 'ℹ️'
    }

    return dbc.Card(
        dbc.CardBody([
            html.Div([
                html.Span(
                    f"{severity_icons.get(severity, '•')} {severity}",
                    className=f"alert-{severity.lower()}",
                    style={'marginRight': '15px', 'fontSize': '0.9rem', 'fontWeight': 'bold'}
                ),
                dbc.Badge(category, color="secondary", className="me-2"),
                html.Span(
                    timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                    style={'color': '#8a9ba8', 'fontSize': '0.8rem'}
                ),
            ], style={'marginBottom': '10px'}),
            html.P(message, style={'color': '#e0e0e0', 'marginBottom': '10px'}),
            html.Div([
                dbc.Button("Acknowledge", size="sm", color="primary", outline=True, id=f"ack-{alert_id}", className="me-2"),
                dbc.Button("Resolve", size="sm", color="success", outline=True, id=f"resolve-{alert_id}"),
            ])
        ]),
        className="mb-2",
        style={'borderLeft': f"4px solid {severity_colors.get(severity, 'gray')}"}
    )


def get_sample_data():
    """Generate sample returns data"""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=1000, freq='D')
    returns = pd.Series(
        np.random.normal(0.0008, 0.012, 1000),
        index=dates
    )
    return returns


def generate_sample_alerts():
    """Generate sample alerts for demonstration"""
    return [
        {
            'id': 1,
            'severity': 'WARNING',
            'category': 'DRAWDOWN',
            'message': 'Current drawdown (-8.2%) approaching warning threshold (-10%)',
            'timestamp': datetime.now() - timedelta(hours=2)
        },
        {
            'id': 2,
            'severity': 'INFO',
            'category': 'REGIME',
            'message': 'Market regime changed from BULL to SIDEWAYS',
            'timestamp': datetime.now() - timedelta(hours=5)
        },
    ]


# Layout
layout = dbc.Container([
    # Header
    dbc.Row([
        dbc.Col([
            html.H2("⚠️ Risk Monitoring", style={'color': '#00d4ff', 'margin': '20px 0'}),
            html.P("Real-time risk tracking with drawdown monitoring, VaR/CVaR analysis, and alert management",
                   style={'color': '#8a9ba8'})
        ])
    ]),

    # Circuit Breaker Status Banner
    dbc.Row([
        dbc.Col([
            dbc.Alert([
                html.Div([
                    html.H5("🟢 CIRCUIT BREAKER STATUS: ACTIVE", className="alert-heading mb-0", style={'display': 'inline-block'}),
                    html.Span(" | ", style={'margin': '0 15px', 'color': '#666'}),
                    html.Span("All systems operational", style={'fontSize': '0.9rem'}),
                    html.Span(" | ", style={'margin': '0 15px', 'color': '#666'}),
                    html.Span("Drawdown: -7.8% / -20% limit", style={'fontSize': '0.9rem'}),
                ], style={'textAlign': 'center'})
            ], color="success", className="mb-4")
        ])
    ]),

    # Risk Metrics Cards
    dbc.Row([
        dbc.Col(create_risk_metric_card(
            "Current Drawdown",
            "-7.8%",
            "good",
            "Limit: -10% warning, -20% halt"
        ), md=3),
        dbc.Col(create_risk_metric_card(
            "VaR (95%)",
            "-1.48%",
            "good",
            "Expected daily loss (95% confidence)"
        ), md=3),
        dbc.Col(create_risk_metric_card(
            "CVaR (95%)",
            "-1.88%",
            "good",
            "Average loss in worst 5% of days"
        ), md=3),
        dbc.Col(create_risk_metric_card(
            "Volatility",
            "15.5%",
            "good",
            "Target: 15% annual"
        ), md=3),
    ], className="mb-4"),

    # Drawdown Tracker (Full Width)
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📉 Drawdown Tracker"),
                dbc.CardBody([
                    dcc.Graph(
                        id='drawdown-tracker-chart',
                        figure=create_drawdown_tracker_chart(get_sample_data()),
                        config={'displayModeBar': False}
                    )
                ])
            ])
        ])
    ], className="mb-4"),

    # VaR/CVaR and Volatility Charts
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Value at Risk (VaR) & Expected Shortfall"),
                dbc.CardBody([
                    dcc.Graph(
                        id='var-cvar-chart',
                        figure=create_var_cvar_chart(get_sample_data()),
                        config={'displayModeBar': False}
                    )
                ])
            ])
        ], md=7),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📈 Rolling Volatility"),
                dbc.CardBody([
                    dcc.Graph(
                        id='volatility-chart',
                        figure=create_volatility_chart(get_sample_data()),
                        config={'displayModeBar': False}
                    )
                ])
            ]),
            dbc.Card([
                dbc.CardHeader("🎯 Risk Limits"),
                dbc.CardBody([
                    html.Div([
                        html.Div([
                            html.Span("Max Drawdown Limit: ", style={'color': '#8a9ba8'}),
                            html.Span("-20%", style={'color': '#ff4444', 'fontWeight': 'bold', 'fontSize': '1.2rem'})
                        ], className="mb-2"),
                        html.Div([
                            html.Span("Current Proximity: ", style={'color': '#8a9ba8'}),
                            html.Span("39% to limit", style={'color': '#00ff88', 'fontWeight': 'bold'})
                        ], className="mb-2"),
                        html.Hr(style={'borderColor': '#2a3f5f'}),
                        html.Div([
                            html.Span("Volatility Target: ", style={'color': '#8a9ba8'}),
                            html.Span("15%", style={'color': '#00d4ff', 'fontWeight': 'bold'})
                        ], className="mb-2"),
                        html.Div([
                            html.Span("Current: ", style={'color': '#8a9ba8'}),
                            html.Span("15.5%", style={'color': '#ffaa00', 'fontWeight': 'bold'})
                        ]),
                    ])
                ])
            ], className="mt-3")
        ], md=5),
    ], className="mb-4"),

    # Active Alerts Section
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader([
                    html.Span("🔔 Active Alerts"),
                    dbc.Badge("2", color="warning", className="ms-2")
                ]),
                dbc.CardBody([
                    html.Div(
                        id='alerts-container',
                        children=[
                            create_alert_card(
                                alert['severity'],
                                alert['category'],
                                alert['message'],
                                alert['timestamp'],
                                alert['id']
                            ) for alert in generate_sample_alerts()
                        ]
                    ),
                    html.Div(
                        "No active alerts",
                        id='no-alerts-message',
                        style={'display': 'none', 'color': '#8a9ba8', 'textAlign': 'center', 'padding': '20px'}
                    )
                ])
            ])
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📋 Alert History (Last 24h)"),
                dbc.CardBody([
                    dbc.Table([
                        html.Thead(html.Tr([
                            html.Th("Time"),
                            html.Th("Severity"),
                            html.Th("Category"),
                            html.Th("Message"),
                        ])),
                        html.Tbody([
                            html.Tr([
                                html.Td("18:34", style={'fontSize': '0.85rem'}),
                                html.Td(dbc.Badge("INFO", color="info")),
                                html.Td("REGIME"),
                                html.Td("Regime: BULL → SIDEWAYS", style={'fontSize': '0.85rem'}),
                            ]),
                            html.Tr([
                                html.Td("15:22", style={'fontSize': '0.85rem'}),
                                html.Td(dbc.Badge("WARNING", color="warning")),
                                html.Td("DRAWDOWN"),
                                html.Td("DD approaching -10%", style={'fontSize': '0.85rem'}),
                            ]),
                            html.Tr([
                                html.Td("12:05", style={'fontSize': '0.85rem'}),
                                html.Td(dbc.Badge("INFO", color="info")),
                                html.Td("SYSTEM"),
                                html.Td("Data refresh completed", style={'fontSize': '0.85rem'}),
                            ]),
                        ])
                    ], bordered=True, hover=True, responsive=True, color="dark", size="sm")
                ])
            ])
        ], md=6),
    ], className="mb-4"),

    # Risk Statistics Summary
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Risk Statistics Summary"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.H6("Drawdown Metrics", className="text-muted mb-3"),
                            html.Div([
                                html.Div([
                                    html.Span("Max Drawdown: "),
                                    html.Strong("-18.2%", style={'color': '#ff4444'})
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("Avg Drawdown: "),
                                    html.Strong("-5.3%")
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("Days in Drawdown: "),
                                    html.Strong("127 of 1000", style={'color': '#ffaa00'})
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("Longest Drawdown: "),
                                    html.Strong("45 days")
                                ], className="mb-1"),
                            ])
                        ], md=4),
                        dbc.Col([
                            html.H6("Risk Measures", className="text-muted mb-3"),
                            html.Div([
                                html.Div([
                                    html.Span("VaR 95% (daily): "),
                                    html.Strong("-1.48%", style={'color': '#ffaa00'})
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("VaR 99% (daily): "),
                                    html.Strong("-2.15%", style={'color': '#ff4444'})
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("CVaR 95%: "),
                                    html.Strong("-1.88%", style={'color': '#ff8888'})
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("Downside Deviation: "),
                                    html.Strong("10.2%")
                                ], className="mb-1"),
                            ])
                        ], md=4),
                        dbc.Col([
                            html.H6("Volatility Metrics", className="text-muted mb-3"),
                            html.Div([
                                html.Div([
                                    html.Span("30-day Vol: "),
                                    html.Strong("16.2%", style={'color': '#ffaa00'})
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("252-day Vol: "),
                                    html.Strong("15.5%", style={'color': '#00ff88'})
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("Skewness: "),
                                    html.Strong("0.12", style={'color': '#00d4ff'})
                                ], className="mb-1"),
                                html.Div([
                                    html.Span("Kurtosis: "),
                                    html.Strong("0.07", style={'color': '#00d4ff'})
                                ], className="mb-1"),
                            ])
                        ], md=4),
                    ])
                ])
            ])
        ])
    ]),

    # Auto-refresh interval
    dcc.Interval(
        id='risk-interval-component',
        interval=10*1000,  # 10 seconds for risk monitoring
        n_intervals=0
    )

], fluid=True)


# Callbacks for alert acknowledgment/resolution would go here
# @callback(...)
