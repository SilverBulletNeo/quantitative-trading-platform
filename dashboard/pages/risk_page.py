"""
Page 2: Risk Monitoring

Risk tracking with drawdown monitoring, VaR, regime detection,
and alert management.
"""

from dash import html
import dash_bootstrap_components as dbc

layout = dbc.Container([
    html.H2("⚠️ Risk Monitoring", style={'color': '#00d4ff', 'margin': '20px 0'}),
    html.P("Coming soon: Real-time risk tracking and alerts", style={'color': '#8a9ba8'}),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Drawdown Tracker"),
                dbc.CardBody([
                    html.P("Track current drawdown vs. historical and thresholds")
                ])
            ])
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Risk Metrics"),
                dbc.CardBody([
                    html.P("VaR, CVaR, volatility monitoring")
                ])
            ])
        ], md=6),
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Active Alerts"),
                dbc.CardBody([
                    html.P("System alerts and circuit breakers")
                ])
            ])
        ])
    ])

], fluid=True)
