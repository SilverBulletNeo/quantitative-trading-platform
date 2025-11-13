"""
Page 3: Performance Attribution

Decomposes returns into benchmark, selection alpha, regime contribution,
and transaction costs. Shows what's actually driving performance.
"""

from dash import html
import dash_bootstrap_components as dbc

layout = dbc.Container([
    html.H2("📈 Performance Attribution", style={'color': '#00d4ff', 'margin': '20px 0'}),
    html.P("Coming soon: Return decomposition and attribution analysis", style={'color': '#8a9ba8'}),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Return Decomposition"),
                dbc.CardBody([
                    html.P("Breakdown: Benchmark + Selection Alpha + Regime Filter + Transaction Costs")
                ])
            ])
        ])
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Regime Contribution"),
                dbc.CardBody([
                    html.P("Impact of regime filtering on returns and Sharpe ratio")
                ])
            ])
        ], md=6),
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Factor Exposures"),
                dbc.CardBody([
                    html.P("Sector, size, value/growth exposures")
                ])
            ])
        ], md=6),
    ])

], fluid=True)
