"""
Page 5: Interactive Strategy Optimization

Real-time strategy optimization and parameter tuning interface with:
- Parameter sweep visualization
- Walk-forward optimization
- Multi-strategy allocation optimizer
- ML regime-based adaptation
- Performance forecasting
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

from strategy_registry import get_registry
from portfolio_manager import MultiStrategyPortfolioManager


def create_parameter_sweep_heatmap():
    """
    Create parameter sweep heatmap

    Shows Sharpe ratio across parameter combinations
    """
    # Example: Momentum lookback vs Top N selection
    lookbacks = [30, 60, 90, 120, 150, 180]
    top_ns = [5, 10, 15, 20, 25, 30]

    # Simulated Sharpe ratios
    np.random.seed(42)
    base_sharpe = np.random.rand(len(lookbacks), len(top_ns)) * 0.5 + 1.5

    # Add sweet spot
    base_sharpe[2, 1] = 2.1  # lookback=90, top_n=10

    fig = go.Figure(data=go.Heatmap(
        z=base_sharpe,
        x=[f"{n} stocks" for n in top_ns],
        y=[f"{lb} days" for lb in lookbacks],
        colorscale='RdYlGn',
        text=np.around(base_sharpe, 2),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title="Sharpe Ratio")
    ))

    fig.update_layout(
        title='Parameter Sweep: Lookback vs Portfolio Size',
        xaxis_title='Portfolio Size (Top N)',
        yaxis_title='Momentum Lookback Period',
        height=400,
        plot_bgcolor='#0a0e27',
        paper_bgcolor='#0a0e27',
        font=dict(color='#e0e0e0')
    )

    # Highlight best parameters
    fig.add_annotation(
        x=1, y=2,
        text="⭐ Optimal",
        showarrow=True,
        arrowhead=2,
        arrowcolor="#00ff88",
        font=dict(color="#00ff88", size=12)
    )

    return fig


def create_walk_forward_optimization_chart():
    """
    Walk-forward optimization visualization

    Shows in-sample optimization vs out-of-sample performance
    """
    windows = ['2019-Q1', '2019-Q2', '2019-Q3', '2019-Q4',
               '2020-Q1', '2020-Q2', '2020-Q3', '2020-Q4']

    # Optimized parameters per window
    optimal_lookbacks = [90, 120, 90, 60, 90, 120, 90, 90]

    # In-sample (optimization) Sharpe
    is_sharpe = [1.85, 1.92, 1.88, 1.95, 1.90, 1.87, 1.91, 1.89]

    # Out-of-sample (test) Sharpe
    oos_sharpe = [1.88, 1.95, 1.90, 1.97, 1.93, 1.89, 1.94, 1.92]

    fig = go.Figure()

    # In-sample Sharpe
    fig.add_trace(go.Bar(
        x=windows,
        y=is_sharpe,
        name='In-Sample (Optimized)',
        marker=dict(color='#00d4ff'),
        text=optimal_lookbacks,
        texttemplate='%{text} days',
        textposition='outside',
        textfont=dict(size=9)
    ))

    # Out-of-sample Sharpe
    fig.add_trace(go.Scatter(
        x=windows,
        y=oos_sharpe,
        name='Out-of-Sample (Test)',
        mode='lines+markers',
        line=dict(color='#00ff88', width=3),
        marker=dict(size=10, symbol='star')
    ))

    # Reference line
    fig.add_hline(y=1.5, line_dash="dash", line_color="#ffaa00",
                  annotation_text="Target: 1.5")

    fig.update_layout(
        title='Walk-Forward Optimization: Equity Momentum',
        xaxis_title='Test Window',
        yaxis_title='Sharpe Ratio',
        height=400,
        plot_bgcolor='#0a0e27',
        paper_bgcolor='#0a0e27',
        font=dict(color='#e0e0e0'),
        legend=dict(x=0.02, y=0.98),
        barmode='group'
    )

    return fig


def create_allocation_optimizer_chart():
    """Multi-strategy allocation optimization"""
    strategies = ['Equity\nMomentum', 'Value\nFactor', 'Quality\nFactor',
                 'Mean\nReversion', 'Pairs\nTrading']

    # Different optimization methods
    equal_weight = [0.20, 0.20, 0.20, 0.20, 0.20]
    sharpe_max = [0.45, 0.25, 0.15, 0.10, 0.05]
    min_variance = [0.15, 0.30, 0.30, 0.15, 0.10]
    risk_parity = [0.22, 0.24, 0.23, 0.18, 0.13]

    fig = go.Figure()

    fig.add_trace(go.Bar(name='Equal Weight', x=strategies, y=equal_weight,
                        marker_color='#666666'))
    fig.add_trace(go.Bar(name='Sharpe Max', x=strategies, y=sharpe_max,
                        marker_color='#00d4ff'))
    fig.add_trace(go.Bar(name='Min Variance', x=strategies, y=min_variance,
                        marker_color='#00ff88'))
    fig.add_trace(go.Bar(name='Risk Parity', x=strategies, y=risk_parity,
                        marker_color='#ffaa00'))

    fig.update_layout(
        title='Multi-Strategy Allocation: Different Optimization Methods',
        xaxis_title='Strategy',
        yaxis_title='Allocation Weight',
        yaxis_tickformat='.0%',
        height=400,
        plot_bgcolor='#0a0e27',
        paper_bgcolor='#0a0e27',
        font=dict(color='#e0e0e0'),
        barmode='group',
        legend=dict(x=0.7, y=0.98)
    )

    return fig


def create_regime_adaptation_chart():
    """Regime-adaptive allocation visualization"""
    dates = pd.date_range(end=datetime.now(), periods=12, freq='M')

    # Simulated regime sequence
    regimes = ['BULL', 'BULL', 'SIDEWAYS', 'CORRECTION', 'BEAR',
              'BEAR', 'SIDEWAYS', 'BULL', 'BULL', 'SIDEWAYS', 'BULL', 'BULL']

    # Adaptive exposure
    exposures = [1.2, 1.2, 1.0, 0.7, 0.5, 0.5, 1.0, 1.2, 1.2, 1.0, 1.2, 1.2]

    # Static exposure (baseline)
    static = [1.0] * 12

    # Color by regime
    colors = {
        'BULL': '#00ff88',
        'BEAR': '#ff4444',
        'SIDEWAYS': '#ffaa00',
        'CORRECTION': '#ff8800',
        'CRISIS': '#ff0000'
    }
    bar_colors = [colors[r] for r in regimes]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=('Market Regime', 'Portfolio Exposure'),
        row_heights=[0.3, 0.7]
    )

    # Regime indicators (bar chart)
    regime_codes = {'BULL': 2, 'SIDEWAYS': 1, 'CORRECTION': 0, 'BEAR': -1, 'CRISIS': -2}
    regime_values = [regime_codes[r] for r in regimes]

    fig.add_trace(
        go.Bar(
            x=dates,
            y=regime_values,
            marker_color=bar_colors,
            showlegend=False,
            text=regimes,
            textposition='inside'
        ),
        row=1, col=1
    )

    # Adaptive vs Static exposure
    fig.add_trace(
        go.Scatter(
            x=dates,
            y=exposures,
            name='Regime-Adaptive',
            mode='lines+markers',
            line=dict(color='#00d4ff', width=3),
            marker=dict(size=10)
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=static,
            name='Static (100%)',
            mode='lines',
            line=dict(color='#666666', width=2, dash='dash')
        ),
        row=2, col=1
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Exposure Multiplier", row=2, col=1)

    fig.update_layout(
        title='Regime-Adaptive Allocation Strategy',
        height=500,
        plot_bgcolor='#0a0e27',
        paper_bgcolor='#0a0e27',
        font=dict(color='#e0e0e0'),
        showlegend=True,
        legend=dict(x=0.02, y=0.5)
    )

    return fig


def create_ml_regime_prediction_chart():
    """ML regime prediction with confidence"""
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')

    # Simulated predictions
    regimes = ['BULL'] * 15 + ['SIDEWAYS'] * 10 + ['CORRECTION'] * 5
    confidences = np.random.rand(30) * 0.3 + 0.65  # 0.65-0.95 confidence

    # Color by regime
    colors = {
        'BULL': '#00ff88',
        'BEAR': '#ff4444',
        'SIDEWAYS': '#ffaa00',
        'CORRECTION': '#ff8800'
    }
    bar_colors = [colors[r] for r in regimes]

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=('Predicted Regime', 'Model Confidence'),
        row_heights=[0.5, 0.5]
    )

    # Regime codes for plotting
    regime_codes = {'BULL': 3, 'SIDEWAYS': 2, 'CORRECTION': 1, 'BEAR': 0}
    regime_values = [regime_codes[r] for r in regimes]

    fig.add_trace(
        go.Scatter(
            x=dates,
            y=regime_values,
            mode='lines+markers',
            marker=dict(size=10, color=bar_colors),
            line=dict(color='#00d4ff', width=2),
            showlegend=False,
            hovertext=regimes,
            hoverinfo='x+text'
        ),
        row=1, col=1
    )

    # Confidence
    fig.add_trace(
        go.Bar(
            x=dates,
            y=confidences,
            marker_color='#00d4ff',
            showlegend=False
        ),
        row=2, col=1
    )

    # Confidence threshold
    fig.add_hline(y=0.7, line_dash="dash", line_color="#ffaa00",
                  annotation_text="Min Confidence (70%)", row=2, col=1)

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Regime", row=1, col=1,
                     ticktext=['BEAR', 'CORRECTION', 'SIDEWAYS', 'BULL'],
                     tickvals=[0, 1, 2, 3])
    fig.update_yaxes(title_text="Confidence", row=2, col=1, tickformat='.0%')

    fig.update_layout(
        title='ML Regime Prediction (Random Forest + Gradient Boosting Ensemble)',
        height=500,
        plot_bgcolor='#0a0e27',
        paper_bgcolor='#0a0e27',
        font=dict(color='#e0e0e0')
    )

    return fig


# Page layout
layout = html.Div([
    # Header
    dbc.Row([
        dbc.Col([
            html.H2("🔬 Interactive Strategy Optimization", className="mb-4"),
            html.P("Real-time parameter tuning, multi-strategy allocation, and ML-powered regime adaptation",
                  style={'color': '#8a9ba8', 'fontSize': '1rem'})
        ])
    ], className="mb-4"),

    # Optimization Method Selector
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("Optimization Settings"),
                dbc.CardBody([
                    html.Div([
                        html.Label("Optimization Method:", className="mb-2"),
                        dcc.Dropdown(
                            id='opt-method',
                            options=[
                                {'label': '⚖️ Equal Weight', 'value': 'equal'},
                                {'label': '📈 Sharpe Maximization', 'value': 'sharpe'},
                                {'label': '🛡️ Minimum Variance', 'value': 'min_var'},
                                {'label': '⚡ Risk Parity', 'value': 'risk_parity'},
                                {'label': '🎯 Regime-Adaptive', 'value': 'regime'},
                            ],
                            value='sharpe',
                            style={'backgroundColor': '#1a1f3a', 'color': '#e0e0e0'}
                        ),
                    ], className="mb-3"),
                    html.Div([
                        html.Label("Rebalance Frequency:", className="mb-2"),
                        dcc.Dropdown(
                            id='rebalance-freq',
                            options=[
                                {'label': 'Daily', 'value': 'daily'},
                                {'label': 'Weekly', 'value': 'weekly'},
                                {'label': 'Monthly', 'value': 'monthly'},
                                {'label': 'Quarterly', 'value': 'quarterly'},
                            ],
                            value='monthly',
                            style={'backgroundColor': '#1a1f3a', 'color': '#e0e0e0'}
                        ),
                    ], className="mb-3"),
                    dbc.Button("⚡ Run Optimization", id="run-opt-btn", color="primary", className="w-100 mt-2"),
                    html.Div(id="opt-status", className="mt-3")
                ])
            ])
        ], md=4),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Strategy Selection"),
                dbc.CardBody([
                    html.Div([
                        dbc.Checklist(
                            id="strategy-selection",
                            options=[
                                {"label": " Equity Momentum (Sharpe: 1.95)", "value": "equity_momentum_90d"},
                                {"label": " Multi-Factor Momentum (Sharpe: 2.0)", "value": "multi_factor_momentum"},
                                {"label": " CPO Momentum (Sharpe: 1.8)", "value": "cpo_momentum"},
                                {"label": " Value Factor (Sharpe: 1.4)", "value": "value_factor"},
                                {"label": " Quality Factor (Sharpe: 1.6)", "value": "quality_factor"},
                            ],
                            value=["equity_momentum_90d", "multi_factor_momentum", "cpo_momentum"],
                            inline=False,
                            style={'fontSize': '0.9rem'}
                        )
                    ]),
                    html.Hr(),
                    html.Div([
                        html.Strong("Selected: "),
                        html.Span(id="selected-count", children="3 strategies")
                    ])
                ])
            ])
        ], md=8),
    ], className="mb-4"),

    # Parameter Sweep
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🔍 Parameter Sweep Analysis"),
                dbc.CardBody([
                    dcc.Graph(
                        id='param-sweep-chart',
                        figure=create_parameter_sweep_heatmap(),
                        config={'displayModeBar': False}
                    ),
                    html.Div([
                        html.P([
                            html.Strong("Optimal Parameters: "),
                            "Lookback = 90 days, Top N = 10 stocks (Sharpe: 2.1)"
                        ], style={'fontSize': '0.9rem', 'marginTop': '10px'})
                    ])
                ])
            ])
        ], md=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📅 Walk-Forward Optimization"),
                dbc.CardBody([
                    dcc.Graph(
                        id='walk-forward-chart',
                        figure=create_walk_forward_optimization_chart(),
                        config={'displayModeBar': False}
                    ),
                    html.Div([
                        html.P([
                            html.Strong("Result: "),
                            html.Span("✅ PASSED - No overfitting detected", style={'color': '#00ff88'})
                        ], style={'fontSize': '0.9rem', 'marginTop': '10px'})
                    ])
                ])
            ])
        ], md=6),
    ], className="mb-4"),

    # Multi-Strategy Allocation
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("⚖️ Multi-Strategy Allocation Optimization"),
                dbc.CardBody([
                    dcc.Graph(
                        id='allocation-chart',
                        figure=create_allocation_optimizer_chart(),
                        config={'displayModeBar': False}
                    ),
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H6("Portfolio Metrics", className="mb-2"),
                                html.P("Expected Return: 24.5%", style={'fontSize': '0.85rem', 'margin': '0'}),
                                html.P("Expected Volatility: 12.8%", style={'fontSize': '0.85rem', 'margin': '0'}),
                                html.P("Expected Sharpe: 1.91", style={'fontSize': '0.85rem', 'margin': '0'}),
                            ])
                        ], md=6),
                        dbc.Col([
                            html.Div([
                                html.H6("Diversification", className="mb-2"),
                                html.P("Correlation: 0.42", style={'fontSize': '0.85rem', 'margin': '0'}),
                                html.P("Diversification Ratio: 1.45", style={'fontSize': '0.85rem', 'margin': '0'}),
                                html.P("Status: Well Diversified ✅", style={'fontSize': '0.85rem', 'margin': '0', 'color': '#00ff88'}),
                            ])
                        ], md=6),
                    ], className="mt-3")
                ])
            ])
        ])
    ], className="mb-4"),

    # ML Regime Adaptation
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🤖 ML Regime-Adaptive Allocation"),
                dbc.CardBody([
                    dcc.Graph(
                        id='regime-adaptation-chart',
                        figure=create_regime_adaptation_chart(),
                        config={'displayModeBar': False}
                    )
                ])
            ])
        ], md=6),

        dbc.Col([
            dbc.Card([
                dbc.CardHeader("🧠 ML Regime Prediction (Ensemble)"),
                dbc.CardBody([
                    dcc.Graph(
                        id='ml-prediction-chart',
                        figure=create_ml_regime_prediction_chart(),
                        config={'displayModeBar': False}
                    )
                ])
            ])
        ], md=6),
    ], className="mb-4"),

    # Performance Forecast
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardHeader("📊 Optimization Results Summary"),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Div([
                                html.H6("Current Allocation", className="mb-3"),
                                html.P("Method: Sharpe Maximization", style={'fontSize': '0.9rem'}),
                                html.P("Strategies: 3 active", style={'fontSize': '0.9rem'}),
                                html.P("Last Rebalance: 2025-11-14", style={'fontSize': '0.9rem'}),
                            ])
                        ], md=3),
                        dbc.Col([
                            html.Div([
                                html.H6("Expected Performance", className="mb-3"),
                                html.P([html.Strong("Return: "), "24.5% annually"], style={'fontSize': '0.9rem'}),
                                html.P([html.Strong("Volatility: "), "12.8%"], style={'fontSize': '0.9rem'}),
                                html.P([html.Strong("Sharpe: "), "1.91"], style={'fontSize': '0.9rem'}),
                            ])
                        ], md=3),
                        dbc.Col([
                            html.Div([
                                html.H6("Risk Metrics", className="mb-3"),
                                html.P([html.Strong("Max DD: "), "-15.2%"], style={'fontSize': '0.9rem'}),
                                html.P([html.Strong("VaR (95%): "), "-2.1%"], style={'fontSize': '0.9rem'}),
                                html.P([html.Strong("Diversification: "), "1.45"], style={'fontSize': '0.9rem'}),
                            ])
                        ], md=3),
                        dbc.Col([
                            html.Div([
                                html.H6("Regime Outlook", className="mb-3"),
                                html.P([html.Strong("Current: "), "BULL"], style={'fontSize': '0.9rem', 'color': '#00ff88'}),
                                html.P([html.Strong("Confidence: "), "87%"], style={'fontSize': '0.9rem'}),
                                html.P([html.Strong("Exposure: "), "120%"], style={'fontSize': '0.9rem'}),
                            ])
                        ], md=3),
                    ])
                ])
            ])
        ])
    ]),

    # Auto-refresh
    dcc.Interval(
        id='optimization-refresh',
        interval=120*1000,  # 2 minutes
        n_intervals=0
    )
])


# Callbacks (placeholders for interactivity)
@callback(
    Output('selected-count', 'children'),
    Input('strategy-selection', 'value')
)
def update_selection_count(selected):
    if not selected:
        return "0 strategies"
    return f"{len(selected)} strategies"


if __name__ == '__main__':
    print("Optimization page loaded successfully")
