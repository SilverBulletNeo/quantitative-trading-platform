"""
Quantitative Trading Dashboard

Professional multi-page Dash application for monitoring trading strategies
with real-time performance tracking, risk management, and analytics.

Built with Dash + Plotly + Bootstrap Components
Inspired by Bloomberg Terminal aesthetics
"""

import dash
from dash import Dash, html, dcc
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import plotly.graph_objects as go

# Initialize app with Bootstrap theme
app = Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.CYBORG],  # Dark theme base
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ]
)

app.title = "Quantitative Trading Dashboard"
server = app.server  # For deployment

# Custom CSS for Bloomberg-style aesthetics
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Bloomberg Terminal Dark Theme */
            body {
                background-color: #0a0e27;
                color: #e0e0e0;
                font-family: 'Roboto Mono', 'Courier New', monospace;
            }

            /* Navbar styling */
            .navbar {
                background-color: #1a1f3a !important;
                border-bottom: 2px solid #00d4ff;
                box-shadow: 0 2px 10px rgba(0, 212, 255, 0.3);
            }

            .navbar-brand {
                color: #00d4ff !important;
                font-weight: bold;
                font-size: 1.5rem;
                letter-spacing: 2px;
            }

            .nav-link {
                color: #e0e0e0 !important;
                transition: all 0.3s;
                border-bottom: 2px solid transparent;
            }

            .nav-link:hover {
                color: #00d4ff !important;
                border-bottom: 2px solid #00d4ff;
            }

            .nav-link.active {
                color: #00d4ff !important;
                border-bottom: 2px solid #00d4ff;
            }

            /* Card styling */
            .card {
                background-color: #1a1f3a;
                border: 1px solid #2a3f5f;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
                margin-bottom: 20px;
            }

            .card-header {
                background-color: #151a30;
                border-bottom: 1px solid #00d4ff;
                color: #00d4ff;
                font-weight: bold;
                letter-spacing: 1px;
            }

            /* Metric cards */
            .metric-card {
                background: linear-gradient(135deg, #1a1f3a 0%, #2a3f5f 100%);
                border-left: 4px solid #00d4ff;
                padding: 20px;
                margin: 10px 0;
                border-radius: 8px;
            }

            .metric-label {
                color: #8a9ba8;
                font-size: 0.85rem;
                text-transform: uppercase;
                letter-spacing: 1px;
                margin-bottom: 5px;
            }

            .metric-value {
                color: #00d4ff;
                font-size: 2rem;
                font-weight: bold;
                font-family: 'Roboto Mono', monospace;
            }

            .metric-value.positive {
                color: #00ff88;
            }

            .metric-value.negative {
                color: #ff4444;
            }

            .metric-value.warning {
                color: #ffaa00;
            }

            /* Alert badges */
            .alert-critical {
                background-color: #ff4444;
                color: white;
                padding: 5px 15px;
                border-radius: 20px;
                animation: pulse 2s infinite;
            }

            .alert-warning {
                background-color: #ffaa00;
                color: #0a0e27;
                padding: 5px 15px;
                border-radius: 20px;
            }

            .alert-info {
                background-color: #00d4ff;
                color: #0a0e27;
                padding: 5px 15px;
                border-radius: 20px;
            }

            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.6; }
            }

            /* Chart styling */
            .js-plotly-plot {
                background-color: transparent !important;
            }

            /* Regime indicator */
            .regime-bull {
                color: #00ff88;
                font-size: 1.5rem;
                font-weight: bold;
            }

            .regime-bear {
                color: #ff4444;
                font-size: 1.5rem;
                font-weight: bold;
            }

            .regime-sideways {
                color: #ffaa00;
                font-size: 1.5rem;
                font-weight: bold;
            }

            /* Table styling */
            table {
                color: #e0e0e0 !important;
            }

            thead {
                background-color: #151a30 !important;
                color: #00d4ff !important;
            }

            tbody tr:hover {
                background-color: #2a3f5f !important;
            }

            /* Scrollbar */
            ::-webkit-scrollbar {
                width: 10px;
            }

            ::-webkit-scrollbar-track {
                background: #0a0e27;
            }

            ::-webkit-scrollbar-thumb {
                background: #2a3f5f;
                border-radius: 5px;
            }

            ::-webkit-scrollbar-thumb:hover {
                background: #00d4ff;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# Navbar
navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("⚡ QUANTITATIVE TRADING DASHBOARD", className="ms-2"),
            dbc.Nav(
                [
                    dbc.NavLink("📊 Performance", href="/", active="exact", className="nav-link"),
                    dbc.NavLink("⚠️ Risk", href="/risk", active="exact", className="nav-link"),
                    dbc.NavLink("📈 Attribution", href="/attribution", active="exact", className="nav-link"),
                    dbc.NavLink("📉 Analytics", href="/analytics", active="exact", className="nav-link"),
                ],
                className="ms-auto",
                navbar=True,
            ),
        ],
        fluid=True,
    ),
    className="navbar",
    dark=True,
)

# App layout
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content', style={'padding': '20px'})
])

# Router callback
@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    """Route to different pages based on URL"""
    if pathname == '/risk':
        from pages import risk_page
        return risk_page.layout
    elif pathname == '/attribution':
        from pages import attribution_page
        return attribution_page.layout
    elif pathname == '/analytics':
        from pages import analytics_page
        return analytics_page.layout
    else:  # Default to performance page
        from pages import performance_page
        return performance_page.layout


if __name__ == '__main__':
    print("="*80)
    print("🚀 Starting Quantitative Trading Dashboard")
    print("="*80)
    print("\n📊 Dashboard Features:")
    print("   - Real-time performance monitoring")
    print("   - Risk management and alerts")
    print("   - Performance attribution analysis")
    print("   - Walk-forward validation tracking")
    print("   - Monte Carlo stress testing")
    print("\n🌐 Access dashboard at: http://localhost:8050")
    print("="*80)
    print()

    app.run(debug=True, host='0.0.0.0', port=8050)
