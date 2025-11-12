# Quick Start Guide

Get the quantitative trading platform running in 5 minutes.

## Prerequisites

- Python 3.11 or higher
- Git
- Virtual environment tool (venv, conda, or similar)

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/quantitative-trading-platform.git
cd quantitative-trading-platform
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (macOS/Linux)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate
```

### 3. Install Core Dependencies

For the starter demo, you only need:

```bash
pip install numpy pandas yfinance
```

For the full platform:

```bash
pip install -r requirements.txt
```

Note: Some packages (like ta-lib) require system-level dependencies. See [Installation Troubleshooting](#installation-troubleshooting) below.

### 4. Run the Starter Demo

```bash
python src/platform_starter.py
```

This will:
- Fetch 5 years of market data for a multi-asset universe
- Run a momentum strategy backtest
- Display performance metrics
- Show current positions

Expected output:
```
============================================================
QUANTITATIVE TRADING PLATFORM - STARTER DEMO
============================================================

Fetching data for 8 symbols...
Downloaded 1259 days of data
Running backtest from 2020-01-01 to 2024-12-31...
Universe: 8 assets
Initial Capital: $1,000,000
------------------------------------------------------------
2020-03-02: Portfolio Value = $998,450, Return = -0.16%
2020-06-01: Portfolio Value = $1,045,230, Return = 4.52%
...
------------------------------------------------------------
Backtest complete!
============================================================
BACKTEST RESULTS
============================================================
Total Return        : 45.23%
Sharpe Ratio        : 1.23
Max Drawdown        : -15.67%
Volatility          : 12.34%
Final Value         : $1,452,300
Total Positions     : 6

Total Trades: 124

Current Positions:
------------------------------------------------------------
SPY       :   $243,500 ( 12.45% P&L)
QQQ       :   $198,750 ( 18.23% P&L)
...
```

## Configuration

### 1. Copy Example Config

```bash
cp config/config.example.yaml config/config.yaml
```

### 2. Edit Configuration

Open `config/config.yaml` and customize:

- **Portfolio Settings**: Initial capital, leverage, target volatility
- **Risk Limits**: Regional, asset class, sector, and position limits
- **Investment Universe**: Add/remove tickers
- **Strategy Parameters**: Momentum lookback, rebalancing frequency
- **Execution Settings**: Commission rates, slippage assumptions

### 3. Set Environment Variables (Optional)

For API keys and credentials:

```bash
# Create .env file
cp .env.example .env

# Edit .env with your credentials
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here
POLYGON_API_KEY=your_key_here
```

## Next Steps

### Explore the Code

1. **Core Components** (`src/core/`)
   - Asset classes and instruments
   - Portfolio management
   - Order and position tracking

2. **Strategies** (`src/strategies/`)
   - Momentum, mean reversion, carry strategies
   - Create your own by extending the Strategy base class

3. **Risk Management** (`src/risk/`)
   - Pre-trade checks
   - Real-time monitoring
   - Risk metrics calculation

4. **Backtesting** (`backtest/`)
   - Historical simulation
   - Performance analysis
   - Transaction cost modeling

### Run Your First Custom Strategy

```python
from src.core.portfolio import Portfolio
from src.strategies.momentum import MomentumStrategy
from src.risk.risk_manager import RiskManager
from backtest.engine import BacktestEngine

# Define your universe
universe = {
    'equity': ['AAPL', 'MSFT', 'GOOGL'],
    'fixed_income': ['TLT', 'AGG']
}

# Create strategy
strategy = MomentumStrategy(universe, lookback=60)

# Run backtest
portfolio = Portfolio(initial_capital=100_000)
risk_manager = RiskManager(portfolio)

engine = BacktestEngine(portfolio, strategy, risk_manager)
results = engine.run('2020-01-01', '2024-12-31')

print(results.metrics())
```

### Development Roadmap

Follow our phased development plan:

- **Phase 1** (Now): Run starter demo, understand architecture
- **Phase 2** (Weeks 1-2): Add more strategies, expand universe
- **Phase 3** (Months 1-2): Integrate real market data APIs
- **Phase 4** (Months 3-4): Build advanced risk analytics
- **Phase 5** (Months 5-6): Paper trading with live data
- **Phase 6** (Months 7+): Production deployment

## Installation Troubleshooting

### TA-Lib Installation

TA-Lib requires a C library. Install it first:

**macOS:**
```bash
brew install ta-lib
pip install ta-lib
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ta-lib
pip install ta-lib
```

**Windows:**
```bash
# Download pre-built wheel from:
# https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
pip install TA_Lib‑0.4.XX‑cpXX‑cpXX‑win_amd64.whl
```

### PyPortfolioOpt Issues

If PyPortfolioOpt fails to install:

```bash
# Install dependencies first
pip install numpy pandas scipy cvxpy

# Then install PyPortfolioOpt
pip install PyPortfolioOpt
```

### Arctic (MongoDB) Setup

Arctic requires MongoDB:

```bash
# macOS
brew tap mongodb/brew
brew install mongodb-community
brew services start mongodb-community

# Ubuntu
sudo apt-get install mongodb
sudo systemctl start mongodb
```

Then install Arctic:

```bash
pip install arctic
```

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test module
pytest tests/unit/test_portfolio.py
```

## Documentation

- **Architecture**: [docs/architecture/SYSTEM_DESIGN.md](docs/architecture/SYSTEM_DESIGN.md)
- **Strategies**: [docs/strategies/](docs/strategies/)
- **API Reference**: [docs/api/](docs/api/)

## Getting Help

- **GitHub Issues**: Report bugs or request features
- **Documentation**: Check the docs/ folder
- **Code Comments**: All modules are thoroughly documented

## Common Issues

### "No module named 'yfinance'"
```bash
pip install yfinance
```

### "Permission denied" on scripts
```bash
chmod +x scripts/*.sh
```

### Data download fails
- Check internet connection
- Verify Yahoo Finance is accessible
- Try alternative data source (see config.yaml)

### Backtest runs slowly
- Reduce universe size
- Increase rebalancing frequency
- Use vectorized operations (already optimized in starter)

## Performance Tips

1. **Use vectorized operations** (NumPy/Pandas) instead of loops
2. **Enable JIT compilation** with Numba for hot paths
3. **Cache expensive calculations** (correlation matrices, factor models)
4. **Use appropriate data structures** (dict for lookups, list for iteration)
5. **Profile before optimizing** with cProfile or line_profiler

## What's Next?

1. ✅ Run the starter demo
2. ✅ Explore the codebase
3. [ ] Read the architecture documentation
4. [ ] Customize the configuration
5. [ ] Create your first strategy
6. [ ] Run a backtest on your strategy
7. [ ] Analyze results and iterate
8. [ ] Paper trade with live data
9. [ ] Deploy to production

---

**Ready to build institutional-grade quant strategies!** 🚀

For detailed documentation, see [README.md](README.md) and [docs/](docs/).
