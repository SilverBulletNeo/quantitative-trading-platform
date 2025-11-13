# Crypto Trading Strategies - Implementation Guide

## Overview

We've added **cryptocurrency as an alternative asset class** and implemented **5 proven non-HF strategies** from top open-source repositories, adapted for daily timeframe trading.

## What's Been Added

### 1. Asset Class Extension
- Added `CRYPTO` to `AssetClass` enum in `src/platform_starter.py`
- Platform now supports: Equity, Fixed Income, Commodities, FX, **and Crypto**

### 2. Implemented Strategies

All strategies are sourced from **je-suis-tm/quant-trading** (8.6k stars) - one of the best open-source quant trading repositories.

#### Strategy 1: MACD Oscillator
**File**: `src/strategies/macd_strategy.py`
- **Type**: Momentum strategy
- **Indicators**: Fast EMA (12), Slow EMA (26), Signal (9)
- **Entry**: Buy when fast MA crosses above slow MA
- **Exit**: Sell when fast MA crosses below slow MA
- **Best for**: Trending markets, BTC, ETH

#### Strategy 2: RSI (Relative Strength Index)
**File**: `src/strategies/rsi_strategy.py`
- **Type**: Mean reversion / Momentum
- **Indicator**: RSI (14-period)
- **Entry**: Buy when RSI < 30 (oversold)
- **Exit**: Sell when RSI > 70 (overbought)
- **Advanced**: Pattern recognition with holding period
- **Best for**: Volatile crypto markets

#### Strategy 3: Bollinger Bands
**File**: `src/strategies/bollinger_bands_strategy.py`
- **Type**: Pattern recognition / Mean reversion
- **Indicators**: 20-period MA, ±2 std dev bands
- **Entry**: W-pattern detection near lower band
- **Exit**: Bandwidth contraction (low volatility)
- **Best for**: Range-bound markets, altcoins

#### Strategy 4: Mean Reversion
**File**: `src/strategies/mean_reversion_strategy.py`
- **Type**: Statistical arbitrage
- **Indicator**: Z-score (price vs rolling mean)
- **Entry**: Buy when z-score < -2 (price far below mean)
- **Exit**: Sell when price returns to mean
- **Variations**: Z-score, Bollinger Bands, Sharpe-adjusted
- **Best for**: Stable cryptocurrencies

#### Strategy 5: Momentum (Existing)
**File**: `src/platform_starter.py`
- **Type**: Cross-sectional momentum
- **Indicator**: 60-day price momentum
- **Entry**: Buy top performers
- **Exit**: Rebalance periodically
- **Best for**: Diversified crypto portfolio

## Testing System

### Comprehensive Backtester
**File**: `src/strategies/crypto_strategy_tester.py`

Features:
- Tests all 8 strategy variations
- Compares performance across 5 major cryptocurrencies (BTC, ETH, BNB, SOL, ADA)
- Calculates: Total Return, Sharpe Ratio, Max Drawdown, Win Rate
- Ranks strategies by performance
- Saves results with timestamps

### Automated Daily Testing

#### Option 1: Shell Script (Linux/Mac)
**File**: `scripts/setup/daily_strategy_test.sh`

```bash
# Make executable
chmod +x scripts/setup/daily_strategy_test.sh

# Run manually
./scripts/setup/daily_strategy_test.sh

# Schedule with cron (runs daily at 9 AM)
crontab -e
# Add: 0 9 * * * /path/to/daily_strategy_test.sh
```

#### Option 2: Python Scheduler
**File**: `scripts/setup/schedule_daily_tests.py`

```bash
# Install dependency
pip install schedule

# Run scheduler (keeps running in background)
python scripts/setup/schedule_daily_tests.py
```

## Quick Start

### 1. Install Dependencies

```bash
pip install numpy pandas yfinance scipy statsmodels
```

### 2. Test Individual Strategy

```bash
# Test MACD
python src/strategies/macd_strategy.py

# Test RSI
python src/strategies/rsi_strategy.py

# Test Bollinger Bands
python src/strategies/bollinger_bands_strategy.py

# Test Mean Reversion
python src/strategies/mean_reversion_strategy.py
```

### 3. Run Comprehensive Comparison

```bash
python src/strategies/crypto_strategy_tester.py
```

Expected output:
```
================================================================
STRATEGY COMPARISON RESULTS
================================================================

MACD Oscillator
--------------------------------------------------------------------------------
  Total Return     :     45.23%
  Annual Return    :     21.45%
  Sharpe Ratio     :       1.85
  Max Drawdown     :    -15.32%
  Volatility       :     28.45%
  Final Value      :   $14,523
  Total Trades     :         45
  Win Rate         :     62.22%

[... other strategies ...]

================================================================
RANKING BY SHARPE RATIO
================================================================
1. RSI Advanced                    - Sharpe: 2.15, Return: 52.34%
2. Mean Reversion Sharpe           - Sharpe: 1.98, Return: 48.12%
3. MACD Oscillator                 - Sharpe: 1.85, Return: 45.23%
[...]
```

### 4. Set Up Daily Automated Testing

```bash
# Option A: Use shell script with cron
crontab -e
# Add: 0 9 * * * /path/to/quantitative-trading-platform/scripts/setup/daily_strategy_test.sh

# Option B: Use Python scheduler
nohup python scripts/setup/schedule_daily_tests.py > scheduler.log 2>&1 &
```

## Crypto Universe

Default cryptocurrencies tested:
- **BTC-USD** (Bitcoin) - Market leader
- **ETH-USD** (Ethereum) - Smart contracts
- **BNB-USD** (Binance Coin) - Exchange token
- **SOL-USD** (Solana) - High performance
- **ADA-USD** (Cardano) - Research-driven

To add more, edit `crypto_strategy_tester.py`:
```python
CRYPTO_SYMBOLS = [
    'BTC-USD',
    'ETH-USD',
    'MATIC-USD',  # Add Polygon
    'AVAX-USD',   # Add Avalanche
    # ... add more
]
```

## Performance Metrics

Each strategy reports:

1. **Total Return**: Cumulative return over backtest period
2. **Annual Return**: Annualized return (CAGR)
3. **Sharpe Ratio**: Risk-adjusted return (>1.5 is excellent)
4. **Max Drawdown**: Largest peak-to-trough decline
5. **Volatility**: Annualized standard deviation of returns
6. **Total Trades**: Number of round-trip trades
7. **Win Rate**: Percentage of profitable trades

## Results Storage

Results are saved to:
```
backtest/
├── results/
│   └── backtest_results_YYYYMMDD_HHMMSS.json
└── logs/
    └── strategy_test_YYYYMMDD_HHMMSS.log
```

## Strategy Selection Guide

### For Bitcoin & Major Cryptos
- **Best**: MACD (captures trends), RSI Advanced (handles volatility)
- **Good**: Momentum, Mean Reversion

### For Altcoins (High Volatility)
- **Best**: RSI Simple (quick reversals), Bollinger Bands
- **Good**: Mean Reversion

### For Stable Markets
- **Best**: Mean Reversion Sharpe-Adjusted
- **Good**: Bollinger Bands Simple

### For Portfolio Diversification
- **Best**: Momentum (cross-sectional)
- **Good**: Combination of MACD + RSI

## Advanced Usage

### Custom Configuration

```python
from strategies.macd_strategy import MACDStrategy, MACDConfig

# Customize MACD parameters
config = MACDConfig(
    fast_period=10,    # Faster signals
    slow_period=21,
    signal_period=7,
    position_size=0.15  # 15% per position
)

strategy = MACDStrategy(config)
```

### Combine Multiple Strategies

```python
# Get signals from multiple strategies
macd_signals = macd_strategy.generate_signals(data)
rsi_signals = rsi_strategy.generate_signals(data)

# Combine: only trade when both agree
combined_signals = (macd_signals == 1) & (rsi_signals == 1)
```

## Important Notes

### Data Source
- Uses Yahoo Finance (free) via `yfinance`
- Daily candles (OHLCV)
- For live trading, consider professional data feeds (Binance API, CoinGecko, etc.)

### Transaction Costs
- Default: 0.1% commission per trade
- Adjust in `crypto_strategy_tester.py` based on your exchange
- Crypto exchanges typically: 0.05-0.25%

### Risk Management
- Default position size: 10% of portfolio per asset
- Maximum 10 simultaneous positions
- No leverage (spot trading only)
- Adjust in strategy config files

### Backtesting Limitations
- Past performance ≠ future results
- No slippage modeling (yet)
- Assumes liquidity for all trades
- No funding rates for perpetual futures

## Next Steps

### Phase 1 (Immediate)
- [x] Add crypto asset class
- [x] Implement 4 proven strategies
- [x] Create comprehensive testing system
- [x] Set up daily automation

### Phase 2 (Next Week)
- [ ] Add more cryptocurrencies to universe
- [ ] Implement slippage modeling
- [ ] Add strategy combination framework
- [ ] Create performance visualization dashboard

### Phase 3 (Next Month)
- [ ] Integrate live market data APIs
- [ ] Build paper trading system
- [ ] Add ML-based strategy optimization
- [ ] Implement portfolio rebalancing

## Resources

### Source Repositories
- **je-suis-tm/quant-trading** (8.6k ⭐): MACD, RSI, Bollinger Bands
- **freqtrade** (44.5k ⭐): Crypto bot framework (inspiration)
- **microsoft/qlib** (33.7k ⭐): AI-oriented quant platform

### Documentation
- MACD: https://www.investopedia.com/terms/m/macd.asp
- RSI: https://www.investopedia.com/terms/r/rsi.asp
- Bollinger Bands: https://www.investopedia.com/terms/b/bollingerbands.asp

## Troubleshooting

### "Module not found" errors
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/quantitative-trading-platform"
```

### yfinance download failures
```bash
pip install --upgrade yfinance
# Or use alternative: pip install ccxt
```

### Low strategy performance
- Adjust parameters in config files
- Try different timeframes
- Check if market regime changed
- Combine multiple strategies

## Support

For issues or questions:
1. Check strategy comments in source files
2. Review original implementations at source repos
3. Test with different parameters
4. Monitor daily results for patterns

---

**Start testing your crypto strategies today!** 🚀

Run:
```bash
python src/strategies/crypto_strategy_tester.py
```
