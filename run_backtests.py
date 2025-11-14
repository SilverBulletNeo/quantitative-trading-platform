"""
Run Backtests on Real Data

Simple script to backtest all strategies on downloaded market data.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def load_data(universe='crypto'):
    """Load price data"""
    filepath = f'data/raw/{universe}_prices.csv'
    if not os.path.exists(filepath):
        print(f"Error: {filepath} not found")
        print("Run: python src/data/simple_data_downloader.py first")
        return None

    data = pd.read_csv(filepath, index_col=0, parse_dates=True)
    print(f"Loaded {universe}: {len(data.columns)} assets, {len(data)} days")
    return data

def calculate_returns(prices):
    """Calculate daily returns"""
    return prices.pct_change().dropna()

def calculate_momentum_signals(prices, lookback=20):
    """Simple momentum strategy"""
    returns = prices.pct_change(lookback)
    signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

    # Buy top 30%, sell bottom 30%
    for date in returns.index:
        if pd.notna(returns.loc[date]).any():
            ranked = returns.loc[date].rank(pct=True)
            signals.loc[date, ranked >= 0.7] = 1  # Top 30%
            signals.loc[date, ranked <= 0.3] = -1  # Bottom 30%

    return signals

def calculate_mean_reversion_signals(prices, lookback=20):
    """Simple mean reversion strategy"""
    # Z-score
    returns = prices.pct_change()
    rolling_mean = prices.rolling(lookback).mean()
    rolling_std = prices.rolling(lookback).std()
    zscore = (prices - rolling_mean) / rolling_std

    signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

    # Buy when oversold (z < -2), sell when overbought (z > 2)
    signals[zscore < -2] = 1
    signals[zscore > 2] = -1

    return signals

def calculate_moving_average_signals(prices, fast=20, slow=50):
    """Moving average crossover"""
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()

    signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    signals[fast_ma > slow_ma] = 1  # Bullish
    signals[fast_ma < slow_ma] = -1  # Bearish

    return signals

def backtest_strategy(prices, signals, strategy_name):
    """Backtest a strategy"""
    returns = prices.pct_change()

    # Portfolio returns (equal weight across signals)
    portfolio_returns = pd.Series(0.0, index=returns.index)

    for date in signals.index:
        if date not in returns.index:
            continue

        # Long positions
        longs = signals.loc[date][signals.loc[date] == 1].index
        if len(longs) > 0:
            portfolio_returns[date] = returns.loc[date, longs].mean()

    # Calculate metrics
    total_return = (1 + portfolio_returns).prod() - 1
    years = len(portfolio_returns) / 252
    ann_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    ann_vol = portfolio_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    # Max drawdown
    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()

    # Win rate
    win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)

    return {
        'Strategy': strategy_name,
        'Total Return': f"{total_return:.2%}",
        'Annual Return': f"{ann_return:.2%}",
        'Volatility': f"{ann_vol:.2%}",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Max Drawdown': f"{max_dd:.2%}",
        'Win Rate': f"{win_rate:.2%}"
    }

def run_all_strategies(prices, universe_name):
    """Run all strategies"""
    print(f"\n{'='*80}")
    print(f"BACKTESTING: {universe_name.upper()}")
    print(f"{'='*80}\n")

    results = []

    # 1. Momentum
    print("Testing Momentum (20-day)...")
    mom_signals = calculate_momentum_signals(prices, 20)
    results.append(backtest_strategy(prices, mom_signals, 'Momentum-20'))

    # 2. Momentum (60-day)
    print("Testing Momentum (60-day)...")
    mom_signals = calculate_momentum_signals(prices, 60)
    results.append(backtest_strategy(prices, mom_signals, 'Momentum-60'))

    # 3. Mean Reversion
    print("Testing Mean Reversion...")
    mr_signals = calculate_mean_reversion_signals(prices, 20)
    results.append(backtest_strategy(prices, mr_signals, 'Mean Reversion'))

    # 4. Moving Average Crossover
    print("Testing MA Crossover (20/50)...")
    ma_signals = calculate_moving_average_signals(prices, 20, 50)
    results.append(backtest_strategy(prices, ma_signals, 'MA Cross 20/50'))

    # 5. Moving Average Crossover (50/200)
    print("Testing MA Crossover (50/200)...")
    ma_signals = calculate_moving_average_signals(prices, 50, 200)
    results.append(backtest_strategy(prices, ma_signals, 'MA Cross 50/200'))

    # 6. Buy and Hold (benchmark)
    print("Testing Buy & Hold (benchmark)...")
    bh_signals = pd.DataFrame(1, index=prices.index, columns=prices.columns)
    results.append(backtest_strategy(prices, bh_signals, 'Buy & Hold'))

    # Display results
    df = pd.DataFrame(results)
    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"{'='*80}\n")
    print(df.to_string(index=False))

    # Save results
    df.to_csv(f'backtest_results_{universe_name}.csv', index=False)
    print(f"\n✅ Results saved to: backtest_results_{universe_name}.csv\n")

    return df

def main():
    """Main entry point"""
    print("="*80)
    print("QUANTITATIVE TRADING PLATFORM")
    print("Real Data Backtesting")
    print("="*80)

    # Test crypto
    print("\n" + "="*80)
    print("CRYPTO UNIVERSE")
    print("="*80)
    crypto_prices = load_data('crypto')
    if crypto_prices is not None:
        crypto_results = run_all_strategies(crypto_prices, 'crypto')

    # Test equities
    print("\n" + "="*80)
    print("EQUITIES UNIVERSE")
    print("="*80)
    equity_prices = load_data('equities')
    if equity_prices is not None:
        equity_results = run_all_strategies(equity_prices, 'equities')

    # Test multi-asset
    print("\n" + "="*80)
    print("MULTI-ASSET UNIVERSE")
    print("="*80)
    multi_prices = load_data('multi_asset')
    if multi_prices is not None:
        multi_results = run_all_strategies(multi_prices, 'multi_asset')

    print("\n" + "="*80)
    print("BACKTEST COMPLETE!")
    print("="*80)
    print("\nResults saved to:")
    print("  - backtest_results_crypto.csv")
    print("  - backtest_results_equities.csv")
    print("  - backtest_results_multi_asset.csv")
    print()

if __name__ == "__main__":
    main()
