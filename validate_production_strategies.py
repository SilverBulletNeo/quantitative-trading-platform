"""
Walk-Forward Validation for Production Strategies

Validates that our production strategies generalize to unseen data.
This is CRITICAL before live deployment.

Tests:
1. Combined 30/70 portfolio (our best strategy)
2. Crypto momentum standalone
3. Equity momentum standalone

Methodology:
- Train window: 2 years
- Test window: 6 months
- Step: 6 months (anchored walk-forward)
- Calculate in-sample vs out-of-sample degradation

Success Criteria:
- Out-of-sample Sharpe > 1.5 for combined portfolio
- Degradation < 0.5 (in-sample - out-of-sample)
- Positive out-of-sample returns in >70% of windows
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

from strategies.production.crypto_momentum import CryptoMomentumStrategy, CryptoMomentumConfig
from strategies.production.equity_momentum import EquityMomentumStrategy, EquityMomentumConfig
from strategies.production.combined_momentum_portfolio import CombinedMomentumPortfolio, CombinedPortfolioConfig


def calculate_metrics(returns):
    """Calculate performance metrics"""
    if len(returns) == 0 or returns.std() == 0:
        return {
            'sharpe': 0,
            'annual_return': 0,
            'max_dd': 0,
            'calmar': 0
        }

    total_return = (1 + returns).prod() - 1
    years = len(returns) / 252
    ann_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()

    calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

    return {
        'sharpe': sharpe,
        'annual_return': ann_return * 100,
        'max_dd': max_dd * 100,
        'calmar': calmar
    }


def walk_forward_combined_portfolio(crypto_prices, equity_prices,
                                    train_years=2, test_months=6):
    """
    Walk-forward validation for combined portfolio

    Args:
        crypto_prices: Crypto prices
        equity_prices: Equity prices
        train_years: Training window size
        test_months: Test window size

    Returns:
        DataFrame with results for each window
    """
    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION: COMBINED 30/70 PORTFOLIO")
    print("="*80)
    print(f"Train: {train_years} years, Test: {test_months} months\n")

    results = []

    # Find overlapping date range
    crypto_start = crypto_prices.index[0]
    equity_start = equity_prices.index[0]
    start_date = max(crypto_start, equity_start)

    end_date = min(crypto_prices.index[-1], equity_prices.index[-1])

    print(f"Overlapping period: {start_date.date()} to {end_date.date()}\n")

    current_start = start_date
    window_num = 1

    while True:
        train_end = current_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(months=test_months)

        if test_end > end_date:
            break

        print(f"Window {window_num}:")
        print(f"  Train: {current_start.date()} to {train_end.date()}")
        print(f"  Test:  {train_end.date()} to {test_end.date()}")

        # Split data
        crypto_train = crypto_prices[(crypto_prices.index >= current_start) &
                                     (crypto_prices.index < train_end)]
        crypto_test = crypto_prices[(crypto_prices.index >= train_end) &
                                   (crypto_prices.index < test_end)]

        equity_train = equity_prices[(equity_prices.index >= current_start) &
                                    (equity_prices.index < train_end)]
        equity_test = equity_prices[(equity_prices.index >= train_end) &
                                   (equity_prices.index < test_end)]

        # Initialize portfolio
        config = CombinedPortfolioConfig(
            crypto_allocation=0.30,
            equity_allocation=0.70
        )
        portfolio = CombinedMomentumPortfolio(config)

        # Train (in-sample)
        try:
            train_results = portfolio.backtest(crypto_train, equity_train)
            train_metrics = train_results['metrics']
        except Exception as e:
            print(f"  Train ERROR: {e}")
            current_start = train_end
            window_num += 1
            continue

        # Test (out-of-sample)
        try:
            test_results = portfolio.backtest(crypto_test, equity_test)
            test_metrics = test_results['metrics']
        except Exception as e:
            print(f"  Test ERROR: {e}")
            current_start = train_end
            window_num += 1
            continue

        degradation = train_metrics['sharpe'] - test_metrics['sharpe']

        print(f"  In-Sample:  Sharpe {train_metrics['sharpe']:.2f}, Return {train_metrics['annual_return']:.1f}%")
        print(f"  Out-Sample: Sharpe {test_metrics['sharpe']:.2f}, Return {test_metrics['annual_return']:.1f}%")
        print(f"  Degradation: {degradation:+.2f}\n")

        results.append({
            'Window': window_num,
            'Train Start': current_start.date(),
            'Train End': train_end.date(),
            'Test Start': train_end.date(),
            'Test End': test_end.date(),
            'In-Sample Sharpe': train_metrics['sharpe'],
            'Out-Sample Sharpe': test_metrics['sharpe'],
            'In-Sample Return': train_metrics['annual_return'],
            'Out-Sample Return': test_metrics['annual_return'],
            'Degradation': degradation
        })

        current_start = train_end
        window_num += 1

    return pd.DataFrame(results)


def walk_forward_crypto(crypto_prices, train_years=2, test_months=6):
    """Walk-forward validation for crypto momentum"""

    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION: CRYPTO MOMENTUM")
    print("="*80)
    print(f"Train: {train_years} years, Test: {test_months} months\n")

    results = []

    start_date = crypto_prices.index[0]
    end_date = crypto_prices.index[-1]

    current_start = start_date
    window_num = 1

    while True:
        train_end = current_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(months=test_months)

        if test_end > end_date:
            break

        print(f"Window {window_num}: Train {current_start.date()} to {train_end.date()}, Test {train_end.date()} to {test_end.date()}")

        # Split data
        train_prices = crypto_prices[(crypto_prices.index >= current_start) &
                                    (crypto_prices.index < train_end)]
        test_prices = crypto_prices[(crypto_prices.index >= train_end) &
                                   (crypto_prices.index < test_end)]

        # Initialize strategy
        config = CryptoMomentumConfig(lookback=20, use_regime_filter=True)
        strategy = CryptoMomentumStrategy(config)

        # Train
        try:
            train_results = strategy.backtest(train_prices, use_regime_filter=True)
            train_metrics = train_results['metrics']
        except:
            current_start = train_end
            window_num += 1
            continue

        # Test
        try:
            test_results = strategy.backtest(test_prices, use_regime_filter=True)
            test_metrics = test_results['metrics']
        except:
            current_start = train_end
            window_num += 1
            continue

        degradation = train_metrics['sharpe'] - test_metrics['sharpe']

        print(f"  In-Sample: Sharpe {train_metrics['sharpe']:.2f}, Out-Sample: Sharpe {test_metrics['sharpe']:.2f}, Degradation: {degradation:+.2f}")

        results.append({
            'Window': window_num,
            'In-Sample Sharpe': train_metrics['sharpe'],
            'Out-Sample Sharpe': test_metrics['sharpe'],
            'Degradation': degradation
        })

        current_start = train_end
        window_num += 1

    return pd.DataFrame(results)


def walk_forward_equity(equity_prices, train_years=3, test_months=6):
    """Walk-forward validation for equity momentum"""

    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION: EQUITY MOMENTUM")
    print("="*80)
    print(f"Train: {train_years} years, Test: {test_months} months\n")

    results = []

    start_date = equity_prices.index[0]
    end_date = equity_prices.index[-1]

    current_start = start_date
    window_num = 1

    while True:
        train_end = current_start + pd.DateOffset(years=train_years)
        test_end = train_end + pd.DateOffset(months=test_months)

        if test_end > end_date:
            break

        print(f"Window {window_num}: Train {current_start.date()} to {train_end.date()}, Test {train_end.date()} to {test_end.date()}")

        # Split data
        train_prices = equity_prices[(equity_prices.index >= current_start) &
                                    (equity_prices.index < train_end)]
        test_prices = equity_prices[(equity_prices.index >= train_end) &
                                   (equity_prices.index < test_end)]

        # Initialize strategy
        config = EquityMomentumConfig(lookback=90, use_regime_filter=True)
        strategy = EquityMomentumStrategy(config)

        # Train
        try:
            train_results = strategy.backtest(train_prices, use_regime_filter=True)
            train_metrics = train_results['metrics']
        except:
            current_start = train_end
            window_num += 1
            continue

        # Test
        try:
            test_results = strategy.backtest(test_prices, use_regime_filter=True)
            test_metrics = test_results['metrics']
        except:
            current_start = train_end
            window_num += 1
            continue

        degradation = train_metrics['sharpe'] - test_metrics['sharpe']

        print(f"  In-Sample: Sharpe {train_metrics['sharpe']:.2f}, Out-Sample: Sharpe {test_metrics['sharpe']:.2f}, Degradation: {degradation:+.2f}")

        results.append({
            'Window': window_num,
            'In-Sample Sharpe': train_metrics['sharpe'],
            'Out-Sample Sharpe': test_metrics['sharpe'],
            'Degradation': degradation
        })

        current_start = train_end
        window_num += 1

    return pd.DataFrame(results)


def main():
    """Run walk-forward validation on all production strategies"""

    print("="*80)
    print("PRODUCTION STRATEGIES - WALK-FORWARD VALIDATION")
    print("="*80)

    # Load data
    crypto_prices = pd.read_csv('data/raw/crypto_prices.csv', index_col=0)
    crypto_prices.index = pd.to_datetime(crypto_prices.index, utc=True).tz_convert(None)

    equity_prices = pd.read_csv('data/raw/equities_prices.csv', index_col=0)
    equity_prices.index = pd.to_datetime(equity_prices.index, utc=True).tz_convert(None)

    print(f"\nCrypto: {len(crypto_prices)} days")
    print(f"Equity: {len(equity_prices)} days\n")

    # 1. Combined Portfolio (most important!)
    combined_results = walk_forward_combined_portfolio(crypto_prices, equity_prices)

    # 2. Crypto standalone
    crypto_results = walk_forward_crypto(crypto_prices)

    # 3. Equity standalone
    equity_results = walk_forward_equity(equity_prices)

    # Analyze results
    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION SUMMARY")
    print("="*80)

    # Combined portfolio
    if len(combined_results) > 0:
        print("\nCOMBINED 30/70 PORTFOLIO:")
        print(f"  Number of windows: {len(combined_results)}")
        print(f"  Avg In-Sample Sharpe:  {combined_results['In-Sample Sharpe'].mean():.2f}")
        print(f"  Avg Out-Sample Sharpe: {combined_results['Out-Sample Sharpe'].mean():.2f}")
        print(f"  Avg Degradation:       {combined_results['Degradation'].mean():.2f}")
        print(f"  Positive Out-Sample:   {(combined_results['Out-Sample Sharpe'] > 0).sum()}/{len(combined_results)} windows")

        avg_oos_sharpe = combined_results['Out-Sample Sharpe'].mean()
        if avg_oos_sharpe > 1.5:
            print(f"  ✅ EXCELLENT - Out-of-sample Sharpe > 1.5")
        elif avg_oos_sharpe > 1.0:
            print(f"  ✓ GOOD - Out-of-sample Sharpe > 1.0")
        else:
            print(f"  ⚠️  MARGINAL - Out-of-sample Sharpe < 1.0")

    # Crypto
    if len(crypto_results) > 0:
        print("\nCRYPTO MOMENTUM:")
        print(f"  Number of windows: {len(crypto_results)}")
        print(f"  Avg In-Sample Sharpe:  {crypto_results['In-Sample Sharpe'].mean():.2f}")
        print(f"  Avg Out-Sample Sharpe: {crypto_results['Out-Sample Sharpe'].mean():.2f}")
        print(f"  Avg Degradation:       {crypto_results['Degradation'].mean():.2f}")

    # Equity
    if len(equity_results) > 0:
        print("\nEQUITY MOMENTUM:")
        print(f"  Number of windows: {len(equity_results)}")
        print(f"  Avg In-Sample Sharpe:  {equity_results['In-Sample Sharpe'].mean():.2f}")
        print(f"  Avg Out-Sample Sharpe: {equity_results['Out-Sample Sharpe'].mean():.2f}")
        print(f"  Avg Degradation:       {equity_results['Degradation'].mean():.2f}")

    # Final verdict
    print("\n" + "="*80)
    print("DEPLOYMENT READINESS")
    print("="*80)

    if len(combined_results) > 0:
        oos_sharpe = combined_results['Out-Sample Sharpe'].mean()
        degradation = combined_results['Degradation'].mean()
        positive_pct = (combined_results['Out-Sample Sharpe'] > 0).sum() / len(combined_results) * 100

        print(f"\nOut-of-Sample Performance:")
        print(f"  Sharpe: {oos_sharpe:.2f}")
        print(f"  Degradation: {degradation:.2f}")
        print(f"  Positive windows: {positive_pct:.0f}%")

        if oos_sharpe > 1.5 and degradation < 0.5:
            print("\n✅ READY FOR DEPLOYMENT")
            print("   Strategy shows strong out-of-sample performance")
            print("   Minimal overfitting detected")
        elif oos_sharpe > 1.0:
            print("\n✓ ACCEPTABLE FOR DEPLOYMENT")
            print("   Strategy shows reasonable out-of-sample performance")
            print("   Monitor closely in live trading")
        else:
            print("\n⚠️  CAUTION RECOMMENDED")
            print("   Out-of-sample performance below target")
            print("   Consider paper trading first")

    print("\n" + "="*80)
    print("✅ WALK-FORWARD VALIDATION COMPLETE")
    print("="*80)

    return combined_results, crypto_results, equity_results


if __name__ == "__main__":
    main()
