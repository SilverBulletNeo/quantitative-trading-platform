"""
Multi-Asset Strategy Testing

Tests momentum and enhanced strategies across different asset classes:
1. Equities (2012-2025, 13.5 years)
2. Crypto (2020-2025, 5.6 years)
3. Commodities (2010-2025, 16 years)
4. Bonds (2010-2025, 16 years)

Key Questions:
- Does momentum work across all asset classes?
- Do crypto assets behave differently (higher vol)?
- Do commodities trend better (traditional momentum assets)?
- What's the optimal lookback for each asset class?
- Can we build a diversified multi-asset momentum portfolio?
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def calculate_metrics(returns):
    """Calculate standard performance metrics"""
    if len(returns) == 0 or returns.std() == 0:
        return {
            'total_return': 0,
            'annual_return': 0,
            'volatility': 0,
            'sharpe': 0,
            'max_drawdown': 0,
            'calmar': 0,
            'win_rate': 0
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
    win_rate = (returns > 0).sum() / len(returns[returns != 0]) if (returns != 0).sum() > 0 else 0

    return {
        'total_return': total_return * 100,
        'annual_return': ann_return * 100,
        'volatility': ann_vol * 100,
        'sharpe': sharpe,
        'max_drawdown': max_dd * 100,
        'calmar': calmar,
        'win_rate': win_rate * 100
    }


def momentum_strategy(prices, lookback=20, transaction_cost=0.0010, long_pct=0.7, short_pct=0.3):
    """
    Baseline momentum strategy

    Args:
        prices: Asset prices (time x assets)
        lookback: Momentum lookback period
        transaction_cost: Cost per trade (10 bps default)
        long_pct: Percentile threshold for longs (0.7 = top 30%)
        short_pct: Percentile threshold for shorts (0.3 = bottom 30%)

    Returns:
        Dict with performance metrics and returns
    """
    returns = prices.pct_change()
    momentum = prices.pct_change(lookback).shift(1)

    portfolio_returns = pd.Series(0.0, index=returns.index)
    prev_longs = []

    for date in momentum.index[lookback+1:]:
        if momentum.loc[date].notna().sum() > 0:
            ranked = momentum.loc[date].rank(pct=True)
            longs = list(ranked[ranked >= long_pct].index)

            if len(longs) > 0:
                daily_return = returns.loc[date, longs].mean()

                # Calculate turnover
                turnover = len(set(longs).symmetric_difference(set(prev_longs))) / max(len(longs), 1)
                daily_return -= turnover * transaction_cost

                portfolio_returns[date] = daily_return
                prev_longs = longs

    metrics = calculate_metrics(portfolio_returns)
    metrics['returns'] = portfolio_returns
    metrics['lookback'] = lookback

    return metrics


def test_asset_class(prices, asset_class_name, lookbacks=[5, 10, 20, 40, 60, 90, 120]):
    """
    Test momentum strategy across different lookback periods for an asset class

    Args:
        prices: Price data for asset class
        asset_class_name: Name of asset class (for reporting)
        lookbacks: List of lookback periods to test

    Returns:
        DataFrame with results for each lookback
    """
    print(f"\n{'='*80}")
    print(f"TESTING: {asset_class_name.upper()}")
    print(f"{'='*80}")
    print(f"Assets: {len(prices.columns)}")
    print(f"Period: {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Days: {len(prices)}")
    print(f"Years: {len(prices)/252:.1f}\n")

    results = []

    for lookback in lookbacks:
        print(f"Testing {lookback}-day momentum...", end=' ')

        try:
            metrics = momentum_strategy(prices, lookback=lookback, transaction_cost=0.0010)

            results.append({
                'Asset Class': asset_class_name,
                'Lookback': f"{lookback}d",
                'Annual Return': f"{metrics['annual_return']:>6.1f}%",
                'Volatility': f"{metrics['volatility']:>5.1f}%",
                'Sharpe': f"{metrics['sharpe']:>5.2f}",
                'Max DD': f"{metrics['max_drawdown']:>6.1f}%",
                'Calmar': f"{metrics['calmar']:>5.2f}",
                'Win Rate': f"{metrics['win_rate']:>5.1f}%",
                '_sharpe_raw': metrics['sharpe'],
                '_return_raw': metrics['annual_return'],
                '_returns': metrics['returns']
            })

            print(f"✓ Sharpe: {metrics['sharpe']:.2f}")

        except Exception as e:
            print(f"✗ Error: {e}")

    return pd.DataFrame(results)


def analyze_asset_class_characteristics(prices, asset_class_name):
    """
    Analyze key characteristics of an asset class

    Args:
        prices: Price data
        asset_class_name: Name of asset class

    Returns:
        Dict with characteristics
    """
    returns = prices.mean(axis=1).pct_change()

    # Volatility
    ann_vol = returns.std() * np.sqrt(252) * 100

    # Trend persistence (autocorrelation)
    autocorr_1d = returns.autocorr(1)
    autocorr_5d = returns.autocorr(5)
    autocorr_20d = returns.autocorr(20)

    # Tail risk
    skewness = returns.skew()
    kurtosis = returns.kurtosis()

    # Drawdown
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min() * 100

    # Buy and hold
    bh_metrics = calculate_metrics(returns)

    return {
        'Asset Class': asset_class_name,
        'Ann Vol': f"{ann_vol:.1f}%",
        'Autocorr 1d': f"{autocorr_1d:.3f}",
        'Autocorr 5d': f"{autocorr_5d:.3f}",
        'Autocorr 20d': f"{autocorr_20d:.3f}",
        'Skewness': f"{skewness:.2f}",
        'Kurtosis': f"{kurtosis:.1f}",
        'Max DD': f"{max_dd:.1f}%",
        'BH Sharpe': f"{bh_metrics['sharpe']:.2f}",
        'BH Return': f"{bh_metrics['annual_return']:.1f}%"
    }


def find_best_strategy_per_asset(results_df):
    """Find best lookback period for each asset class"""

    print(f"\n{'='*80}")
    print("BEST STRATEGY PER ASSET CLASS")
    print(f"{'='*80}\n")

    best_strategies = []

    for asset_class in results_df['Asset Class'].unique():
        asset_results = results_df[results_df['Asset Class'] == asset_class]

        # Find best by Sharpe
        best_idx = asset_results['_sharpe_raw'].idxmax()
        best = asset_results.loc[best_idx]

        best_strategies.append({
            'Asset Class': asset_class,
            'Best Lookback': best['Lookback'],
            'Sharpe': best['Sharpe'],
            'Annual Return': best['Annual Return'],
            'Max DD': best['Max DD'],
            'Calmar': best['Calmar']
        })

        print(f"{asset_class}:")
        print(f"  Best Lookback: {best['Lookback']}")
        print(f"  Sharpe:        {best['Sharpe']}")
        print(f"  Annual Return: {best['Annual Return']}")
        print(f"  Max DD:        {best['Max DD']}")
        print()

    return pd.DataFrame(best_strategies)


def build_multi_asset_momentum_portfolio(all_results):
    """
    Build a diversified multi-asset momentum portfolio

    Uses best lookback for each asset class, equal weight across assets
    """
    print(f"\n{'='*80}")
    print("MULTI-ASSET MOMENTUM PORTFOLIO")
    print(f"{'='*80}\n")

    # Get best strategy returns for each asset class
    portfolio_returns = {}

    for asset_class in all_results['Asset Class'].unique():
        asset_results = all_results[all_results['Asset Class'] == asset_class]
        best_idx = asset_results['_sharpe_raw'].idxmax()
        best_returns = asset_results.loc[best_idx, '_returns']
        portfolio_returns[asset_class] = best_returns

    # Combine into DataFrame and align dates
    portfolio_df = pd.DataFrame(portfolio_returns)

    # Equal weight across asset classes
    combined_returns = portfolio_df.mean(axis=1)

    metrics = calculate_metrics(combined_returns)

    print(f"Combined Multi-Asset Portfolio:")
    print(f"  Annual Return:   {metrics['annual_return']:>10.1f}%")
    print(f"  Volatility:      {metrics['volatility']:>10.1f}%")
    print(f"  Sharpe Ratio:    {metrics['sharpe']:>10.2f}")
    print(f"  Max Drawdown:    {metrics['max_drawdown']:>10.1f}%")
    print(f"  Calmar Ratio:    {metrics['calmar']:>10.2f}")
    print(f"  Win Rate:        {metrics['win_rate']:>10.1f}%")

    # Correlation between asset class strategies
    print(f"\nCorrelation Matrix:")
    corr = portfolio_df.corr()
    print(corr.to_string())

    return metrics, portfolio_df


def main():
    """Main multi-asset testing suite"""

    print("="*80)
    print("MULTI-ASSET MOMENTUM STRATEGY TESTING")
    print("="*80)
    print("\nTesting momentum across Equities, Crypto, Commodities, and Bonds")
    print("Testing lookbacks: 5, 10, 20, 40, 60, 90, 120 days\n")

    # Load all asset classes
    print("Loading data...")

    # Equities
    equities = pd.read_csv('data/raw/equities_prices.csv', index_col=0)
    equities.index = pd.to_datetime(equities.index, utc=True).tz_convert(None)
    print(f"✓ Equities: {len(equities.columns)} assets, {len(equities)} days")

    # Crypto
    crypto = pd.read_csv('data/raw/crypto_prices.csv', index_col=0)
    crypto.index = pd.to_datetime(crypto.index, utc=True).tz_convert(None)
    print(f"✓ Crypto: {len(crypto.columns)} assets, {len(crypto)} days")

    # Commodities
    commodities = pd.read_csv('data/raw/commodities_prices.csv', index_col=0)
    commodities.index = pd.to_datetime(commodities.index, utc=True).tz_convert(None)
    print(f"✓ Commodities: {len(commodities.columns)} assets, {len(commodities)} days")

    # Bonds
    bonds = pd.read_csv('data/raw/bonds_prices.csv', index_col=0)
    bonds.index = pd.to_datetime(bonds.index, utc=True).tz_convert(None)
    print(f"✓ Bonds: {len(bonds.columns)} assets, {len(bonds)} days")

    # Analyze characteristics
    print(f"\n{'='*80}")
    print("ASSET CLASS CHARACTERISTICS")
    print(f"{'='*80}\n")

    characteristics = []
    characteristics.append(analyze_asset_class_characteristics(equities, 'Equities'))
    characteristics.append(analyze_asset_class_characteristics(crypto, 'Crypto'))
    characteristics.append(analyze_asset_class_characteristics(commodities, 'Commodities'))
    characteristics.append(analyze_asset_class_characteristics(bonds, 'Bonds'))

    char_df = pd.DataFrame(characteristics)
    print(char_df.to_string(index=False))

    # Test each asset class
    all_results = []

    # Equities
    eq_results = test_asset_class(equities, 'Equities')
    all_results.append(eq_results)

    # Crypto
    crypto_results = test_asset_class(crypto, 'Crypto')
    all_results.append(crypto_results)

    # Commodities
    comm_results = test_asset_class(commodities, 'Commodities')
    all_results.append(comm_results)

    # Bonds
    bond_results = test_asset_class(bonds, 'Bonds')
    all_results.append(bond_results)

    # Combine results
    combined_results = pd.concat(all_results, ignore_index=True)

    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPLETE RESULTS - ALL ASSET CLASSES")
    print(f"{'='*80}\n")

    display_cols = ['Asset Class', 'Lookback', 'Annual Return', 'Sharpe', 'Max DD', 'Calmar']
    print(combined_results[display_cols].to_string(index=False))

    # Find best for each asset class
    best_df = find_best_strategy_per_asset(combined_results)

    # Build multi-asset portfolio
    multi_metrics, multi_returns = build_multi_asset_momentum_portfolio(combined_results)

    # Key insights
    print(f"\n{'='*80}")
    print("KEY INSIGHTS")
    print(f"{'='*80}\n")

    # Which asset class has best momentum?
    best_asset = best_df.loc[best_df['Sharpe'].str.replace('%', '').astype(float).idxmax()]
    print(f"1. Best Asset Class for Momentum:")
    print(f"   {best_asset['Asset Class']} - Sharpe {best_asset['Sharpe']}, Return {best_asset['Annual Return']}\n")

    # Optimal lookback varies by asset?
    print(f"2. Optimal Lookback by Asset Class:")
    for _, row in best_df.iterrows():
        print(f"   {row['Asset Class']:12s}: {row['Best Lookback']}")
    print()

    # Does multi-asset diversification help?
    eq_best_sharpe = float(best_df[best_df['Asset Class'] == 'Equities']['Sharpe'].values[0])
    print(f"3. Diversification Benefit:")
    print(f"   Best Single Asset (Equities): Sharpe {eq_best_sharpe:.2f}")
    print(f"   Multi-Asset Portfolio:        Sharpe {multi_metrics['sharpe']:.2f}")

    if multi_metrics['sharpe'] > eq_best_sharpe:
        print(f"   ✅ Diversification improves Sharpe by {multi_metrics['sharpe'] - eq_best_sharpe:.2f}")
    else:
        print(f"   ⚠️  No diversification benefit (negative correlation?)")

    print(f"\n{'='*80}")
    print("✅ MULTI-ASSET TESTING COMPLETE")
    print(f"{'='*80}")

    return combined_results, best_df, multi_metrics


if __name__ == "__main__":
    results, best, multi = main()
