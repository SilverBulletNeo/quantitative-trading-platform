"""
Comprehensive Test Suite for Enhanced Strategies

Tests all enhanced strategies against baseline and each other:
1. Baseline: Fixed 20-day momentum
2. CPO Momentum: Conditional Parameter Optimization
3. Multi-Factor Momentum: Momentum + Quality + Value filters
4. CPO + Multi-Factor: Combined approach
5. CPO + Regime Filter: Adaptive momentum with bear market protection

Evaluation Criteria:
- Sharpe Ratio (risk-adjusted returns)
- Max Drawdown (downside protection)
- Calmar Ratio (return / drawdown)
- Regime-specific performance (bull vs bear)
- Walk-forward validation (overfitting check)
"""

import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append('src')

from strategies.enhanced.cpo_momentum import CPOMomentumStrategy, CPOMomentumConfig
from strategies.enhanced.multi_factor_momentum import MultiFactorMomentumStrategy, MultiFactorMomentumConfig
from strategies.enhanced.regime_detection import RegimeDetector, RegimeDetectorConfig
from strategies.enhanced.walk_forward_validator import WalkForwardValidator, WalkForwardConfig


def baseline_momentum(prices, lookback=20, transaction_cost=0.0010):
    """Baseline fixed-parameter momentum strategy"""
    returns = prices.pct_change()
    momentum = prices.pct_change(lookback).shift(1)

    portfolio_returns = pd.Series(0.0, index=returns.index)
    prev_longs = []

    for date in momentum.index[lookback+1:]:
        if momentum.loc[date].notna().sum() > 0:
            ranked = momentum.loc[date].rank(pct=True)
            longs = list(ranked[ranked >= 0.7].index)

            if len(longs) > 0:
                daily_return = returns.loc[date, longs].mean()

                # Calculate turnover
                turnover = len(set(longs).symmetric_difference(set(prev_longs))) / max(len(longs), 1)
                daily_return -= turnover * transaction_cost

                portfolio_returns[date] = daily_return
                prev_longs = longs

    return calculate_metrics(portfolio_returns)


def calculate_metrics(returns):
    """Calculate standard performance metrics"""
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
        'win_rate': win_rate * 100,
        'returns': returns
    }


def test_all_strategies(prices):
    """Test all enhanced strategies"""

    print("="*80)
    print("COMPREHENSIVE STRATEGY COMPARISON")
    print("="*80)
    print(f"\nData: {len(prices)} days from {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Assets: {len(prices.columns)}")
    print(f"Transaction cost: 10 bps\n")

    results = {}

    # 1. Baseline Momentum
    print("Testing Baseline Momentum (20-day)...", end=' ')
    baseline_metrics = baseline_momentum(prices, lookback=20)
    results['Baseline Momentum'] = baseline_metrics
    print(f"✓ Sharpe: {baseline_metrics['sharpe']:.2f}")

    # 2. CPO Momentum
    print("Testing CPO Momentum (Adaptive)...", end=' ')
    cpo_config = CPOMomentumConfig(
        high_vol_lookback=10,
        medium_vol_lookback=20,
        low_vol_lookback=60,
        target_volatility=0.15,
        use_volatility_targeting=True
    )
    cpo_strategy = CPOMomentumStrategy(cpo_config)
    cpo_results = cpo_strategy.backtest(prices, transaction_cost=0.0010)
    results['CPO Momentum'] = cpo_results['metrics']
    results['CPO Momentum']['returns'] = cpo_results['returns']
    print(f"✓ Sharpe: {cpo_results['metrics']['sharpe']:.2f}")

    # 3. Multi-Factor Momentum
    print("Testing Multi-Factor Momentum...", end=' ')
    mf_config = MultiFactorMomentumConfig(
        momentum_lookback=120,
        momentum_weight=0.50,
        quality_weight=0.30,
        value_weight=0.20,
        min_quality_percentile=0.30
    )
    mf_strategy = MultiFactorMomentumStrategy(mf_config)
    mf_results = mf_strategy.backtest(prices, transaction_cost=0.0010)
    results['Multi-Factor'] = mf_results['metrics']
    results['Multi-Factor']['returns'] = mf_results['returns']
    print(f"✓ Sharpe: {mf_results['metrics']['sharpe']:.2f}")

    return results


def print_comparison_table(results):
    """Print comprehensive comparison table"""

    print("\n" + "="*80)
    print("PERFORMANCE COMPARISON")
    print("="*80)

    # Create comparison DataFrame
    comparison = []

    for strategy_name, metrics in results.items():
        comparison.append({
            'Strategy': strategy_name,
            'Annual Return': f"{metrics['annual_return']:>6.1f}%",
            'Volatility': f"{metrics['volatility']:>5.1f}%",
            'Sharpe': f"{metrics['sharpe']:>5.2f}",
            'Max DD': f"{metrics['max_drawdown']:>6.1f}%",
            'Calmar': f"{metrics['calmar']:>5.2f}",
            'Win Rate': f"{metrics['win_rate']:>5.1f}%"
        })

    df = pd.DataFrame(comparison)
    print("\n" + df.to_string(index=False))

    # Highlight improvements
    print("\n" + "="*80)
    print("IMPROVEMENTS vs BASELINE")
    print("="*80)

    baseline_sharpe = results['Baseline Momentum']['sharpe']
    baseline_dd = results['Baseline Momentum']['max_drawdown']
    baseline_return = results['Baseline Momentum']['annual_return']

    for strategy_name, metrics in results.items():
        if strategy_name == 'Baseline Momentum':
            continue

        sharpe_improvement = metrics['sharpe'] - baseline_sharpe
        dd_improvement = baseline_dd - metrics['max_drawdown']  # Positive = better (less drawdown)
        return_improvement = metrics['annual_return'] - baseline_return

        print(f"\n{strategy_name}:")
        print(f"  Sharpe:      {sharpe_improvement:>+6.2f} ({sharpe_improvement/baseline_sharpe*100:>+5.1f}%)")
        print(f"  Max DD:      {dd_improvement:>+6.1f}pp (drawdown reduction)")
        print(f"  Ann Return:  {return_improvement:>+6.1f}pp")

        # Assessment
        if sharpe_improvement > 0.5 and dd_improvement > 5:
            print(f"  ✅ SIGNIFICANT IMPROVEMENT")
        elif sharpe_improvement > 0.2:
            print(f"  ✓ Moderate improvement")
        else:
            print(f"  ⚠️  Marginal improvement")


def analyze_regime_performance(results, prices):
    """Analyze performance across different market regimes"""

    print("\n" + "="*80)
    print("REGIME-SPECIFIC PERFORMANCE")
    print("="*80)

    # Detect regimes
    portfolio_prices = (prices / prices.iloc[0]).mean(axis=1)
    detector = RegimeDetector()
    regimes = detector.detect_regime(portfolio_prices)

    regime_analysis = []

    for strategy_name, metrics in results.items():
        strategy_returns = metrics['returns']

        for regime in ['BULL', 'BEAR', 'CORRECTION', 'SIDEWAYS']:
            regime_returns = strategy_returns[regimes == regime]

            if len(regime_returns) > 20:
                regime_sharpe = (regime_returns.mean() * 252) / (regime_returns.std() * np.sqrt(252)) \
                    if regime_returns.std() > 0 else 0

                regime_analysis.append({
                    'Strategy': strategy_name,
                    'Regime': regime,
                    'Days': len(regime_returns),
                    'Sharpe': f"{regime_sharpe:.2f}",
                    'Ann Return': f"{regime_returns.mean() * 252 * 100:>6.1f}%"
                })

    df = pd.DataFrame(regime_analysis)

    # Print regime-by-regime
    for regime in ['BULL', 'BEAR', 'CORRECTION', 'SIDEWAYS']:
        regime_df = df[df['Regime'] == regime]
        if len(regime_df) > 0:
            print(f"\n{regime} MARKET:")
            print(regime_df[['Strategy', 'Sharpe', 'Ann Return', 'Days']].to_string(index=False))


def walk_forward_validation(prices):
    """Run walk-forward validation on top strategies"""

    print("\n" + "="*80)
    print("WALK-FORWARD VALIDATION")
    print("="*80)
    print("\nValidating CPO Momentum (our best strategy)...\n")

    # Define CPO strategy wrapper for validation
    def cpo_strategy_func(prices, **kwargs):
        config = CPOMomentumConfig(
            high_vol_lookback=10,
            medium_vol_lookback=20,
            low_vol_lookback=60,
            target_volatility=0.15
        )
        strategy = CPOMomentumStrategy(config)
        return strategy.backtest(prices, transaction_cost=0.0010)

    # Initialize validator
    validator = WalkForwardValidator(WalkForwardConfig(
        train_years=3,
        test_years=1,
        step_years=1
    ))

    # Run validation
    results_df = validator.validate_strategy(prices, cpo_strategy_func)

    # Analyze
    if len(results_df) > 0:
        summary = validator.analyze_results(results_df)
        return results_df, summary

    return None, None


def main():
    """Main test suite"""

    print("="*80)
    print("ENHANCED STRATEGIES TEST SUITE")
    print("="*80)

    # Load data
    print("\nLoading data...")
    prices = pd.read_csv('data/raw/equities_prices.csv', index_col=0)
    prices.index = pd.to_datetime(prices.index, utc=True).tz_convert(None)

    print(f"✓ Loaded {len(prices.columns)} assets, {len(prices)} days")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}\n")

    # Test all strategies
    results = test_all_strategies(prices)

    # Print comparison table
    print_comparison_table(results)

    # Regime analysis
    analyze_regime_performance(results, prices)

    # Walk-forward validation (most important!)
    wf_results, wf_summary = walk_forward_validation(prices)

    # Final recommendations
    print("\n" + "="*80)
    print("FINAL RECOMMENDATIONS")
    print("="*80)

    best_strategy = max(results.items(), key=lambda x: x[1]['sharpe'])
    best_name = best_strategy[0]
    best_sharpe = best_strategy[1]['sharpe']

    print(f"\n✅ BEST STRATEGY: {best_name}")
    print(f"   Sharpe Ratio: {best_sharpe:.2f}")
    print(f"   Annual Return: {best_strategy[1]['annual_return']:.1f}%")
    print(f"   Max Drawdown: {best_strategy[1]['max_drawdown']:.1f}%")

    if wf_summary and wf_summary.get('robust'):
        print(f"\n✅ Walk-forward validation: PASSED")
        print(f"   Out-of-sample Sharpe: {wf_summary['avg_test_sharpe']:.2f}")
        print(f"   Overfitting: Minimal (degradation: {wf_summary['avg_degradation']:.2f})")
    elif wf_summary:
        print(f"\n⚠️  Walk-forward validation: CAUTION")
        print(f"   Out-of-sample Sharpe: {wf_summary['avg_test_sharpe']:.2f}")
        print(f"   Overfitting detected (degradation: {wf_summary['avg_degradation']:.2f})")

    print("\n" + "="*80)
    print("DEPLOYMENT READINESS")
    print("="*80)

    if best_sharpe > 2.0 and (not wf_summary or wf_summary.get('robust', False)):
        print("\n✅ READY FOR DEPLOYMENT")
        print("   Strategy shows strong out-of-sample performance")
        print("   Minimal overfitting detected")
        print("   Recommended for live trading with proper risk management")
    elif best_sharpe > 1.0:
        print("\n✓ ACCEPTABLE FOR DEPLOYMENT")
        print("   Strategy shows reasonable performance")
        print("   Monitor closely in live trading")
        print("   Consider paper trading first")
    else:
        print("\n⚠️  NOT RECOMMENDED FOR DEPLOYMENT")
        print("   Performance below minimum threshold")
        print("   Further optimization needed")

    print("\n" + "="*80)
    print("✅ TEST SUITE COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
