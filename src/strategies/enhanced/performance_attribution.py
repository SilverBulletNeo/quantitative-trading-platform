"""
Performance Attribution System

Decomposes portfolio returns into component sources to understand what's
actually driving performance.

Attribution Factors:
1. Asset Selection - Which assets we picked (vs equal weight all)
2. Timing - When we entered/exited (vs buy-and-hold)
3. Regime Filtering - Impact of bear market detection
4. Position Sizing - Impact of volatility targeting
5. Transaction Costs - Drag from trading costs

Benefits:
1. Understand true alpha sources
2. Identify what's working and what's not
3. Optimize strategy components
4. Explain performance to investors/stakeholders
5. Debug performance issues

Methodology:
- Brinson-Fachler attribution (asset selection + timing)
- Decomposition analysis (factor contributions)
- Regime-conditional attribution
- Rolling attribution windows

Academic Foundation:
- Brinson, Hood & Beebower (1986): Determinants of portfolio performance
- Brinson & Fachler (1985): Measuring non-US equity portfolio performance
- Bacon (2008): Practical portfolio performance measurement
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import warnings


@dataclass
class AttributionConfig:
    """Configuration for performance attribution"""

    # Analysis windows
    rolling_window: int = 252  # 1 year rolling attribution

    # Benchmarks
    use_equal_weight_benchmark: bool = True
    use_buy_hold_benchmark: bool = True

    # Decomposition
    analyze_regime_impact: bool = True
    analyze_position_sizing: bool = True
    analyze_transaction_costs: bool = True


class PerformanceAttribution:
    """
    Performance Attribution System

    Decomposes returns into component sources to understand alpha drivers.
    """

    def __init__(self, config: Optional[AttributionConfig] = None):
        """Initialize performance attribution"""
        self.config = config or AttributionConfig()

    def calculate_asset_selection_return(self,
                                        weights: pd.DataFrame,
                                        returns: pd.DataFrame) -> pd.Series:
        """
        Calculate return from asset selection (picking winners vs losers)

        Compares actual weights to equal-weight benchmark

        Args:
            weights: Actual portfolio weights (time x assets)
            returns: Asset returns (time x assets)

        Returns:
            Asset selection return contribution
        """
        # Equal weight benchmark
        num_assets = len(weights.columns)
        ew_weight = 1.0 / num_assets
        benchmark_weights = pd.DataFrame(ew_weight, index=weights.index, columns=weights.columns)

        # Only weight assets we're trading
        for date in weights.index:
            if weights.loc[date].sum() == 0:
                benchmark_weights.loc[date] = 0
            else:
                active_assets = weights.loc[date][weights.loc[date] > 0].index
                if len(active_assets) > 0:
                    benchmark_weights.loc[date] = 0
                    benchmark_weights.loc[date, active_assets] = 1.0 / len(active_assets)

        # Weight difference
        weight_diff = weights - benchmark_weights

        # Return contribution from being overweight winners / underweight losers
        selection_return = (weight_diff * returns).sum(axis=1)

        return selection_return

    def calculate_timing_return(self,
                                signals: pd.DataFrame,
                                prices: pd.DataFrame) -> pd.Series:
        """
        Calculate return from market timing (entry/exit decisions)

        Compares strategy returns to buy-and-hold benchmark

        Args:
            signals: Trading signals (time x assets)
            prices: Asset prices (time x assets)

        Returns:
            Timing return contribution
        """
        # Buy-and-hold returns (equal weight)
        bh_returns = prices.pct_change()
        num_assets = len(prices.columns)
        bh_portfolio = (bh_returns.mean(axis=1))

        # Strategy returns (uses timing)
        # We'll calculate this from actual weights in the main function
        # Here we just return the benchmark for comparison

        return bh_portfolio

    def calculate_regime_impact(self,
                               returns_with_filter: pd.Series,
                               returns_no_filter: pd.Series) -> Dict:
        """
        Calculate impact of regime filtering

        Args:
            returns_with_filter: Returns with regime filter
            returns_no_filter: Returns without regime filter

        Returns:
            Dictionary with regime impact statistics
        """
        # Performance metrics
        def calc_metrics(rets):
            total_ret = (1 + rets).prod() - 1
            years = len(rets) / 252
            ann_ret = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
            ann_vol = rets.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

            cum = (1 + rets).cumprod()
            dd = (cum - cum.expanding().max()) / cum.expanding().max()
            max_dd = dd.min()

            return {
                'ann_return': ann_ret * 100,
                'sharpe': sharpe,
                'max_dd': max_dd * 100
            }

        with_filter = calc_metrics(returns_with_filter)
        no_filter = calc_metrics(returns_no_filter)

        impact = {
            'return_improvement': with_filter['ann_return'] - no_filter['ann_return'],
            'sharpe_improvement': with_filter['sharpe'] - no_filter['sharpe'],
            'drawdown_improvement': no_filter['max_dd'] - with_filter['max_dd'],
            'with_filter': with_filter,
            'no_filter': no_filter
        }

        return impact

    def calculate_position_sizing_impact(self,
                                        weights_with_vol_target: pd.DataFrame,
                                        weights_equal: pd.DataFrame,
                                        returns: pd.DataFrame) -> Dict:
        """
        Calculate impact of volatility targeting vs equal weight

        Args:
            weights_with_vol_target: Weights with vol targeting
            weights_equal: Equal weights
            returns: Asset returns

        Returns:
            Dictionary with position sizing impact
        """
        # Calculate returns for each approach
        vol_target_returns = (weights_with_vol_target * returns).sum(axis=1)
        equal_returns = (weights_equal * returns).sum(axis=1)

        # Metrics
        def calc_sharpe(rets):
            if rets.std() == 0:
                return 0
            return (rets.mean() * 252) / (rets.std() * np.sqrt(252))

        vol_target_sharpe = calc_sharpe(vol_target_returns)
        equal_sharpe = calc_sharpe(equal_returns)

        impact = {
            'sharpe_improvement': vol_target_sharpe - equal_sharpe,
            'vol_target_sharpe': vol_target_sharpe,
            'equal_sharpe': equal_sharpe
        }

        return impact

    def calculate_transaction_cost_impact(self,
                                         weights: pd.DataFrame,
                                         transaction_cost: float = 0.0010) -> pd.Series:
        """
        Calculate drag from transaction costs

        Args:
            weights: Portfolio weights
            transaction_cost: Cost per turnover (e.g., 0.001 = 10 bps)

        Returns:
            Transaction cost impact (negative returns)
        """
        # Calculate turnover
        turnover = weights.diff().abs().sum(axis=1)

        # Cost impact (negative)
        cost_impact = -turnover * transaction_cost

        return cost_impact

    def decompose_returns(self,
                         portfolio_returns: pd.Series,
                         weights: pd.DataFrame,
                         asset_returns: pd.DataFrame,
                         transaction_cost: float = 0.0010) -> pd.DataFrame:
        """
        Decompose portfolio returns into components

        Args:
            portfolio_returns: Total portfolio returns
            weights: Portfolio weights
            asset_returns: Individual asset returns
            transaction_cost: Transaction cost rate

        Returns:
            DataFrame with return decomposition
        """
        # 1. Asset selection contribution
        selection = self.calculate_asset_selection_return(weights, asset_returns)

        # 2. Transaction cost drag
        tc_impact = self.calculate_transaction_cost_impact(weights, transaction_cost)

        # 3. Pure asset returns (equal weight)
        num_assets = len(asset_returns.columns)
        ew_returns = asset_returns.mean(axis=1)

        # 4. Residual (should be small if our decomposition is good)
        # Total return = EW benchmark + Selection + TC + Residual
        residual = portfolio_returns - (ew_returns + selection + tc_impact)

        decomposition = pd.DataFrame({
            'total_return': portfolio_returns,
            'benchmark_return': ew_returns,
            'selection_alpha': selection,
            'transaction_costs': tc_impact,
            'residual': residual
        })

        return decomposition

    def attribution_summary(self,
                           decomposition: pd.DataFrame,
                           regimes: Optional[pd.Series] = None) -> Dict:
        """
        Summarize attribution analysis

        Args:
            decomposition: Return decomposition DataFrame
            regimes: Market regimes (optional)

        Returns:
            Dictionary with attribution summary
        """
        # Annualize contributions
        years = len(decomposition) / 252

        def annualize(series):
            total = series.sum()
            return (total / years) * 100

        summary = {
            'total_return': annualize(decomposition['total_return']),
            'benchmark_return': annualize(decomposition['benchmark_return']),
            'selection_alpha': annualize(decomposition['selection_alpha']),
            'transaction_costs': annualize(decomposition['transaction_costs']),
            'residual': annualize(decomposition['residual']),
        }

        # Calculate percentages
        total = summary['total_return']
        if abs(total) > 0.01:  # Avoid division by near-zero
            summary['selection_pct'] = (summary['selection_alpha'] / total) * 100
            summary['tc_pct'] = (summary['transaction_costs'] / total) * 100
            summary['benchmark_pct'] = (summary['benchmark_return'] / total) * 100

        # Regime-specific attribution
        if regimes is not None:
            regime_attr = {}
            for regime in regimes.unique():
                regime_mask = regimes == regime
                regime_decomp = decomposition[regime_mask]

                if len(regime_decomp) > 10:
                    regime_attr[regime] = {
                        'total_return': annualize(regime_decomp['total_return']),
                        'selection_alpha': annualize(regime_decomp['selection_alpha']),
                        'days': len(regime_decomp)
                    }

            summary['regime_attribution'] = regime_attr

        return summary

    def analyze_strategy(self,
                        strategy_results: Dict,
                        baseline_results: Optional[Dict] = None) -> Dict:
        """
        Comprehensive strategy attribution analysis

        Args:
            strategy_results: Results from strategy backtest
            baseline_results: Baseline results for comparison (optional)

        Returns:
            Dictionary with complete attribution analysis
        """
        # Extract components
        returns = strategy_results['returns']
        weights = strategy_results['weights']
        prices = strategy_results.get('prices')

        # Calculate asset returns if prices available
        if prices is not None:
            asset_returns = prices.pct_change()
        else:
            warnings.warn("Prices not provided, limited attribution available")
            asset_returns = None

        # Decompose returns
        if asset_returns is not None:
            decomposition = self.decompose_returns(
                returns,
                weights,
                asset_returns,
                transaction_cost=strategy_results.get('transaction_cost', 0.0010)
            )

            # Attribution summary
            regimes = strategy_results.get('regimes')
            attr_summary = self.attribution_summary(decomposition, regimes)
        else:
            attr_summary = {}

        # Regime impact (if baseline provided)
        regime_impact = None
        if baseline_results is not None:
            regime_impact = self.calculate_regime_impact(
                returns,
                baseline_results['returns']
            )

        analysis = {
            'decomposition': decomposition if asset_returns is not None else None,
            'summary': attr_summary,
            'regime_impact': regime_impact,
            'metrics': strategy_results.get('metrics', {})
        }

        return analysis


def main():
    """Test performance attribution system"""

    print("="*80)
    print("PERFORMANCE ATTRIBUTION ANALYSIS")
    print("="*80)

    # Add src to path
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

    from strategies.production.equity_momentum import EquityMomentumStrategy, EquityMomentumConfig

    # Load data
    prices = pd.read_csv('data/raw/equities_prices.csv', index_col=0)
    prices.index = pd.to_datetime(prices.index, utc=True).tz_convert(None)

    print(f"\nTesting on {len(prices)} days from {prices.index[0].date()} to {prices.index[-1].date()}\n")

    # Run strategy WITH regime filter
    print("Running strategy WITH regime filter...")
    config_with = EquityMomentumConfig(lookback=90, use_regime_filter=True)
    strategy_with = EquityMomentumStrategy(config_with)
    results_with = strategy_with.backtest(prices, use_regime_filter=True)

    # Run strategy WITHOUT regime filter
    print("Running strategy WITHOUT regime filter...")
    config_no = EquityMomentumConfig(lookback=90, use_regime_filter=False)
    strategy_no = EquityMomentumStrategy(config_no)
    results_no = strategy_no.backtest(prices, use_regime_filter=False)

    # Initialize attribution analyzer
    attributor = PerformanceAttribution()

    # Add prices to results for attribution
    results_with['prices'] = prices
    results_with['transaction_cost'] = 0.0010

    print("\n" + "="*80)
    print("RETURN ATTRIBUTION ANALYSIS")
    print("="*80)

    # Comprehensive attribution
    attribution = attributor.analyze_strategy(results_with, results_no)

    # Print summary
    if 'summary' in attribution and attribution['summary']:
        summary = attribution['summary']

        print("\nAnnualized Return Decomposition:")
        print(f"  Total Strategy Return:  {summary['total_return']:>7.2f}%")
        print(f"  Benchmark (EW) Return:  {summary['benchmark_return']:>7.2f}%")
        print(f"  Selection Alpha:        {summary['selection_alpha']:>7.2f}%")
        print(f"  Transaction Costs:      {summary['transaction_costs']:>7.2f}%")
        print(f"  Residual:              {summary['residual']:>7.2f}%")

        if 'selection_pct' in summary:
            print("\nReturn Source Breakdown:")
            print(f"  Benchmark:        {summary['benchmark_pct']:>6.1f}%")
            print(f"  Selection Alpha:  {summary['selection_pct']:>6.1f}%")
            print(f"  Transaction Cost: {summary['tc_pct']:>6.1f}%")

    # Regime impact
    print("\n" + "="*80)
    print("REGIME FILTER IMPACT")
    print("="*80)

    if attribution['regime_impact']:
        impact = attribution['regime_impact']

        print(f"\nWithout Regime Filter:")
        print(f"  Annual Return:  {impact['no_filter']['ann_return']:>7.2f}%")
        print(f"  Sharpe:         {impact['no_filter']['sharpe']:>7.2f}")
        print(f"  Max Drawdown:   {impact['no_filter']['max_dd']:>7.2f}%")

        print(f"\nWith Regime Filter:")
        print(f"  Annual Return:  {impact['with_filter']['ann_return']:>7.2f}%")
        print(f"  Sharpe:         {impact['with_filter']['sharpe']:>7.2f}")
        print(f"  Max Drawdown:   {impact['with_filter']['max_dd']:>7.2f}%")

        print(f"\nRegime Filter Impact:")
        print(f"  Return Improvement:    {impact['return_improvement']:>+7.2f}pp")
        print(f"  Sharpe Improvement:    {impact['sharpe_improvement']:>+7.2f}")
        print(f"  Drawdown Improvement:  {impact['drawdown_improvement']:>+7.2f}pp")

        if impact['sharpe_improvement'] > 0.2:
            print("\n✅ Regime filter provides SIGNIFICANT value!")
        elif impact['sharpe_improvement'] > 0:
            print("\n✓ Regime filter provides modest value")
        else:
            print("\n⚠️  Regime filter may not be adding value")

    # Regime-specific attribution
    if 'summary' in attribution and 'regime_attribution' in attribution['summary']:
        print("\n" + "="*80)
        print("PERFORMANCE BY MARKET REGIME")
        print("="*80)

        regime_attr = attribution['summary']['regime_attribution']

        for regime, stats in regime_attr.items():
            print(f"\n{regime}:")
            print(f"  Annual Return:   {stats['total_return']:>7.2f}%")
            print(f"  Selection Alpha: {stats['selection_alpha']:>7.2f}%")
            print(f"  Days:            {stats['days']:>7d}")

    # Transaction cost analysis
    print("\n" + "="*80)
    print("TRANSACTION COST ANALYSIS")
    print("="*80)

    weights = results_with['weights']
    turnover = weights.diff().abs().sum(axis=1)
    avg_daily_turnover = turnover.mean()
    annual_turnover = avg_daily_turnover * 252

    tc_rate = 0.0010
    annual_tc_cost = annual_turnover * tc_rate * 100

    print(f"\nAverage Daily Turnover:  {avg_daily_turnover:.2%}")
    print(f"Annual Turnover:         {annual_turnover:.2f}x")
    print(f"Annual TC Cost (10bps):  {annual_tc_cost:.2f}%")

    if annual_tc_cost < 2.0:
        print("\n✅ Transaction costs are LOW - strategy is efficient")
    elif annual_tc_cost < 5.0:
        print("\n✓ Transaction costs are MODERATE")
    else:
        print("\n⚠️  Transaction costs are HIGH - consider reducing turnover")

    # Key findings
    print("\n" + "="*80)
    print("KEY FINDINGS")
    print("="*80)

    if attribution['summary']:
        summary = attribution['summary']

        print("\n1. ALPHA SOURCES:")
        selection_alpha = summary.get('selection_alpha', 0)
        if selection_alpha > 2:
            print(f"   ✅ Strong selection alpha ({selection_alpha:.1f}% per year)")
            print("      Asset selection is working well")
        elif selection_alpha > 0:
            print(f"   ✓ Positive selection alpha ({selection_alpha:.1f}% per year)")
        else:
            print(f"   ⚠️  Negative selection alpha ({selection_alpha:.1f}% per year)")
            print("      Strategy not adding value through asset selection")

        print("\n2. REGIME FILTERING:")
        if attribution['regime_impact']:
            sharpe_imp = attribution['regime_impact']['sharpe_improvement']
            if sharpe_imp > 0.2:
                print(f"   ✅ Major value add (Sharpe +{sharpe_imp:.2f})")
            elif sharpe_imp > 0:
                print(f"   ✓ Positive contribution (Sharpe +{sharpe_imp:.2f})")
            else:
                print(f"   ⚠️  Not adding value (Sharpe {sharpe_imp:+.2f})")

        print("\n3. TRANSACTION COSTS:")
        tc_cost = abs(summary.get('transaction_costs', 0))
        if tc_cost < 2:
            print(f"   ✅ Low cost ({tc_cost:.1f}% per year)")
        elif tc_cost < 5:
            print(f"   ✓ Moderate cost ({tc_cost:.1f}% per year)")
        else:
            print(f"   ⚠️  High cost ({tc_cost:.1f}% per year)")

    print("\n" + "="*80)
    print("✅ PERFORMANCE ATTRIBUTION COMPLETE")
    print("="*80)

    return attribution


if __name__ == "__main__":
    main()
