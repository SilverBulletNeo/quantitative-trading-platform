"""
Monte Carlo Stress Testing

Simulates thousands of alternative market scenarios to validate strategy
robustness and understand performance distribution.

Simulation Methods:
1. Bootstrap Resampling - Shuffle historical returns
2. Parametric Simulation - Draw from fitted distributions
3. Block Bootstrap - Preserve autocorrelation structure
4. Stress Scenarios - Extreme market conditions

Metrics Analyzed:
- Sharpe ratio distribution
- Drawdown distribution
- Probability of loss
- Value at Risk (VaR)
- Expected Shortfall (CVaR)
- Recovery time distribution

Benefits:
1. Understand performance range (best/worst cases)
2. Estimate probability of losses
3. Test robustness to different market conditions
4. Validate that backtest results aren't lucky
5. Set realistic expectations for live trading

Academic Foundation:
- Efron (1979): Bootstrap methods
- Jorion (2006): Value at Risk
- Glasserman (2003): Monte Carlo Methods in Financial Engineering
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import warnings
from scipy import stats


@dataclass
class MonteCarloConfig:
    """Configuration for Monte Carlo stress testing"""

    # Simulation parameters
    n_simulations: int = 10000        # Number of simulations
    block_size: int = 21              # Block size for block bootstrap (~1 month)

    # Risk metrics
    confidence_levels: List[float] = None  # Will be set in __post_init__

    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.90, 0.95, 0.99]  # 90%, 95%, 99%

    # Stress scenarios
    include_stress_tests: bool = True
    crash_magnitude: float = -0.30    # -30% crash
    bear_market_length: int = 252     # 1 year bear market


class MonteCarloStressTester:
    """
    Monte Carlo Stress Testing

    Simulates thousands of scenarios to validate strategy robustness.
    """

    def __init__(self, config: Optional[MonteCarloConfig] = None):
        """Initialize Monte Carlo stress tester"""
        self.config = config or MonteCarloConfig()

    def bootstrap_resample(self, returns: pd.Series,
                          n_simulations: Optional[int] = None) -> pd.DataFrame:
        """
        Bootstrap resampling of returns

        Randomly shuffles historical returns to create alternative scenarios

        Args:
            returns: Historical returns
            n_simulations: Number of simulations (default from config)

        Returns:
            DataFrame with simulated return series
        """
        n_sim = n_simulations or self.config.n_simulations
        n_periods = len(returns)

        simulations = pd.DataFrame(
            index=range(n_periods),
            columns=range(n_sim)
        )

        for sim in range(n_sim):
            # Random sample with replacement
            sampled_returns = np.random.choice(returns.values, size=n_periods, replace=True)
            simulations[sim] = sampled_returns

        return simulations

    def block_bootstrap(self, returns: pd.Series,
                       block_size: Optional[int] = None,
                       n_simulations: Optional[int] = None) -> pd.DataFrame:
        """
        Block bootstrap resampling

        Preserves short-term autocorrelation by sampling blocks

        Args:
            returns: Historical returns
            block_size: Size of blocks (default from config)
            n_simulations: Number of simulations (default from config)

        Returns:
            DataFrame with simulated return series
        """
        n_sim = n_simulations or self.config.n_simulations
        blk_size = block_size or self.config.block_size
        n_periods = len(returns)

        simulations = pd.DataFrame(
            index=range(n_periods),
            columns=range(n_sim)
        )

        returns_array = returns.values

        for sim in range(n_sim):
            simulated = []

            while len(simulated) < n_periods:
                # Random starting point
                start_idx = np.random.randint(0, n_periods - blk_size)

                # Extract block
                block = returns_array[start_idx:start_idx + blk_size]
                simulated.extend(block)

            # Trim to exact length
            simulations[sim] = simulated[:n_periods]

        return simulations

    def parametric_simulation(self, returns: pd.Series,
                             distribution: str = 'normal',
                             n_simulations: Optional[int] = None) -> pd.DataFrame:
        """
        Parametric simulation using fitted distributions

        Args:
            returns: Historical returns
            distribution: 'normal', 't', or 'skewt'
            n_simulations: Number of simulations (default from config)

        Returns:
            DataFrame with simulated return series
        """
        n_sim = n_simulations or self.config.n_simulations
        n_periods = len(returns)

        # Fit distribution parameters
        if distribution == 'normal':
            mu = returns.mean()
            sigma = returns.std()

            simulations = pd.DataFrame(
                np.random.normal(mu, sigma, size=(n_periods, n_sim)),
                columns=range(n_sim)
            )

        elif distribution == 't':
            # Fit Student's t distribution
            params = stats.t.fit(returns.values)
            df, loc, scale = params

            simulations = pd.DataFrame(
                stats.t.rvs(df, loc=loc, scale=scale, size=(n_periods, n_sim)),
                columns=range(n_sim)
            )

        else:
            raise ValueError(f"Unknown distribution: {distribution}")

        return simulations

    def create_stress_scenarios(self, returns: pd.Series) -> Dict[str, pd.Series]:
        """
        Create stress test scenarios

        Args:
            returns: Historical returns

        Returns:
            Dictionary of stress scenarios
        """
        scenarios = {}

        n_periods = len(returns)

        # 1. Market Crash (sudden -30%)
        crash_returns = returns.copy()
        crash_day = n_periods // 2
        crash_returns.iloc[crash_day] = self.config.crash_magnitude
        scenarios['market_crash'] = crash_returns

        # 2. Extended Bear Market (-50% over 1 year)
        bear_returns = returns.copy()
        bear_start = n_periods // 3
        bear_end = bear_start + self.config.bear_market_length

        # Gradual decline
        bear_daily_return = -0.50 / self.config.bear_market_length
        for i in range(bear_start, min(bear_end, n_periods)):
            bear_returns.iloc[i] = bear_daily_return
        scenarios['bear_market'] = bear_returns

        # 3. High Volatility (2x normal volatility)
        vol_multiplier = 2.0
        high_vol_returns = returns.copy()
        mu = returns.mean()
        sigma = returns.std()
        high_vol_returns = np.random.normal(mu, sigma * vol_multiplier, size=n_periods)
        scenarios['high_volatility'] = pd.Series(high_vol_returns, index=returns.index)

        # 4. Low Returns (halve all returns)
        low_return_returns = returns.copy() * 0.5
        scenarios['low_returns'] = low_return_returns

        # 5. Reversed Returns (flip signs - worst case)
        reversed_returns = -returns.copy()
        scenarios['reversed_returns'] = reversed_returns

        return scenarios

    def calculate_metrics(self, returns: pd.Series) -> Dict:
        """Calculate performance metrics for a return series"""

        if len(returns) == 0 or returns.std() == 0:
            return {
                'total_return': 0,
                'annual_return': 0,
                'sharpe': 0,
                'max_drawdown': 0,
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
            'total_return': total_return * 100,
            'annual_return': ann_return * 100,
            'volatility': ann_vol * 100,
            'sharpe': sharpe,
            'max_drawdown': max_dd * 100,
            'calmar': calmar
        }

    def run_simulations(self, returns: pd.Series,
                       method: str = 'block_bootstrap') -> pd.DataFrame:
        """
        Run Monte Carlo simulations

        Args:
            returns: Historical strategy returns
            method: 'bootstrap', 'block_bootstrap', or 'parametric'

        Returns:
            DataFrame with metrics for each simulation
        """
        print(f"Running {self.config.n_simulations} {method} simulations...")

        # Generate simulated returns
        if method == 'bootstrap':
            simulated_returns = self.bootstrap_resample(returns)
        elif method == 'block_bootstrap':
            simulated_returns = self.block_bootstrap(returns)
        elif method == 'parametric':
            simulated_returns = self.parametric_simulation(returns)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate metrics for each simulation
        results = []

        for sim in simulated_returns.columns:
            sim_returns = simulated_returns[sim]
            metrics = self.calculate_metrics(sim_returns)
            metrics['simulation'] = sim
            results.append(metrics)

        results_df = pd.DataFrame(results)

        return results_df

    def analyze_distribution(self, simulation_results: pd.DataFrame,
                           actual_metrics: Dict) -> Dict:
        """
        Analyze distribution of simulated results

        Args:
            simulation_results: Results from Monte Carlo simulations
            actual_metrics: Actual strategy metrics

        Returns:
            Dictionary with distribution analysis
        """
        analysis = {}

        # Sharpe ratio analysis
        sharpes = simulation_results['sharpe']
        actual_sharpe = actual_metrics['sharpe']

        analysis['sharpe'] = {
            'actual': actual_sharpe,
            'mean': sharpes.mean(),
            'median': sharpes.median(),
            'std': sharpes.std(),
            'percentile_5': sharpes.quantile(0.05),
            'percentile_25': sharpes.quantile(0.25),
            'percentile_75': sharpes.quantile(0.75),
            'percentile_95': sharpes.quantile(0.95),
            'percentile_rank': stats.percentileofscore(sharpes, actual_sharpe)
        }

        # Annual return analysis
        returns = simulation_results['annual_return']
        actual_return = actual_metrics['annual_return']

        analysis['annual_return'] = {
            'actual': actual_return,
            'mean': returns.mean(),
            'median': returns.median(),
            'std': returns.std(),
            'percentile_5': returns.quantile(0.05),
            'percentile_95': returns.quantile(0.95),
            'prob_positive': (returns > 0).mean() * 100,
            'prob_above_10pct': (returns > 10).mean() * 100
        }

        # Drawdown analysis
        drawdowns = simulation_results['max_drawdown']
        actual_dd = actual_metrics['max_drawdown']

        analysis['max_drawdown'] = {
            'actual': actual_dd,
            'mean': drawdowns.mean(),
            'median': drawdowns.median(),
            'std': drawdowns.std(),
            'percentile_5': drawdowns.quantile(0.05),
            'percentile_95': drawdowns.quantile(0.95),
            'prob_below_10pct': (drawdowns > -10).mean() * 100,
            'prob_below_20pct': (drawdowns > -20).mean() * 100
        }

        return analysis

    def run_stress_tests(self, returns: pd.Series,
                        strategy_func) -> pd.DataFrame:
        """
        Run stress test scenarios

        Args:
            returns: Historical returns
            strategy_func: Function to apply strategy to new returns

        Returns:
            DataFrame with stress test results
        """
        scenarios = self.create_stress_scenarios(returns)

        stress_results = []

        for scenario_name, scenario_returns in scenarios.items():
            metrics = self.calculate_metrics(scenario_returns)
            metrics['scenario'] = scenario_name
            stress_results.append(metrics)

        return pd.DataFrame(stress_results)

    def comprehensive_analysis(self, returns: pd.Series,
                              actual_metrics: Dict) -> Dict:
        """
        Comprehensive Monte Carlo analysis

        Args:
            returns: Historical strategy returns
            actual_metrics: Actual backtest metrics

        Returns:
            Dictionary with complete analysis
        """
        print("="*80)
        print("MONTE CARLO STRESS TESTING")
        print("="*80)

        # 1. Block Bootstrap (preserves autocorrelation)
        print("\n1. Block Bootstrap Simulations...")
        bootstrap_results = self.run_simulations(returns, method='block_bootstrap')

        # 2. Parametric Simulations (assumes normal distribution)
        print("2. Parametric Simulations...")
        parametric_results = self.run_simulations(returns, method='parametric')

        # 3. Stress Scenarios
        print("3. Stress Test Scenarios...")
        stress_results = self.run_stress_tests(returns, None)

        # Analyze distributions
        print("4. Analyzing results...")
        bootstrap_analysis = self.analyze_distribution(bootstrap_results, actual_metrics)
        parametric_analysis = self.analyze_distribution(parametric_results, actual_metrics)

        analysis = {
            'bootstrap_results': bootstrap_results,
            'parametric_results': parametric_results,
            'stress_results': stress_results,
            'bootstrap_analysis': bootstrap_analysis,
            'parametric_analysis': parametric_analysis,
            'actual_metrics': actual_metrics
        }

        return analysis


def main():
    """Test Monte Carlo stress testing"""

    print("="*80)
    print("MONTE CARLO STRESS TESTING")
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

    # Run baseline strategy
    print("Running baseline strategy...")
    config = EquityMomentumConfig(lookback=90, use_regime_filter=True)
    strategy = EquityMomentumStrategy(config)
    results = strategy.backtest(prices, use_regime_filter=True)

    actual_returns = results['returns']
    actual_metrics = results['metrics']

    print(f"Actual Sharpe: {actual_metrics['sharpe']:.2f}")
    print(f"Actual Return: {actual_metrics['annual_return']:.1f}%\n")

    # Initialize Monte Carlo tester
    mc_config = MonteCarloConfig(n_simulations=10000)
    mc_tester = MonteCarloStressTester(mc_config)

    # Run comprehensive analysis
    analysis = mc_tester.comprehensive_analysis(actual_returns, actual_metrics)

    # Print results
    print("\n" + "="*80)
    print("SHARPE RATIO DISTRIBUTION")
    print("="*80)

    sharpe_analysis = analysis['bootstrap_analysis']['sharpe']

    print(f"\nActual Sharpe:        {sharpe_analysis['actual']:.2f}")
    print(f"Mean Simulated:       {sharpe_analysis['mean']:.2f}")
    print(f"Median Simulated:     {sharpe_analysis['median']:.2f}")
    print(f"Std Dev:              {sharpe_analysis['std']:.2f}")
    print(f"\n5th Percentile:       {sharpe_analysis['percentile_5']:.2f}")
    print(f"25th Percentile:      {sharpe_analysis['percentile_25']:.2f}")
    print(f"75th Percentile:      {sharpe_analysis['percentile_75']:.2f}")
    print(f"95th Percentile:      {sharpe_analysis['percentile_95']:.2f}")
    print(f"\nActual Percentile Rank: {sharpe_analysis['percentile_rank']:.1f}%")

    if sharpe_analysis['percentile_rank'] > 75:
        print("\n✅ Actual performance in TOP QUARTILE of simulations!")
        print("   Backtest results appear ROBUST")
    elif sharpe_analysis['percentile_rank'] > 50:
        print("\n✓ Actual performance ABOVE MEDIAN")
    elif sharpe_analysis['percentile_rank'] > 25:
        print("\n⚠️  Actual performance BELOW MEDIAN")
    else:
        print("\n⚠️  Actual performance in BOTTOM QUARTILE")
        print("   Backtest may be lucky - exercise caution")

    # Return distribution
    print("\n" + "="*80)
    print("ANNUAL RETURN DISTRIBUTION")
    print("="*80)

    return_analysis = analysis['bootstrap_analysis']['annual_return']

    print(f"\nActual Return:        {return_analysis['actual']:.1f}%")
    print(f"Mean Simulated:       {return_analysis['mean']:.1f}%")
    print(f"Median Simulated:     {return_analysis['median']:.1f}%")
    print(f"\n5th Percentile:       {return_analysis['percentile_5']:.1f}%")
    print(f"95th Percentile:      {return_analysis['percentile_95']:.1f}%")
    print(f"\nProb(Positive):       {return_analysis['prob_positive']:.1f}%")
    print(f"Prob(>10%):           {return_analysis['prob_above_10pct']:.1f}%")

    # Drawdown distribution
    print("\n" + "="*80)
    print("MAXIMUM DRAWDOWN DISTRIBUTION")
    print("="*80)

    dd_analysis = analysis['bootstrap_analysis']['max_drawdown']

    print(f"\nActual Max DD:        {dd_analysis['actual']:.1f}%")
    print(f"Mean Simulated:       {dd_analysis['mean']:.1f}%")
    print(f"Median Simulated:     {dd_analysis['median']:.1f}%")
    print(f"\n5th Percentile:       {dd_analysis['percentile_5']:.1f}% (best)")
    print(f"95th Percentile:      {dd_analysis['percentile_95']:.1f}% (worst)")
    print(f"\nProb(DD < -10%):      {100 - dd_analysis['prob_below_10pct']:.1f}%")
    print(f"Prob(DD < -20%):      {100 - dd_analysis['prob_below_20pct']:.1f}%")

    # Stress test results
    print("\n" + "="*80)
    print("STRESS TEST SCENARIOS")
    print("="*80)

    stress_results = analysis['stress_results']

    print("\n" + stress_results[['scenario', 'sharpe', 'annual_return', 'max_drawdown']].to_string(index=False))

    # Find worst case
    worst_scenario = stress_results.loc[stress_results['sharpe'].idxmin()]

    print(f"\n⚠️  WORST CASE: {worst_scenario['scenario']}")
    print(f"    Sharpe: {worst_scenario['sharpe']:.2f}")
    print(f"    Return: {worst_scenario['annual_return']:.1f}%")
    print(f"    Max DD: {worst_scenario['max_drawdown']:.1f}%")

    # Overall assessment
    print("\n" + "="*80)
    print("ROBUSTNESS ASSESSMENT")
    print("="*80)

    percentile_rank = sharpe_analysis['percentile_rank']
    prob_positive = return_analysis['prob_positive']
    worst_sharpe = stress_results['sharpe'].min()

    print("\nRobustness Checks:")

    # Check 1: Percentile rank
    if percentile_rank > 60:
        print(f"  ✅ Backtest in top {100-percentile_rank:.0f}% of simulations")
    else:
        print(f"  ⚠️  Backtest only at {percentile_rank:.0f}th percentile")

    # Check 2: Probability of positive returns
    if prob_positive > 80:
        print(f"  ✅ High probability of profit ({prob_positive:.0f}%)")
    elif prob_positive > 60:
        print(f"  ✓ Reasonable probability of profit ({prob_positive:.0f}%)")
    else:
        print(f"  ⚠️  Low probability of profit ({prob_positive:.0f}%)")

    # Check 3: Stress test survivability
    if worst_sharpe > 0:
        print(f"  ✅ Positive Sharpe in all stress scenarios")
    elif worst_sharpe > -0.5:
        print(f"  ✓ Survives most stress scenarios")
    else:
        print(f"  ⚠️  Poor performance in stress scenarios")

    # Final verdict
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)

    if percentile_rank > 60 and prob_positive > 75:
        print("\n✅ STRATEGY IS ROBUST")
        print("   Performance appears genuine and repeatable")
        print("   Suitable for live deployment with monitoring")
    elif percentile_rank > 40 and prob_positive > 60:
        print("\n✓ STRATEGY IS REASONABLY ROBUST")
        print("   Performance is decent but not exceptional")
        print("   Consider paper trading first")
    else:
        print("\n⚠️  STRATEGY ROBUSTNESS CONCERNS")
        print("   Backtest may not be representative")
        print("   Exercise caution and extensive paper trading")

    print("\n" + "="*80)
    print("✅ MONTE CARLO STRESS TESTING COMPLETE")
    print("="*80)

    return analysis


if __name__ == "__main__":
    main()
