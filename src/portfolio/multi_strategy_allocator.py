"""
Multi-Strategy Portfolio Allocator

Combines multiple trading strategies into an optimal portfolio allocation.
Treats each strategy as an "asset" and optimally allocates capital across them.

This is the core of a multi-strategy hedge fund:
- Monitor performance of all strategies
- Calculate correlation between strategies
- Dynamically allocate capital based on risk-adjusted performance
- Rebalance periodically

Academic Foundation:
- Grinold & Kahn - "Active Portfolio Management"
- AQR Capital - "Multi-Strategy Diversification"
- Ang, Papanikolaou, Westerfield (2014) - "Portfolio Choice with Illiquid Assets"

Allocation Methods:
1. Risk Parity - Equal risk contribution from each strategy
2. Mean-Variance - Maximize Sharpe ratio across strategies
3. Inverse Sharpe - Weight by historical Sharpe ratios
4. Equal Weight - Simple 1/N allocation (benchmark)

Why Multi-Strategy Works:
- Diversification across alpha sources
- Lower drawdowns (strategies uncorrelated)
- More consistent returns
- Reduced tail risk
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

# Import optimization frameworks
from portfolio.risk_parity import RiskParity, RiskParityConfig
from portfolio.mean_variance_optimization import MeanVarianceOptimization, MVOConfig


@dataclass
class MultiStrategyConfig:
    """Configuration for Multi-Strategy Allocator"""
    allocation_method: str = 'risk_parity'  # 'risk_parity', 'max_sharpe', 'inverse_sharpe', 'equal_weight'
    lookback_period: int = 60  # Days to calculate strategy performance
    rebalance_frequency: int = 20  # Rebalance every N days (monthly)
    min_sharpe_threshold: float = 0.0  # Minimum Sharpe to include strategy
    max_allocation: float = 0.25  # Maximum 25% to single strategy
    min_allocation: float = 0.05  # Minimum 5% to each strategy
    exclude_negative_sharpe: bool = True  # Exclude strategies with negative Sharpe


class MultiStrategyAllocator:
    """
    Multi-Strategy Portfolio Allocator

    Optimally allocates capital across multiple trading strategies.
    """

    def __init__(self, config: MultiStrategyConfig = None):
        self.config = config or MultiStrategyConfig()
        self.rp = RiskParity()
        self.mvo = MeanVarianceOptimization()

    def calculate_strategy_returns(
        self,
        signals: Dict[str, pd.DataFrame],
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate returns for each strategy

        Args:
            signals: Dict of {strategy_name: signals_dataframe}
            prices: DataFrame of asset prices

        Returns:
            DataFrame with columns = strategy names, values = daily returns
        """
        strategy_returns = {}

        for strategy_name, strategy_signals in signals.items():
            # Convert signals to positions (simple equal-weight per signal)
            weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            positions = {col: 0.0 for col in prices.columns}

            for i, (idx, row) in enumerate(strategy_signals.iterrows()):
                for symbol in strategy_signals.columns:
                    if symbol not in prices.columns:
                        continue

                    signal = row[symbol]
                    if signal == 1:
                        # Buy signal - equal weight allocation
                        positions[symbol] = 0.10  # 10% per position
                    elif signal == -1:
                        # Sell/close signal
                        positions[symbol] = 0.0

                    weights.loc[idx, symbol] = positions[symbol]

            # Calculate strategy returns
            price_returns = prices.pct_change()
            strategy_daily_returns = (weights.shift(1) * price_returns).sum(axis=1)

            strategy_returns[strategy_name] = strategy_daily_returns

        return pd.DataFrame(strategy_returns)

    def calculate_strategy_metrics(
        self,
        strategy_returns: pd.DataFrame,
        lookback: int = None
    ) -> pd.DataFrame:
        """
        Calculate performance metrics for each strategy

        Returns DataFrame with metrics:
        - Total Return
        - Sharpe Ratio
        - Volatility
        - Max Drawdown
        - Win Rate
        """
        if lookback is None:
            lookback = self.config.lookback_period

        metrics = []

        for strategy_name in strategy_returns.columns:
            returns = strategy_returns[strategy_name].iloc[-lookback:]

            if len(returns) == 0:
                continue

            # Calculate metrics
            total_return = (1 + returns).prod() - 1
            sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
            volatility = returns.std() * np.sqrt(252)

            # Max drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_dd = drawdown.min()

            # Win rate
            win_rate = (returns > 0).sum() / len(returns) if len(returns) > 0 else 0

            metrics.append({
                'Strategy': strategy_name,
                'Total Return': total_return,
                'Sharpe Ratio': sharpe,
                'Volatility': volatility,
                'Max Drawdown': max_dd,
                'Win Rate': win_rate
            })

        return pd.DataFrame(metrics).set_index('Strategy')

    def filter_strategies(
        self,
        strategy_returns: pd.DataFrame,
        metrics: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Filter strategies based on performance criteria

        Excludes:
        - Strategies with Sharpe < threshold
        - Strategies with negative Sharpe (optional)
        """
        valid_strategies = []

        for strategy in strategy_returns.columns:
            if strategy not in metrics.index:
                continue

            sharpe = metrics.loc[strategy, 'Sharpe Ratio']

            # Check thresholds
            if sharpe < self.config.min_sharpe_threshold:
                continue

            if self.config.exclude_negative_sharpe and sharpe < 0:
                continue

            valid_strategies.append(strategy)

        return strategy_returns[valid_strategies]

    def allocate_equal_weight(self, strategy_returns: pd.DataFrame) -> pd.Series:
        """Equal weight allocation (1/N benchmark)"""
        n = len(strategy_returns.columns)
        weights = pd.Series([1/n] * n, index=strategy_returns.columns)
        return weights

    def allocate_inverse_sharpe(
        self,
        strategy_returns: pd.DataFrame,
        metrics: pd.DataFrame
    ) -> pd.Series:
        """
        Weight by Sharpe ratio

        Higher Sharpe = Higher allocation
        """
        sharpes = metrics.loc[strategy_returns.columns, 'Sharpe Ratio']

        # Ensure all Sharpes are positive
        sharpes = sharpes.clip(lower=0.01)

        # Weight proportional to Sharpe
        weights = sharpes / sharpes.sum()

        # Apply constraints
        weights = weights.clip(
            lower=self.config.min_allocation,
            upper=self.config.max_allocation
        )

        # Renormalize
        weights = weights / weights.sum()

        return weights

    def allocate_risk_parity(self, strategy_returns: pd.DataFrame) -> pd.Series:
        """
        Risk Parity allocation across strategies

        Each strategy contributes equal risk
        """
        # Convert returns to prices for Risk Parity interface
        prices = (1 + strategy_returns).cumprod()

        # Use Risk Parity with custom config
        rp_config = RiskParityConfig(
            method='equal_risk',
            max_position=self.config.max_allocation,
            min_position=self.config.min_allocation
        )
        rp = RiskParity(rp_config)

        result = rp.optimize_portfolio(prices, method='equal_risk')

        return result['weights']

    def allocate_max_sharpe(self, strategy_returns: pd.DataFrame) -> pd.Series:
        """
        Maximum Sharpe Ratio allocation across strategies

        Uses Mean-Variance Optimization
        """
        # Convert returns to prices
        prices = (1 + strategy_returns).cumprod()

        # Use MVO with custom config
        mvo_config = MVOConfig(
            max_position=self.config.max_allocation,
            min_position=self.config.min_allocation
        )
        mvo = MeanVarianceOptimization(mvo_config)

        result = mvo.optimize_portfolio(prices, method='max_sharpe')

        return result['weights']

    def allocate_capital(
        self,
        strategy_returns: pd.DataFrame,
        metrics: pd.DataFrame,
        method: str = None
    ) -> Dict:
        """
        Main allocation function

        Args:
            strategy_returns: DataFrame of strategy returns
            metrics: DataFrame of strategy metrics
            method: Allocation method (overrides config)

        Returns:
            Dict with 'weights', 'method', 'metrics'
        """
        if method is None:
            method = self.config.allocation_method

        # Filter strategies
        filtered_returns = self.filter_strategies(strategy_returns, metrics)

        if len(filtered_returns.columns) == 0:
            print("Warning: No strategies passed filters!")
            return {
                'weights': pd.Series(dtype=float),
                'method': method,
                'metrics': metrics,
                'status': 'no_valid_strategies'
            }

        # Allocate based on method
        if method == 'equal_weight':
            weights = self.allocate_equal_weight(filtered_returns)
        elif method == 'inverse_sharpe':
            weights = self.allocate_inverse_sharpe(filtered_returns, metrics)
        elif method == 'risk_parity':
            try:
                weights = self.allocate_risk_parity(filtered_returns)
            except Exception as e:
                print(f"Risk Parity failed: {e}, falling back to Inverse Sharpe")
                weights = self.allocate_inverse_sharpe(filtered_returns, metrics)
        elif method == 'max_sharpe':
            try:
                weights = self.allocate_max_sharpe(filtered_returns)
            except Exception as e:
                print(f"Max Sharpe failed: {e}, falling back to Inverse Sharpe")
                weights = self.allocate_inverse_sharpe(filtered_returns, metrics)
        else:
            raise ValueError(f"Unknown allocation method: {method}")

        return {
            'weights': weights,
            'method': method,
            'metrics': metrics.loc[weights.index],
            'status': 'success'
        }

    def backtest_allocation(
        self,
        signals: Dict[str, pd.DataFrame],
        prices: pd.DataFrame,
        rebalance_frequency: int = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Backtest multi-strategy allocation with periodic rebalancing

        Args:
            signals: Dict of {strategy_name: signals_dataframe}
            prices: DataFrame of asset prices
            rebalance_frequency: Days between rebalances

        Returns:
            Tuple of (portfolio_returns, allocation_history)
        """
        if rebalance_frequency is None:
            rebalance_frequency = self.config.rebalance_frequency

        # Calculate strategy returns
        print("Calculating strategy returns...")
        strategy_returns = self.calculate_strategy_returns(signals, prices)

        # Initialize tracking
        portfolio_returns = pd.Series(0.0, index=strategy_returns.index)
        allocation_history = []

        # Track current allocation
        current_weights = None
        last_rebalance = 0

        print(f"Running backtest with {len(strategy_returns.columns)} strategies...")

        for i in range(self.config.lookback_period, len(strategy_returns)):
            # Check if it's time to rebalance
            if i - last_rebalance >= rebalance_frequency:
                # Calculate metrics over lookback period
                lookback_returns = strategy_returns.iloc[i-self.config.lookback_period:i]
                metrics = self.calculate_strategy_metrics(lookback_returns)

                # Allocate capital
                allocation = self.allocate_capital(lookback_returns, metrics)

                if allocation['status'] == 'success':
                    current_weights = allocation['weights']

                    allocation_history.append({
                        'date': strategy_returns.index[i],
                        'weights': current_weights.to_dict(),
                        'method': allocation['method']
                    })

                    last_rebalance = i

            # Calculate portfolio return using current weights
            if current_weights is not None and len(current_weights) > 0:
                strategy_ret = strategy_returns.iloc[i]
                # Only use strategies we have weights for
                common_strategies = current_weights.index.intersection(strategy_ret.index)
                portfolio_returns.iloc[i] = (
                    current_weights[common_strategies] * strategy_ret[common_strategies]
                ).sum()

        return portfolio_returns, pd.DataFrame(allocation_history)

    def compare_allocation_methods(
        self,
        signals: Dict[str, pd.DataFrame],
        prices: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Compare all allocation methods

        Shows which method produces best risk-adjusted returns
        """
        methods = ['equal_weight', 'inverse_sharpe', 'risk_parity', 'max_sharpe']
        results = []

        for method in methods:
            print(f"\nTesting {method}...")

            # Temporarily change config
            original_method = self.config.allocation_method
            self.config.allocation_method = method

            # Backtest
            try:
                portfolio_returns, _ = self.backtest_allocation(signals, prices)

                # Calculate metrics
                total_return = (1 + portfolio_returns).prod() - 1
                sharpe = (
                    portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
                    if portfolio_returns.std() != 0 else 0
                )

                cumulative = (1 + portfolio_returns).cumprod()
                running_max = cumulative.expanding().max()
                drawdown = (cumulative - running_max) / running_max
                max_dd = drawdown.min()

                results.append({
                    'Method': method,
                    'Total Return': total_return,
                    'Sharpe Ratio': sharpe,
                    'Max Drawdown': max_dd,
                    'Volatility': portfolio_returns.std() * np.sqrt(252),
                    'Calmar Ratio': total_return / abs(max_dd) if max_dd != 0 else 0
                })

            except Exception as e:
                print(f"  Failed: {e}")
                results.append({
                    'Method': method,
                    'Total Return': 0,
                    'Sharpe Ratio': 0,
                    'Max Drawdown': 0,
                    'Volatility': 0,
                    'Calmar Ratio': 0
                })

            # Restore original
            self.config.allocation_method = original_method

        return pd.DataFrame(results)


if __name__ == "__main__":
    """Test Multi-Strategy Allocator with synthetic strategies"""

    print("=" * 80)
    print("MULTI-STRATEGY PORTFOLIO ALLOCATOR TEST")
    print("=" * 80)
    print()

    # Create synthetic price data
    print("Creating synthetic test data...")
    np.random.seed(42)
    n_days = 500
    n_assets = 4

    dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
    assets = ['BTC-USD', 'ETH-USD', 'SPY', 'TLT']

    # Generate price data
    prices_data = {}
    for asset in assets:
        returns = np.random.normal(0.001, 0.02, n_days)
        prices_data[asset] = (1 + returns).cumprod() * 100

    prices = pd.DataFrame(prices_data, index=dates)

    # Create synthetic strategies with different characteristics
    print("Creating synthetic strategies...")
    print("- Strategy 1: High Sharpe, low frequency")
    print("- Strategy 2: Medium Sharpe, medium frequency")
    print("- Strategy 3: Low Sharpe, high frequency")
    print("- Strategy 4: Negative Sharpe (should be excluded)")
    print()

    # Strategy 1: Conservative, high win rate
    signals_1 = pd.DataFrame(0, index=dates, columns=assets)
    for i in range(50, len(dates), 30):  # Monthly signals
        if np.random.random() > 0.3:  # 70% win rate
            signals_1.iloc[i, 0] = 1  # Buy

    # Strategy 2: Moderate
    signals_2 = pd.DataFrame(0, index=dates, columns=assets)
    for i in range(30, len(dates), 15):  # Bi-weekly
        if np.random.random() > 0.45:  # 55% win rate
            signals_2.iloc[i, 1] = 1

    # Strategy 3: Aggressive, lower win rate
    signals_3 = pd.DataFrame(0, index=dates, columns=assets)
    for i in range(10, len(dates), 5):  # Weekly
        if np.random.random() > 0.5:  # 50% win rate
            signals_3.iloc[i, 2] = 1

    # Strategy 4: Poor performance (negative Sharpe)
    signals_4 = pd.DataFrame(0, index=dates, columns=assets)
    for i in range(20, len(dates), 10):
        if np.random.random() > 0.6:  # 40% win rate
            signals_4.iloc[i, 3] = 1

    signals = {
        'Conservative High Sharpe': signals_1,
        'Moderate Medium Sharpe': signals_2,
        'Aggressive Low Sharpe': signals_3,
        'Poor Negative Sharpe': signals_4
    }

    # Initialize allocator
    config = MultiStrategyConfig(
        allocation_method='risk_parity',
        lookback_period=60,
        rebalance_frequency=30,
        exclude_negative_sharpe=True
    )
    allocator = MultiStrategyAllocator(config)

    # Calculate strategy returns
    print("=" * 80)
    print("INDIVIDUAL STRATEGY PERFORMANCE")
    print("=" * 80)

    strategy_returns = allocator.calculate_strategy_returns(signals, prices)
    metrics = allocator.calculate_strategy_metrics(strategy_returns)

    print()
    print(metrics.round(4).to_string())

    # Test single allocation
    print("\n" + "=" * 80)
    print("CAPITAL ALLOCATION (Risk Parity)")
    print("=" * 80)

    allocation = allocator.allocate_capital(strategy_returns, metrics, method='risk_parity')

    if allocation['status'] == 'success':
        print("\nAllocated Weights:")
        for strategy, weight in allocation['weights'].sort_values(ascending=False).items():
            print(f"  {strategy:30s}: {weight:>6.2%}")

        print(f"\nTotal Allocation: {allocation['weights'].sum():.2%}")
    else:
        print(f"\nAllocation failed: {allocation['status']}")

    # Compare all methods
    print("\n" + "=" * 80)
    print("COMPARING ALL ALLOCATION METHODS")
    print("=" * 80)

    comparison = allocator.compare_allocation_methods(signals, prices)
    print()
    print(comparison.round(4).to_string(index=False))

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("Multi-Strategy Allocation:")
    print("  - Diversifies across alpha sources")
    print("  - Reduces single-strategy risk")
    print("  - More stable returns through market cycles")
    print()
    print("Best Method:")
    best_method = comparison.loc[comparison['Sharpe Ratio'].idxmax(), 'Method']
    print(f"  - {best_method} (highest Sharpe ratio)")
    print()
    print("=" * 80)
    print("Test complete!")
    print("=" * 80)
