"""
Kelly Criterion Position Sizing

Implements Kelly Criterion for optimal position sizing to maximize
long-term growth while accounting for risk.

The Kelly Criterion Formula:
f* = (p * b - q) / b

Where:
- f* = optimal fraction of capital to bet
- p = probability of winning (win rate)
- q = probability of losing (1 - p)
- b = payoff ratio (average win / average loss)

For Trading:
- Estimate p from historical win rate
- Estimate b from win/loss ratio
- Apply fractional Kelly (0.25x - 0.5x) for safety

Benefits:
1. Mathematically optimal for long-term growth
2. Accounts for both win rate AND payoff ratio
3. Automatically scales position size with edge
4. Prevents over-leveraging

Limitations:
1. Assumes edge is stable (can change)
2. Full Kelly is aggressive (use fractional)
3. Requires accurate estimation of parameters
4. Path-dependent (drawdowns still occur)

Academic Foundation:
- Kelly (1956): A new interpretation of information rate
- Thorp (1969): Optimal gambling systems for favorable games
- MacLean, Thorp & Ziemba (2011): Kelly Capital Growth Investment Criterion
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
import warnings


@dataclass
class KellyConfig:
    """Configuration for Kelly Criterion position sizing"""

    # Kelly fraction (reduce for safety)
    kelly_fraction: float = 0.25  # Use 1/4 Kelly for safety

    # Estimation windows
    win_rate_lookback: int = 252  # 1 year for win rate estimation
    payoff_lookback: int = 252    # 1 year for payoff estimation

    # Minimum requirements
    min_trades_required: int = 30  # Need at least 30 trades to estimate
    min_win_rate: float = 0.40     # Must have >40% win rate
    min_payoff_ratio: float = 1.0  # Must have positive payoff

    # Leverage limits
    max_leverage: float = 2.0      # Cap at 2x leverage
    min_leverage: float = 0.1      # Floor at 10% (always stay invested)

    # Risk management
    use_rolling_estimation: bool = True  # Re-estimate over time
    use_ensemble_kelly: bool = True      # Average multiple estimation methods


class KellyPositionSizer:
    """
    Kelly Criterion Position Sizing

    Calculates optimal position sizes using Kelly Criterion based on
    historical win rates and payoff ratios.
    """

    def __init__(self, config: Optional[KellyConfig] = None):
        """Initialize Kelly position sizer"""
        self.config = config or KellyConfig()

    def calculate_win_rate(self, returns: pd.Series,
                          lookback: Optional[int] = None) -> float:
        """
        Calculate win rate (probability of positive return)

        Args:
            returns: Strategy returns
            lookback: Lookback period (default from config)

        Returns:
            Win rate (0 to 1)
        """
        lookback = lookback or self.config.win_rate_lookback

        if len(returns) < lookback:
            recent_returns = returns
        else:
            recent_returns = returns.iloc[-lookback:]

        # Only count trading days (non-zero returns)
        trading_returns = recent_returns[recent_returns != 0]

        if len(trading_returns) < self.config.min_trades_required:
            return 0.0

        win_rate = (trading_returns > 0).sum() / len(trading_returns)

        return win_rate

    def calculate_payoff_ratio(self, returns: pd.Series,
                               lookback: Optional[int] = None) -> float:
        """
        Calculate payoff ratio (average win / average loss)

        Args:
            returns: Strategy returns
            lookback: Lookback period (default from config)

        Returns:
            Payoff ratio (>0)
        """
        lookback = lookback or self.config.payoff_lookback

        if len(returns) < lookback:
            recent_returns = returns
        else:
            recent_returns = returns.iloc[-lookback:]

        # Only count trading days
        trading_returns = recent_returns[recent_returns != 0]

        if len(trading_returns) < self.config.min_trades_required:
            return 0.0

        # Separate wins and losses
        wins = trading_returns[trading_returns > 0]
        losses = trading_returns[trading_returns < 0]

        if len(wins) == 0 or len(losses) == 0:
            return 0.0

        avg_win = wins.mean()
        avg_loss = abs(losses.mean())

        if avg_loss == 0:
            return 0.0

        payoff_ratio = avg_win / avg_loss

        return payoff_ratio

    def calculate_kelly_fraction(self, win_rate: float,
                                 payoff_ratio: float) -> float:
        """
        Calculate Kelly fraction

        Args:
            win_rate: Probability of winning (p)
            payoff_ratio: Average win / average loss (b)

        Returns:
            Kelly fraction (can be negative if no edge)
        """
        if payoff_ratio <= 0:
            return 0.0

        # Kelly formula: f* = (p * b - q) / b
        p = win_rate
        q = 1 - win_rate
        b = payoff_ratio

        kelly_f = (p * b - q) / b

        return kelly_f

    def calculate_optimal_leverage(self, returns: pd.Series,
                                   method: str = 'standard') -> float:
        """
        Calculate optimal leverage using Kelly Criterion

        Args:
            returns: Strategy returns
            method: 'standard', 'simplified', or 'ensemble'

        Returns:
            Optimal leverage multiplier
        """
        if len(returns) < self.config.min_trades_required:
            warnings.warn(f"Insufficient data: {len(returns)} returns, need {self.config.min_trades_required}")
            return self.config.min_leverage

        if method == 'standard':
            leverage = self._kelly_standard(returns)
        elif method == 'simplified':
            leverage = self._kelly_simplified(returns)
        elif method == 'ensemble':
            leverage = self._kelly_ensemble(returns)
        else:
            raise ValueError(f"Unknown method: {method}")

        return leverage

    def _kelly_standard(self, returns: pd.Series) -> float:
        """Standard Kelly Criterion calculation"""

        win_rate = self.calculate_win_rate(returns)
        payoff_ratio = self.calculate_payoff_ratio(returns)

        # Check minimum requirements
        if win_rate < self.config.min_win_rate:
            return self.config.min_leverage

        if payoff_ratio < self.config.min_payoff_ratio:
            return self.config.min_leverage

        # Calculate Kelly fraction
        kelly_f = self.calculate_kelly_fraction(win_rate, payoff_ratio)

        # Apply fractional Kelly for safety
        fractional_kelly = kelly_f * self.config.kelly_fraction

        # Apply limits
        leverage = np.clip(fractional_kelly,
                          self.config.min_leverage,
                          self.config.max_leverage)

        return leverage

    def _kelly_simplified(self, returns: pd.Series) -> float:
        """
        Simplified Kelly using Sharpe ratio approximation

        For normally distributed returns:
        f* ≈ (μ / σ²) = Sharpe / σ
        """
        if len(returns) < self.config.min_trades_required:
            return self.config.min_leverage

        # Annualize
        mean_return = returns.mean() * 252
        volatility = returns.std() * np.sqrt(252)

        if volatility == 0:
            return self.config.min_leverage

        # Simplified Kelly
        kelly_f = mean_return / (volatility ** 2)

        # Apply fractional Kelly
        fractional_kelly = kelly_f * self.config.kelly_fraction

        # Apply limits
        leverage = np.clip(fractional_kelly,
                          self.config.min_leverage,
                          self.config.max_leverage)

        return leverage

    def _kelly_ensemble(self, returns: pd.Series) -> float:
        """Ensemble Kelly: average multiple estimation methods"""

        standard_kelly = self._kelly_standard(returns)
        simplified_kelly = self._kelly_simplified(returns)

        # Average the two methods
        ensemble_kelly = (standard_kelly + simplified_kelly) / 2

        return ensemble_kelly

    def calculate_rolling_leverage(self, returns: pd.Series,
                                   method: str = 'ensemble') -> pd.Series:
        """
        Calculate rolling optimal leverage over time

        Args:
            returns: Strategy returns
            method: Estimation method

        Returns:
            Time series of optimal leverage
        """
        leverage_series = pd.Series(self.config.min_leverage, index=returns.index)

        lookback = max(self.config.win_rate_lookback, self.config.payoff_lookback)

        for i in range(lookback, len(returns)):
            historical_returns = returns.iloc[:i]

            optimal_leverage = self.calculate_optimal_leverage(
                historical_returns,
                method=method
            )

            leverage_series.iloc[i] = optimal_leverage

        return leverage_series

    def analyze_kelly_statistics(self, returns: pd.Series) -> Dict:
        """
        Analyze Kelly Criterion parameters and optimal leverage

        Args:
            returns: Strategy returns

        Returns:
            Dictionary with Kelly statistics
        """
        # Calculate components
        win_rate = self.calculate_win_rate(returns)
        payoff_ratio = self.calculate_payoff_ratio(returns)

        # Calculate Kelly fractions
        full_kelly = self.calculate_kelly_fraction(win_rate, payoff_ratio)
        fractional_kelly = full_kelly * self.config.kelly_fraction

        # Calculate leverages
        standard_leverage = self._kelly_standard(returns)
        simplified_leverage = self._kelly_simplified(returns)
        ensemble_leverage = self._kelly_ensemble(returns)

        # Trading statistics
        trading_returns = returns[returns != 0]
        wins = trading_returns[trading_returns > 0]
        losses = trading_returns[trading_returns < 0]

        return {
            'win_rate': win_rate * 100,
            'loss_rate': (1 - win_rate) * 100,
            'payoff_ratio': payoff_ratio,
            'avg_win': wins.mean() * 100 if len(wins) > 0 else 0,
            'avg_loss': losses.mean() * 100 if len(losses) > 0 else 0,
            'num_trades': len(trading_returns),
            'num_wins': len(wins),
            'num_losses': len(losses),
            'full_kelly': full_kelly,
            'fractional_kelly': fractional_kelly,
            'standard_leverage': standard_leverage,
            'simplified_leverage': simplified_leverage,
            'ensemble_leverage': ensemble_leverage,
            'recommended_leverage': ensemble_leverage
        }

    def apply_kelly_sizing(self, base_weights: pd.DataFrame,
                          returns: pd.Series,
                          method: str = 'ensemble') -> pd.DataFrame:
        """
        Apply Kelly Criterion sizing to base portfolio weights

        Args:
            base_weights: Base portfolio weights (time x assets)
            returns: Historical strategy returns
            method: Kelly estimation method

        Returns:
            Adjusted weights with Kelly sizing
        """
        # Calculate rolling leverage
        leverage_series = self.calculate_rolling_leverage(returns, method=method)

        # Apply leverage to weights
        kelly_weights = base_weights.copy()

        for date in kelly_weights.index:
            if date in leverage_series.index:
                leverage = leverage_series.loc[date]
                kelly_weights.loc[date] = kelly_weights.loc[date] * leverage

        return kelly_weights


def main():
    """Test Kelly Criterion position sizing"""

    print("="*80)
    print("KELLY CRITERION POSITION SIZING")
    print("="*80)

    # Add src to path for imports
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

    # Load equity strategy results
    from strategies.production.equity_momentum import EquityMomentumStrategy, EquityMomentumConfig

    prices = pd.read_csv('data/raw/equities_prices.csv', index_col=0)
    prices.index = pd.to_datetime(prices.index, utc=True).tz_convert(None)

    print(f"\nTesting on {len(prices)} days from {prices.index[0].date()} to {prices.index[-1].date()}\n")

    # Run baseline equity momentum
    equity_config = EquityMomentumConfig(lookback=90, use_regime_filter=True)
    equity_strategy = EquityMomentumStrategy(equity_config)

    print("Running baseline equity momentum strategy...")
    baseline_results = equity_strategy.backtest(prices, use_regime_filter=True)
    baseline_returns = baseline_results['returns']

    print(f"Baseline Sharpe: {baseline_results['metrics']['sharpe']:.2f}")
    print(f"Baseline Return: {baseline_results['metrics']['annual_return']:.1f}%\n")

    # Initialize Kelly position sizer
    print("="*80)
    print("KELLY CRITERION ANALYSIS")
    print("="*80)

    kelly_sizer = KellyPositionSizer()

    # Analyze Kelly statistics
    kelly_stats = kelly_sizer.analyze_kelly_statistics(baseline_returns)

    print(f"\nStrategy Statistics:")
    print(f"  Win Rate:        {kelly_stats['win_rate']:.1f}%")
    print(f"  Loss Rate:       {kelly_stats['loss_rate']:.1f}%")
    print(f"  Payoff Ratio:    {kelly_stats['payoff_ratio']:.2f}")
    print(f"  Avg Win:         {kelly_stats['avg_win']:.3f}%")
    print(f"  Avg Loss:        {kelly_stats['avg_loss']:.3f}%")
    print(f"  Total Trades:    {kelly_stats['num_trades']}")

    print(f"\nKelly Fractions:")
    print(f"  Full Kelly:      {kelly_stats['full_kelly']:.2f}x")
    print(f"  1/4 Kelly:       {kelly_stats['fractional_kelly']:.2f}x")

    print(f"\nRecommended Leverage:")
    print(f"  Standard Method:    {kelly_stats['standard_leverage']:.2f}x")
    print(f"  Simplified Method:  {kelly_stats['simplified_leverage']:.2f}x")
    print(f"  Ensemble Method:    {kelly_stats['ensemble_leverage']:.2f}x")

    print(f"\n✅ RECOMMENDED: {kelly_stats['recommended_leverage']:.2f}x leverage")

    # Apply Kelly sizing to strategy
    print("\n" + "="*80)
    print("STRATEGY WITH KELLY SIZING")
    print("="*80)

    kelly_weights = kelly_sizer.apply_kelly_sizing(
        baseline_results['weights'],
        baseline_returns,
        method='ensemble'
    )

    # Backtest with Kelly weights
    returns = prices.pct_change()
    kelly_returns = pd.Series(0.0, index=returns.index)

    prev_weights = pd.Series(0.0, index=prices.columns)
    transaction_cost = 0.0010

    for date in returns.index[1:]:
        current_weights = kelly_weights.loc[date]

        if current_weights.sum() > 0:
            daily_return = (returns.loc[date] * current_weights).sum()

            # Transaction costs
            turnover = (current_weights - prev_weights).abs().sum()
            daily_return -= turnover * transaction_cost

            kelly_returns[date] = daily_return
            prev_weights = current_weights
        else:
            prev_weights = pd.Series(0.0, index=prices.columns)

    # Calculate Kelly metrics
    total_return = (1 + kelly_returns).prod() - 1
    years = len(kelly_returns) / 252
    ann_return = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    ann_vol = kelly_returns.std() * np.sqrt(252)
    sharpe = ann_return / ann_vol if ann_vol > 0 else 0

    cum_returns = (1 + kelly_returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()

    print(f"\nKelly-Sized Strategy:")
    print(f"  Annual Return:  {ann_return*100:.1f}%")
    print(f"  Volatility:     {ann_vol*100:.1f}%")
    print(f"  Sharpe:         {sharpe:.2f}")
    print(f"  Max Drawdown:   {max_dd*100:.1f}%")

    # Comparison
    print("\n" + "="*80)
    print("BASELINE vs KELLY COMPARISON")
    print("="*80)

    comparison = pd.DataFrame([
        {
            'Strategy': 'Baseline (No Kelly)',
            'Annual Return': f"{baseline_results['metrics']['annual_return']:.1f}%",
            'Sharpe': f"{baseline_results['metrics']['sharpe']:.2f}",
            'Max DD': f"{baseline_results['metrics']['max_drawdown']:.1f}%",
            'Leverage': '1.00x'
        },
        {
            'Strategy': 'Kelly Sized',
            'Annual Return': f"{ann_return*100:.1f}%",
            'Sharpe': f"{sharpe:.2f}",
            'Max DD': f"{max_dd*100:.1f}%",
            'Leverage': f"{kelly_stats['recommended_leverage']:.2f}x"
        }
    ])

    print("\n" + comparison.to_string(index=False))

    # Improvement analysis
    sharpe_improvement = sharpe - baseline_results['metrics']['sharpe']
    return_improvement = (ann_return*100) - baseline_results['metrics']['annual_return']

    print("\n" + "="*80)
    print("KELLY SIZING IMPACT")
    print("="*80)

    print(f"\nSharpe Improvement:  {sharpe_improvement:+.2f}")
    print(f"Return Improvement:  {return_improvement:+.1f}pp")

    if sharpe_improvement > 0.2:
        print("\n✅ Kelly sizing SIGNIFICANTLY improves performance!")
        print("   Use Kelly-based position sizing for optimal growth")
    elif sharpe_improvement > 0:
        print("\n✓ Kelly sizing provides modest improvement")
    else:
        print("\n⚠️  Kelly sizing does not improve performance")
        print("   Strategy may already be near-optimal or parameters need tuning")

    print("\n" + "="*80)
    print("✅ KELLY CRITERION POSITION SIZING READY")
    print("="*80)

    return kelly_returns, kelly_stats


if __name__ == "__main__":
    main()
