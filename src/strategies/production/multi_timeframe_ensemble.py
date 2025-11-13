"""
Multi-Timeframe Ensemble Momentum Strategy

Combines multiple momentum lookback periods to reduce overfitting and
improve robustness. Hedges against parameter sensitivity.

Key Idea:
Instead of betting everything on one lookback (e.g., 90-day), combine:
- Short-term (40-60 days): Captures recent trends
- Medium-term (90 days): Our best single parameter
- Long-term (120-150 days): Captures institutional flows

Benefits:
1. Less sensitive to specific parameter choice
2. More robust across different market conditions
3. Diversification across timeframes
4. Better out-of-sample performance

Expected Improvement:
- Sharpe: +0.1 to +0.3 vs single best parameter
- Drawdown: -2 to -5% reduction
- More consistent across regimes

Academic Foundation:
- Moskowitz, Ooi & Pedersen (2012): Time series momentum
- Goyal & Jegadeesh (2018): Cross-sectional vs time series
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from strategies.enhanced.regime_detection import RegimeDetector, MarketRegime


@dataclass
class EnsembleConfig:
    """Configuration for multi-timeframe ensemble"""

    # Lookback periods and weights
    lookbacks: List[int] = None  # Will be set in __post_init__
    weights: List[float] = None  # Will be set in __post_init__

    # Default lookbacks
    def __post_init__(self):
        if self.lookbacks is None:
            # Default: 60, 90, 120 day for equities
            self.lookbacks = [60, 90, 120]

        if self.weights is None:
            # Equal weight by default
            n = len(self.lookbacks)
            self.weights = [1.0 / n] * n

        # Validate
        assert len(self.lookbacks) == len(self.weights), "Lookbacks and weights must match"
        assert abs(sum(self.weights) - 1.0) < 0.01, "Weights must sum to 1.0"

    # Signal generation
    long_percentile: float = 0.70
    skip_days: int = 5

    # Position sizing
    equal_weight: bool = True
    max_position_size: float = 0.15

    # Risk management
    target_volatility: float = 0.15
    use_vol_targeting: bool = True
    use_regime_filter: bool = True
    transaction_cost: float = 0.0010


class MultiTimeframeEnsemble:
    """
    Multi-Timeframe Ensemble Momentum

    Combines momentum signals from multiple lookback periods.
    """

    def __init__(self, config: Optional[EnsembleConfig] = None):
        """Initialize ensemble strategy"""
        self.config = config or EnsembleConfig()
        self.regime_detector = RegimeDetector() if self.config.use_regime_filter else None

    def calculate_momentum_multi(self, prices: pd.DataFrame) -> Dict[int, pd.DataFrame]:
        """
        Calculate momentum for multiple lookback periods

        Args:
            prices: Asset prices

        Returns:
            Dictionary mapping lookback -> momentum scores
        """
        momentum_dict = {}

        for lookback in self.config.lookbacks:
            total_lookback = lookback + self.config.skip_days

            momentum = (
                prices.shift(self.config.skip_days) /
                prices.shift(total_lookback)
            ) - 1

            momentum_dict[lookback] = momentum

        return momentum_dict

    def calculate_ensemble_signal(self, momentum_dict: Dict[int, pd.DataFrame],
                                  date: pd.Timestamp) -> pd.Series:
        """
        Calculate ensemble signal by combining multiple timeframes

        Args:
            momentum_dict: Dictionary of momentum scores
            date: Current date

        Returns:
            Ensemble signal (-1, 0, 1)
        """
        # Get momentum scores for this date from each timeframe
        scores = []
        for lookback, weight in zip(self.config.lookbacks, self.config.weights):
            mom = momentum_dict[lookback].loc[date]

            if mom.notna().sum() > 0:
                # Rank and convert to signal
                ranked = mom.rank(pct=True)

                # Binary signal: 1 if top percentile, 0 otherwise
                signal = (ranked >= self.config.long_percentile).astype(float)

                # Weight this signal
                scores.append(signal * weight)

        if len(scores) == 0:
            return pd.Series(0, index=momentum_dict[self.config.lookbacks[0]].columns)

        # Sum weighted signals
        ensemble_signal = sum(scores)

        # Convert to discrete signals
        # If ensemble > 0.5, it means majority of timeframes agree
        final_signal = (ensemble_signal >= 0.5).astype(int)

        return final_signal

    def generate_signals(self, prices: pd.DataFrame,
                        regimes: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate ensemble trading signals

        Args:
            prices: Asset prices
            regimes: Market regimes (optional)

        Returns:
            Trading signals
        """
        # Calculate momentum for all timeframes
        momentum_dict = self.calculate_momentum_multi(prices)

        # Initialize signals
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        # Get max lookback
        max_lookback = max(self.config.lookbacks) + self.config.skip_days

        for date in prices.index[max_lookback:]:
            # Check regime if filtering enabled
            if regimes is not None and date in regimes.index:
                current_regime = regimes.loc[date]

                # Don't trade during bear markets/corrections
                if current_regime in [MarketRegime.BEAR.value,
                                     MarketRegime.CORRECTION.value,
                                     MarketRegime.CRISIS.value]:
                    signals.loc[date] = 0
                    continue

            # Calculate ensemble signal
            ensemble_signal = self.calculate_ensemble_signal(momentum_dict, date)
            signals.loc[date] = ensemble_signal

        return signals

    def calculate_position_sizes(self, signals: pd.DataFrame,
                                 prices: pd.DataFrame,
                                 regimes: Optional[pd.Series] = None) -> pd.DataFrame:
        """Calculate position sizes with volatility targeting"""

        returns = prices.pct_change()
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        vol_lookback = 60

        for date_idx, date in enumerate(signals.index[vol_lookback:], vol_lookback):
            current_signals = signals.loc[date]
            longs = current_signals[current_signals == 1].index

            if len(longs) == 0:
                continue

            # Base weights
            base_weight = 1.0 / len(longs)
            base_weights = pd.Series(base_weight, index=longs)

            # Volatility targeting
            if self.config.use_vol_targeting:
                recent_returns = returns.iloc[date_idx-vol_lookback:date_idx].mean(axis=1)
                realized_vol = recent_returns.std() * np.sqrt(252)

                if realized_vol > 0:
                    vol_scalar = self.config.target_volatility / realized_vol
                    vol_scalar = np.clip(vol_scalar, 0.3, 1.5)
                else:
                    vol_scalar = 1.0
            else:
                vol_scalar = 1.0

            # Regime adjustment
            leverage_mult = 1.0
            if regimes is not None and date in regimes.index:
                current_regime = regimes.loc[date]
                if current_regime == MarketRegime.SIDEWAYS.value:
                    leverage_mult = 0.7
                elif current_regime == MarketRegime.UNKNOWN.value:
                    leverage_mult = 0.5

            final_scalar = vol_scalar * leverage_mult

            # Apply weights
            for asset in longs:
                weights.loc[date, asset] = base_weights[asset] * final_scalar

            # Cap individual positions
            weights.loc[date] = weights.loc[date].clip(upper=self.config.max_position_size)

        return weights

    def backtest(self, prices: pd.DataFrame,
                use_regime_filter: bool = True) -> Dict:
        """
        Backtest ensemble strategy

        Args:
            prices: Asset prices
            use_regime_filter: Whether to use regime detection

        Returns:
            Dictionary with results
        """
        returns = prices.pct_change()

        # Detect regimes if enabled
        regimes = None
        if use_regime_filter and self.regime_detector:
            portfolio_prices = (prices / prices.iloc[0]).mean(axis=1)
            regimes = self.regime_detector.detect_regime(portfolio_prices)

        # Generate signals
        signals = self.generate_signals(prices, regimes)

        # Calculate position sizes
        weights = self.calculate_position_sizes(signals, prices, regimes)

        # Calculate portfolio returns
        portfolio_returns = pd.Series(0.0, index=returns.index)
        prev_weights = pd.Series(0.0, index=prices.columns)

        for date in returns.index[1:]:
            current_weights = weights.loc[date]

            if current_weights.sum() > 0:
                daily_return = (returns.loc[date] * current_weights).sum()

                # Transaction costs
                turnover = (current_weights - prev_weights).abs().sum()
                daily_return -= turnover * self.config.transaction_cost

                portfolio_returns[date] = daily_return
                prev_weights = current_weights
            else:
                prev_weights = pd.Series(0.0, index=prices.columns)

        # Calculate metrics
        metrics = self._calculate_metrics(portfolio_returns)

        return {
            'returns': portfolio_returns,
            'signals': signals,
            'weights': weights,
            'regimes': regimes,
            'metrics': metrics
        }

    def _calculate_metrics(self, returns: pd.Series) -> Dict:
        """Calculate performance metrics"""

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


def main():
    """Test multi-timeframe ensemble"""

    print("="*80)
    print("MULTI-TIMEFRAME ENSEMBLE MOMENTUM")
    print("="*80)

    # Load data
    prices = pd.read_csv('data/raw/equities_prices.csv', index_col=0)
    prices.index = pd.to_datetime(prices.index, utc=True).tz_convert(None)

    print(f"\nTesting on {len(prices)} days from {prices.index[0].date()} to {prices.index[-1].date()}\n")

    # Test different ensemble configurations

    # 1. Default: 60/90/120 equal weight
    print("="*80)
    print("ENSEMBLE 1: 60/90/120-day Equal Weight")
    print("="*80)

    config1 = EnsembleConfig(
        lookbacks=[60, 90, 120],
        weights=[0.33, 0.34, 0.33]
    )
    ensemble1 = MultiTimeframeEnsemble(config1)
    results1 = ensemble1.backtest(prices, use_regime_filter=True)

    # 2. 90-day focused: 40/90/120 with 90-day emphasis
    print("\n" + "="*80)
    print("ENSEMBLE 2: 40/90/120-day (90-day emphasis)")
    print("="*80)

    config2 = EnsembleConfig(
        lookbacks=[40, 90, 120],
        weights=[0.25, 0.50, 0.25]  # 50% weight to our best parameter
    )
    ensemble2 = MultiTimeframeEnsemble(config2)
    results2 = ensemble2.backtest(prices, use_regime_filter=True)

    # 3. Wide range: 30/60/90/120/150
    print("\n" + "="*80)
    print("ENSEMBLE 3: 30/60/90/120/150-day Wide Range")
    print("="*80)

    config3 = EnsembleConfig(
        lookbacks=[30, 60, 90, 120, 150],
        weights=[0.15, 0.20, 0.30, 0.20, 0.15]
    )
    ensemble3 = MultiTimeframeEnsemble(config3)
    results3 = ensemble3.backtest(prices, use_regime_filter=True)

    # Compare results
    print("\n" + "="*80)
    print("ENSEMBLE COMPARISON")
    print("="*80)

    comparison = pd.DataFrame([
        {
            'Ensemble': '60/90/120 Equal',
            'Sharpe': f"{results1['metrics']['sharpe']:.2f}",
            'Ann Return': f"{results1['metrics']['annual_return']:.1f}%",
            'Max DD': f"{results1['metrics']['max_drawdown']:.1f}%",
            'Calmar': f"{results1['metrics']['calmar']:.2f}"
        },
        {
            'Ensemble': '40/90/120 (90 emphasis)',
            'Sharpe': f"{results2['metrics']['sharpe']:.2f}",
            'Ann Return': f"{results2['metrics']['annual_return']:.1f}%",
            'Max DD': f"{results2['metrics']['max_drawdown']:.1f}%",
            'Calmar': f"{results2['metrics']['calmar']:.2f}"
        },
        {
            'Ensemble': '30/60/90/120/150 Wide',
            'Sharpe': f"{results3['metrics']['sharpe']:.2f}",
            'Ann Return': f"{results3['metrics']['annual_return']:.1f}%",
            'Max DD': f"{results3['metrics']['max_drawdown']:.1f}%",
            'Calmar': f"{results3['metrics']['calmar']:.2f}"
        }
    ])

    print("\n" + comparison.to_string(index=False))

    # Compare to single 90-day
    print("\n" + "="*80)
    print("vs SINGLE 90-DAY BENCHMARK")
    print("="*80)

    # Import our production equity strategy
    from strategies.production.equity_momentum import EquityMomentumStrategy, EquityMomentumConfig

    equity_config = EquityMomentumConfig(lookback=90)
    equity_strategy = EquityMomentumStrategy(equity_config)
    equity_results = equity_strategy.backtest(prices, use_regime_filter=True)

    print(f"\nSingle 90-day: Sharpe {equity_results['metrics']['sharpe']:.2f}, Return {equity_results['metrics']['annual_return']:.1f}%")

    # Find best ensemble
    sharpes = {
        'Ensemble 1': results1['metrics']['sharpe'],
        'Ensemble 2': results2['metrics']['sharpe'],
        'Ensemble 3': results3['metrics']['sharpe']
    }

    best = max(sharpes, key=sharpes.get)
    best_sharpe = sharpes[best]

    improvement = best_sharpe - equity_results['metrics']['sharpe']

    print(f"\n{'='*80}")
    print("RECOMMENDATION")
    print(f"{'='*80}")

    print(f"\nBest Ensemble: {best}")
    print(f"Sharpe: {best_sharpe:.2f}")
    print(f"Improvement vs 90-day: {improvement:+.2f}")

    if improvement > 0.1:
        print("\n✅ Ensemble IMPROVES performance")
        print("   Use ensemble for better robustness")
    elif improvement > 0:
        print("\n✓ Ensemble provides modest improvement")
    else:
        print("\n⚠️  Ensemble does not improve performance")
        print("   Stick with single 90-day parameter")

    print("\n" + "="*80)
    print("✅ MULTI-TIMEFRAME ENSEMBLE READY")
    print("="*80)

    return results1, results2, results3


if __name__ == "__main__":
    main()
