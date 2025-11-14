"""
Multi-Factor Framework

Combines multiple factors (Value, Quality, Momentum) for optimal portfolio construction.
This is the approach used by institutional quant funds (AQR, Dimensional, etc.).

Academic Foundation:
- Fama & French (1993) - Three-Factor Model (Market, Size, Value)
- Carhart (1997) - Four-Factor Model (+ Momentum)
- Fama & French (2015) - Five-Factor Model (+ Profitability, Investment)
- AQR Capital - Multi-factor research across asset classes
- Asness et al (2013) - "Value and Momentum Everywhere"

Why Combine Factors:
1. Diversification - Factors have low correlation
2. Robustness - Different factors work in different regimes
3. Higher Sharpe - Combined portfolio smoother than individual factors
4. Risk management - Balanced factor exposure

Factor Correlations:
- Value vs Momentum: ~-0.2 to 0 (negative/zero correlation)
- Value vs Quality: ~0.3 (positive, quality avoids value traps)
- Momentum vs Quality: ~0.1 to 0.3 (low positive)

Combination Methods:
1. Equal Weight - Simple, diversified (1/N across factors)
2. IC-Weighted - Weight by Information Coefficient (predictive power)
3. Sharpe-Weighted - Weight by historical Sharpe ratios
4. Optimization - Maximize portfolio Sharpe (factor allocation)
5. Score-Based - Composite score from all factors

Historical Performance:
- Single Factor: Sharpe 0.4-0.8
- Multi-Factor (equal weight): Sharpe 1.0-1.5
- Multi-Factor (optimized): Sharpe 1.5-2.0

Factor Timing:
- Value: Works in mean reversion regimes, value rallies
- Momentum: Works in trending markets, bull/bear runs
- Quality: Works in all regimes, especially crises
- Combining all 3 provides all-weather exposure

Fama-French-Carhart Four-Factor Model:
- Market (Beta): CAPM market risk premium
- SMB (Size): Small Minus Big market cap
- HML (Value): High Minus Low book-to-market
- UMD (Momentum): Up Minus Down past returns

Expected Returns:
- Multi-Factor (academic): 8-12% annual
- Multi-Factor (real-world after costs): 5-8% annual
- Best when rebalanced quarterly/monthly
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

# Import factor strategies
from factors.value_factor import ValueFactor, ValueFactorConfig
from factors.quality_factor import QualityFactor, QualityFactorConfig
from factors.momentum_factor import MomentumFactor, MomentumFactorConfig


@dataclass
class MultiFactorConfig:
    """Configuration for Multi-Factor Framework"""
    factors: List[str] = None  # ['value', 'quality', 'momentum']
    combination_method: str = 'equal_weight'  # 'equal_weight', 'ic_weighted', 'sharpe_weighted', 'optimized'
    long_percentile: float = 0.20  # Top 20% composite score
    short_percentile: float = 0.80  # Bottom 20%
    rebalance_frequency: int = 21  # Monthly
    use_sector_neutrality: bool = False  # Sector-neutral factors

    def __post_init__(self):
        if self.factors is None:
            self.factors = ['value', 'quality', 'momentum']


class MultiFactorFramework:
    """
    Multi-Factor Framework

    Combines Value, Quality, and Momentum factors optimally.
    """

    def __init__(self, config: MultiFactorConfig = None):
        self.config = config or MultiFactorConfig()

        # Initialize individual factor strategies
        self.value_factor = ValueFactor()
        self.quality_factor = QualityFactor()
        self.momentum_factor = MomentumFactor()

    def normalize_scores(self, scores: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize factor scores to 0-1 range using percentile ranks

        This makes different factor scores comparable
        """
        normalized = scores.rank(axis=1, pct=True)
        return normalized

    def calculate_composite_score(
        self,
        factor_scores: Dict[str, pd.DataFrame],
        weights: Dict[str, float] = None
    ) -> pd.DataFrame:
        """
        Calculate composite score from multiple factors

        Args:
            factor_scores: Dict of {factor_name: score_dataframe}
            weights: Dict of {factor_name: weight} (default: equal weight)

        Returns:
            Composite score DataFrame
        """
        if weights is None:
            # Equal weight
            n_factors = len(factor_scores)
            weights = {name: 1/n_factors for name in factor_scores.keys()}

        # Normalize all factor scores
        normalized_scores = {}
        for name, scores in factor_scores.items():
            normalized_scores[name] = self.normalize_scores(scores)

        # Weighted combination
        composite = pd.DataFrame(0.0, index=list(factor_scores.values())[0].index,
                                columns=list(factor_scores.values())[0].columns)

        for name, scores in normalized_scores.items():
            if name in weights:
                composite += weights[name] * scores

        return composite

    def calculate_factor_correlations(
        self,
        factor_scores: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate correlation between factor scores

        Shows which factors are diversifying
        """
        # Stack all factor scores
        all_scores = []
        factor_names = []

        for name, scores in factor_scores.items():
            # Average score across assets for each date
            avg_score = scores.mean(axis=1)
            all_scores.append(avg_score)
            factor_names.append(name)

        scores_df = pd.DataFrame(all_scores, index=factor_names).T

        # Calculate correlation
        correlation = scores_df.corr()

        return correlation

    def calculate_ic_weights(
        self,
        factor_scores: Dict[str, pd.DataFrame],
        forward_returns: pd.DataFrame,
        lookback: int = 126  # 6 months
    ) -> Dict[str, float]:
        """
        Calculate Information Coefficient (IC) weights

        IC = Correlation between factor score and forward returns
        Higher IC = More predictive factor = Higher weight

        This is adaptive - weights change based on recent predictive power
        """
        ics = {}

        for name, scores in factor_scores.items():
            # Calculate IC for each period
            period_ics = []

            for i in range(lookback, len(scores)):
                # Factor scores at time t
                current_scores = scores.iloc[i]

                # Forward returns from t to t+21 (1 month)
                if i + 21 < len(forward_returns):
                    future_ret = forward_returns.iloc[i+21]

                    # Correlation between scores and returns
                    ic = current_scores.corr(future_ret)

                    if not np.isnan(ic):
                        period_ics.append(ic)

            # Average IC over lookback period
            avg_ic = np.mean(period_ics) if period_ics else 0
            ics[name] = max(avg_ic, 0)  # Ensure non-negative

        # Normalize to sum to 1
        total_ic = sum(ics.values())
        if total_ic > 0:
            ic_weights = {name: ic / total_ic for name, ic in ics.items()}
        else:
            # Fall back to equal weight
            n = len(ics)
            ic_weights = {name: 1/n for name in ics.keys()}

        return ic_weights

    def calculate_sharpe_weights(
        self,
        factor_returns: Dict[str, pd.Series],
        lookback: int = 252  # 1 year
    ) -> Dict[str, float]:
        """
        Calculate Sharpe-ratio-based weights

        Weight factors by their historical Sharpe ratios
        Better Sharpe = Higher weight
        """
        sharpes = {}

        for name, returns in factor_returns.items():
            recent_returns = returns.iloc[-lookback:]

            if len(recent_returns) > 0 and recent_returns.std() > 0:
                sharpe = recent_returns.mean() / recent_returns.std() * np.sqrt(252)
                sharpes[name] = max(sharpe, 0)  # Ensure non-negative
            else:
                sharpes[name] = 0

        # Normalize to sum to 1
        total_sharpe = sum(sharpes.values())
        if total_sharpe > 0:
            sharpe_weights = {name: s / total_sharpe for name, s in sharpes.items()}
        else:
            n = len(sharpes)
            sharpe_weights = {name: 1/n for name in sharpes.keys()}

        return sharpe_weights

    def generate_signals(
        self,
        composite_score: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate signals from composite multi-factor score

        Long: Highest composite scores
        Short: Lowest composite scores
        """
        # Rank by composite score
        ranks = composite_score.rank(axis=1, ascending=False, pct=True)

        signals = pd.DataFrame(0, index=ranks.index, columns=ranks.columns)

        # Long: Top percentile
        signals[ranks <= self.config.long_percentile] = 1

        # Short: Bottom percentile
        if self.config.short_percentile < 1.0:
            signals[ranks >= self.config.short_percentile] = -1

        return signals

    def backtest(
        self,
        prices: pd.DataFrame,
        value_scores: pd.DataFrame,
        quality_scores: pd.DataFrame,
        momentum_scores: pd.DataFrame
    ) -> Dict:
        """
        Backtest multi-factor strategy

        Args:
            prices: Asset prices
            value_scores: Value factor scores
            quality_scores: Quality factor scores
            momentum_scores: Momentum factor scores

        Returns:
            Performance metrics and diagnostics
        """
        # Combine factor scores
        factor_scores = {}

        if 'value' in self.config.factors:
            factor_scores['value'] = value_scores

        if 'quality' in self.config.factors:
            factor_scores['quality'] = quality_scores

        if 'momentum' in self.config.factors:
            factor_scores['momentum'] = momentum_scores

        # Calculate forward returns for IC calculation
        returns = prices.pct_change()
        forward_returns = prices.pct_change(21)  # 1-month forward

        # Calculate weights based on method
        if self.config.combination_method == 'equal_weight':
            weights = None  # Equal weight (default)

        elif self.config.combination_method == 'ic_weighted':
            weights = self.calculate_ic_weights(factor_scores, forward_returns)

        elif self.config.combination_method == 'sharpe_weighted':
            # Need factor returns first (simplified: use factor scores as proxy)
            factor_returns = {name: scores.mean(axis=1) for name, scores in factor_scores.items()}
            weights = self.calculate_sharpe_weights(factor_returns)

        else:
            weights = None

        # Calculate composite score
        composite_score = self.calculate_composite_score(factor_scores, weights)

        # Generate signals
        signals = self.generate_signals(composite_score)

        # Portfolio construction
        portfolio_weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        for date in signals.index:
            date_signals = signals.loc[date]

            longs = date_signals[date_signals == 1].index
            shorts = date_signals[date_signals == -1].index

            if len(longs) > 0:
                portfolio_weights.loc[date, longs] = 1.0 / len(longs)

            if len(shorts) > 0:
                portfolio_weights.loc[date, shorts] = -1.0 / len(shorts)

        # Calculate portfolio returns
        portfolio_returns = (portfolio_weights.shift(1) * returns).sum(axis=1)

        # Performance metrics
        total_return = (1 + portfolio_returns).prod() - 1
        ann_return = (1 + total_return) ** (252 / len(portfolio_returns)) - 1
        ann_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        # Max drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        # Win rate
        win_rate = (portfolio_returns > 0).sum() / len(portfolio_returns)

        # Factor correlations
        factor_corr = self.calculate_factor_correlations(factor_scores)

        return {
            'total_return': total_return,
            'annual_return': ann_return,
            'volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'portfolio_returns': portfolio_returns,
            'composite_score': composite_score,
            'signals': signals,
            'weights': portfolio_weights,
            'factor_correlations': factor_corr,
            'factor_weights': weights if weights else {name: 1/len(factor_scores) for name in factor_scores.keys()}
        }


if __name__ == "__main__":
    """Test Multi-Factor Framework"""

    print("=" * 80)
    print("MULTI-FACTOR FRAMEWORK TEST")
    print("=" * 80)
    print()

    # Create synthetic data
    print("Creating synthetic test data...")
    print("- 10 stocks with various characteristics")
    print("- Value, Quality, and Momentum signals")
    print("- 3 years of data")
    print()

    np.random.seed(42)
    n_days = 252 * 3
    n_stocks = 10

    dates = pd.date_range('2021-01-01', periods=n_days, freq='D')
    stocks = [f'Stock_{i}' for i in range(1, n_stocks + 1)]

    # Generate data where different stocks excel in different factors
    prices_data = {}
    value_data = {}  # Low P/E = high value
    quality_data = {}  # High ROE = high quality
    momentum_data = {}

    for i, stock in enumerate(stocks):
        # Stock characteristics:
        # 1-2: High value, low quality, neutral momentum
        # 3-4: Low value, high quality, positive momentum
        # 5-6: Neutral value, high quality, high momentum (best)
        # 7-8: High value, high quality, negative momentum
        # 9-10: Low value, low quality, neutral momentum (worst)

        if i < 2:
            # Cheap value stocks, poor quality
            pe = 8 + np.random.normal(0, 1, n_days)  # Low P/E (good value)
            roe = 0.08 + np.random.normal(0, 0.01, n_days)  # Low ROE (poor quality)
            trend = 0.0004
            vol = 0.020
        elif i < 4:
            # Expensive growth stocks, high quality
            pe = 30 + np.random.normal(0, 2, n_days)  # High P/E (poor value)
            roe = 0.20 + np.random.normal(0, 0.02, n_days)  # High ROE (good quality)
            trend = 0.0008
            vol = 0.015
        elif i < 6:
            # Best: Decent value, high quality, strong momentum
            pe = 15 + np.random.normal(0, 1, n_days)  # Medium P/E
            roe = 0.18 + np.random.normal(0, 0.02, n_days)  # High ROE
            trend = 0.0012  # Strong trend
            vol = 0.015
        elif i < 8:
            # Value traps: Cheap but low momentum
            pe = 10 + np.random.normal(0, 1, n_days)
            roe = 0.15 + np.random.normal(0, 0.02, n_days)
            trend = -0.0002  # Downtrend
            vol = 0.022
        else:
            # Worst: Expensive, poor quality, neutral
            pe = 35 + np.random.normal(0, 2, n_days)
            roe = 0.05 + np.random.normal(0, 0.01, n_days)
            trend = 0.0002
            vol = 0.025

        # Generate prices
        returns = np.random.normal(trend, vol, n_days)
        prices = (1 + returns).cumprod() * 100
        prices_data[stock] = prices

        # Factor scores (invert P/E for value score: low P/E = high value)
        value_data[stock] = 1 / pe  # Higher = better value
        quality_data[stock] = roe  # Higher = better quality

    prices_df = pd.DataFrame(prices_data, index=dates)
    value_df = pd.DataFrame(value_data, index=dates)
    quality_df = pd.DataFrame(quality_data, index=dates)

    # Calculate momentum
    momentum_calc = MomentumFactor()
    momentum_df = momentum_calc.calculate_momentum(prices_df)

    # Initialize Multi-Factor Framework
    config = MultiFactorConfig(
        factors=['value', 'quality', 'momentum'],
        combination_method='equal_weight',
        long_percentile=0.20,
        short_percentile=0.80
    )

    multi_factor = MultiFactorFramework(config)

    # Test equal weight combination
    print("=" * 80)
    print("METHOD 1: EQUAL WEIGHT MULTI-FACTOR")
    print("=" * 80)
    print()

    print("Combines Value + Quality + Momentum equally")
    print()

    result_equal = multi_factor.backtest(prices_df, value_df, quality_df, momentum_df)

    print("PERFORMANCE METRICS:")
    print(f"  Annual Return: {result_equal['annual_return']:>10.2%}")
    print(f"  Volatility: {result_equal['volatility']:>10.2%}")
    print(f"  Sharpe Ratio: {result_equal['sharpe_ratio']:>10.2f}")
    print(f"  Max Drawdown: {result_equal['max_drawdown']:>10.2%}")
    print(f"  Win Rate: {result_equal['win_rate']:>10.2%}")
    print()

    print("FACTOR WEIGHTS:")
    for factor, weight in result_equal['factor_weights'].items():
        print(f"  {factor:10s}: {weight:>6.2%}")

    print("\n" + "=" * 80)
    print("FACTOR CORRELATIONS")
    print("=" * 80)
    print()
    print(result_equal['factor_correlations'].round(2))
    print()
    print("Low correlation = Good diversification")

    # Compare with single factors
    print("\n" + "=" * 80)
    print("COMPARISON: Single Factors vs Multi-Factor")
    print("=" * 80)
    print()

    # Test individual factors
    value_only = MultiFactorFramework(MultiFactorConfig(factors=['value']))
    quality_only = MultiFactorFramework(MultiFactorConfig(factors=['quality']))
    momentum_only = MultiFactorFramework(MultiFactorConfig(factors=['momentum']))

    result_value = value_only.backtest(prices_df, value_df, quality_df, momentum_df)
    result_quality = quality_only.backtest(prices_df, value_df, quality_df, momentum_df)
    result_momentum = momentum_only.backtest(prices_df, value_df, quality_df, momentum_df)

    comparison = pd.DataFrame({
        'Factor': ['Value Only', 'Quality Only', 'Momentum Only', 'Multi-Factor (All 3)'],
        'Annual Return': [
            result_value['annual_return'],
            result_quality['annual_return'],
            result_momentum['annual_return'],
            result_equal['annual_return']
        ],
        'Sharpe Ratio': [
            result_value['sharpe_ratio'],
            result_quality['sharpe_ratio'],
            result_momentum['sharpe_ratio'],
            result_equal['sharpe_ratio']
        ],
        'Max Drawdown': [
            result_value['max_drawdown'],
            result_quality['max_drawdown'],
            result_momentum['max_drawdown'],
            result_equal['max_drawdown']
        ]
    })

    print(comparison.to_string(index=False))

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("Why Combine Factors:")
    print("  - Diversification: Factors have low correlation")
    print("  - Robustness: Different factors work in different regimes")
    print("  - Higher Sharpe: Combined > Individual")
    print("  - Risk management: Balanced exposure")
    print()
    print("Factor Interactions:")
    print("  - Value + Quality: Quality avoids value traps")
    print("  - Value + Momentum: Negative correlation (diversifying)")
    print("  - Quality + Momentum: Both work in bull markets")
    print()
    print("Fama-French-Carhart Four-Factor Model:")
    print("  - Market (Beta): Systematic risk premium")
    print("  - HML (Value): High book-to-market Minus Low")
    print("  - UMD (Momentum): Up Minus Down past returns")
    print("  - SMB (Size): Small Minus Big market cap")
    print()
    print("Expected Performance:")
    print("  - Single Factor: Sharpe 0.4-0.8")
    print("  - Multi-Factor: Sharpe 1.0-2.0")
    print("  - Real-world (after costs): 5-10% annual")
    print()
    print("Best Practices:")
    print("  - Rebalance monthly/quarterly")
    print("  - Equal weight is robust starting point")
    print("  - Can use IC-weighting for adaptivity")
    print("  - Combine 3+ factors for diversification")
    print()
    print("=" * 80)
    print("Test complete!")
    print("=" * 80)
