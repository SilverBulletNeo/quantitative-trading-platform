"""
Momentum Factor Strategy

Momentum: "Buy winners, sell losers" - Past winners continue to outperform.
One of the most robust anomalies in finance.

Academic Foundation:
- Jegadeesh & Titman (1993) - "Returns to Buying Winners and Selling Losers"
  * Momentum premium: ~1% per month
  * 6-month formation period, 6-month holding period
  * Works across stocks, countries, asset classes
- Moskowitz, Ooi, Pedersen (2012) - "Time Series Momentum Everywhere"
  * TSM works in 58 liquid instruments across 4 asset classes
  * ~6-7% annual return
  * Positive in all asset classes
- Asness, Moskowitz, Pedersen (2013) - "Value and Momentum Everywhere"
  * Momentum works globally across assets
  * Low correlation with value factor
- Carhart (1997) - Four-Factor Model (added UMD to Fama-French)

Types of Momentum:
1. Cross-Sectional (Relative): Rank assets, buy winners vs losers
2. Time-Series (Absolute): Buy if price > moving average
3. Risk-Adjusted: Momentum scaled by volatility

UMD Factor (Up Minus Down):
- Fama-French-Carhart four-factor model
- Long: Top 30% performers (winners)
- Short: Bottom 30% performers (losers)
- Historical return: ~0.7% monthly, 8-9% annually

Why Momentum Works:
- Underreaction to news (gradual information diffusion)
- Behavioral biases (anchoring, confirmation bias)
- Risk-based (dynamic risk exposure)
- Herding behavior
- Trend following in markets

Key Characteristics:
- Formation period: 6-12 months (measure past performance)
- Skip recent month (avoid short-term reversal)
- Holding period: 1-12 months
- Rebalance: Monthly
- Works across all asset classes

Momentum Crash Risk:
- Can experience severe drawdowns during market reversals
- Famous crashes: 2009 (+84% before, -73% crash), 1932, 2001
- Risk management critical: stop-losses, volatility targeting
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MomentumFactorConfig:
    """Configuration for Momentum Factor strategy"""
    formation_period: int = 126  # 6 months (126 trading days)
    skip_period: int = 21  # Skip recent month to avoid reversal
    holding_period: int = 21  # 1 month holding
    long_percentile: float = 0.30  # Top 30% (winners)
    short_percentile: float = 0.70  # Bottom 30% (losers)
    momentum_type: str = 'cross_sectional'  # 'cross_sectional' or 'time_series'
    use_multiple_horizons: bool = True  # Combine multiple lookback periods
    vol_adjust: bool = False  # Volatility-adjusted momentum
    rebalance_frequency: int = 21  # Monthly


class MomentumFactor:
    """
    Momentum Factor Strategy (UMD - Up Minus Down)

    Implements Jegadeesh & Titman (1993) momentum strategy.
    """

    def __init__(self, config: MomentumFactorConfig = None):
        self.config = config or MomentumFactorConfig()

    def calculate_momentum(
        self,
        prices: pd.DataFrame,
        lookback: int = None,
        skip_period: int = None
    ) -> pd.DataFrame:
        """
        Calculate momentum score

        Momentum = (Price[t-skip] / Price[t-lookback-skip]) - 1

        Skip recent period to avoid short-term reversal effect
        """
        if lookback is None:
            lookback = self.config.formation_period

        if skip_period is None:
            skip_period = self.config.skip_period

        # Total lookback
        total_lookback = lookback + skip_period

        # Calculate returns from t-total_lookback to t-skip_period
        momentum = (prices.shift(skip_period) / prices.shift(total_lookback)) - 1

        return momentum

    def calculate_multi_horizon_momentum(
        self,
        prices: pd.DataFrame,
        horizons: List[int] = None
    ) -> pd.DataFrame:
        """
        Calculate momentum across multiple horizons

        Combines 1M, 3M, 6M, 12M momentum
        Moskowitz et al (2012) approach
        """
        if horizons is None:
            # Default: 1M, 3M, 6M, 12M
            horizons = [21, 63, 126, 252]

        momentum_scores = []

        for horizon in horizons:
            mom = self.calculate_momentum(prices, lookback=horizon)
            momentum_scores.append(mom)

        # Average across horizons (equal weight)
        avg_momentum = pd.concat(momentum_scores).groupby(level=0).mean()

        return avg_momentum

    def calculate_volatility_adjusted_momentum(
        self,
        prices: pd.DataFrame,
        lookback: int = None
    ) -> pd.DataFrame:
        """
        Calculate volatility-adjusted momentum

        Vol-Adjusted Momentum = Raw Momentum / Volatility

        This scales momentum by risk (Sharpe-like adjustment)
        """
        # Raw momentum
        momentum = self.calculate_momentum(prices, lookback=lookback)

        # Calculate volatility
        returns = prices.pct_change()
        volatility = returns.rolling(window=63).std() * np.sqrt(252)  # 3-month vol

        # Adjust momentum by volatility
        vol_adj_momentum = momentum / volatility

        return vol_adj_momentum

    def rank_by_momentum(
        self,
        momentum: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Rank assets by momentum score

        Returns percentile ranks (0-1)
        """
        ranks = momentum.rank(axis=1, ascending=False, pct=True)
        return ranks

    def generate_cross_sectional_signals(
        self,
        momentum: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate cross-sectional momentum signals

        Cross-sectional = Relative momentum (rank-based)

        Long: Top performers (winners)
        Short: Bottom performers (losers)
        """
        # Rank by momentum
        ranks = self.rank_by_momentum(momentum)

        signals = pd.DataFrame(0, index=ranks.index, columns=ranks.columns)

        # Long: Top percentile (winners)
        signals[ranks <= self.config.long_percentile] = 1

        # Short: Bottom percentile (losers)
        if self.config.short_percentile < 1.0:
            signals[ranks >= self.config.short_percentile] = -1

        return signals

    def generate_time_series_signals(
        self,
        prices: pd.DataFrame,
        momentum: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate time-series momentum signals

        Time-series = Absolute momentum (trend following)

        Long: If price > moving average (positive momentum)
        Short: If price < moving average (negative momentum)
        """
        signals = pd.DataFrame(0, index=momentum.index, columns=momentum.columns)

        # Long: Positive momentum
        signals[momentum > 0] = 1

        # Short: Negative momentum (optional)
        if self.config.short_percentile < 1.0:
            signals[momentum < 0] = -1

        return signals

    def generate_signals(
        self,
        prices: pd.DataFrame,
        momentum: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Generate momentum signals

        Args:
            prices: Price data
            momentum: Pre-calculated momentum scores (optional)

        Returns:
            DataFrame of signals (+1 long, 0 neutral, -1 short)
        """
        # Calculate momentum if not provided
        if momentum is None:
            if self.config.use_multiple_horizons:
                momentum = self.calculate_multi_horizon_momentum(prices)
            elif self.config.vol_adjust:
                momentum = self.calculate_volatility_adjusted_momentum(prices)
            else:
                momentum = self.calculate_momentum(prices)

        # Generate signals based on momentum type
        if self.config.momentum_type == 'cross_sectional':
            signals = self.generate_cross_sectional_signals(momentum)
        elif self.config.momentum_type == 'time_series':
            signals = self.generate_time_series_signals(prices, momentum)
        else:
            raise ValueError(f"Unknown momentum type: {self.config.momentum_type}")

        return signals

    def backtest(
        self,
        prices: pd.DataFrame
    ) -> Dict:
        """
        Backtest momentum factor strategy

        Args:
            prices: Asset prices

        Returns:
            Performance metrics
        """
        # Generate signals
        signals = self.generate_signals(prices)

        # Equal weight portfolio
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        for date in signals.index:
            date_signals = signals.loc[date]

            longs = date_signals[date_signals == 1].index
            shorts = date_signals[date_signals == -1].index

            if len(longs) > 0:
                weights.loc[date, longs] = 1.0 / len(longs)

            if len(shorts) > 0:
                weights.loc[date, shorts] = -1.0 / len(shorts)

        # Calculate returns
        returns = prices.pct_change()
        portfolio_returns = (weights.shift(1) * returns).sum(axis=1)

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

        # Long/short performance
        long_returns = (weights[weights > 0].shift(1) * returns).sum(axis=1)
        short_returns = (weights[weights < 0].shift(1) * returns).sum(axis=1)

        long_sharpe = (long_returns.mean() / long_returns.std() * np.sqrt(252)
                      if long_returns.std() > 0 else 0)
        short_sharpe = (short_returns.mean() / short_returns.std() * np.sqrt(252)
                       if short_returns.std() > 0 else 0)

        return {
            'total_return': total_return,
            'annual_return': ann_return,
            'volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'portfolio_returns': portfolio_returns,
            'long_sharpe': long_sharpe,
            'short_sharpe': short_sharpe,
            'signals': signals,
            'weights': weights
        }


if __name__ == "__main__":
    """Test Momentum Factor Strategy"""

    print("=" * 80)
    print("MOMENTUM FACTOR STRATEGY TEST")
    print("=" * 80)
    print()

    # Create synthetic data
    print("Creating synthetic test data...")
    print("- 10 stocks with trending behavior")
    print("- 3 years of data")
    print()

    np.random.seed(42)
    n_days = 252 * 3
    n_stocks = 10

    dates = pd.date_range('2021-01-01', periods=n_days, freq='D')
    stocks = [f'Stock_{i}' for i in range(1, n_stocks + 1)]

    # Generate data with momentum characteristics
    prices_data = {}

    for i, stock in enumerate(stocks):
        # Some stocks with strong uptrends (winners)
        # Some with downtrends (losers)
        # Some sideways (neutral)

        if i < 3:
            # Winners: Strong uptrend
            trend = 0.0012
            vol = 0.015
        elif i < 6:
            # Neutral: Sideways
            trend = 0.0003
            vol = 0.018
        else:
            # Losers: Downtrend
            trend = -0.0005
            vol = 0.020

        # Generate returns with trend + noise
        returns = np.random.normal(trend, vol, n_days)
        prices = (1 + returns).cumprod() * 100
        prices_data[stock] = prices

    prices_df = pd.DataFrame(prices_data, index=dates)

    # Initialize Momentum Factor
    config = MomentumFactorConfig(
        formation_period=126,  # 6 months
        skip_period=21,  # Skip 1 month
        momentum_type='cross_sectional',
        use_multiple_horizons=False
    )
    momentum_factor = MomentumFactor(config)

    # Calculate momentum scores
    print("=" * 80)
    print("MOMENTUM SCORES (6-Month Formation)")
    print("=" * 80)
    print()

    momentum_scores = momentum_factor.calculate_momentum(prices_df)

    latest_momentum = momentum_scores.iloc[-1].sort_values(ascending=False)
    print("Latest Momentum Scores:")
    for stock, score in latest_momentum.items():
        print(f"  {stock}: {score:>8.2%}")

    # Backtest
    print("\n" + "=" * 80)
    print("BACKTEST: CROSS-SECTIONAL MOMENTUM STRATEGY")
    print("=" * 80)
    print()

    print("Strategy (Jegadeesh & Titman 1993):")
    print("  - Formation: 6 months (126 days)")
    print("  - Skip: 1 month (21 days) to avoid reversal")
    print("  - Long: Top 30% performers (winners)")
    print("  - Short: Bottom 30% performers (losers)")
    print("  - Rebalance: Monthly")
    print()

    result = momentum_factor.backtest(prices_df)

    print("PERFORMANCE METRICS:")
    print(f"  Total Return: {result['total_return']:>10.2%}")
    print(f"  Annual Return: {result['annual_return']:>10.2%}")
    print(f"  Volatility: {result['volatility']:>10.2%}")
    print(f"  Sharpe Ratio: {result['sharpe_ratio']:>10.2f}")
    print(f"  Max Drawdown: {result['max_drawdown']:>10.2%}")
    print(f"  Win Rate: {result['win_rate']:>10.2%}")
    print()

    print("LONG/SHORT BREAKDOWN:")
    print(f"  Long (Winners) Sharpe: {result['long_sharpe']:>10.2f}")
    print(f"  Short (Losers) Sharpe: {result['short_sharpe']:>10.2f}")
    print()

    # Test multi-horizon momentum
    print("=" * 80)
    print("MULTI-HORIZON MOMENTUM (Moskowitz et al 2012)")
    print("=" * 80)
    print()

    config_multi = MomentumFactorConfig(
        momentum_type='cross_sectional',
        use_multiple_horizons=True  # Combine 1M, 3M, 6M, 12M
    )
    momentum_multi = MomentumFactor(config_multi)

    result_multi = momentum_multi.backtest(prices_df)

    print("Strategy:")
    print("  - Combines 1M, 3M, 6M, 12M momentum")
    print("  - Equal weight across horizons")
    print("  - More robust to momentum crashes")
    print()

    print("PERFORMANCE METRICS:")
    print(f"  Annual Return: {result_multi['annual_return']:>10.2%}")
    print(f"  Sharpe Ratio: {result_multi['sharpe_ratio']:>10.2f}")
    print(f"  Max Drawdown: {result_multi['max_drawdown']:>10.2%}")
    print()

    # Show sample holdings
    print("=" * 80)
    print("SAMPLE PORTFOLIO HOLDINGS (Latest Date)")
    print("=" * 80)
    print()

    latest_signals = result['signals'].iloc[-1]

    longs = latest_signals[latest_signals == 1].index.tolist()
    shorts = latest_signals[latest_signals == -1].index.tolist()

    print("LONG POSITIONS (Winners):")
    for stock in longs:
        mom_score = momentum_scores.iloc[-1][stock]
        price_change = (prices_df[stock].iloc[-1] / prices_df[stock].iloc[-127] - 1)
        print(f"  {stock}: 6M Return={price_change:>7.2%}, Momentum Score={mom_score:>7.2%}")

    print("\nSHORT POSITIONS (Losers):")
    for stock in shorts:
        mom_score = momentum_scores.iloc[-1][stock]
        price_change = (prices_df[stock].iloc[-1] / prices_df[stock].iloc[-127] - 1)
        print(f"  {stock}: 6M Return={price_change:>7.2%}, Momentum Score={mom_score:>7.2%}")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("Momentum Factor (UMD):")
    print("  - Jegadeesh & Titman (1993): ~1% monthly return")
    print("  - Carhart Four-Factor Model: 0.7% monthly (8-9% annual)")
    print("  - Works across all asset classes and countries")
    print()
    print("Formation Period:")
    print("  - Standard: 6-12 months")
    print("  - Skip recent month to avoid reversal")
    print("  - Short-term (<1M) tends to reverse")
    print("  - Medium-term (3-12M) strongest momentum")
    print()
    print("Types of Momentum:")
    print("  - Cross-Sectional: Relative (rank stocks)")
    print("  - Time-Series: Absolute (trend following)")
    print("  - Both work, but different exposures")
    print()
    print("Why Momentum Works:")
    print("  - Underreaction to news (gradual diffusion)")
    print("  - Behavioral biases (anchoring, herding)")
    print("  - Risk-based (dynamic exposure)")
    print()
    print("Momentum Crashes:")
    print("  - 2009: +84% then -73% crash")
    print("  - Occurs during market reversals")
    print("  - Risk management critical (stop-losses, vol targeting)")
    print()
    print("Best Practices:")
    print("  - Combine multiple horizons (1M, 3M, 6M, 12M)")
    print("  - Volatility adjust momentum scores")
    print("  - Use with other factors (value, quality)")
    print("  - Rebalance monthly (not too frequent)")
    print()
    print("=" * 80)
    print("Test complete!")
    print("=" * 80)
