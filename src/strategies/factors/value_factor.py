"""
Value Factor Strategy

Value investing: Buy cheap assets, sell expensive assets.
One of the most robust and oldest investment strategies.

Academic Foundation:
- Graham & Dodd (1934) - "Security Analysis" (original value investing)
- Fama & French (1992) - "The Cross-Section of Expected Stock Returns"
  * HML factor (High Minus Low) - value premium
  * Value stocks outperform growth by ~5% annually (1926-1991)
- Basu (1977) - "Investment Performance of Common Stocks in Relation to P/E Ratios"
- Asness, Moskowitz, Pedersen (2013) - "Value and Momentum Everywhere"

Value Metrics:
1. Price-to-Earnings (P/E) - Most common
2. Price-to-Book (P/B) - Buffett's favorite
3. Price-to-Sales (P/S) - Revenue-based
4. Dividend Yield - Income-focused
5. EV/EBITDA - Enterprise value based
6. Free Cash Flow Yield - Cash generation

Strategy:
- Rank assets by value metrics
- Long cheapest quintile (bottom 20% by P/E)
- Short/avoid most expensive quintile (top 20%)
- Rebalance monthly/quarterly

Why Value Works:
- Mean reversion - prices overreact
- Risk premium - value stocks riskier
- Behavioral - investors overextrapolate growth
- Fundamental - cheap stocks have margin of safety

Fama-French HML Factor:
- HML = High book-to-market Minus Low book-to-market
- Historical return: ~5% annually
- Works across countries, asset classes, time periods
- Nobel Prize (Fama 2013)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class ValueFactorConfig:
    """Configuration for Value Factor strategy"""
    metric: str = 'pe_ratio'  # 'pe_ratio', 'pb_ratio', 'ps_ratio', 'dividend_yield', 'ev_ebitda'
    long_percentile: float = 0.20  # Bottom 20% (cheapest)
    short_percentile: float = 0.80  # Top 20% (most expensive)
    rebalance_frequency: int = 21  # Monthly (21 trading days)
    min_stocks: int = 5  # Minimum stocks in portfolio
    weight_method: str = 'equal'  # 'equal' or 'factor_weighted'


class ValueFactor:
    """
    Value Factor Strategy

    Buys cheap assets based on valuation metrics.
    """

    def __init__(self, config: ValueFactorConfig = None):
        self.config = config or ValueFactorConfig()

    def calculate_pe_ratio(
        self,
        prices: pd.DataFrame,
        earnings_per_share: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate Price-to-Earnings (P/E) Ratio

        P/E = Price / Earnings Per Share

        Lower P/E = Cheaper (better value)
        """
        pe_ratio = prices / earnings_per_share
        return pe_ratio

    def calculate_pb_ratio(
        self,
        prices: pd.DataFrame,
        book_value_per_share: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate Price-to-Book (P/B) Ratio

        P/B = Price / Book Value Per Share

        Warren Buffett's favorite metric
        Lower P/B = Cheaper
        """
        pb_ratio = prices / book_value_per_share
        return pb_ratio

    def calculate_ps_ratio(
        self,
        prices: pd.DataFrame,
        sales_per_share: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate Price-to-Sales (P/S) Ratio

        P/S = Price / Sales Per Share

        Useful when earnings are negative
        Lower P/S = Cheaper
        """
        ps_ratio = prices / sales_per_share
        return ps_ratio

    def calculate_dividend_yield(
        self,
        prices: pd.DataFrame,
        annual_dividends: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate Dividend Yield

        Dividend Yield = Annual Dividend / Price

        Higher yield = Better value (more income per $)
        """
        div_yield = annual_dividends / prices
        return div_yield

    def calculate_fcf_yield(
        self,
        prices: pd.DataFrame,
        free_cash_flow_per_share: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate Free Cash Flow Yield

        FCF Yield = FCF Per Share / Price

        Higher = Better (more cash generation per $)
        """
        fcf_yield = free_cash_flow_per_share / prices
        return fcf_yield

    def rank_by_value(
        self,
        value_metric: pd.DataFrame,
        ascending: bool = True
    ) -> pd.DataFrame:
        """
        Rank assets by value metric

        Args:
            value_metric: DataFrame of value metrics
            ascending: True if lower is better (P/E, P/B)
                      False if higher is better (Dividend Yield)

        Returns:
            DataFrame of percentile ranks (0-1)
        """
        # Rank each row (each date)
        ranks = value_metric.rank(axis=1, ascending=ascending, pct=True)
        return ranks

    def generate_signals(
        self,
        value_metric: pd.DataFrame,
        ascending: bool = True
    ) -> pd.DataFrame:
        """
        Generate long/short signals based on value ranks

        Args:
            value_metric: Value metric (P/E, P/B, etc.)
            ascending: True if lower is better

        Returns:
            DataFrame of signals:
            +1 = Long (cheap/value)
             0 = Neutral
            -1 = Short (expensive/growth)
        """
        # Rank assets by value
        ranks = self.rank_by_value(value_metric, ascending=ascending)

        # Generate signals
        signals = pd.DataFrame(0, index=ranks.index, columns=ranks.columns)

        # Long: Bottom percentile (cheapest)
        signals[ranks <= self.config.long_percentile] = 1

        # Short: Top percentile (most expensive)
        if self.config.short_percentile < 1.0:
            signals[ranks >= self.config.short_percentile] = -1

        return signals

    def calculate_weights(
        self,
        signals: pd.DataFrame,
        value_metric: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Calculate portfolio weights from signals

        Methods:
        - equal: Equal weight across all positions
        - factor_weighted: Weight by factor score (more extreme value = higher weight)
        """
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        for date in signals.index:
            date_signals = signals.loc[date]

            # Get long and short positions
            longs = date_signals[date_signals == 1].index
            shorts = date_signals[date_signals == -1].index

            if self.config.weight_method == 'equal':
                # Equal weight
                n_longs = len(longs)
                n_shorts = len(shorts)

                if n_longs > 0:
                    weights.loc[date, longs] = 1.0 / n_longs

                if n_shorts > 0:
                    weights.loc[date, shorts] = -1.0 / n_shorts

            elif self.config.weight_method == 'factor_weighted':
                # Weight by factor score (more extreme = higher weight)
                if value_metric is not None:
                    date_values = value_metric.loc[date]

                    if len(longs) > 0:
                        long_values = date_values[longs]
                        # Inverse weighting (lower P/E = higher weight)
                        long_weights = (1 / long_values) / (1 / long_values).sum()
                        weights.loc[date, longs] = long_weights

                    if len(shorts) > 0:
                        short_values = date_values[shorts]
                        # Higher P/E = higher short weight
                        short_weights = short_values / short_values.sum()
                        weights.loc[date, shorts] = -short_weights

        return weights

    def backtest(
        self,
        prices: pd.DataFrame,
        value_metric: pd.DataFrame,
        ascending: bool = True
    ) -> Dict:
        """
        Backtest value factor strategy

        Args:
            prices: Asset prices
            value_metric: Value metric (P/E, P/B, etc.)
            ascending: True if lower is better

        Returns:
            Dictionary with performance metrics
        """
        # Generate signals
        signals = self.generate_signals(value_metric, ascending=ascending)

        # Calculate weights
        weights = self.calculate_weights(signals, value_metric)

        # Calculate returns
        returns = prices.pct_change()

        # Portfolio returns (weights from previous day)
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
    """Test Value Factor Strategy"""

    print("=" * 80)
    print("VALUE FACTOR STRATEGY TEST")
    print("=" * 80)
    print()

    # Create synthetic data
    print("Creating synthetic test data...")
    print("- 10 stocks with varying P/E ratios")
    print("- 3 years of daily data")
    print()

    np.random.seed(42)
    n_days = 252 * 3
    n_stocks = 10

    dates = pd.date_range('2021-01-01', periods=n_days, freq='D')
    stocks = [f'Stock_{i}' for i in range(1, n_stocks + 1)]

    # Generate prices with different growth rates
    prices_data = {}
    pe_data = {}

    for i, stock in enumerate(stocks):
        # Stock 1-3: Low P/E, value stocks (slower growth, higher returns)
        # Stock 4-7: Medium P/E
        # Stock 8-10: High P/E, growth stocks (fast growth, lower returns)

        if i < 3:
            # Value stocks: Low P/E, good returns
            growth = 0.0010
            vol = 0.018
            pe = 8 + i * 2  # P/E 8-12
        elif i < 7:
            # Medium stocks
            growth = 0.0006
            vol = 0.015
            pe = 15 + i  # P/E 15-21
        else:
            # Growth stocks: High P/E, lower returns
            growth = 0.0003
            vol = 0.020
            pe = 30 + i * 3  # P/E 30-39

        # Generate returns (value stocks perform better over time)
        returns = np.random.normal(growth, vol, n_days)
        prices = (1 + returns).cumprod() * 100
        prices_data[stock] = prices

        # P/E ratio (gradually mean reverts)
        pe_ratio = pe + np.random.normal(0, 2, n_days)
        pe_data[stock] = pe_ratio

    prices_df = pd.DataFrame(prices_data, index=dates)
    pe_df = pd.DataFrame(pe_data, index=dates)

    # Initialize Value Factor
    config = ValueFactorConfig(
        metric='pe_ratio',
        long_percentile=0.30,  # Buy cheapest 30%
        short_percentile=0.70,  # Short most expensive 30%
        weight_method='equal'
    )
    value_factor = ValueFactor(config)

    # Backtest strategy
    print("=" * 80)
    print("BACKTEST: P/E-BASED VALUE STRATEGY")
    print("=" * 80)
    print()

    print("Strategy:")
    print("  - Long: Lowest 30% by P/E (value stocks)")
    print("  - Short: Highest 30% by P/E (growth stocks)")
    print("  - Rebalance: Monthly")
    print()

    result = value_factor.backtest(prices_df, pe_df, ascending=True)

    print("PERFORMANCE METRICS:")
    print(f"  Total Return: {result['total_return']:>10.2%}")
    print(f"  Annual Return: {result['annual_return']:>10.2%}")
    print(f"  Volatility: {result['volatility']:>10.2%}")
    print(f"  Sharpe Ratio: {result['sharpe_ratio']:>10.2f}")
    print(f"  Max Drawdown: {result['max_drawdown']:>10.2%}")
    print(f"  Win Rate: {result['win_rate']:>10.2%}")
    print()

    print("LONG/SHORT BREAKDOWN:")
    print(f"  Long Sharpe: {result['long_sharpe']:>10.2f}")
    print(f"  Short Sharpe: {result['short_sharpe']:>10.2f}")
    print()

    # Compare with buy-and-hold
    print("=" * 80)
    print("COMPARISON: Value Strategy vs. Equal-Weight Buy & Hold")
    print("=" * 80)
    print()

    # Equal weight buy and hold
    equal_returns = prices_df.pct_change().mean(axis=1)
    equal_total = (1 + equal_returns).prod() - 1
    equal_ann = (1 + equal_total) ** (252 / len(equal_returns)) - 1
    equal_vol = equal_returns.std() * np.sqrt(252)
    equal_sharpe = equal_ann / equal_vol if equal_vol > 0 else 0

    comparison = pd.DataFrame({
        'Strategy': ['Value Factor', 'Equal Weight'],
        'Annual Return': [result['annual_return'], equal_ann],
        'Volatility': [result['volatility'], equal_vol],
        'Sharpe Ratio': [result['sharpe_ratio'], equal_sharpe],
        'Max Drawdown': [result['max_drawdown'],
                         ((1 + equal_returns).cumprod() / (1 + equal_returns).cumprod().expanding().max() - 1).min()]
    })

    print(comparison.to_string(index=False))
    print()

    outperformance = result['annual_return'] - equal_ann
    print(f"Value Premium (Outperformance): {outperformance:+.2%} annually")

    # Show sample holdings
    print("\n" + "=" * 80)
    print("SAMPLE PORTFOLIO HOLDINGS (Latest Date)")
    print("=" * 80)
    print()

    latest_signals = result['signals'].iloc[-1]
    latest_pe = pe_df.iloc[-1]

    longs = latest_signals[latest_signals == 1].index.tolist()
    shorts = latest_signals[latest_signals == -1].index.tolist()

    print("LONG POSITIONS (Value Stocks):")
    for stock in longs:
        print(f"  {stock}: P/E = {latest_pe[stock]:.1f}")

    print("\nSHORT POSITIONS (Growth Stocks):")
    for stock in shorts:
        print(f"  {stock}: P/E = {latest_pe[stock]:.1f}")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("Value Factor (HML):")
    print("  - Fama-French: ~5% annual premium (1926-present)")
    print("  - Works across countries, asset classes")
    print("  - Mean reversion + behavioral bias")
    print()
    print("Best Value Metrics:")
    print("  - P/E Ratio: Most common, earnings-based")
    print("  - P/B Ratio: Buffett's favorite, book value")
    print("  - Dividend Yield: Income-focused")
    print("  - FCF Yield: Cash generation")
    print()
    print("Implementation:")
    print("  - Rebalance monthly/quarterly (not daily)")
    print("  - Long-short or long-only")
    print("  - Can combine with momentum, quality")
    print()
    print("Risks:")
    print("  - Value traps (cheap for a reason)")
    print("  - Long periods of underperformance (2007-2020)")
    print("  - Works best with quality filters")
    print()
    print("=" * 80)
    print("Test complete!")
    print("=" * 80)
