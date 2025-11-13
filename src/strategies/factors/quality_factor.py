"""
Quality Factor Strategy

Quality investing: Buy high-quality businesses, avoid low-quality ones.
Complementary to value factor - helps avoid "value traps".

Academic Foundation:
- Piotroski (2000) - "Value Investing: The Use of Historical Financial Statement Information"
  * F-Score: 9-point quality score
  * Combines value + quality → 23% annual returns
- Novy-Marx (2013) - "The Other Side of Value: The Gross Profitability Premium"
  * Profitable firms outperform unprofitable by ~0.3% monthly
- Asness, Frazzini, Pedersen (2019) - "Quality Minus Junk"
  * QMJ factor: 3.0% annual excess return
  * AQR Capital's quality factor

Quality Metrics:
1. Profitability:
   - ROE (Return on Equity) - % return on shareholder capital
   - ROA (Return on Assets) - % return on total assets
   - Gross Profit Margin - Revenue - COGS
   - Operating Margin - Operating income / Revenue

2. Growth:
   - Earnings growth (YoY)
   - Revenue growth (YoY)
   - Free cash flow growth

3. Safety:
   - Debt-to-Equity ratio (lower = safer)
   - Interest coverage (EBIT / Interest)
   - Current ratio (current assets / current liabilities)

4. Efficiency:
   - Asset turnover (Revenue / Assets)
   - Inventory turnover
   - Cash conversion cycle

Piotroski F-Score (9 points):
Profitability (4 points):
- Positive ROA: +1
- Positive operating cash flow: +1
- ROA increasing: +1
- Cash flow > Net income (quality earnings): +1

Leverage (3 points):
- Debt-to-assets decreasing: +1
- Current ratio increasing: +1
- No new share issuance: +1

Efficiency (2 points):
- Gross margin increasing: +1
- Asset turnover increasing: +1

Strategy:
- Score 8-9: Buy (high quality)
- Score 0-2: Sell/short (low quality)
- Combine with value for best results

Why Quality Works:
- Sustainable competitive advantages (moats)
- Earnings persistence
- Lower bankruptcy risk
- Behavioral - investors undervalue quality
- Flight to quality during crises
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class QualityFactorConfig:
    """Configuration for Quality Factor strategy"""
    metrics: List[str] = None  # Metrics to use ['roe', 'roa', 'debt_to_equity', 'margin']
    use_fscore: bool = True  # Use Piotroski F-Score
    long_percentile: float = 0.20  # Top 20% quality
    short_percentile: float = 0.80  # Bottom 20% quality
    rebalance_frequency: int = 63  # Quarterly (63 trading days)
    min_stocks: int = 5

    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['roe', 'roa', 'gross_margin', 'debt_to_equity']


class QualityFactor:
    """
    Quality Factor Strategy

    Buys high-quality businesses based on profitability, safety, efficiency.
    """

    def __init__(self, config: QualityFactorConfig = None):
        self.config = config or QualityFactorConfig()

    def calculate_roe(
        self,
        net_income: pd.DataFrame,
        shareholders_equity: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate Return on Equity (ROE)

        ROE = Net Income / Shareholders' Equity

        Higher = Better (more profitable use of equity)
        Buffett target: ROE > 15%
        """
        roe = net_income / shareholders_equity
        return roe

    def calculate_roa(
        self,
        net_income: pd.DataFrame,
        total_assets: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate Return on Assets (ROA)

        ROA = Net Income / Total Assets

        Higher = Better (efficient asset use)
        """
        roa = net_income / total_assets
        return roa

    def calculate_gross_margin(
        self,
        revenue: pd.DataFrame,
        cogs: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate Gross Profit Margin

        Gross Margin = (Revenue - COGS) / Revenue

        Higher = Better (pricing power, efficiency)
        """
        gross_margin = (revenue - cogs) / revenue
        return gross_margin

    def calculate_operating_margin(
        self,
        operating_income: pd.DataFrame,
        revenue: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate Operating Margin

        Operating Margin = Operating Income / Revenue

        Higher = Better
        """
        operating_margin = operating_income / revenue
        return operating_margin

    def calculate_debt_to_equity(
        self,
        total_debt: pd.DataFrame,
        shareholders_equity: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate Debt-to-Equity Ratio

        D/E = Total Debt / Shareholders' Equity

        Lower = Better (less leverage, safer)
        """
        debt_to_equity = total_debt / shareholders_equity
        return debt_to_equity

    def calculate_interest_coverage(
        self,
        ebit: pd.DataFrame,
        interest_expense: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate Interest Coverage Ratio

        Interest Coverage = EBIT / Interest Expense

        Higher = Better (can easily pay interest)
        Minimum recommended: > 2.5
        """
        interest_coverage = ebit / interest_expense
        return interest_coverage

    def calculate_current_ratio(
        self,
        current_assets: pd.DataFrame,
        current_liabilities: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate Current Ratio

        Current Ratio = Current Assets / Current Liabilities

        Higher = Better (can pay short-term debts)
        Minimum: > 1.0
        """
        current_ratio = current_assets / current_liabilities
        return current_ratio

    def calculate_piotroski_fscore(
        self,
        roa: pd.DataFrame,
        ocf: pd.DataFrame,  # Operating cash flow
        debt: pd.DataFrame,
        current_ratio: pd.DataFrame,
        shares_outstanding: pd.DataFrame,
        gross_margin: pd.DataFrame,
        asset_turnover: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Calculate Piotroski F-Score (0-9 points)

        Higher score = Higher quality
        8-9: Strong buy
        0-2: Avoid/short
        """
        fscore = pd.DataFrame(0, index=roa.index, columns=roa.columns)

        # Profitability (4 points)
        fscore += (roa > 0).astype(int)  # Positive ROA
        fscore += (ocf > 0).astype(int)  # Positive operating CF
        fscore += (roa.diff() > 0).astype(int)  # Increasing ROA
        fscore += (ocf > roa).astype(int)  # CF > Net income (quality earnings)

        # Leverage (3 points)
        fscore += (debt.diff() < 0).astype(int)  # Decreasing debt
        fscore += (current_ratio.diff() > 0).astype(int)  # Increasing liquidity
        fscore += (shares_outstanding.diff() <= 0).astype(int)  # No dilution

        # Efficiency (2 points)
        fscore += (gross_margin.diff() > 0).astype(int)  # Improving margins
        fscore += (asset_turnover.diff() > 0).astype(int)  # Improving efficiency

        return fscore

    def composite_quality_score(
        self,
        metrics_dict: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Calculate composite quality score from multiple metrics

        Combines and normalizes all quality metrics
        """
        scores = []

        for metric_name, metric_data in metrics_dict.items():
            # Normalize to 0-1 range (percentile rank)
            if metric_name == 'debt_to_equity':
                # Lower is better for debt
                score = metric_data.rank(axis=1, ascending=True, pct=True)
            else:
                # Higher is better for profitability metrics
                score = metric_data.rank(axis=1, ascending=False, pct=True)

            scores.append(score)

        # Average all scores
        composite = pd.concat(scores).groupby(level=0).mean()

        return composite

    def generate_signals(
        self,
        quality_score: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate long/short signals from quality scores

        Args:
            quality_score: Composite quality scores or F-Score

        Returns:
            DataFrame of signals (+1 long, 0 neutral, -1 short)
        """
        # Rank by quality
        ranks = quality_score.rank(axis=1, ascending=False, pct=True)

        signals = pd.DataFrame(0, index=ranks.index, columns=ranks.columns)

        # Long: Highest quality (top percentile)
        signals[ranks <= self.config.long_percentile] = 1

        # Short: Lowest quality (bottom percentile)
        if self.config.short_percentile < 1.0:
            signals[ranks >= self.config.short_percentile] = -1

        return signals

    def backtest(
        self,
        prices: pd.DataFrame,
        quality_score: pd.DataFrame
    ) -> Dict:
        """
        Backtest quality factor strategy

        Args:
            prices: Asset prices
            quality_score: Quality scores (F-Score or composite)

        Returns:
            Performance metrics
        """
        # Generate signals
        signals = self.generate_signals(quality_score)

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

        return {
            'total_return': total_return,
            'annual_return': ann_return,
            'volatility': ann_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'portfolio_returns': portfolio_returns,
            'signals': signals,
            'weights': weights
        }


if __name__ == "__main__":
    """Test Quality Factor Strategy"""

    print("=" * 80)
    print("QUALITY FACTOR STRATEGY TEST")
    print("=" * 80)
    print()

    # Create synthetic data
    print("Creating synthetic test data...")
    print("- 10 stocks with varying quality metrics")
    print("- 3 years of data")
    print()

    np.random.seed(42)
    n_days = 252 * 3
    n_stocks = 10

    dates = pd.date_range('2021-01-01', periods=n_days, freq='D')
    stocks = [f'Stock_{i}' for i in range(1, n_stocks + 1)]

    # Generate data with quality-based performance
    prices_data = {}
    roe_data = {}
    roa_data = {}
    margin_data = {}
    debt_data = {}

    for i, stock in enumerate(stocks):
        # High quality stocks (Stock 1-3): High ROE, low debt, strong returns
        # Medium quality (Stock 4-7): Average metrics
        # Low quality (Stock 8-10): Low ROE, high debt, weak returns

        if i < 3:
            # High quality
            roe = 0.18 + np.random.normal(0, 0.02, n_days)  # 18% ROE
            roa = 0.10 + np.random.normal(0, 0.01, n_days)
            margin = 0.25 + np.random.normal(0, 0.02, n_days)
            debt_to_eq = 0.3 + np.random.normal(0, 0.05, n_days)
            growth = 0.0010
            vol = 0.015
        elif i < 7:
            # Medium quality
            roe = 0.12 + np.random.normal(0, 0.03, n_days)  # 12% ROE
            roa = 0.06 + np.random.normal(0, 0.02, n_days)
            margin = 0.15 + np.random.normal(0, 0.03, n_days)
            debt_to_eq = 0.8 + np.random.normal(0, 0.1, n_days)
            growth = 0.0005
            vol = 0.018
        else:
            # Low quality
            roe = 0.05 + np.random.normal(0, 0.04, n_days)  # 5% ROE
            roa = 0.02 + np.random.normal(0, 0.02, n_days)
            margin = 0.08 + np.random.normal(0, 0.03, n_days)
            debt_to_eq = 1.5 + np.random.normal(0, 0.2, n_days)
            growth = 0.0001
            vol = 0.025

        # Generate prices (quality drives returns)
        returns = np.random.normal(growth, vol, n_days)
        prices = (1 + returns).cumprod() * 100
        prices_data[stock] = prices
        roe_data[stock] = roe
        roa_data[stock] = roa
        margin_data[stock] = margin
        debt_data[stock] = debt_to_eq

    prices_df = pd.DataFrame(prices_data, index=dates)
    roe_df = pd.DataFrame(roe_data, index=dates)
    roa_df = pd.DataFrame(roa_data, index=dates)
    margin_df = pd.DataFrame(margin_data, index=dates)
    debt_df = pd.DataFrame(debt_data, index=dates)

    # Initialize Quality Factor
    quality_factor = QualityFactor()

    # Create composite quality score
    metrics_dict = {
        'roe': roe_df,
        'roa': roa_df,
        'gross_margin': margin_df,
        'debt_to_equity': debt_df
    }

    print("=" * 80)
    print("CALCULATING COMPOSITE QUALITY SCORE")
    print("=" * 80)
    print()

    print("Metrics used:")
    print("  - ROE (Return on Equity) - Higher is better")
    print("  - ROA (Return on Assets) - Higher is better")
    print("  - Gross Margin - Higher is better")
    print("  - Debt-to-Equity - Lower is better")
    print()

    quality_score = quality_factor.composite_quality_score(metrics_dict)

    # Backtest
    print("=" * 80)
    print("BACKTEST: QUALITY FACTOR STRATEGY")
    print("=" * 80)
    print()

    print("Strategy:")
    print("  - Long: Top 20% quality (high ROE, low debt)")
    print("  - Short: Bottom 20% quality (low ROE, high debt)")
    print("  - Rebalance: Quarterly")
    print()

    result = quality_factor.backtest(prices_df, quality_score)

    print("PERFORMANCE METRICS:")
    print(f"  Total Return: {result['total_return']:>10.2%}")
    print(f"  Annual Return: {result['annual_return']:>10.2%}")
    print(f"  Volatility: {result['volatility']:>10.2%}")
    print(f"  Sharpe Ratio: {result['sharpe_ratio']:>10.2f}")
    print(f"  Max Drawdown: {result['max_drawdown']:>10.2%}")
    print(f"  Win Rate: {result['win_rate']:>10.2%}")
    print()

    # Show sample holdings
    print("=" * 80)
    print("SAMPLE PORTFOLIO HOLDINGS (Latest Date)")
    print("=" * 80)
    print()

    latest_signals = result['signals'].iloc[-1]
    latest_scores = quality_score.iloc[-1]
    latest_roe = roe_df.iloc[-1]
    latest_debt = debt_df.iloc[-1]

    longs = latest_signals[latest_signals == 1].index.tolist()
    shorts = latest_signals[latest_signals == -1].index.tolist()

    print("LONG POSITIONS (High Quality):")
    for stock in longs:
        print(f"  {stock}: ROE={latest_roe[stock]:.1%}, D/E={latest_debt[stock]:.2f}, "
              f"Quality Score={latest_scores[stock]:.3f}")

    print("\nSHORT POSITIONS (Low Quality):")
    for stock in shorts:
        print(f"  {stock}: ROE={latest_roe[stock]:.1%}, D/E={latest_debt[stock]:.2f}, "
              f"Quality Score={latest_scores[stock]:.3f}")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("Quality Factor (QMJ):")
    print("  - AQR: 3.0% annual excess return")
    print("  - Novy-Marx: Profitability premium 0.3%/month")
    print("  - Piotroski: Value + Quality → 23% annual")
    print()
    print("Best Quality Metrics:")
    print("  - ROE > 15% (Buffett's threshold)")
    print("  - Stable/growing profit margins")
    print("  - Low debt-to-equity")
    print("  - High interest coverage (>3x)")
    print()
    print("Piotroski F-Score:")
    print("  - 9-point score (0-9)")
    print("  - Score 8-9: Strong buy")
    print("  - Score 0-2: Avoid/short")
    print("  - Combines profitability, leverage, efficiency")
    print()
    print("Quality + Value:")
    print("  - Quality avoids 'value traps'")
    print("  - Best combination: Cheap + High quality")
    print("  - Multi-factor approach recommended")
    print()
    print("Crisis Performance:")
    print("  - Quality outperforms in recessions")
    print("  - 'Flight to quality' effect")
    print("  - Lower drawdowns vs. junk stocks")
    print()
    print("=" * 80)
    print("Test complete!")
    print("=" * 80)
