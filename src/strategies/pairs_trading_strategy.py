"""
Pairs Trading Strategy - Statistical Arbitrage

Based on cointegration theory and mean reversion
Sourced from je-suis-tm/quant-trading (8.6k stars)

Strategy:
- Find pairs of assets that are cointegrated (move together long-term)
- Calculate spread (difference) between the pair
- Trade when spread deviates from mean
- Entry: Spread >2 std devs from mean
- Exit: Spread returns to mean

Academic Foundation:
- Engle-Granger two-step cointegration test
- Augmented Dickey-Fuller test for stationarity
- Statistical arbitrage with mean-reverting spreads

Timeframe: Daily
Best For: Highly correlated asset pairs (e.g., BTC-ETH, SPY-QQQ)
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

try:
    from statsmodels.tsa.stattools import adfuller, coint
    from statsmodels.regression.linear_model import OLS
except ImportError:
    print("Warning: statsmodels not installed. Install with: pip install statsmodels")
    # Provide fallback implementations
    adfuller = None
    coint = None
    OLS = None


@dataclass
class PairsTradingConfig:
    """Configuration for Pairs Trading strategy"""
    lookback_period: int = 60  # Period for cointegration testing
    entry_threshold: float = 2.0  # Z-score threshold for entry
    exit_threshold: float = 0.5  # Z-score threshold for exit
    position_size: float = 0.10  # 10% per pair (5% each side)
    max_holding_period: int = 30  # Maximum days to hold position


class PairsTradingStrategy:
    """
    Pairs Trading Strategy using Cointegration

    Statistical arbitrage strategy that exploits mean-reverting relationships
    between cointegrated asset pairs.
    """

    def __init__(self, config: PairsTradingConfig = None):
        self.config = config or PairsTradingConfig()
        self.pairs = []  # List of cointegrated pairs

    def test_cointegration(self, asset1: pd.Series, asset2: pd.Series) -> Tuple[bool, float, float]:
        """
        Test if two assets are cointegrated using Engle-Granger method

        Returns:
            (is_cointegrated, p_value, hedge_ratio)
        """
        if coint is None or OLS is None:
            # Fallback: simple correlation test
            correlation = asset1.corr(asset2)
            return correlation > 0.7, 0.05 if correlation > 0.7 else 0.5, 1.0

        # Step 1: Estimate hedge ratio using OLS regression
        model = OLS(asset1, asset2)
        results = model.fit()
        hedge_ratio = results.params[0]

        # Step 2: Calculate spread
        spread = asset1 - hedge_ratio * asset2

        # Step 3: Test spread for stationarity using ADF test
        if adfuller is not None:
            adf_result = adfuller(spread.dropna())
            p_value = adf_result[1]
            is_cointegrated = p_value < 0.05  # 95% confidence
        else:
            # Fallback
            p_value = 0.05
            is_cointegrated = True

        return is_cointegrated, p_value, hedge_ratio

    def find_pairs(self, prices: pd.DataFrame) -> List[Tuple[str, str, float]]:
        """
        Find all cointegrated pairs in the universe

        Returns:
            List of (asset1, asset2, hedge_ratio) tuples
        """
        pairs = []
        symbols = prices.columns.tolist()

        print(f"Testing {len(symbols)} assets for cointegration...")

        # Test all combinations
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                asset1_name = symbols[i]
                asset2_name = symbols[j]

                asset1 = prices[asset1_name].dropna()
                asset2 = prices[asset2_name].dropna()

                # Align series
                common_index = asset1.index.intersection(asset2.index)
                if len(common_index) < self.config.lookback_period:
                    continue

                asset1 = asset1.loc[common_index]
                asset2 = asset2.loc[common_index]

                # Test cointegration
                is_coint, p_value, hedge_ratio = self.test_cointegration(asset1, asset2)

                if is_coint:
                    pairs.append((asset1_name, asset2_name, hedge_ratio))
                    print(f"  Found pair: {asset1_name}-{asset2_name} (p={p_value:.4f}, hedge={hedge_ratio:.2f})")

        self.pairs = pairs
        print(f"Total cointegrated pairs found: {len(pairs)}")
        return pairs

    def calculate_spread(
        self,
        asset1: pd.Series,
        asset2: pd.Series,
        hedge_ratio: float
    ) -> pd.Series:
        """Calculate normalized spread between two assets"""
        spread = asset1 - hedge_ratio * asset2
        return spread

    def calculate_zscore(self, spread: pd.Series) -> pd.Series:
        """Calculate z-score of spread"""
        mean = spread.rolling(window=self.config.lookback_period).mean()
        std = spread.rolling(window=self.config.lookback_period).std()
        zscore = (spread - mean) / std
        return zscore

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate pairs trading signals

        For each pair:
        - Long Asset1, Short Asset2 when z-score < -threshold (Asset1 undervalued)
        - Short Asset1, Long Asset2 when z-score > +threshold (Asset1 overvalued)
        - Exit when z-score returns to mean
        """
        # First, find cointegrated pairs
        if not self.pairs:
            self.find_pairs(prices)

        if not self.pairs:
            print("Warning: No cointegrated pairs found!")
            return pd.DataFrame(0, index=prices.index, columns=prices.columns)

        # Initialize signals DataFrame
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        # Generate signals for each pair
        for asset1_name, asset2_name, hedge_ratio in self.pairs:
            asset1 = prices[asset1_name]
            asset2 = prices[asset2_name]

            # Calculate spread and z-score
            spread = self.calculate_spread(asset1, asset2, hedge_ratio)
            zscore = self.calculate_zscore(spread)

            # Track positions for this pair
            position = 0  # 0=flat, 1=long spread, -1=short spread
            days_held = 0

            for i in range(self.config.lookback_period, len(prices)):
                current_z = zscore.iloc[i]

                if pd.isna(current_z):
                    continue

                if position == 0:
                    # Look for entry
                    if current_z < -self.config.entry_threshold:
                        # Long spread: Long asset1, Short asset2
                        signals.iloc[i][asset1_name] = 1
                        signals.iloc[i][asset2_name] = -1
                        position = 1
                        days_held = 0
                    elif current_z > self.config.entry_threshold:
                        # Short spread: Short asset1, Long asset2
                        signals.iloc[i][asset1_name] = -1
                        signals.iloc[i][asset2_name] = 1
                        position = -1
                        days_held = 0

                elif position != 0:
                    days_held += 1

                    # Exit conditions
                    exit_mean_reversion = abs(current_z) < self.config.exit_threshold
                    exit_max_holding = days_held >= self.config.max_holding_period
                    exit_reversal = (position == 1 and current_z > 0) or (position == -1 and current_z < 0)

                    if exit_mean_reversion or exit_max_holding or exit_reversal:
                        # Exit positions
                        signals.iloc[i][asset1_name] = -position  # Reverse position
                        signals.iloc[i][asset2_name] = position
                        position = 0
                        days_held = 0

        return signals

    def get_position_weights(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Convert signals to position weights"""
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        # Track current positions
        positions = {col: 0.0 for col in signals.columns}

        for i, (idx, row) in enumerate(signals.iterrows()):
            for symbol in signals.columns:
                signal = row[symbol]

                if signal == 1:  # Buy
                    positions[symbol] = self.config.position_size / 2  # Half position per leg
                elif signal == -1:  # Sell/Close
                    positions[symbol] = 0

                weights.iloc[i][symbol] = positions[symbol]

        return weights

    def get_statistics(self, prices: pd.DataFrame) -> Dict:
        """Calculate strategy statistics"""
        if not self.pairs:
            return {"total_pairs": 0}

        stats = {
            "total_pairs": len(self.pairs),
            "pairs": []
        }

        for asset1_name, asset2_name, hedge_ratio in self.pairs:
            asset1 = prices[asset1_name]
            asset2 = prices[asset2_name]

            spread = self.calculate_spread(asset1, asset2, hedge_ratio)
            zscore = self.calculate_zscore(spread)

            pair_stats = {
                "pair": f"{asset1_name}-{asset2_name}",
                "hedge_ratio": hedge_ratio,
                "current_zscore": zscore.iloc[-1] if not zscore.empty else np.nan,
                "mean_zscore": zscore.mean(),
                "std_zscore": zscore.std(),
                "correlation": asset1.corr(asset2)
            }
            stats["pairs"].append(pair_stats)

        return stats


if __name__ == "__main__":
    """Test Pairs Trading strategy"""
    import yfinance as yf

    print("Testing Pairs Trading Strategy...")
    print("-" * 80)

    # Test with assets that are likely cointegrated
    symbols = [
        'BTC-USD',  # Bitcoin
        'ETH-USD',  # Ethereum (often cointegrated with BTC)
        'SPY',      # S&P 500
        'QQQ',      # Nasdaq (often cointegrated with SPY)
        'GLD',      # Gold
        'SLV',      # Silver (often cointegrated with GLD)
    ]

    data = yf.download(symbols, start='2023-01-01', end='2024-12-31', progress=False)['Adj Close']

    # Initialize strategy
    strategy = PairsTradingStrategy()

    # Find pairs
    pairs = strategy.find_pairs(data)

    if pairs:
        print("\n" + "=" * 80)
        print("PAIR STATISTICS")
        print("=" * 80)

        stats = strategy.get_statistics(data)
        for pair_stat in stats["pairs"]:
            print(f"\nPair: {pair_stat['pair']}")
            print(f"  Hedge Ratio: {pair_stat['hedge_ratio']:.4f}")
            print(f"  Correlation: {pair_stat['correlation']:.4f}")
            print(f"  Current Z-Score: {pair_stat['current_zscore']:.2f}")
            print(f"  Mean Z-Score: {pair_stat['mean_zscore']:.4f}")
            print(f"  Std Z-Score: {pair_stat['std_zscore']:.4f}")

        # Generate signals
        print("\n" + "=" * 80)
        print("GENERATING SIGNALS")
        print("=" * 80)

        signals = strategy.generate_signals(data)
        print(f"\nTotal signals generated: {(signals != 0).sum().sum()}")

        # Show recent signals
        recent_signals = signals.tail(10)
        print("\nRecent signals (last 10 days):")
        print(recent_signals[recent_signals.any(axis=1)])

        weights = strategy.get_position_weights(signals)
        print("\nCurrent positions:")
        print(weights.iloc[-1])
    else:
        print("\nNo cointegrated pairs found. Try different assets or longer timeframe.")

    print("\n" + "=" * 80)
    print("Test complete!")
