"""
Bollinger Bands W-Pattern Strategy

Based on open-source implementation from je-suis-tm/quant-trading
Adapted for crypto trading with daily timeframe

Strategy:
- Identify "W" pattern (double bottom) near lower Bollinger Band
- Enter long when price breaks above upper band after forming W
- Exit when bandwidth contracts (low volatility)

Timeframe: Daily
Asset Classes: Crypto, Equity, Commodities
"""

import numpy as np
import pandas as pd
from typing import List, Optional
from dataclasses import dataclass


@dataclass
class BollingerBandsConfig:
    """Configuration for Bollinger Bands strategy"""
    period: int = 20  # Moving average period
    std_dev: float = 2.0  # Standard deviation multiplier
    lookback: int = 75  # Lookback period for W pattern
    alpha: float = 0.0001  # Tolerance for band proximity
    beta: float = 0.0001  # Bandwidth contraction threshold
    position_size: float = 0.10


class BollingerBandsStrategy:
    """
    Bollinger Bands W-Pattern Recognition Strategy

    Identifies double-bottom (W) patterns using Bollinger Bands as a framework.
    The strategy looks for reversal signals when price forms a W near the lower band
    and then breaks above the upper band.
    """

    def __init__(self, config: BollingerBandsConfig = None):
        self.config = config or BollingerBandsConfig()

    def calculate_bands(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate Bollinger Bands

        Returns DataFrame with: middle_band, upper_band, lower_band, bandwidth
        """
        # Middle band = Simple Moving Average
        middle = prices.rolling(window=self.config.period).mean()

        # Calculate standard deviation
        std = prices.rolling(window=self.config.period).std()

        # Upper and lower bands
        upper = middle + (self.config.std_dev * std)
        lower = middle - (self.config.std_dev * std)

        # Bandwidth (indicator of volatility)
        bandwidth = (upper - lower) / middle

        return pd.DataFrame({
            'middle': middle,
            'upper': upper,
            'lower': lower,
            'std': std,
            'bandwidth': bandwidth
        })

    def detect_w_pattern(self, prices: pd.Series, bands: pd.DataFrame, index: int) -> bool:
        """
        Detect W pattern (double bottom) formation

        The W pattern consists of 5 nodes:
        - Node 1: First bottom (touching lower band)
        - Node 2: Middle peak (near middle band)
        - Node 3: Second bottom (near lower band, higher than Node 1)
        - Node 4: Price breaks above upper band
        - Node 5: Current point

        Args:
            prices: Price series
            bands: Bollinger Bands DataFrame
            index: Current index to check

        Returns:
            True if W pattern is detected
        """
        if index < self.config.lookback:
            return False

        lookback_start = max(0, index - self.config.lookback)
        price_window = prices.iloc[lookback_start:index + 1]
        bands_window = bands.iloc[lookback_start:index + 1]

        try:
            # Condition 4: Current price breaks above upper band
            if prices.iloc[index] <= bands['upper'].iloc[index] * (1 - self.config.alpha):
                return False

            # Search backward for pattern nodes
            search_range = range(len(price_window) - 2, 0, -1)

            for i in search_range:
                # Node 2: Middle peak near middle band
                if abs(price_window.iloc[i] - bands_window['middle'].iloc[i]) < self.config.alpha:

                    # Node 1: First bottom near lower band
                    for j in range(i - 1, 0, -1):
                        if abs(price_window.iloc[j] - bands_window['lower'].iloc[j]) < self.config.alpha:
                            first_bottom = price_window.iloc[j]

                            # Node 3: Second bottom near lower band, higher than first
                            for k in range(i + 1, len(price_window) - 1):
                                near_lower = abs(price_window.iloc[k] - bands_window['lower'].iloc[k]) < self.config.alpha
                                higher_than_first = price_window.iloc[k] > first_bottom

                                if near_lower and higher_than_first:
                                    return True

            return False

        except (IndexError, KeyError):
            return False

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on Bollinger Bands

        Simple strategy:
        - Buy when price touches lower band (oversold)
        - Sell when price touches upper band (overbought) or bandwidth contracts
        """
        signals = pd.DataFrame(index=prices.index)

        for column in prices.columns:
            bands = self.calculate_bands(prices[column])

            asset_signals = np.zeros(len(prices))
            position = False

            for i in range(self.config.period, len(prices)):
                if pd.isna(bands['lower'].iloc[i]):
                    continue

                if not position:
                    # Entry: Price near lower band (oversold)
                    if prices[column].iloc[i] <= bands['lower'].iloc[i] * (1 + self.config.alpha):
                        asset_signals[i] = 1
                        position = True
                else:
                    # Exit conditions
                    upper_touch = prices[column].iloc[i] >= bands['upper'].iloc[i] * (1 - self.config.alpha)
                    low_volatility = bands['bandwidth'].iloc[i] < self.config.beta

                    if upper_touch or low_volatility:
                        asset_signals[i] = -1
                        position = False

            signals[column] = asset_signals

        return signals

    def generate_w_pattern_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced strategy using W-pattern recognition
        """
        signals = pd.DataFrame(index=prices.index)

        for column in prices.columns:
            bands = self.calculate_bands(prices[column])

            asset_signals = np.zeros(len(prices))
            position = False

            for i in range(self.config.lookback, len(prices)):
                if not position:
                    # Look for W pattern
                    if self.detect_w_pattern(prices[column], bands, i):
                        asset_signals[i] = 1
                        position = True
                else:
                    # Exit when bandwidth contracts (low volatility)
                    if bands['bandwidth'].iloc[i] < self.config.beta:
                        asset_signals[i] = -1
                        position = False

            signals[column] = asset_signals

        return signals

    def get_position_weights(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Convert signals to position weights"""
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        for column in signals.columns:
            position = 0
            for i, signal in enumerate(signals[column]):
                if signal == 1:
                    position = self.config.position_size
                elif signal == -1:
                    position = 0
                weights.iloc[i][column] = position

        return weights


if __name__ == "__main__":
    """Test Bollinger Bands strategy"""
    import yfinance as yf

    print("Testing Bollinger Bands Strategy...")
    print("-" * 60)

    symbols = ['BTC-USD', 'ETH-USD', 'SPY']
    data = yf.download(symbols, start='2023-01-01', end='2024-12-31', progress=False)['Adj Close']

    strategy = BollingerBandsStrategy()

    # Simple strategy
    print("\n=== SIMPLE BOLLINGER BANDS STRATEGY ===")
    signals = strategy.generate_signals(data)
    print("Total trades per asset:")
    print((signals != 0).sum())

    # W-pattern strategy
    print("\n=== W-PATTERN RECOGNITION STRATEGY ===")
    w_signals = strategy.generate_w_pattern_signals(data)
    print("Total trades per asset:")
    print((w_signals != 0).sum())

    weights = strategy.get_position_weights(w_signals)
    print("\nCurrent positions:")
    print(weights.iloc[-1])

    print("\nStrategy test complete!")
