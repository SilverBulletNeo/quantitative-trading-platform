"""
MACD (Moving Average Convergence Divergence) Strategy

Based on open-source implementation from je-suis-tm/quant-trading
Adapted for our quantitative trading platform with crypto support

Strategy:
- Buy when fast MA crosses above slow MA (bullish crossover)
- Sell when fast MA crosses below slow MA (bearish crossover)

Timeframe: Daily
Asset Classes: All (Equity, Crypto, Commodities, FX)
"""

import numpy as np
import pandas as pd
from typing import List
from dataclasses import dataclass


@dataclass
class MACDConfig:
    """Configuration for MACD strategy"""
    fast_period: int = 12  # Fast EMA period
    slow_period: int = 26  # Slow EMA period
    signal_period: int = 9  # Signal line period
    position_size: float = 0.10  # 10% per position


class MACDStrategy:
    """
    MACD Oscillator Strategy

    Generates signals based on the difference between fast and slow exponential
    moving averages. When fast MA > slow MA, the trend is bullish (buy signal).
    When fast MA < slow MA, the trend is bearish (sell signal).
    """

    def __init__(self, config: MACDConfig = None):
        self.config = config or MACDConfig()

    def calculate_macd(self, prices: pd.Series) -> pd.DataFrame:
        """
        Calculate MACD indicator components

        Returns DataFrame with columns: fast_ma, slow_ma, macd, signal, histogram
        """
        # Calculate exponential moving averages
        fast_ma = prices.ewm(span=self.config.fast_period, adjust=False).mean()
        slow_ma = prices.ewm(span=self.config.slow_period, adjust=False).mean()

        # MACD line = fast_ma - slow_ma
        macd = fast_ma - slow_ma

        # Signal line = EMA of MACD
        signal = macd.ewm(span=self.config.signal_period, adjust=False).mean()

        # Histogram = MACD - Signal
        histogram = macd - signal

        return pd.DataFrame({
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'macd': macd,
            'signal': signal,
            'histogram': histogram
        })

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals for all assets in the DataFrame

        Args:
            prices: DataFrame with asset prices as columns

        Returns:
            DataFrame with signals (1=buy, -1=sell, 0=hold) for each asset
        """
        signals = pd.DataFrame(index=prices.index)

        for column in prices.columns:
            # Calculate MACD for this asset
            macd_data = self.calculate_macd(prices[column])

            # Generate position signals
            # Position = 1 when fast_ma > slow_ma (bullish)
            # Position = -1 when fast_ma < slow_ma (bearish)
            positions = np.where(
                macd_data['fast_ma'] > macd_data['slow_ma'],
                1,  # Long position
                0   # No position (can add -1 for short if desired)
            )

            # Generate trade signals (difference in positions)
            # Signal = 1 when crossing into bullish (buy)
            # Signal = -1 when crossing into bearish (sell)
            trade_signals = np.diff(positions, prepend=0)

            signals[column] = trade_signals

        return signals

    def get_position_weights(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Convert signals to position weights

        Args:
            signals: DataFrame with trading signals

        Returns:
            DataFrame with position weights for each asset
        """
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        for column in signals.columns:
            # Track current position
            position = 0
            for i, signal in enumerate(signals[column]):
                if signal == 1:  # Buy signal
                    position = self.config.position_size
                elif signal == -1:  # Sell signal
                    position = 0

                weights.iloc[i][column] = position

        return weights


if __name__ == "__main__":
    """Test the MACD strategy"""
    import yfinance as yf

    # Download sample data
    print("Testing MACD Strategy...")
    print("-" * 60)

    # Test with BTC and ETH
    symbols = ['BTC-USD', 'ETH-USD', 'SPY']
    data = yf.download(symbols, start='2023-01-01', end='2024-12-31', progress=False)['Adj Close']

    # Initialize strategy
    strategy = MACDStrategy()

    # Generate signals
    signals = strategy.generate_signals(data)
    weights = strategy.get_position_weights(signals)

    # Display results
    print("\nLast 10 signals:")
    print(signals.tail(10))

    print("\nCurrent positions:")
    print(weights.iloc[-1])

    print("\nTotal trades per asset:")
    print((signals != 0).sum())

    print("\nStrategy test complete!")
