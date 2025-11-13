"""
RSI (Relative Strength Index) Strategy

Based on open-source implementation from je-suis-tm/quant-trading
Adapted for crypto and daily timeframe trading

Strategy:
- Buy when RSI < 30 (oversold)
- Sell when RSI > 70 (overbought)
- Hold when 30 <= RSI <= 70 (neutral zone)

Timeframe: Daily
Asset Classes: All (best for Crypto, Equity)
"""

import numpy as np
import pandas as pd
from typing import List
from dataclasses import dataclass


@dataclass
class RSIConfig:
    """Configuration for RSI strategy"""
    period: int = 14  # RSI calculation period
    oversold: int = 30  # Oversold threshold
    overbought: int = 70  # Overbought threshold
    position_size: float = 0.10  # 10% per position
    hold_days: int = 5  # Minimum holding period
    exit_rsi_change: float = 4.0  # Exit if RSI increases by this much


class RSIStrategy:
    """
    RSI Oscillator Strategy

    Uses the Relative Strength Index to identify overbought/oversold conditions.
    Buys when assets are oversold (RSI < 30) and sells when overbought (RSI > 70).
    """

    def __init__(self, config: RSIConfig = None):
        self.config = config or RSIConfig()

    def calculate_rsi(self, prices: pd.Series) -> pd.Series:
        """
        Calculate RSI indicator

        RSI = 100 - (100 / (1 + RS))
        where RS = Average Gain / Average Loss
        """
        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)

        # Calculate exponential moving averages
        avg_gains = gains.ewm(span=self.config.period, adjust=False).mean()
        avg_losses = losses.ewm(span=self.config.period, adjust=False).mean()

        # Calculate RS and RSI
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))

        return rsi

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on RSI

        Args:
            prices: DataFrame with asset prices as columns

        Returns:
            DataFrame with signals (1=buy, -1=sell, 0=hold)
        """
        signals = pd.DataFrame(index=prices.index)

        for column in prices.columns:
            # Calculate RSI
            rsi = self.calculate_rsi(prices[column])

            # Simple oversold/overbought strategy
            # Buy when RSI < 30 (oversold)
            # Sell when RSI > 70 (overbought)
            positions = np.select(
                [rsi < self.config.oversold, rsi > self.config.overbought],
                [1, -1],  # 1 = buy, -1 = sell
                default=0  # 0 = hold
            )

            # Generate trade signals (changes in position)
            trade_signals = np.diff(positions, prepend=0)

            signals[column] = trade_signals

        return signals

    def generate_advanced_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Advanced RSI strategy with holding period and exit conditions

        This implements the pattern recognition approach:
        - Enter on oversold conditions (RSI < 30)
        - Exit after holding period OR if RSI increases significantly
        """
        signals = pd.DataFrame(index=prices.index)

        for column in prices.columns:
            rsi = self.calculate_rsi(prices[column])

            asset_signals = np.zeros(len(rsi))
            position = False
            entry_rsi = 0
            days_held = 0

            for i in range(len(rsi)):
                if pd.isna(rsi.iloc[i]):
                    continue

                if not position:
                    # Look for entry: oversold condition
                    if rsi.iloc[i] < self.config.oversold:
                        asset_signals[i] = 1  # Buy signal
                        position = True
                        entry_rsi = rsi.iloc[i]
                        days_held = 0
                else:
                    # Check exit conditions
                    days_held += 1
                    rsi_change = rsi.iloc[i] - entry_rsi

                    exit_condition = (
                        days_held >= self.config.hold_days or
                        rsi_change >= self.config.exit_rsi_change or
                        rsi.iloc[i] > self.config.overbought
                    )

                    if exit_condition:
                        asset_signals[i] = -1  # Sell signal
                        position = False

            signals[column] = asset_signals

        return signals

    def get_position_weights(self, signals: pd.DataFrame) -> pd.DataFrame:
        """
        Convert signals to position weights
        """
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        for column in signals.columns:
            position = 0
            for i, signal in enumerate(signals[column]):
                if signal == 1:  # Buy signal
                    position = self.config.position_size
                elif signal == -1:  # Sell signal
                    position = 0

                weights.iloc[i][column] = position

        return weights


if __name__ == "__main__":
    """Test the RSI strategy"""
    import yfinance as yf

    print("Testing RSI Strategy...")
    print("-" * 60)

    # Test with crypto and equity
    symbols = ['BTC-USD', 'ETH-USD', 'SPY']
    data = yf.download(symbols, start='2023-01-01', end='2024-12-31', progress=False)['Adj Close']

    # Initialize strategy
    strategy = RSIStrategy()

    # Test simple strategy
    print("\n=== SIMPLE RSI STRATEGY ===")
    signals = strategy.generate_signals(data)
    print("\nTotal trades per asset:")
    print((signals != 0).sum())

    # Test advanced strategy
    print("\n=== ADVANCED RSI STRATEGY ===")
    advanced_signals = strategy.generate_advanced_signals(data)
    print("\nTotal trades per asset:")
    print((advanced_signals != 0).sum())

    # Display current positions
    weights = strategy.get_position_weights(advanced_signals)
    print("\nCurrent positions:")
    print(weights.iloc[-1])

    print("\nStrategy test complete!")
