"""
Mean Reversion Strategy

Statistical arbitrage strategy that trades on price deviations from mean
Suitable for crypto and equity markets on daily timeframe

Strategy:
- Calculate z-score of price vs moving average
- Buy when price is > 2 std devs below mean (oversold)
- Sell when price is > 2 std devs above mean (overbought)
- Exit when price returns to mean

Timeframe: Daily
Asset Classes: Crypto, Equity
"""

import numpy as np
import pandas as pd
from typing import List
from dataclasses import dataclass


@dataclass
class MeanReversionConfig:
    """Configuration for Mean Reversion strategy"""
    lookback_period: int = 20  # Period for calculating mean and std dev
    entry_threshold: float = 2.0  # Z-score threshold for entry (std devs)
    exit_threshold: float = 0.5  # Z-score threshold for exit
    position_size: float = 0.10
    use_bollinger_bands: bool = True  # Use Bollinger Bands approach


class MeanReversionStrategy:
    """
    Mean Reversion Strategy

    Exploits the tendency of prices to revert to their historical mean.
    Uses z-scores to identify extreme deviations and trade reversals.
    """

    def __init__(self, config: MeanReversionConfig = None):
        self.config = config or MeanReversionConfig()

    def calculate_zscore(self, prices: pd.Series) -> pd.Series:
        """
        Calculate z-score: (price - mean) / std_dev

        Positive z-score = price above mean (potentially overbought)
        Negative z-score = price below mean (potentially oversold)
        """
        rolling_mean = prices.rolling(window=self.config.lookback_period).mean()
        rolling_std = prices.rolling(window=self.config.lookback_period).std()

        zscore = (prices - rolling_mean) / rolling_std

        return zscore

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate mean reversion signals

        Buy when z-score < -entry_threshold (price well below mean)
        Sell when z-score > +entry_threshold (price well above mean)
        Exit when z-score crosses back toward zero
        """
        signals = pd.DataFrame(index=prices.index)

        for column in prices.columns:
            zscore = self.calculate_zscore(prices[column])

            asset_signals = np.zeros(len(prices))
            position = 0  # 0 = no position, 1 = long, -1 = short

            for i in range(self.config.lookback_period, len(prices)):
                if pd.isna(zscore.iloc[i]):
                    continue

                current_z = zscore.iloc[i]

                if position == 0:
                    # Look for entry signals
                    if current_z < -self.config.entry_threshold:
                        # Price far below mean - buy
                        asset_signals[i] = 1
                        position = 1
                    elif current_z > self.config.entry_threshold:
                        # Price far above mean - short (if allowed)
                        # For crypto, we'll skip shorting and just wait
                        pass

                elif position == 1:
                    # Currently long - look for exit
                    if current_z > -self.config.exit_threshold:
                        # Price returned to near mean - exit
                        asset_signals[i] = -1
                        position = 0

                elif position == -1:
                    # Currently short - look for exit
                    if current_z < self.config.exit_threshold:
                        asset_signals[i] = 1  # Cover short
                        position = 0

            signals[column] = asset_signals

        return signals

    def generate_bb_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate signals using Bollinger Bands approach

        Buy when price touches lower band
        Sell when price touches upper band
        """
        signals = pd.DataFrame(index=prices.index)

        for column in prices.columns:
            # Calculate Bollinger Bands
            rolling_mean = prices[column].rolling(window=self.config.lookback_period).mean()
            rolling_std = prices[column].rolling(window=self.config.lookback_period).std()

            upper_band = rolling_mean + (2 * rolling_std)
            lower_band = rolling_mean - (2 * rolling_std)

            asset_signals = np.zeros(len(prices))
            position = False

            for i in range(self.config.lookback_period, len(prices)):
                if pd.isna(rolling_mean.iloc[i]):
                    continue

                price = prices[column].iloc[i]

                if not position:
                    # Entry: price at or below lower band
                    if price <= lower_band.iloc[i]:
                        asset_signals[i] = 1
                        position = True
                else:
                    # Exit: price at or above upper band, or crosses mean
                    if price >= upper_band.iloc[i] or price >= rolling_mean.iloc[i]:
                        asset_signals[i] = -1
                        position = False

            signals[column] = asset_signals

        return signals

    def calculate_sharpe_adjusted_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced signals using Sharpe ratio to filter trades

        Only take trades when recent Sharpe ratio is favorable
        """
        signals = pd.DataFrame(index=prices.index)

        for column in prices.columns:
            zscore = self.calculate_zscore(prices[column])

            # Calculate rolling returns and Sharpe
            returns = prices[column].pct_change()
            rolling_sharpe = (
                returns.rolling(window=self.config.lookback_period).mean() /
                returns.rolling(window=self.config.lookback_period).std()
            ) * np.sqrt(252)

            asset_signals = np.zeros(len(prices))
            position = 0

            for i in range(self.config.lookback_period, len(prices)):
                if pd.isna(zscore.iloc[i]) or pd.isna(rolling_sharpe.iloc[i]):
                    continue

                current_z = zscore.iloc[i]
                current_sharpe = rolling_sharpe.iloc[i]

                # Only trade if Sharpe is reasonable (> 0.5)
                if position == 0 and current_sharpe > 0.5:
                    if current_z < -self.config.entry_threshold:
                        asset_signals[i] = 1
                        position = 1

                elif position == 1:
                    if current_z > -self.config.exit_threshold:
                        asset_signals[i] = -1
                        position = 0

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
    """Test Mean Reversion strategy"""
    import yfinance as yf

    print("Testing Mean Reversion Strategy...")
    print("-" * 60)

    symbols = ['BTC-USD', 'ETH-USD', 'SPY']
    data = yf.download(symbols, start='2023-01-01', end='2024-12-31', progress=False)['Adj Close']

    strategy = MeanReversionStrategy()

    # Z-score strategy
    print("\n=== Z-SCORE MEAN REVERSION ===")
    signals = strategy.generate_signals(data)
    print("Total trades per asset:")
    print((signals != 0).sum())

    # Bollinger Bands approach
    print("\n=== BOLLINGER BANDS MEAN REVERSION ===")
    bb_signals = strategy.generate_bb_signals(data)
    print("Total trades per asset:")
    print((bb_signals != 0).sum())

    # Sharpe-adjusted
    print("\n=== SHARPE-ADJUSTED MEAN REVERSION ===")
    sharpe_signals = strategy.calculate_sharpe_adjusted_signals(data)
    print("Total trades per asset:")
    print((sharpe_signals != 0).sum())

    weights = strategy.get_position_weights(sharpe_signals)
    print("\nCurrent positions:")
    print(weights.iloc[-1])

    print("\nStrategy test complete!")
