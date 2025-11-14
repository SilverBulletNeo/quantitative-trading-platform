"""
Parabolic SAR (Stop and Reverse) Strategy

Trend-following strategy using the Parabolic SAR indicator.
Developed by J. Welles Wilder Jr. (also created RSI, ATR).

Strategy:
- Parabolic SAR creates a trailing stop that accelerates with the trend
- Buy when price crosses above SAR (uptrend)
- Sell when price crosses below SAR (downtrend)
- SAR dots appear below price in uptrend, above price in downtrend

Why It Works:
- Excellent for trending markets
- Provides clear entry/exit signals
- Built-in stop loss (the SAR level)
- Acceleration factor increases profit in strong trends

Timeframe: Daily (also works on intraday)
Best For: Trending assets (crypto during bull/bear markets, trending stocks)
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class ParabolicSARConfig:
    """Configuration for Parabolic SAR strategy"""
    acceleration_start: float = 0.02  # Initial acceleration factor
    acceleration_max: float = 0.20  # Maximum acceleration factor
    acceleration_increment: float = 0.02  # Increment per new extreme
    position_size: float = 0.10  # 10% per position


class ParabolicSARStrategy:
    """
    Parabolic SAR (Stop and Reverse) Strategy

    Uses Parabolic SAR indicator to identify trend direction and generate
    buy/sell signals based on SAR crossovers.
    """

    def __init__(self, config: ParabolicSARConfig = None):
        self.config = config or ParabolicSARConfig()

    def calculate_sar(
        self,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate Parabolic SAR indicator

        Algorithm:
        1. SAR starts as previous period's extreme point
        2. If long: SAR = Prior SAR + AF * (Prior EP - Prior SAR)
        3. If short: SAR = Prior SAR - AF * (Prior SAR - Prior EP)
        4. AF starts at 0.02, increases by 0.02 when new extreme is reached (max 0.20)
        5. EP = Extreme Point (highest high if long, lowest low if short)

        Returns DataFrame with: sar, trend (1=up, -1=down), af, ep
        """
        # Initialize
        sar = pd.Series(index=close.index, dtype=float)
        trend = pd.Series(index=close.index, dtype=int)
        af = pd.Series(index=close.index, dtype=float)
        ep = pd.Series(index=close.index, dtype=float)

        # Start values
        sar.iloc[0] = low.iloc[0]
        trend.iloc[0] = 1  # Start bullish
        af.iloc[0] = self.config.acceleration_start
        ep.iloc[0] = high.iloc[0]

        for i in range(1, len(close)):
            # Previous values
            prev_sar = sar.iloc[i-1]
            prev_trend = trend.iloc[i-1]
            prev_af = af.iloc[i-1]
            prev_ep = ep.iloc[i-1]

            if prev_trend == 1:  # Uptrend
                # Calculate SAR for uptrend
                sar.iloc[i] = prev_sar + prev_af * (prev_ep - prev_sar)

                # SAR cannot be above prior two lows
                sar.iloc[i] = min(sar.iloc[i], low.iloc[i-1])
                if i >= 2:
                    sar.iloc[i] = min(sar.iloc[i], low.iloc[i-2])

                # Check for trend reversal
                if low.iloc[i] < sar.iloc[i]:
                    # Trend reverses to downtrend
                    trend.iloc[i] = -1
                    sar.iloc[i] = prev_ep  # SAR becomes previous EP
                    ep.iloc[i] = low.iloc[i]
                    af.iloc[i] = self.config.acceleration_start
                else:
                    # Continue uptrend
                    trend.iloc[i] = 1

                    # Update EP if new high
                    if high.iloc[i] > prev_ep:
                        ep.iloc[i] = high.iloc[i]
                        af.iloc[i] = min(
                            prev_af + self.config.acceleration_increment,
                            self.config.acceleration_max
                        )
                    else:
                        ep.iloc[i] = prev_ep
                        af.iloc[i] = prev_af

            else:  # Downtrend (prev_trend == -1)
                # Calculate SAR for downtrend
                sar.iloc[i] = prev_sar - prev_af * (prev_sar - prev_ep)

                # SAR cannot be below prior two highs
                sar.iloc[i] = max(sar.iloc[i], high.iloc[i-1])
                if i >= 2:
                    sar.iloc[i] = max(sar.iloc[i], high.iloc[i-2])

                # Check for trend reversal
                if high.iloc[i] > sar.iloc[i]:
                    # Trend reverses to uptrend
                    trend.iloc[i] = 1
                    sar.iloc[i] = prev_ep  # SAR becomes previous EP
                    ep.iloc[i] = high.iloc[i]
                    af.iloc[i] = self.config.acceleration_start
                else:
                    # Continue downtrend
                    trend.iloc[i] = -1

                    # Update EP if new low
                    if low.iloc[i] < prev_ep:
                        ep.iloc[i] = low.iloc[i]
                        af.iloc[i] = min(
                            prev_af + self.config.acceleration_increment,
                            self.config.acceleration_max
                        )
                    else:
                        ep.iloc[i] = prev_ep
                        af.iloc[i] = prev_af

        return pd.DataFrame({
            'sar': sar,
            'trend': trend,
            'af': af,
            'ep': ep
        })

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Parabolic SAR trading signals

        Long: When SAR crosses below price (trend changes to up)
        Short/Exit: When SAR crosses above price (trend changes to down)
        """
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        for column in prices.columns:
            # Need OHLC data - use close as proxy for all
            close = prices[column]
            high = prices[column]  # In practice, use actual high
            low = prices[column]   # In practice, use actual low

            # Calculate Parabolic SAR
            sar_data = self.calculate_sar(high, low, close)

            # Generate signals based on trend changes
            trend_changes = sar_data['trend'].diff()

            for i in range(1, len(prices)):
                if trend_changes.iloc[i] == 2:  # Changed from -1 to 1 (uptrend)
                    signals.iloc[i][column] = 1  # Buy signal
                elif trend_changes.iloc[i] == -2:  # Changed from 1 to -1 (downtrend)
                    signals.iloc[i][column] = -1  # Sell signal

        return signals

    def get_position_weights(self, signals: pd.DataFrame) -> pd.DataFrame:
        """Convert signals to position weights"""
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        positions = {col: 0.0 for col in signals.columns}

        for i, (idx, row) in enumerate(signals.iterrows()):
            for symbol in signals.columns:
                signal = row[symbol]

                if signal == 1:  # Buy
                    positions[symbol] = self.config.position_size
                elif signal == -1:  # Sell
                    positions[symbol] = 0

                weights.iloc[i][symbol] = positions[symbol]

        return weights

    def get_sar_statistics(self, prices: pd.DataFrame) -> Dict:
        """Get current SAR statistics for all assets"""
        stats = {}

        for column in prices.columns:
            close = prices[column]
            high = prices[column]
            low = prices[column]

            sar_data = self.calculate_sar(high, low, close)

            stats[column] = {
                'current_sar': sar_data['sar'].iloc[-1],
                'current_price': close.iloc[-1],
                'current_trend': 'UPTREND' if sar_data['trend'].iloc[-1] == 1 else 'DOWNTREND',
                'acceleration_factor': sar_data['af'].iloc[-1],
                'extreme_point': sar_data['ep'].iloc[-1],
                'distance_to_sar': (close.iloc[-1] - sar_data['sar'].iloc[-1]) / close.iloc[-1]
            }

        return stats


if __name__ == "__main__":
    """Test Parabolic SAR strategy"""
    import yfinance as yf

    print("=" * 80)
    print("PARABOLIC SAR STRATEGY TEST")
    print("=" * 80)
    print()

    # Test with trending assets
    symbols = ['BTC-USD', 'ETH-USD', 'SPY', 'QQQ']

    print(f"Fetching data for {len(symbols)} assets...")
    data = yf.download(symbols, start='2023-01-01', end='2024-12-31', progress=False)['Adj Close']

    # Initialize strategy
    strategy = ParabolicSARStrategy()

    # Get SAR statistics
    print("\n" + "=" * 80)
    print("PARABOLIC SAR ANALYSIS")
    print("=" * 80)

    stats = strategy.get_sar_statistics(data)

    for symbol, stat in stats.items():
        print(f"\n{symbol}")
        print(f"  Current Price     : ${stat['current_price']:.2f}")
        print(f"  Current SAR       : ${stat['current_sar']:.2f}")
        print(f"  Trend             : {stat['current_trend']}")
        print(f"  Accel Factor (AF) : {stat['acceleration_factor']:.3f}")
        print(f"  Extreme Point (EP): ${stat['extreme_point']:.2f}")
        print(f"  Distance to SAR   : {stat['distance_to_sar']:.2%}")

    # Generate signals
    print("\n" + "=" * 80)
    print("GENERATING SIGNALS")
    print("=" * 80)

    signals = strategy.generate_signals(data)
    print(f"\nTotal signals: {(signals != 0).sum().sum()}")

    # Show signal distribution
    print("\nSignals per asset:")
    for col in signals.columns:
        buy_signals = (signals[col] == 1).sum()
        sell_signals = (signals[col] == -1).sum()
        print(f"  {col:12s}: {buy_signals:3d} buys, {sell_signals:3d} sells")

    # Get position weights
    weights = strategy.get_position_weights(signals)

    print("\n" + "=" * 80)
    print("CURRENT POSITIONS")
    print("=" * 80)

    current_weights = weights.iloc[-1]
    current_positions = current_weights[current_weights > 0]

    if len(current_positions) > 0:
        print("\nPositions (in uptrend):")
        for symbol, weight in current_positions.items():
            print(f"  {symbol:12s}: {weight:>6.2%}")
    else:
        print("\n  No positions (all in downtrend)")

    # Calculate backtest performance
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    returns = data.pct_change()
    portfolio_returns = (weights.shift(1) * returns).sum(axis=1)

    cumulative = (1 + portfolio_returns).cumprod()
    total_return = cumulative.iloc[-1] - 1
    sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() != 0 else 0

    cum_max = cumulative.expanding().max()
    drawdown = (cumulative - cum_max) / cum_max
    max_dd = drawdown.min()

    print(f"\nTotal Return        : {total_return:>10.2%}")
    print(f"Sharpe Ratio        : {sharpe:>10.2f}")
    print(f"Max Drawdown        : {max_dd:>10.2%}")
    print(f"Volatility          : {portfolio_returns.std() * np.sqrt(252):>10.2%}")
    print(f"Total Signals       : {(signals != 0).sum().sum():>10.0f}")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)
