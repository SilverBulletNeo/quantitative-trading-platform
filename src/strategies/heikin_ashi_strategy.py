"""
Heikin-Ashi Candlestick Strategy

Modified candlestick technique that filters out market noise and
identifies trend direction more clearly than traditional candlesticks.

Strategy:
- Transform OHLC data using Heikin-Ashi formulas
- Buy on consecutive green/white candles (uptrend)
- Sell on consecutive red/black candles (downtrend)
- Filter noise better than traditional candlesticks

Why It Works:
- Averages price data to smooth out volatility
- Clearer trend identification
- Fewer false signals in choppy markets
- Better for trend-following

Timeframe: Daily (also works intraday)
Best For: Trending markets, reducing whipsaws
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class HeikinAshiConfig:
    """Configuration for Heikin-Ashi strategy"""
    consecutive_candles: int = 2  # Need N consecutive same-color candles for signal
    position_size: float = 0.10  # 10% per position
    use_shadows: bool = True  # Use upper/lower shadows for confirmation


class HeikinAshiStrategy:
    """
    Heikin-Ashi Candlestick Strategy

    Uses modified candlestick calculations to filter noise and
    identify stronger trend signals.
    """

    def __init__(self, config: HeikinAshiConfig = None):
        self.config = config or HeikinAshiConfig()

    def calculate_heikin_ashi(
        self,
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.DataFrame:
        """
        Calculate Heikin-Ashi candlesticks

        Formulas:
        - HA Close = (O + H + L + C) / 4
        - HA Open = (Previous HA Open + Previous HA Close) / 2
        - HA High = max(H, HA Open, HA Close)
        - HA Low = min(L, HA Open, HA Close)

        Returns DataFrame with ha_open, ha_high, ha_low, ha_close
        """
        ha_close = (open_ + high + low + close) / 4

        ha_open = pd.Series(index=open_.index, dtype=float)
        ha_open.iloc[0] = (open_.iloc[0] + close.iloc[0]) / 2

        for i in range(1, len(open_)):
            ha_open.iloc[i] = (ha_open.iloc[i-1] + ha_close.iloc[i-1]) / 2

        ha_high = pd.concat([high, ha_open, ha_close], axis=1).max(axis=1)
        ha_low = pd.concat([low, ha_open, ha_close], axis=1).min(axis=1)

        return pd.DataFrame({
            'ha_open': ha_open,
            'ha_high': ha_high,
            'ha_low': ha_low,
            'ha_close': ha_close
        })

    def identify_candle_color(self, ha_data: pd.DataFrame) -> pd.Series:
        """
        Identify candle color
        1 = Green/White (bullish)
        -1 = Red/Black (bearish)
        """
        return np.where(ha_data['ha_close'] >= ha_data['ha_open'], 1, -1)

    def check_trend_strength(self, ha_data: pd.DataFrame, index: int) -> Tuple[bool, bool]:
        """
        Check trend strength using shadows

        Strong Uptrend:
        - No lower shadow (ha_low == ha_open)
        - Small upper shadow

        Strong Downtrend:
        - No upper shadow (ha_high == ha_open)
        - Small lower shadow

        Returns: (is_strong_uptrend, is_strong_downtrend)
        """
        if not self.config.use_shadows:
            return False, False

        ha_open = ha_data['ha_open'].iloc[index]
        ha_high = ha_data['ha_high'].iloc[index]
        ha_low = ha_data['ha_low'].iloc[index]
        ha_close = ha_data['ha_close'].iloc[index]

        body_size = abs(ha_close - ha_open)
        upper_shadow = ha_high - max(ha_open, ha_close)
        lower_shadow = min(ha_open, ha_close) - ha_low

        # Strong uptrend: Small/no lower shadow, small upper shadow
        is_strong_up = (
            ha_close > ha_open and
            lower_shadow < body_size * 0.1 and
            upper_shadow < body_size * 0.3
        )

        # Strong downtrend: Small/no upper shadow, small lower shadow
        is_strong_down = (
            ha_close < ha_open and
            upper_shadow < body_size * 0.1 and
            lower_shadow < body_size * 0.3
        )

        return is_strong_up, is_strong_down

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate Heikin-Ashi trading signals

        Entry Rules:
        - LONG: N consecutive green candles (strong uptrend forming)
        - SHORT/EXIT: N consecutive red candles (strong downtrend forming)

        Optional: Use shadow analysis for confirmation
        """
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        for column in prices.columns:
            # For simplified version, use close as proxy for OHLC
            # In practice, use actual OHLC data
            close = prices[column]
            open_ = prices[column]
            high = prices[column]
            low = prices[column]

            # Calculate Heikin-Ashi
            ha_data = self.calculate_heikin_ashi(open_, high, low, close)

            # Identify candle colors
            colors = self.identify_candle_color(ha_data)

            # Track position and consecutive candles
            current_position = 0
            consecutive_green = 0
            consecutive_red = 0

            for i in range(len(prices)):
                candle_color = colors[i]

                # Count consecutive candles
                if candle_color == 1:  # Green
                    consecutive_green += 1
                    consecutive_red = 0
                else:  # Red
                    consecutive_red += 1
                    consecutive_green = 0

                # Check for trend strength if enabled
                if self.config.use_shadows:
                    is_strong_up, is_strong_down = self.check_trend_strength(ha_data, i)
                else:
                    is_strong_up, is_strong_down = False, False

                # Generate signals
                if current_position == 0:
                    # Look for entry
                    if (consecutive_green >= self.config.consecutive_candles or is_strong_up):
                        signals.iloc[i][column] = 1  # Buy
                        current_position = 1
                        consecutive_green = 0  # Reset counter

                elif current_position == 1:
                    # Look for exit
                    if (consecutive_red >= self.config.consecutive_candles or is_strong_down):
                        signals.iloc[i][column] = -1  # Sell
                        current_position = 0
                        consecutive_red = 0  # Reset counter

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

    def get_heikin_ashi_statistics(self, prices: pd.DataFrame) -> Dict:
        """Get current Heikin-Ashi statistics"""
        stats = {}

        for column in prices.columns:
            close = prices[column]
            open_ = prices[column]
            high = prices[column]
            low = prices[column]

            ha_data = self.calculate_heikin_ashi(open_, high, low, close)
            colors = self.identify_candle_color(ha_data)

            # Count recent consecutive candles
            recent_colors = colors[-10:]  # Last 10 candles
            current_color = recent_colors.iloc[-1]
            consecutive = 1

            for i in range(len(recent_colors)-2, -1, -1):
                if recent_colors.iloc[i] == current_color:
                    consecutive += 1
                else:
                    break

            is_strong_up, is_strong_down = self.check_trend_strength(ha_data, -1)

            stats[column] = {
                'current_trend': 'UPTREND' if current_color == 1 else 'DOWNTREND',
                'consecutive_candles': consecutive,
                'strong_trend': is_strong_up or is_strong_down,
                'ha_open': ha_data['ha_open'].iloc[-1],
                'ha_close': ha_data['ha_close'].iloc[-1],
                'body_size': abs(ha_data['ha_close'].iloc[-1] - ha_data['ha_open'].iloc[-1])
            }

        return stats


if __name__ == "__main__":
    """Test Heikin-Ashi strategy"""
    import yfinance as yf

    print("=" * 80)
    print("HEIKIN-ASHI STRATEGY TEST")
    print("=" * 80)
    print()

    # Test with various assets
    symbols = ['BTC-USD', 'ETH-USD', 'SPY', 'QQQ']

    print(f"Fetching data for {len(symbols)} assets...")
    data = yf.download(symbols, start='2023-01-01', end='2024-12-31', progress=False)['Adj Close']

    # Initialize strategy
    config = HeikinAshiConfig(
        consecutive_candles=3,  # Need 3 consecutive candles
        use_shadows=True  # Use shadow analysis
    )
    strategy = HeikinAshiStrategy(config)

    # Get Heikin-Ashi statistics
    print("\n" + "=" * 80)
    print("HEIKIN-ASHI ANALYSIS")
    print("=" * 80)

    stats = strategy.get_heikin_ashi_statistics(data)

    for symbol, stat in stats.items():
        print(f"\n{symbol}")
        print(f"  Current Trend         : {stat['current_trend']}")
        print(f"  Consecutive Candles   : {stat['consecutive_candles']}")
        print(f"  Strong Trend Signal   : {'YES' if stat['strong_trend'] else 'NO'}")
        print(f"  HA Open               : ${stat['ha_open']:.2f}")
        print(f"  HA Close              : ${stat['ha_close']:.2f}")
        print(f"  Body Size             : ${stat['body_size']:.2f}")

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
        print("\n  No positions")

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
    print("\nNote: For best results, use actual OHLC data instead of just Close prices")
