"""
Time-Series Momentum Strategy (Trend Following)

Also known as "Absolute Momentum" - trades each asset based on its own
past performance, not relative to others.

Academic Foundation:
- "Time Series Momentum" - Moskowitz, Ooi, Pedersen (2012)
- "A Century of Evidence on Trend-Following Investing" - AQR (2014)
- Works across ALL asset classes and timeframes

Strategy:
- For each asset independently:
  - Buy if recent returns are positive (uptrend)
  - Sell/avoid if recent returns are negative (downtrend)
- Use multiple lookback periods (1M, 3M, 6M, 12M)
- Combine signals for robustness

Why It Works:
- Behavioral: Delayed overreaction, disposition effect
- Risk-based: Compensation for time-varying risk
- Structural: Slow institutional capital flows

Timeframe: Daily
Best For: Any tradeable asset, works on crypto, equity, commodities, FX
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class TimeSeriesMomentumConfig:
    """Configuration for Time-Series Momentum strategy"""
    lookback_periods: List[int] = None  # Multiple lookbacks for robustness
    holding_period: int = 5  # Days between rebalances
    position_size: float = 0.10  # 10% per position
    signal_threshold: float = 0.0  # Minimum return to trigger signal
    use_volatility_scaling: bool = True  # Scale positions by inverse volatility
    vol_lookback: int = 60  # Lookback for volatility calculation

    def __post_init__(self):
        if self.lookback_periods is None:
            # Default: 1-month, 3-month, 6-month, 12-month
            self.lookback_periods = [20, 60, 120, 252]


class TimeSeriesMomentumStrategy:
    """
    Time-Series Momentum Strategy

    Trend-following strategy that goes long assets in uptrends and
    avoids/shorts assets in downtrends.
    """

    def __init__(self, config: TimeSeriesMomentumConfig = None):
        self.config = config or TimeSeriesMomentumConfig()

    def calculate_momentum(self, prices: pd.Series, lookback: int) -> pd.Series:
        """Calculate momentum over specified lookback period"""
        return prices.pct_change(lookback)

    def calculate_volatility(self, prices: pd.Series) -> pd.Series:
        """Calculate realized volatility"""
        returns = prices.pct_change()
        vol = returns.rolling(window=self.config.vol_lookback).std()
        return vol * np.sqrt(252)  # Annualize

    def combine_momentum_signals(self, momentum_dict: Dict[int, pd.Series]) -> pd.Series:
        """
        Combine multiple momentum signals

        Simple average of signs (positive/negative) across timeframes
        More robust than single timeframe
        """
        # Convert to signs (-1, 0, +1)
        signs = pd.DataFrame({
            period: np.sign(momentum)
            for period, momentum in momentum_dict.items()
        })

        # Average across periods
        combined = signs.mean(axis=1)

        return combined

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate time-series momentum signals

        For each asset:
        - Calculate momentum over multiple periods
        - Combine signals
        - Generate buy/sell signals based on trend direction
        """
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        # Calculate momentum for each lookback period
        for column in prices.columns:
            asset_prices = prices[column]

            # Calculate momentum over all lookback periods
            momentum_dict = {}
            for lookback in self.config.lookback_periods:
                mom = self.calculate_momentum(asset_prices, lookback)
                momentum_dict[lookback] = mom

            # Combine signals
            combined_signal = self.combine_momentum_signals(momentum_dict)

            # Generate buy/sell signals
            last_rebalance = 0
            current_position = 0

            for i in range(max(self.config.lookback_periods), len(prices)):
                # Check if it's time to rebalance
                if i - last_rebalance >= self.config.holding_period:
                    signal_value = combined_signal.iloc[i]

                    if pd.isna(signal_value):
                        continue

                    # Determine new position
                    if signal_value > self.config.signal_threshold:
                        new_position = 1  # Long (uptrend)
                    elif signal_value < -self.config.signal_threshold:
                        new_position = 0  # Flat (downtrend) - or -1 for short
                    else:
                        new_position = 0  # Neutral

                    # Generate signal if position changes
                    if new_position != current_position:
                        if new_position == 1:
                            signals.iloc[i][column] = 1  # Buy
                        elif current_position == 1:
                            signals.iloc[i][column] = -1  # Sell

                        current_position = new_position

                    last_rebalance = i

        return signals

    def get_position_weights(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Convert signals to position weights

        Optionally use volatility scaling for risk-adjusted positions
        """
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        # Calculate volatility for scaling if needed
        volatility = {}
        if self.config.use_volatility_scaling and prices is not None:
            for column in prices.columns:
                volatility[column] = self.calculate_volatility(prices[column])

        # Track current positions
        positions = {col: 0.0 for col in signals.columns}

        for i, (idx, row) in enumerate(signals.iterrows()):
            for symbol in signals.columns:
                signal = row[symbol]

                if signal == 1:  # Buy
                    # Calculate position size
                    if self.config.use_volatility_scaling and symbol in volatility:
                        vol = volatility[symbol].iloc[i]
                        if pd.notna(vol) and vol > 0:
                            # Inverse volatility weighting (target ~15% volatility)
                            target_vol = 0.15
                            scale = target_vol / vol
                            size = self.config.position_size * np.clip(scale, 0.5, 2.0)
                        else:
                            size = self.config.position_size
                    else:
                        size = self.config.position_size

                    positions[symbol] = size

                elif signal == -1:  # Sell
                    positions[symbol] = 0

                weights.iloc[i][symbol] = positions[symbol]

        return weights

    def get_strategy_statistics(
        self,
        prices: pd.DataFrame,
        signals: pd.DataFrame
    ) -> Dict:
        """Calculate per-asset statistics"""
        stats = {}

        for column in prices.columns:
            asset_signals = signals[column]
            trades = (asset_signals != 0).sum()

            # Calculate momentum over different periods
            momentum_values = {}
            for lookback in self.config.lookback_periods:
                mom = self.calculate_momentum(prices[column], lookback)
                momentum_values[f'{lookback}d'] = mom.iloc[-1] if not mom.empty else np.nan

            stats[column] = {
                'total_trades': trades,
                'momentum': momentum_values,
                'current_signal': 'LONG' if asset_signals.iloc[-1] > 0 else 'FLAT'
            }

        return stats


if __name__ == "__main__":
    """Test Time-Series Momentum strategy"""
    import yfinance as yf

    print("=" * 80)
    print("TIME-SERIES MOMENTUM STRATEGY TEST")
    print("=" * 80)
    print()

    # Test with diversified universe
    symbols = [
        'BTC-USD', 'ETH-USD',  # Crypto
        'SPY', 'QQQ',          # Equity
        'GLD',                 # Commodities
        'TLT',                 # Fixed Income
    ]

    print(f"Fetching data for {len(symbols)} assets...")
    data = yf.download(symbols, start='2022-01-01', end='2024-12-31', progress=False)['Adj Close']

    # Initialize strategy with multiple timeframes
    config = TimeSeriesMomentumConfig(
        lookback_periods=[20, 60, 120, 252],  # 1M, 3M, 6M, 12M
        holding_period=5,  # Weekly rebalance
        use_volatility_scaling=True
    )
    strategy = TimeSeriesMomentumStrategy(config)

    # Generate signals
    print("\nGenerating signals...")
    signals = strategy.generate_signals(data)

    # Get position weights
    weights = strategy.get_position_weights(signals, data)

    # Show strategy statistics
    print("\n" + "=" * 80)
    print("MOMENTUM STATISTICS (Current)")
    print("=" * 80)

    stats = strategy.get_strategy_statistics(data, signals)
    for symbol, symbol_stats in stats.items():
        print(f"\n{symbol}")
        print(f"  Status: {symbol_stats['current_signal']}")
        print(f"  Total Trades: {symbol_stats['total_trades']}")
        print(f"  Momentum:")
        for period, value in symbol_stats['momentum'].items():
            if pd.notna(value):
                print(f"    {period}: {value:>8.2%}")

    # Current portfolio
    print("\n" + "=" * 80)
    print("CURRENT PORTFOLIO")
    print("=" * 80)

    current_weights = weights.iloc[-1]
    current_positions = current_weights[current_weights > 0].sort_values(ascending=False)

    if len(current_positions) > 0:
        print("\nPositions:")
        for symbol, weight in current_positions.items():
            print(f"  {symbol:12s}: {weight:>6.2%}")
        print(f"\nTotal Allocation: {current_positions.sum():.2%}")
    else:
        print("\n  No positions (all assets in downtrend)")

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
