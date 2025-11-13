"""
Cross-Sectional Momentum Strategy

Relative strength strategy - ranks assets by recent performance and
buys winners, sells/avoids losers.

Academic Foundation:
- "Momentum" - Jegadeesh and Titman (1993)
- "Value and Momentum Everywhere" - AQR Capital (2013)
- Works across ALL asset classes

Strategy:
- Rank all assets by past N-month returns
- Long top decile (best performers)
- Short bottom decile (worst performers) or just avoid
- Rebalance monthly

Why It Works:
- Behavioral: Underreaction to news, herding, confirmation bias
- Risk-based: Compensation for crash risk
- Structural: Slow information diffusion

Timeframe: Daily rebalancing, but uses monthly lookback
Best For: Large universe of assets (10+), works on crypto, equity, commodities
"""

import numpy as np
import pandas as pd
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class CrossSectionalMomentumConfig:
    """Configuration for Cross-Sectional Momentum strategy"""
    lookback_period: int = 60  # 3 months for momentum calculation
    holding_period: int = 20  # Rebalance every 20 days
    long_percentile: float = 0.70  # Top 30% (long positions)
    short_percentile: float = 0.30  # Bottom 30% (short if allowed)
    position_size: float = 0.10  # 10% per position
    allow_shorts: bool = False  # Crypto: typically long-only
    skip_recent: int = 5  # Skip last 5 days to avoid reversal


class CrossSectionalMomentumStrategy:
    """
    Cross-Sectional Momentum Strategy

    Ranks assets by relative performance and constructs a long-short
    (or long-only) portfolio based on momentum signals.
    """

    def __init__(self, config: CrossSectionalMomentumConfig = None):
        self.config = config or CrossSectionalMomentumConfig()

    def calculate_momentum(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum for each asset

        Momentum = Return from t-lookback to t-skip_recent
        (Skip most recent period to avoid short-term reversal)
        """
        # Calculate returns
        if self.config.skip_recent > 0:
            # Skip recent days (avoid reversal)
            momentum = (
                prices.shift(self.config.skip_recent) /
                prices.shift(self.config.lookback_period + self.config.skip_recent) - 1
            )
        else:
            momentum = prices.pct_change(self.config.lookback_period)

        return momentum

    def rank_assets(self, momentum: pd.Series) -> pd.Series:
        """
        Rank assets by momentum (0=worst, 1=best)

        Returns percentile rank for each asset
        """
        return momentum.rank(pct=True)

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate cross-sectional momentum signals

        Long: Top performers (above long_percentile)
        Short: Bottom performers (below short_percentile)
        Neutral: Middle performers
        """
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        # Calculate momentum
        momentum = self.calculate_momentum(prices)

        # Generate signals for each rebalance period
        last_rebalance = 0

        for i in range(self.config.lookback_period, len(prices)):
            # Check if it's time to rebalance
            if i - last_rebalance >= self.config.holding_period:
                # Get current momentum values
                current_momentum = momentum.iloc[i]

                # Rank assets
                ranks = self.rank_assets(current_momentum.dropna())

                # Generate signals
                for symbol in prices.columns:
                    if symbol not in ranks.index:
                        continue

                    rank = ranks[symbol]

                    # Close existing positions first
                    if i > 0:
                        signals.iloc[i][symbol] = -1  # Close signal

                    # Open new positions
                    if rank >= self.config.long_percentile:
                        # Top performers - Long
                        signals.iloc[i][symbol] = 1
                    elif self.config.allow_shorts and rank <= self.config.short_percentile:
                        # Bottom performers - Short
                        signals.iloc[i][symbol] = -1

                last_rebalance = i

        return signals

    def get_position_weights(self, signals: pd.DataFrame, momentum: pd.DataFrame = None) -> pd.DataFrame:
        """
        Convert signals to position weights

        Can weight equally or by momentum strength
        """
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        # Track current positions
        positions = {col: 0.0 for col in signals.columns}

        for i, (idx, row) in enumerate(signals.iterrows()):
            # Process signals
            longs = []
            shorts = []

            for symbol in signals.columns:
                signal = row[symbol]

                if signal == 1:
                    longs.append(symbol)
                elif signal == -1:
                    # Check if closing or opening short
                    if positions[symbol] > 0:
                        # Closing long
                        positions[symbol] = 0
                    elif self.config.allow_shorts:
                        # Opening short
                        shorts.append(symbol)
                        positions[symbol] = -self.config.position_size
                    else:
                        positions[symbol] = 0

            # Allocate to long positions
            if longs:
                equal_weight = 1.0 / len(longs)
                for symbol in longs:
                    positions[symbol] = equal_weight * 0.95  # 95% invested, 5% cash

            # Set weights for this period
            for symbol in signals.columns:
                weights.iloc[i][symbol] = positions[symbol]

        return weights

    def get_portfolio_statistics(
        self,
        prices: pd.DataFrame,
        weights: pd.DataFrame,
        initial_capital: float = 10000
    ) -> Dict:
        """Calculate portfolio statistics"""

        # Calculate daily returns
        returns = prices.pct_change()

        # Calculate portfolio returns
        portfolio_returns = (weights.shift(1) * returns).sum(axis=1)

        # Calculate cumulative returns
        cumulative_returns = (1 + portfolio_returns).cumprod()
        final_value = initial_capital * cumulative_returns.iloc[-1]

        # Calculate metrics
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1

        sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() != 0 else 0

        # Max drawdown
        cum_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - cum_max) / cum_max
        max_drawdown = drawdown.min()

        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'volatility': portfolio_returns.std() * np.sqrt(252),
            'final_value': final_value,
            'total_trades': (weights.diff() != 0).sum().sum() / 2
        }


if __name__ == "__main__":
    """Test Cross-Sectional Momentum strategy"""
    import yfinance as yf

    print("=" * 80)
    print("CROSS-SECTIONAL MOMENTUM STRATEGY TEST")
    print("=" * 80)
    print()

    # Test with crypto universe
    symbols = [
        'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'ADA-USD',
        'XRP-USD', 'DOT-USD', 'MATIC-USD', 'AVAX-USD', 'LINK-USD'
    ]

    print(f"Fetching data for {len(symbols)} cryptocurrencies...")
    data = yf.download(symbols, start='2023-01-01', end='2024-12-31', progress=False)['Adj Close']

    # Initialize strategy
    config = CrossSectionalMomentumConfig(
        lookback_period=60,  # 3-month momentum
        holding_period=20,   # Rebalance monthly
        long_percentile=0.70,  # Top 30%
        allow_shorts=False  # Crypto: long-only
    )
    strategy = CrossSectionalMomentumStrategy(config)

    # Calculate momentum
    print("\nCalculating momentum scores...")
    momentum = strategy.calculate_momentum(data)

    # Show current momentum rankings
    print("\n" + "=" * 80)
    print("CURRENT MOMENTUM RANKINGS")
    print("=" * 80)

    current_momentum = momentum.iloc[-1].dropna().sort_values(ascending=False)
    print("\nTop performers (last 3 months):")
    for i, (symbol, mom_value) in enumerate(current_momentum.head(5).items(), 1):
        print(f"  {i}. {symbol:12s}: {mom_value:>8.2%}")

    print("\nBottom performers (last 3 months):")
    for i, (symbol, mom_value) in enumerate(current_momentum.tail(5).items(), 1):
        print(f"  {i}. {symbol:12s}: {mom_value:>8.2%}")

    # Generate signals
    print("\n" + "=" * 80)
    print("GENERATING SIGNALS")
    print("=" * 80)

    signals = strategy.generate_signals(data)
    print(f"\nTotal rebalances: {(signals != 0).any(axis=1).sum()}")

    # Get position weights
    weights = strategy.get_position_weights(signals)

    # Current positions
    print("\nCurrent portfolio positions:")
    current_weights = weights.iloc[-1]
    current_positions = current_weights[current_weights > 0].sort_values(ascending=False)
    if len(current_positions) > 0:
        for symbol, weight in current_positions.items():
            print(f"  {symbol:12s}: {weight:>6.2%}")
    else:
        print("  No positions")

    # Calculate performance
    print("\n" + "=" * 80)
    print("BACKTEST RESULTS")
    print("=" * 80)

    stats = strategy.get_portfolio_statistics(data, weights)
    print(f"\nTotal Return        : {stats['total_return']:>10.2%}")
    print(f"Annual Return       : {stats['annual_return']:>10.2%}")
    print(f"Sharpe Ratio        : {stats['sharpe_ratio']:>10.2f}")
    print(f"Max Drawdown        : {stats['max_drawdown']:>10.2%}")
    print(f"Volatility          : {stats['volatility']:>10.2%}")
    print(f"Final Value         : ${stats['final_value']:>10,.0f}")
    print(f"Total Trades        : {stats['total_trades']:>10.0f}")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)
