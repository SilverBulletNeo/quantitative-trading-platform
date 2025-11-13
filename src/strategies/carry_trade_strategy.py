"""
Carry Trade Strategy

Harvest yield/carry premium across asset classes:
- FX Carry: Borrow low-yield currency, invest in high-yield currency
- Bond Carry: Roll-down return from holding longer-dated bonds
- Crypto Carry: Funding rate arbitrage in perpetual futures

Academic Foundation:
- "Carry" - Koijen, Moskowitz, Pedersen, Vrugt (2018)
- "Value and Momentum Everywhere" - AQR Capital
- Fama-French: Term premium, credit premium

Strategy:
- Rank assets by carry/yield
- Long high-carry assets
- Short low-carry assets (if applicable)
- Hold to collect carry premium

Why It Works:
- Risk premium: Compensation for crash risk, liquidity risk
- Behavioral: Reaching for yield, extrapolation
- Structural: Insurance-like payoffs

Timeframe: Daily (but hold for weeks/months)
Best For: FX pairs, bond futures, crypto perpetuals
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass


@dataclass
class CarryTradeConfig:
    """Configuration for Carry Trade strategy"""
    lookback_period: int = 20  # Period to calculate carry
    holding_period: int = 20  # Hold for ~1 month
    long_percentile: float = 0.70  # Top 30% carry
    short_percentile: float = 0.30  # Bottom 30% carry
    position_size: float = 0.10
    allow_shorts: bool = False  # Crypto: typically long-only
    use_volatility_scaling: bool = True
    vol_target: float = 0.10  # 10% volatility target


class CarryTradeStrategy:
    """
    Carry Trade Strategy

    Exploits yield differentials across assets to generate returns
    from collecting carry premium.

    Note: For crypto, this is simplified. In practice, you'd use
    funding rates from perpetual futures exchanges.
    """

    def __init__(self, config: CarryTradeConfig = None):
        self.config = config or CarryTradeConfig()

    def estimate_carry(self, prices: pd.Series) -> pd.Series:
        """
        Estimate carry (expected return from holding)

        For simplified implementation:
        - Use recent drift (expected return)
        - In practice: use yield curves, funding rates, dividends

        For crypto: Funding rate from perpetuals
        For FX: Interest rate differential
        For bonds: Roll-down return
        For equity: Dividend yield
        """
        # Calculate recent return trend (proxy for carry)
        returns = prices.pct_change()

        # Rolling mean return (annualized)
        carry = returns.rolling(window=self.config.lookback_period).mean() * 252

        return carry

    def calculate_volatility(self, prices: pd.Series) -> pd.Series:
        """Calculate realized volatility"""
        returns = prices.pct_change()
        vol = returns.rolling(window=self.config.lookback_period).std()
        return vol * np.sqrt(252)  # Annualize

    def calculate_sharpe_ratio(self, prices: pd.Series) -> pd.Series:
        """
        Calculate rolling Sharpe ratio (carry / volatility)

        This is the risk-adjusted carry - better metric than raw carry
        """
        carry = self.estimate_carry(prices)
        vol = self.calculate_volatility(prices)

        # Avoid division by zero
        sharpe = carry / vol.replace(0, np.nan)

        return sharpe

    def rank_by_carry(self, carry_dict: Dict[str, float]) -> pd.Series:
        """Rank assets by carry (percentile)"""
        carry_series = pd.Series(carry_dict)
        return carry_series.rank(pct=True)

    def generate_signals(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Generate carry trade signals

        Rank assets by carry/Sharpe ratio and construct portfolio:
        - Long: High carry assets
        - Short: Low carry assets (if allowed)
        - Rebalance periodically
        """
        signals = pd.DataFrame(0, index=prices.index, columns=prices.columns)

        # Calculate carry for each asset
        carry_dict = {}
        for column in prices.columns:
            carry_dict[column] = self.calculate_sharpe_ratio(prices[column])

        # Generate signals at each rebalance period
        last_rebalance = 0

        for i in range(self.config.lookback_period, len(prices)):
            # Check if it's time to rebalance
            if i - last_rebalance >= self.config.holding_period:
                # Get current carry values
                current_carry = {}
                for symbol, carry_series in carry_dict.items():
                    if pd.notna(carry_series.iloc[i]):
                        current_carry[symbol] = carry_series.iloc[i]

                if not current_carry:
                    continue

                # Rank by carry
                ranks = self.rank_by_carry(current_carry)

                # Close existing positions
                for symbol in prices.columns:
                    if i > 0:
                        signals.iloc[i][symbol] = -1  # Close signal

                # Open new positions based on carry rank
                for symbol, rank in ranks.items():
                    if rank >= self.config.long_percentile:
                        # High carry - Long
                        signals.iloc[i][symbol] = 1
                    elif self.config.allow_shorts and rank <= self.config.short_percentile:
                        # Low carry - Short
                        signals.iloc[i][symbol] = -1

                last_rebalance = i

        return signals

    def get_position_weights(
        self,
        signals: pd.DataFrame,
        prices: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Convert signals to position weights

        Use volatility scaling for risk parity
        """
        weights = pd.DataFrame(0.0, index=signals.index, columns=signals.columns)

        # Calculate volatility for each asset if needed
        volatility = {}
        if self.config.use_volatility_scaling and prices is not None:
            for column in prices.columns:
                volatility[column] = self.calculate_volatility(prices[column])

        # Track current positions
        positions = {col: 0.0 for col in signals.columns}

        for i, (idx, row) in enumerate(signals.iterrows()):
            # Count longs and shorts
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
                        shorts.append(symbol)

            # Allocate to positions
            for symbol in longs:
                if self.config.use_volatility_scaling and symbol in volatility:
                    vol = volatility[symbol].iloc[i]
                    if pd.notna(vol) and vol > 0:
                        # Inverse volatility weighting
                        weight = (self.config.vol_target / vol) * (1.0 / len(longs))
                        weight = np.clip(weight, 0.01, 0.25)  # Limit: 1-25%
                        positions[symbol] = weight
                    else:
                        positions[symbol] = 1.0 / len(longs) if longs else 0
                else:
                    positions[symbol] = 1.0 / len(longs) if longs else 0

            for symbol in shorts:
                if self.config.use_volatility_scaling and symbol in volatility:
                    vol = volatility[symbol].iloc[i]
                    if pd.notna(vol) and vol > 0:
                        weight = -(self.config.vol_target / vol) * (1.0 / len(shorts))
                        weight = np.clip(weight, -0.25, -0.01)
                        positions[symbol] = weight
                    else:
                        positions[symbol] = -1.0 / len(shorts) if shorts else 0
                else:
                    positions[symbol] = -1.0 / len(shorts) if shorts else 0

            # Set weights
            for symbol in signals.columns:
                weights.iloc[i][symbol] = positions[symbol]

        return weights

    def get_carry_statistics(self, prices: pd.DataFrame) -> Dict:
        """Calculate current carry statistics for all assets"""
        stats = {}

        for column in prices.columns:
            carry = self.estimate_carry(prices[column])
            vol = self.calculate_volatility(prices[column])
            sharpe = self.calculate_sharpe_ratio(prices[column])

            stats[column] = {
                'carry': carry.iloc[-1] if not carry.empty else np.nan,
                'volatility': vol.iloc[-1] if not vol.empty else np.nan,
                'sharpe': sharpe.iloc[-1] if not sharpe.empty else np.nan
            }

        return stats


if __name__ == "__main__":
    """Test Carry Trade strategy"""
    import yfinance as yf

    print("=" * 80)
    print("CARRY TRADE STRATEGY TEST")
    print("=" * 80)
    print()

    # Test with assets that have different yield characteristics
    symbols = [
        'BTC-USD', 'ETH-USD', 'BNB-USD',  # Crypto (high vol, potential high carry)
        'TLT',  # Long-term Treasury (moderate carry, low vol)
        'HYG',  # High-yield bonds (higher carry, higher vol)
        'GLD',  # Gold (low carry, medium vol)
    ]

    print(f"Fetching data for {len(symbols)} assets...")
    data = yf.download(symbols, start='2023-01-01', end='2024-12-31', progress=False)['Adj Close']

    # Initialize strategy
    config = CarryTradeConfig(
        lookback_period=60,
        holding_period=20,
        long_percentile=0.60,  # Top 40%
        use_volatility_scaling=True,
        allow_shorts=False
    )
    strategy = CarryTradeStrategy(config)

    # Calculate carry statistics
    print("\n" + "=" * 80)
    print("CARRY STATISTICS (Risk-Adjusted)")
    print("=" * 80)

    carry_stats = strategy.get_carry_statistics(data)

    # Sort by Sharpe ratio (risk-adjusted carry)
    sorted_assets = sorted(
        carry_stats.items(),
        key=lambda x: x[1]['sharpe'] if pd.notna(x[1]['sharpe']) else -999,
        reverse=True
    )

    print("\nAssets ranked by risk-adjusted carry (Sharpe):\n")
    for symbol, stats in sorted_assets:
        print(f"{symbol:12s}:")
        print(f"  Carry (annual)      : {stats['carry']:>8.2%}" if pd.notna(stats['carry']) else "  Carry: N/A")
        print(f"  Volatility (annual) : {stats['volatility']:>8.2%}" if pd.notna(stats['volatility']) else "  Vol: N/A")
        print(f"  Sharpe Ratio        : {stats['sharpe']:>8.2f}" if pd.notna(stats['sharpe']) else "  Sharpe: N/A")
        print()

    # Generate signals
    print("=" * 80)
    print("GENERATING SIGNALS")
    print("=" * 80)

    signals = strategy.generate_signals(data)
    print(f"\nTotal rebalances: {(signals != 0).any(axis=1).sum()}")

    # Get position weights
    weights = strategy.get_position_weights(signals, data)

    # Current portfolio
    print("\n" + "=" * 80)
    print("CURRENT PORTFOLIO")
    print("=" * 80)

    current_weights = weights.iloc[-1]
    current_positions = current_weights[current_weights != 0].sort_values(ascending=False)

    if len(current_positions) > 0:
        print("\nPositions:")
        for symbol, weight in current_positions.items():
            print(f"  {symbol:12s}: {weight:>6.2%}")
        print(f"\nGross Exposure: {current_positions.abs().sum():.2%}")
        print(f"Net Exposure: {current_positions.sum():.2%}")
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
    print(f"Total Rebalances    : {(signals != 0).any(axis=1).sum():>10.0f}")

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)
