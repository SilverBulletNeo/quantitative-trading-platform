"""
Quantitative Trading Platform - Starter Implementation

This file provides a minimal working example of the trading platform.
It demonstrates the core components working together:
- Asset classes
- Portfolio management
- Risk management
- Strategy implementation
- Backtesting

Run this file to see a complete backtest of a momentum strategy.
"""

from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional
from enum import Enum
import numpy as np
import pandas as pd

# For market data (install with: pip install yfinance)
try:
    import yfinance as yf
except ImportError:
    print("Please install yfinance: pip install yfinance")
    exit(1)


# ============================================================================
# ENUMS
# ============================================================================

class AssetClass(Enum):
    """Asset class categories"""
    EQUITY = "equity"
    FIXED_INCOME = "fixed_income"
    COMMODITIES = "commodities"
    FX = "fx"
    CRYPTO = "crypto"  # Added cryptocurrency as alternative asset class


class Region(Enum):
    """Geographic regions"""
    US = "US"
    EUROPE = "Europe"
    ASIA = "Asia"
    LATAM = "LATAM"


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Asset:
    """Represents a tradeable asset"""
    symbol: str
    asset_class: AssetClass
    region: Region
    sector: Optional[str] = None

    def __hash__(self):
        return hash(self.symbol)


@dataclass
class Position:
    """Represents a position in an asset"""
    asset: Asset
    quantity: float
    entry_price: float
    entry_date: date
    current_price: float = 0.0

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def pnl(self) -> float:
        return (self.current_price - self.entry_price) * self.quantity

    @property
    def pnl_pct(self) -> float:
        return (self.current_price / self.entry_price - 1.0) if self.entry_price != 0 else 0.0


@dataclass
class Signal:
    """Trading signal"""
    asset: Asset
    target_weight: float  # Target portfolio weight (-1 to 1, negative = short)
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=datetime.now)


# ============================================================================
# PORTFOLIO MANAGEMENT
# ============================================================================

class Portfolio:
    """Manages portfolio positions and capital"""

    def __init__(self, initial_capital: float):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Position] = {}
        self.equity_curve: List[float] = [initial_capital]
        self.dates: List[date] = []

    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)"""
        return self.cash + sum(pos.market_value for pos in self.positions.values())

    @property
    def positions_value(self) -> float:
        """Total value of all positions"""
        return sum(pos.market_value for pos in self.positions.values())

    @property
    def total_return(self) -> float:
        """Total return since inception"""
        return (self.total_value / self.initial_capital - 1.0)

    def update_prices(self, prices: Dict[str, float], current_date: date):
        """Update current prices for all positions"""
        for symbol, position in self.positions.items():
            if symbol in prices:
                position.current_price = prices[symbol]

        self.equity_curve.append(self.total_value)
        self.dates.append(current_date)

    def get_metrics(self) -> Dict:
        """Calculate portfolio performance metrics"""
        returns = pd.Series(self.equity_curve).pct_change().dropna()

        total_return = self.total_return
        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0
        max_dd = self._calculate_max_drawdown()
        volatility = returns.std() * np.sqrt(252)

        return {
            'Total Return': f"{total_return:.2%}",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_dd:.2%}",
            'Volatility': f"{volatility:.2%}",
            'Final Value': f"${self.total_value:,.0f}",
            'Total Positions': len(self.positions)
        }

    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        equity = pd.Series(self.equity_curve)
        running_max = equity.expanding().max()
        drawdown = (equity - running_max) / running_max
        return drawdown.min()


# ============================================================================
# RISK MANAGEMENT
# ============================================================================

class RiskManager:
    """Manages portfolio risk and enforces limits"""

    def __init__(self, portfolio: Portfolio, config: Dict):
        self.portfolio = portfolio
        self.config = config

    def check_position_limits(self, signal: Signal) -> bool:
        """Check if position passes risk limits"""
        # Check maximum position size
        max_position_pct = self.config.get('max_position_pct', 0.05)
        if abs(signal.target_weight) > max_position_pct:
            return False

        # Check regional limits
        regional_limits = self.config.get('regional_limits', {})
        region_limit = regional_limits.get(signal.asset.region.value, 1.0)
        current_region_weight = self._get_region_weight(signal.asset.region)

        if current_region_weight + abs(signal.target_weight) > region_limit:
            return False

        return True

    def _get_region_weight(self, region: Region) -> float:
        """Calculate current weight in a region"""
        total_value = self.portfolio.total_value
        if total_value == 0:
            return 0.0

        region_value = sum(
            pos.market_value
            for pos in self.portfolio.positions.values()
            if pos.asset.region == region
        )
        return region_value / total_value

    def calculate_position_size(self, signal: Signal, volatility: float) -> float:
        """Calculate position size using volatility targeting"""
        target_vol = self.config.get('target_volatility', 0.10)

        if volatility == 0:
            return 0

        # Volatility-adjusted position size
        vol_adjusted_weight = signal.target_weight * (target_vol / volatility)

        # Apply position limits
        max_position = self.config.get('max_position_pct', 0.05)
        final_weight = np.clip(vol_adjusted_weight, -max_position, max_position)

        return final_weight * self.portfolio.total_value


# ============================================================================
# STRATEGY
# ============================================================================

class MomentumStrategy:
    """Simple momentum strategy across asset classes"""

    def __init__(self, universe: List[Asset], lookback: int = 60):
        self.universe = universe
        self.lookback = lookback

    def generate_signals(self, prices: pd.DataFrame) -> List[Signal]:
        """Generate momentum signals"""
        signals = []

        # Calculate momentum for each asset
        momentum = prices.pct_change(self.lookback)

        for asset in self.universe:
            if asset.symbol not in momentum.columns:
                continue

            mom_value = momentum[asset.symbol].iloc[-1]

            # Skip if momentum is NaN
            if pd.isna(mom_value):
                continue

            # Generate signal based on momentum
            # Positive momentum = long, negative = short (simplified)
            if mom_value > 0:
                target_weight = 0.10  # 10% allocation to each positive momentum asset
                signals.append(Signal(
                    asset=asset,
                    target_weight=target_weight,
                    confidence=abs(mom_value)
                ))

        # Normalize weights to sum to 1.0
        total_weight = sum(abs(s.target_weight) for s in signals)
        if total_weight > 0:
            for signal in signals:
                signal.target_weight = signal.target_weight / total_weight * 0.95  # 95% invested, 5% cash

        return signals


# ============================================================================
# BACKTESTING ENGINE
# ============================================================================

class BacktestEngine:
    """Backtesting engine with realistic transaction costs"""

    def __init__(
        self,
        portfolio: Portfolio,
        strategy: MomentumStrategy,
        risk_manager: RiskManager,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005
    ):
        self.portfolio = portfolio
        self.strategy = strategy
        self.risk_manager = risk_manager
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.trades: List[Dict] = []

    def run(self, prices: pd.DataFrame, rebalance_frequency: int = 20) -> Portfolio:
        """Run backtest on historical data"""
        print(f"Running backtest from {prices.index[0]} to {prices.index[-1]}...")
        print(f"Universe: {len(self.strategy.universe)} assets")
        print(f"Initial Capital: ${self.portfolio.initial_capital:,.0f}")
        print("-" * 60)

        # Iterate through each date
        for i in range(self.strategy.lookback, len(prices)):
            current_date = prices.index[i]
            current_prices = prices.iloc[i].to_dict()

            # Update portfolio prices
            self.portfolio.update_prices(current_prices, current_date.date())

            # Rebalance on schedule
            if i % rebalance_frequency == 0:
                # Generate signals
                historical_data = prices.iloc[:i+1]
                signals = self.strategy.generate_signals(historical_data)

                # Execute rebalance
                self._rebalance(signals, current_prices, current_date.date())

                # Print progress
                if i % 60 == 0:
                    metrics = self.portfolio.get_metrics()
                    print(f"{current_date.date()}: Portfolio Value = ${self.portfolio.total_value:,.0f}, "
                          f"Return = {metrics['Total Return']}")

        print("-" * 60)
        print("Backtest complete!")
        return self.portfolio

    def _rebalance(self, signals: List[Signal], prices: Dict[str, float], current_date: date):
        """Rebalance portfolio based on signals"""
        # Close existing positions not in signals
        signal_symbols = {s.asset.symbol for s in signals}
        positions_to_close = [
            symbol for symbol in self.portfolio.positions.keys()
            if symbol not in signal_symbols
        ]

        for symbol in positions_to_close:
            self._close_position(symbol, prices[symbol], current_date)

        # Open or adjust positions based on signals
        for signal in signals:
            if not self.risk_manager.check_position_limits(signal):
                continue

            # Calculate volatility for position sizing
            volatility = 0.15  # Simplified: assume 15% volatility
            position_value = self.risk_manager.calculate_position_size(signal, volatility)

            if position_value > 100:  # Minimum position size
                self._open_or_adjust_position(signal, position_value, prices[signal.asset.symbol], current_date)

    def _open_or_adjust_position(self, signal: Signal, position_value: float, price: float, current_date: date):
        """Open new position or adjust existing one"""
        symbol = signal.asset.symbol
        quantity = position_value / price

        # Transaction costs
        cost = position_value * (self.commission_pct + self.slippage_pct)
        self.portfolio.cash -= cost

        if symbol in self.portfolio.positions:
            # Adjust existing position
            old_position = self.portfolio.positions[symbol]
            self.portfolio.cash += old_position.market_value

        # Create new position
        self.portfolio.positions[symbol] = Position(
            asset=signal.asset,
            quantity=quantity,
            entry_price=price,
            entry_date=current_date,
            current_price=price
        )

        self.portfolio.cash -= position_value

        # Record trade
        self.trades.append({
            'date': current_date,
            'symbol': symbol,
            'action': 'BUY',
            'quantity': quantity,
            'price': price,
            'value': position_value
        })

    def _close_position(self, symbol: str, price: float, current_date: date):
        """Close existing position"""
        if symbol not in self.portfolio.positions:
            return

        position = self.portfolio.positions[symbol]
        position.current_price = price
        sale_value = position.market_value

        # Transaction costs
        cost = sale_value * (self.commission_pct + self.slippage_pct)
        self.portfolio.cash += sale_value - cost

        # Record trade
        self.trades.append({
            'date': current_date,
            'symbol': symbol,
            'action': 'SELL',
            'quantity': position.quantity,
            'price': price,
            'value': sale_value,
            'pnl': position.pnl
        })

        del self.portfolio.positions[symbol]


# ============================================================================
# DATA FETCHER
# ============================================================================

class MarketDataFetcher:
    """Fetch market data for backtesting"""

    @staticmethod
    def fetch_data(symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance"""
        print(f"Fetching data for {len(symbols)} symbols...")

        data = yf.download(
            symbols,
            start=start_date,
            end=end_date,
            progress=False
        )['Adj Close']

        # Handle single symbol case
        if isinstance(data, pd.Series):
            data = data.to_frame(name=symbols[0])

        # Forward fill missing data
        data = data.fillna(method='ffill')

        print(f"Downloaded {len(data)} days of data")
        return data


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run example backtest"""

    print("=" * 60)
    print("QUANTITATIVE TRADING PLATFORM - STARTER DEMO")
    print("=" * 60)
    print()

    # Define investment universe
    universe = [
        # Equity
        Asset('SPY', AssetClass.EQUITY, Region.US, 'Broad Market'),
        Asset('QQQ', AssetClass.EQUITY, Region.US, 'Technology'),
        Asset('EFA', AssetClass.EQUITY, Region.EUROPE, 'Developed Markets'),
        Asset('EEM', AssetClass.EQUITY, Region.ASIA, 'Emerging Markets'),

        # Fixed Income
        Asset('TLT', AssetClass.FIXED_INCOME, Region.US, 'Long Treasury'),
        Asset('AGG', AssetClass.FIXED_INCOME, Region.US, 'Aggregate Bonds'),

        # Commodities
        Asset('GLD', AssetClass.COMMODITIES, Region.US, 'Gold'),
        Asset('DBC', AssetClass.COMMODITIES, Region.US, 'Broad Commodities'),
    ]

    # Configuration
    config = {
        'initial_capital': 1_000_000,
        'max_position_pct': 0.20,
        'target_volatility': 0.15,
        'regional_limits': {
            'US': 0.70,
            'Europe': 0.20,
            'Asia': 0.10,
        }
    }

    # Fetch market data
    symbols = [asset.symbol for asset in universe]
    prices = MarketDataFetcher.fetch_data(
        symbols,
        start_date='2020-01-01',
        end_date='2024-12-31'
    )

    # Initialize components
    portfolio = Portfolio(config['initial_capital'])
    risk_manager = RiskManager(portfolio, config)
    strategy = MomentumStrategy(universe, lookback=60)

    # Run backtest
    engine = BacktestEngine(
        portfolio=portfolio,
        strategy=strategy,
        risk_manager=risk_manager,
        commission_pct=0.001,
        slippage_pct=0.0005
    )

    results = engine.run(prices, rebalance_frequency=20)

    # Display results
    print()
    print("=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    metrics = results.get_metrics()
    for key, value in metrics.items():
        print(f"{key:20s}: {value}")

    print()
    print(f"Total Trades: {len(engine.trades)}")

    print()
    print("Current Positions:")
    print("-" * 60)
    for symbol, position in results.positions.items():
        print(f"{symbol:10s}: ${position.market_value:>12,.0f} ({position.pnl_pct:>7.2%} P&L)")

    print()
    print("=" * 60)
    print("Demo complete! Check out the code to understand how it works.")
    print("=" * 60)


if __name__ == "__main__":
    main()
