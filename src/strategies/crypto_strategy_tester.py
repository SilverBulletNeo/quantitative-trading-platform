"""
Crypto Strategy Testing & Comparison System

Tests all implemented strategies on cryptocurrency data and compares performance.
Run this daily to monitor strategy effectiveness.

Strategies tested:
1. MACD Oscillator
2. RSI (Simple & Advanced)
3. Bollinger Bands (Simple & W-Pattern)
4. Mean Reversion (Z-Score & Bollinger Bands)
5. Momentum (from platform_starter.py)

Usage:
    python crypto_strategy_tester.py --daily
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import json

# Import our strategies
from strategies.macd_strategy import MACDStrategy, MACDConfig
from strategies.rsi_strategy import RSIStrategy, RSIConfig
from strategies.bollinger_bands_strategy import BollingerBandsStrategy, BollingerBandsConfig
from strategies.mean_reversion_strategy import MeanReversionStrategy, MeanReversionConfig


class StrategyBacktester:
    """Backtest and compare multiple strategies"""

    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.results = {}

    def fetch_crypto_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch cryptocurrency data"""
        print(f"Fetching data for {len(symbols)} cryptocurrencies...")
        data = yf.download(symbols, start=start_date, end=end_date, progress=False)['Adj Close']

        if isinstance(data, pd.Series):
            data = data.to_frame(name=symbols[0])

        print(f"Downloaded {len(data)} days of data")
        return data

    def backtest_strategy(
        self,
        strategy_name: str,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        commission: float = 0.001
    ) -> Dict:
        """
        Backtest a strategy and calculate performance metrics

        Args:
            strategy_name: Name of the strategy
            signals: DataFrame with trading signals (1=buy, -1=sell, 0=hold)
            prices: DataFrame with asset prices
            commission: Commission rate (default 0.1%)

        Returns:
            Dictionary with performance metrics
        """
        print(f"\nBacktesting {strategy_name}...")

        # Initialize portfolio
        cash = self.initial_capital
        positions = {}
        portfolio_value = []
        trades = []

        # Track portfolio over time
        for date in prices.index:
            # Update portfolio value
            positions_value = sum(
                positions.get(sym, 0) * prices.loc[date, sym]
                for sym in prices.columns
                if sym in positions
            )
            total_value = cash + positions_value
            portfolio_value.append(total_value)

            # Process signals for this date
            for symbol in signals.columns:
                signal = signals.loc[date, symbol]

                if signal == 1:  # Buy signal
                    # Calculate position size (10% of portfolio per position)
                    position_size = total_value * 0.10
                    price = prices.loc[date, symbol]
                    quantity = position_size / price

                    # Deduct cost + commission
                    cost = position_size * (1 + commission)
                    if cash >= cost:
                        cash -= cost
                        positions[symbol] = positions.get(symbol, 0) + quantity
                        trades.append({
                            'date': date,
                            'symbol': symbol,
                            'action': 'BUY',
                            'price': price,
                            'quantity': quantity,
                            'value': position_size
                        })

                elif signal == -1 and symbol in positions:  # Sell signal
                    quantity = positions[symbol]
                    price = prices.loc[date, symbol]
                    proceeds = quantity * price * (1 - commission)

                    cash += proceeds
                    del positions[symbol]
                    trades.append({
                        'date': date,
                        'symbol': symbol,
                        'action': 'SELL',
                        'price': price,
                        'quantity': quantity,
                        'value': proceeds
                    })

        # Calculate metrics
        portfolio_series = pd.Series(portfolio_value, index=prices.index)
        returns = portfolio_series.pct_change().dropna()

        total_return = (portfolio_value[-1] / self.initial_capital - 1)
        annual_return = (1 + total_return) ** (252 / len(prices)) - 1

        sharpe = returns.mean() / returns.std() * np.sqrt(252) if returns.std() != 0 else 0

        # Calculate max drawdown
        cummax = portfolio_series.expanding().max()
        drawdown = (portfolio_series - cummax) / cummax
        max_drawdown = drawdown.min()

        # Win rate
        winning_trades = sum(1 for t in trades if t['action'] == 'SELL' and t['value'] > t['quantity'] * prices.loc[t['date'], t['symbol']])
        total_trades = len([t for t in trades if t['action'] == 'SELL'])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        metrics = {
            'strategy': strategy_name,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'volatility': returns.std() * np.sqrt(252),
            'final_value': portfolio_value[-1],
            'total_trades': len(trades),
            'win_rate': win_rate,
            'avg_return_per_trade': total_return / len(trades) if trades else 0
        }

        self.results[strategy_name] = metrics
        return metrics

    def print_results(self):
        """Print formatted results"""
        print("\n" + "=" * 80)
        print("STRATEGY COMPARISON RESULTS")
        print("=" * 80)

        # Sort by Sharpe ratio
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['sharpe_ratio'],
            reverse=True
        )

        for strategy_name, metrics in sorted_results:
            print(f"\n{strategy_name}")
            print("-" * 80)
            print(f"  Total Return     : {metrics['total_return']:>10.2%}")
            print(f"  Annual Return    : {metrics['annual_return']:>10.2%}")
            print(f"  Sharpe Ratio     : {metrics['sharpe_ratio']:>10.2f}")
            print(f"  Max Drawdown     : {metrics['max_drawdown']:>10.2%}")
            print(f"  Volatility       : {metrics['volatility']:>10.2%}")
            print(f"  Final Value      : ${metrics['final_value']:>10,.0f}")
            print(f"  Total Trades     : {metrics['total_trades']:>10}")
            print(f"  Win Rate         : {metrics['win_rate']:>10.2%}")

        print("\n" + "=" * 80)
        print("RANKING BY SHARPE RATIO")
        print("=" * 80)
        for i, (strategy_name, metrics) in enumerate(sorted_results, 1):
            print(f"{i}. {strategy_name:30s} - Sharpe: {metrics['sharpe_ratio']:.2f}, "
                  f"Return: {metrics['total_return']:.2%}")

    def save_results(self, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_results_{timestamp}.json"

        filepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'backtest', 'results', filename
        )

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nResults saved to: {filepath}")


def main():
    """Run comprehensive strategy comparison"""

    print("=" * 80)
    print("CRYPTO STRATEGY TESTING SYSTEM")
    print("=" * 80)
    print()

    # Configuration
    CRYPTO_SYMBOLS = [
        'BTC-USD',   # Bitcoin
        'ETH-USD',   # Ethereum
        'BNB-USD',   # Binance Coin
        'SOL-USD',   # Solana
        'ADA-USD',   # Cardano
    ]

    # Date range (last 2 years for comprehensive test)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    # Initialize backtester
    backtester = StrategyBacktester(initial_capital=10000)

    # Fetch data
    data = backtester.fetch_crypto_data(
        CRYPTO_SYMBOLS,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )

    # Test each strategy
    strategies = []

    # 1. MACD Strategy
    print("\n--- MACD Strategy ---")
    macd = MACDStrategy()
    macd_signals = macd.generate_signals(data)
    strategies.append(('MACD Oscillator', macd_signals))

    # 2. RSI Simple
    print("\n--- RSI Simple Strategy ---")
    rsi = RSIStrategy()
    rsi_signals = rsi.generate_signals(data)
    strategies.append(('RSI Simple', rsi_signals))

    # 3. RSI Advanced
    print("\n--- RSI Advanced Strategy ---")
    rsi_advanced_signals = rsi.generate_advanced_signals(data)
    strategies.append(('RSI Advanced', rsi_advanced_signals))

    # 4. Bollinger Bands Simple
    print("\n--- Bollinger Bands Simple ---")
    bb = BollingerBandsStrategy()
    bb_signals = bb.generate_signals(data)
    strategies.append(('Bollinger Bands Simple', bb_signals))

    # 5. Bollinger Bands W-Pattern
    print("\n--- Bollinger Bands W-Pattern ---")
    bb_w_signals = bb.generate_w_pattern_signals(data)
    strategies.append(('Bollinger Bands W-Pattern', bb_w_signals))

    # 6. Mean Reversion Z-Score
    print("\n--- Mean Reversion Z-Score ---")
    mr = MeanReversionStrategy()
    mr_signals = mr.generate_signals(data)
    strategies.append(('Mean Reversion Z-Score', mr_signals))

    # 7. Mean Reversion Bollinger Bands
    print("\n--- Mean Reversion BB ---")
    mr_bb_signals = mr.generate_bb_signals(data)
    strategies.append(('Mean Reversion BB', mr_bb_signals))

    # 8. Mean Reversion Sharpe-Adjusted
    print("\n--- Mean Reversion Sharpe-Adjusted ---")
    mr_sharpe_signals = mr.calculate_sharpe_adjusted_signals(data)
    strategies.append(('Mean Reversion Sharpe', mr_sharpe_signals))

    # Backtest all strategies
    for strategy_name, signals in strategies:
        backtester.backtest_strategy(strategy_name, signals, data)

    # Print and save results
    backtester.print_results()
    backtester.save_results()

    print("\n" + "=" * 80)
    print("Testing complete! Run this script daily to monitor strategy performance.")
    print("=" * 80)


if __name__ == "__main__":
    main()
