"""
Comprehensive Strategy Testing & Comparison System

Tests ALL implemented strategies on multi-asset universe and compares performance.
Run this daily to monitor strategy effectiveness across asset classes.

Strategies tested:
Category 1: Technical Indicators (5 strategies)
1. MACD Oscillator
2. RSI (Simple & Advanced)
3. Bollinger Bands (Simple & W-Pattern)
4. Mean Reversion (Z-Score, BB, Sharpe-adjusted)

Category 2: Momentum & Trend (3 strategies)
5. Cross-Sectional Momentum
6. Time-Series Momentum
7. Original Momentum (from platform_starter)

Category 3: Statistical Arbitrage (2 strategies)
8. Pairs Trading
9. Carry Trade

Total: 12+ strategy variations

Usage:
    python comprehensive_strategy_tester.py --asset-class crypto
    python comprehensive_strategy_tester.py --asset-class equity
    python comprehensive_strategy_tester.py --asset-class multi
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
import argparse

# Import our strategies
from strategies.macd_strategy import MACDStrategy, MACDConfig
from strategies.rsi_strategy import RSIStrategy, RSIConfig
from strategies.bollinger_bands_strategy import BollingerBandsStrategy, BollingerBandsConfig
from strategies.mean_reversion_strategy import MeanReversionStrategy, MeanReversionConfig
from strategies.cross_sectional_momentum import CrossSectionalMomentumStrategy, CrossSectionalMomentumConfig
from strategies.time_series_momentum import TimeSeriesMomentumStrategy, TimeSeriesMomentumConfig
from strategies.pairs_trading_strategy import PairsTradingStrategy, PairsTradingConfig
from strategies.carry_trade_strategy import CarryTradeStrategy, CarryTradeConfig


class ComprehensiveStrategyTester:
    """Test and compare multiple strategies across asset classes"""

    def __init__(self, initial_capital: float = 10000):
        self.initial_capital = initial_capital
        self.results = {}

    def get_universe(self, asset_class: str = 'crypto') -> Dict[str, List[str]]:
        """Get asset universe for testing"""
        universes = {
            'crypto': [
                'BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'ADA-USD',
                'XRP-USD', 'DOT-USD', 'MATIC-USD', 'AVAX-USD', 'LINK-USD'
            ],
            'equity': [
                'SPY', 'QQQ', 'DIA', 'IWM',  # US Indices
                'VGK', 'VWO', 'EFA',  # International
                'XLF', 'XLK', 'XLE',  # Sectors
            ],
            'multi': [
                # Crypto
                'BTC-USD', 'ETH-USD',
                # Equity
                'SPY', 'QQQ',
                # Commodities
                'GLD', 'SLV',
                # Fixed Income
                'TLT', 'HYG',
            ]
        }

        return {asset_class: universes.get(asset_class, universes['multi'])}

    def fetch_data(self, symbols: List[str], start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch market data"""
        print(f"Fetching data for {len(symbols)} assets...")
        data = yf.download(symbols, start=start_date, end=end_date, progress=False)['Adj Close']

        if isinstance(data, pd.Series):
            data = data.to_frame(name=symbols[0])

        data = data.fillna(method='ffill')
        print(f"Downloaded {len(data)} days of data")
        return data

    def backtest_strategy(
        self,
        strategy_name: str,
        signals: pd.DataFrame,
        prices: pd.DataFrame,
        weights: pd.DataFrame = None,
        commission: float = 0.001
    ) -> Dict:
        """
        Backtest a strategy and calculate performance metrics
        """
        print(f"\nBacktesting {strategy_name}...")

        if weights is None:
            # Convert signals to simple equal-weight positions
            weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
            positions = {col: 0.0 for col in prices.columns}

            for i, (idx, row) in enumerate(signals.iterrows()):
                for symbol in signals.columns:
                    signal = row[symbol]
                    if signal == 1:
                        positions[symbol] = 0.10
                    elif signal == -1:
                        positions[symbol] = 0
                    weights.iloc[i][symbol] = positions[symbol]

        # Calculate returns
        returns = prices.pct_change()
        portfolio_returns = (weights.shift(1) * returns).sum(axis=1)

        # Calculate metrics
        cumulative = (1 + portfolio_returns).cumprod()
        total_return = cumulative.iloc[-1] - 1 if len(cumulative) > 0 else 0
        annual_return = (1 + total_return) ** (252 / len(prices)) - 1 if len(prices) > 0 else 0

        sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252) if portfolio_returns.std() != 0 else 0

        # Max drawdown
        cum_max = cumulative.expanding().max()
        drawdown = (cumulative - cum_max) / cum_max
        max_drawdown = drawdown.min() if len(drawdown) > 0 else 0

        # Win rate (based on daily returns)
        winning_days = (portfolio_returns > 0).sum()
        total_days = (portfolio_returns != 0).sum()
        win_rate = winning_days / total_days if total_days > 0 else 0

        # Total trades
        total_trades = (signals != 0).sum().sum()

        metrics = {
            'strategy': strategy_name,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'volatility': portfolio_returns.std() * np.sqrt(252),
            'final_value': self.initial_capital * (1 + total_return),
            'total_trades': total_trades,
            'win_rate': win_rate,
            'calmar_ratio': annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        }

        self.results[strategy_name] = metrics
        return metrics

    def print_results(self):
        """Print formatted results"""
        print("\n" + "=" * 100)
        print("COMPREHENSIVE STRATEGY COMPARISON RESULTS")
        print("=" * 100)

        # Sort by Sharpe ratio
        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]['sharpe_ratio'],
            reverse=True
        )

        for strategy_name, metrics in sorted_results:
            print(f"\n{strategy_name}")
            print("-" * 100)
            print(f"  Total Return     : {metrics['total_return']:>10.2%}  |  Annual Return    : {metrics['annual_return']:>10.2%}")
            print(f"  Sharpe Ratio     : {metrics['sharpe_ratio']:>10.2f}  |  Calmar Ratio     : {metrics['calmar_ratio']:>10.2f}")
            print(f"  Max Drawdown     : {metrics['max_drawdown']:>10.2%}  |  Volatility       : {metrics['volatility']:>10.2%}")
            print(f"  Final Value      : ${metrics['final_value']:>10,.0f}  |  Total Trades     : {metrics['total_trades']:>10.0f}")
            print(f"  Win Rate         : {metrics['win_rate']:>10.2%}")

        print("\n" + "=" * 100)
        print("RANKING BY SHARPE RATIO")
        print("=" * 100)
        for i, (strategy_name, metrics) in enumerate(sorted_results, 1):
            print(f"{i:2d}. {strategy_name:45s} | Sharpe: {metrics['sharpe_ratio']:>6.2f} | Return: {metrics['total_return']:>8.2%} | MaxDD: {metrics['max_drawdown']:>8.2%}")

        # Category analysis
        print("\n" + "=" * 100)
        print("CATEGORY PERFORMANCE")
        print("=" * 100)

        categories = {
            'Technical Indicators': ['MACD', 'RSI', 'Bollinger', 'Mean Reversion'],
            'Momentum': ['Momentum', 'Cross-Sectional', 'Time-Series'],
            'Statistical Arbitrage': ['Pairs Trading', 'Carry']
        }

        for category, keywords in categories.items():
            cat_strategies = [
                (name, metrics) for name, metrics in self.results.items()
                if any(kw in name for kw in keywords)
            ]

            if cat_strategies:
                avg_sharpe = np.mean([m['sharpe_ratio'] for _, m in cat_strategies])
                avg_return = np.mean([m['total_return'] for _, m in cat_strategies])
                print(f"\n{category}:")
                print(f"  Strategies: {len(cat_strategies)}")
                print(f"  Avg Sharpe: {avg_sharpe:>6.2f}")
                print(f"  Avg Return: {avg_return:>8.2%}")

    def save_results(self, filename: str = None):
        """Save results to JSON file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"comprehensive_results_{timestamp}.json"

        filepath = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'backtest', 'results', filename
        )

        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        print(f"\nResults saved to: {filepath}")


def main(asset_class='crypto'):
    """Run comprehensive strategy comparison"""

    print("=" * 100)
    print("COMPREHENSIVE STRATEGY TESTING SYSTEM")
    print("=" * 100)
    print()

    # Configuration
    tester = ComprehensiveStrategyTester(initial_capital=10000)
    universe = tester.get_universe(asset_class)
    symbols = universe[asset_class]

    print(f"Asset Class: {asset_class.upper()}")
    print(f"Universe: {', '.join(symbols)}")
    print()

    # Date range (last 2 years)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=730)

    # Fetch data
    data = tester.fetch_data(
        symbols,
        start_date.strftime('%Y-%m-%d'),
        end_date.strftime('%Y-%m-%d')
    )

    strategies = []

    # ========== CATEGORY 1: TECHNICAL INDICATORS ==========

    print("\n" + "=" * 100)
    print("CATEGORY 1: TECHNICAL INDICATOR STRATEGIES")
    print("=" * 100)

    # 1. MACD
    print("\n1. MACD Oscillator Strategy")
    macd = MACDStrategy()
    macd_signals = macd.generate_signals(data)
    strategies.append(('MACD Oscillator', macd_signals, None))

    # 2. RSI Simple
    print("2. RSI Simple Strategy")
    rsi = RSIStrategy()
    rsi_signals = rsi.generate_signals(data)
    strategies.append(('RSI Simple', rsi_signals, None))

    # 3. RSI Advanced
    print("3. RSI Advanced Strategy")
    rsi_advanced_signals = rsi.generate_advanced_signals(data)
    strategies.append(('RSI Advanced', rsi_advanced_signals, None))

    # 4. Bollinger Bands Simple
    print("4. Bollinger Bands Simple")
    bb = BollingerBandsStrategy()
    bb_signals = bb.generate_signals(data)
    strategies.append(('Bollinger Bands Simple', bb_signals, None))

    # 5. Bollinger Bands W-Pattern
    print("5. Bollinger Bands W-Pattern")
    bb_w_signals = bb.generate_w_pattern_signals(data)
    strategies.append(('Bollinger Bands W-Pattern', bb_w_signals, None))

    # 6-8. Mean Reversion variations
    print("6. Mean Reversion Z-Score")
    mr = MeanReversionStrategy()
    mr_signals = mr.generate_signals(data)
    strategies.append(('Mean Reversion Z-Score', mr_signals, None))

    print("7. Mean Reversion Bollinger Bands")
    mr_bb_signals = mr.generate_bb_signals(data)
    strategies.append(('Mean Reversion BB', mr_bb_signals, None))

    print("8. Mean Reversion Sharpe-Adjusted")
    mr_sharpe_signals = mr.calculate_sharpe_adjusted_signals(data)
    strategies.append(('Mean Reversion Sharpe', mr_sharpe_signals, None))

    # ========== CATEGORY 2: MOMENTUM STRATEGIES ==========

    print("\n" + "=" * 100)
    print("CATEGORY 2: MOMENTUM & TREND-FOLLOWING STRATEGIES")
    print("=" * 100)

    # 9. Cross-Sectional Momentum
    print("\n9. Cross-Sectional Momentum")
    csm = CrossSectionalMomentumStrategy()
    csm_signals = csm.generate_signals(data)
    csm_weights = csm.get_position_weights(csm_signals)
    strategies.append(('Cross-Sectional Momentum', csm_signals, csm_weights))

    # 10. Time-Series Momentum
    print("10. Time-Series Momentum")
    tsm = TimeSeriesMomentumStrategy()
    tsm_signals = tsm.generate_signals(data)
    tsm_weights = tsm.get_position_weights(tsm_signals, data)
    strategies.append(('Time-Series Momentum', tsm_signals, tsm_weights))

    # ========== CATEGORY 3: STATISTICAL ARBITRAGE ==========

    print("\n" + "=" * 100)
    print("CATEGORY 3: STATISTICAL ARBITRAGE STRATEGIES")
    print("=" * 100)

    # 11. Pairs Trading
    print("\n11. Pairs Trading")
    try:
        pairs = PairsTradingStrategy()
        pairs_signals = pairs.generate_signals(data)
        strategies.append(('Pairs Trading', pairs_signals, None))
    except Exception as e:
        print(f"  Warning: Pairs Trading failed - {e}")

    # 12. Carry Trade
    print("12. Carry Trade")
    carry = CarryTradeStrategy()
    carry_signals = carry.generate_signals(data)
    carry_weights = carry.get_position_weights(carry_signals, data)
    strategies.append(('Carry Trade', carry_signals, carry_weights))

    # ========== BACKTEST ALL STRATEGIES ==========

    print("\n" + "=" * 100)
    print("BACKTESTING ALL STRATEGIES")
    print("=" * 100)

    for strategy_name, signals, weights in strategies:
        tester.backtest_strategy(strategy_name, signals, data, weights)

    # ========== PRINT RESULTS ==========

    tester.print_results()
    tester.save_results()

    print("\n" + "=" * 100)
    print("TESTING COMPLETE!")
    print("=" * 100)
    print("\nRun this script daily to monitor strategy performance across market conditions.")
    print("Use different asset classes: --asset-class crypto|equity|multi")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Comprehensive Strategy Tester')
    parser.add_argument('--asset-class', type=str, default='crypto',
                        choices=['crypto', 'equity', 'multi'],
                        help='Asset class to test (crypto, equity, multi)')

    args = parser.parse_args()
    main(args.asset_class)
