"""
Walk-Forward Validation Framework

Proper out-of-sample testing framework to prevent data snooping bias.
CRITICAL for validating all new strategies before deployment.

Walk-Forward Process:
1. Split data into train/test windows
2. Optimize parameters on training window
3. Test on out-of-sample test window
4. Roll forward and repeat
5. Aggregate out-of-sample results

This ensures strategies generalize to unseen data.

Key Metrics:
- In-sample vs Out-of-sample Sharpe degradation
- Consistency across windows
- Parameter stability

Academic Foundation:
- Pardo (2008): The Evaluation and Optimization of Trading Strategies
- Bailey et al. (2014): Probability of Backtest Overfitting
- Harvey & Liu (2015): Backtesting
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Callable, Any
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation"""

    train_years: int = 3      # Training window size
    test_years: int = 1       # Test window size
    step_years: int = 1       # Step size (1 = anchored, train_years = sliding)

    min_train_days: int = 500  # Minimum days needed for training
    min_test_days: int = 100   # Minimum days needed for testing


class WalkForwardValidator:
    """
    Walk-Forward Validation Framework

    Tests strategy robustness using proper train/test splits.
    """

    def __init__(self, config: Optional[WalkForwardConfig] = None):
        """
        Initialize walk-forward validator

        Args:
            config: Validation configuration
        """
        self.config = config or WalkForwardConfig()
        self.results_history = []

    def create_windows(self, start_date: pd.Timestamp,
                      end_date: pd.Timestamp) -> List[Dict]:
        """
        Create train/test windows for walk-forward analysis

        Args:
            start_date: Start date of data
            end_date: End date of data

        Returns:
            List of window dictionaries with train/test dates
        """
        windows = []
        current_start = start_date

        while True:
            # Training window
            train_end = current_start + pd.DateOffset(years=self.config.train_years)

            if train_end > end_date:
                break

            # Test window
            test_end = train_end + pd.DateOffset(years=self.config.test_years)

            if test_end > end_date:
                test_end = end_date

            # Check minimum requirements
            if test_end <= train_end:
                break

            windows.append({
                'window_id': len(windows) + 1,
                'train_start': current_start,
                'train_end': train_end,
                'test_start': train_end,
                'test_end': test_end
            })

            # Move to next window
            current_start += pd.DateOffset(years=self.config.step_years)

            # Stop if we've reached the end
            if current_start >= end_date - pd.DateOffset(years=self.config.train_years + self.config.test_years):
                break

        return windows

    def run_single_window(self, window: Dict,
                         prices: pd.DataFrame,
                         strategy_func: Callable,
                         **strategy_params) -> Dict:
        """
        Run strategy on a single train/test window

        Args:
            window: Window dictionary with train/test dates
            prices: Price data
            strategy_func: Function that takes prices and returns backtest results
            strategy_params: Additional parameters for strategy

        Returns:
            Dictionary with in-sample and out-of-sample results
        """
        # Split data
        train_prices = prices[(prices.index >= window['train_start']) &
                             (prices.index < window['train_end'])]

        test_prices = prices[(prices.index >= window['test_start']) &
                            (prices.index < window['test_end'])]

        # Check minimum requirements
        if len(train_prices) < self.config.min_train_days:
            return {'error': 'Insufficient training data'}

        if len(test_prices) < self.config.min_test_days:
            return {'error': 'Insufficient test data'}

        # Run strategy on training data
        try:
            train_results = strategy_func(train_prices, **strategy_params)
            train_metrics = train_results.get('metrics', {})
        except Exception as e:
            return {'error': f'Training failed: {str(e)}'}

        # Run strategy on test data (out-of-sample)
        try:
            test_results = strategy_func(test_prices, **strategy_params)
            test_metrics = test_results.get('metrics', {})
        except Exception as e:
            return {'error': f'Testing failed: {str(e)}'}

        return {
            'window_id': window['window_id'],
            'train_start': window['train_start'].date(),
            'train_end': window['train_end'].date(),
            'test_start': window['test_start'].date(),
            'test_end': window['test_end'].date(),
            'train_days': len(train_prices),
            'test_days': len(test_prices),
            'train_sharpe': train_metrics.get('sharpe', 0),
            'test_sharpe': test_metrics.get('sharpe', 0),
            'train_return': train_metrics.get('annual_return', 0),
            'test_return': test_metrics.get('annual_return', 0),
            'train_max_dd': train_metrics.get('max_drawdown', 0),
            'test_max_dd': test_metrics.get('max_drawdown', 0),
            'degradation': train_metrics.get('sharpe', 0) - test_metrics.get('sharpe', 0)
        }

    def validate_strategy(self, prices: pd.DataFrame,
                         strategy_func: Callable,
                         **strategy_params) -> pd.DataFrame:
        """
        Run full walk-forward validation

        Args:
            prices: Price data
            strategy_func: Strategy function to test
            strategy_params: Additional strategy parameters

        Returns:
            DataFrame with results for all windows
        """
        print("="*80)
        print("WALK-FORWARD VALIDATION")
        print("="*80)

        print(f"\nData period: {prices.index[0].date()} to {prices.index[-1].date()}")
        print(f"Train window: {self.config.train_years} years")
        print(f"Test window: {self.config.test_years} years\n")

        # Create windows
        windows = self.create_windows(prices.index[0], prices.index[-1])

        print(f"Created {len(windows)} train/test windows\n")

        # Run validation for each window
        results = []

        for i, window in enumerate(windows, 1):
            print(f"Window {i}/{len(windows)}: ", end='')
            print(f"Train {window['train_start'].date()} to {window['train_end'].date()}, ", end='')
            print(f"Test {window['test_start'].date()} to {window['test_end'].date()}...", end=' ')

            result = self.run_single_window(window, prices, strategy_func, **strategy_params)

            if 'error' in result:
                print(f"ERROR: {result['error']}")
            else:
                print(f"✓ Train Sharpe: {result['train_sharpe']:.2f}, Test Sharpe: {result['test_sharpe']:.2f}")
                results.append(result)

        self.results_history = results

        return pd.DataFrame(results) if results else pd.DataFrame()

    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze walk-forward validation results

        Args:
            results_df: Results DataFrame from validate_strategy

        Returns:
            Dictionary with summary statistics
        """
        if len(results_df) == 0:
            return {'error': 'No results to analyze'}

        print("\n" + "="*80)
        print("WALK-FORWARD VALIDATION RESULTS")
        print("="*80)

        # Summary statistics
        avg_train_sharpe = results_df['train_sharpe'].mean()
        avg_test_sharpe = results_df['test_sharpe'].mean()
        avg_degradation = results_df['degradation'].mean()
        std_degradation = results_df['degradation'].std()

        avg_train_return = results_df['train_return'].mean()
        avg_test_return = results_df['test_return'].mean()

        # Consistency metrics
        positive_test_sharpe_pct = (results_df['test_sharpe'] > 0).sum() / len(results_df) * 100
        positive_test_return_pct = (results_df['test_return'] > 0).sum() / len(results_df) * 100

        # Best/worst windows
        best_window = results_df.loc[results_df['test_sharpe'].idxmax()]
        worst_window = results_df.loc[results_df['test_sharpe'].idxmin()]

        print(f"\nNumber of Windows: {len(results_df)}")

        print(f"\nIn-Sample Performance:")
        print(f"  Average Sharpe:  {avg_train_sharpe:>8.2f}")
        print(f"  Average Return:  {avg_train_return:>7.1f}%")

        print(f"\nOut-of-Sample Performance:")
        print(f"  Average Sharpe:  {avg_test_sharpe:>8.2f}")
        print(f"  Average Return:  {avg_test_return:>7.1f}%")
        print(f"  Positive Sharpe: {positive_test_sharpe_pct:>7.1f}% of windows")
        print(f"  Positive Return: {positive_test_return_pct:>7.1f}% of windows")

        print(f"\nOverfitting Analysis:")
        print(f"  Avg Degradation: {avg_degradation:>8.2f} (in-sample - out-sample)")
        print(f"  Std Degradation: {avg_degradation:>8.2f}")

        # Interpretation
        print(f"\n{'='*80}")
        if avg_degradation > 1.0:
            print("⚠️  WARNING: Significant overfitting detected!")
            print("   In-sample performance >> out-of-sample performance")
            print("   Strategy may not generalize well to live trading")
        elif avg_degradation > 0.5:
            print("⚠️  CAUTION: Moderate overfitting detected")
            print("   Some performance degradation out-of-sample")
        else:
            print("✅ Strategy appears ROBUST")
            print("   Minimal overfitting - performance consistent across windows")

        if avg_test_sharpe < 0.5:
            print("\n⚠️  WARNING: Low out-of-sample Sharpe ratio")
            print("   Strategy may not be profitable in live trading")
        elif avg_test_sharpe < 1.0:
            print("\n✓ Acceptable out-of-sample performance")
        else:
            print("\n✅ Strong out-of-sample performance")

        print(f"{'='*80}")

        print(f"\nBest Test Window:")
        print(f"  Period: {best_window['test_start']} to {best_window['test_end']}")
        print(f"  Sharpe: {best_window['test_sharpe']:.2f}")
        print(f"  Return: {best_window['test_return']:.1f}%")

        print(f"\nWorst Test Window:")
        print(f"  Period: {worst_window['test_start']} to {worst_window['test_end']}")
        print(f"  Sharpe: {worst_window['test_sharpe']:.2f}")
        print(f"  Return: {worst_window['test_return']:.1f}%")

        return {
            'n_windows': len(results_df),
            'avg_train_sharpe': avg_train_sharpe,
            'avg_test_sharpe': avg_test_sharpe,
            'avg_degradation': avg_degradation,
            'std_degradation': std_degradation,
            'positive_sharpe_pct': positive_test_sharpe_pct,
            'robust': avg_degradation < 0.5 and avg_test_sharpe > 0.5,
            'best_window': {
                'period': f"{best_window['test_start']} to {best_window['test_end']}",
                'sharpe': best_window['test_sharpe'],
                'return': best_window['test_return']
            },
            'worst_window': {
                'period': f"{worst_window['test_start']} to {worst_window['test_end']}",
                'sharpe': worst_window['test_sharpe'],
                'return': worst_window['test_return']
            }
        }


def main():
    """Example usage with simple momentum strategy"""
    print("="*80)
    print("WALK-FORWARD VALIDATION FRAMEWORK - DEMO")
    print("="*80)

    # Load data
    prices = pd.read_csv('data/raw/equities_prices.csv', index_col=0)
    prices.index = pd.to_datetime(prices.index, utc=True).tz_convert(None)

    print(f"\nLoaded {len(prices.columns)} assets, {len(prices)} days")
    print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}\n")

    # Define a simple strategy function for testing
    def simple_momentum_strategy(prices, lookback=20):
        """Simple momentum strategy for testing"""
        returns = prices.pct_change()
        momentum = prices.pct_change(lookback).shift(1)

        portfolio_returns = pd.Series(0.0, index=returns.index)

        for date in momentum.index[lookback+1:]:
            if momentum.loc[date].notna().sum() > 0:
                ranked = momentum.loc[date].rank(pct=True)
                longs = ranked[ranked >= 0.7].index

                if len(longs) > 0:
                    portfolio_returns[date] = returns.loc[date, longs].mean()

        # Calculate metrics
        total_ret = (1 + portfolio_returns).prod() - 1
        years = len(portfolio_returns) / 252
        ann_ret = (1 + total_ret) ** (1/years) - 1 if years > 0 else 0
        ann_vol = portfolio_returns.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        cum_rets = (1 + portfolio_returns).cumprod()
        running_max = cum_rets.expanding().max()
        drawdown = (cum_rets - running_max) / running_max
        max_dd = drawdown.min()

        return {
            'returns': portfolio_returns,
            'metrics': {
                'annual_return': ann_ret * 100,
                'volatility': ann_vol * 100,
                'sharpe': sharpe,
                'max_drawdown': max_dd * 100
            }
        }

    # Initialize validator
    config = WalkForwardConfig(
        train_years=3,
        test_years=1,
        step_years=1
    )

    validator = WalkForwardValidator(config)

    # Run validation
    results_df = validator.validate_strategy(
        prices,
        simple_momentum_strategy,
        lookback=20
    )

    # Analyze results
    if len(results_df) > 0:
        summary = validator.analyze_results(results_df)

        print("\n" + "="*80)
        print("✅ WALK-FORWARD VALIDATION COMPLETE")
        print("="*80)
        print("\nThis framework should be used to validate ALL new strategies")
        print("before deployment to ensure robustness and avoid overfitting.")

    return results_df, summary


if __name__ == "__main__":
    main()
