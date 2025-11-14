"""
Critical Backtest Analysis Framework

Analyzes strategy robustness through:
1. Rolling performance analysis (regime detection)
2. Walk-forward validation (out-of-sample testing)
3. Parameter sensitivity (data snooping bias)
4. Transaction cost impact
5. Conditional Parameter Optimization (CPO)
6. Stress testing across regimes
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Market Regime Definitions
REGIMES = {
    'QE3_BULL': ('2012-05-18', '2014-12-31'),      # QE3, steady growth
    'CHINA_CORRECTION': ('2015-01-01', '2016-06-30'),  # China slowdown, oil crash
    'TAX_RALLY': ('2017-01-01', '2018-02-28'),     # Trump tax cuts
    'CORRECTION_2018': ('2018-03-01', '2018-12-31'), # Tech selloff, -20%
    'RECOVERY_2019': ('2019-01-01', '2019-12-31'), # Fed pivot
    'COVID_CRASH': ('2020-01-01', '2020-03-31'),   # COVID panic
    'COVID_RECOVERY': ('2020-04-01', '2021-12-31'), # QE infinity
    'RATE_HIKE_BEAR': ('2022-01-01', '2022-12-31'), # Fed aggression
    'AI_BOOM': ('2023-01-01', '2025-11-12'),       # AI rally, soft landing
}

class BacktestRobustnessAnalyzer:
    """Comprehensive backtest robustness analysis"""

    def __init__(self, prices_file='data/raw/equities_prices.csv'):
        """Load price data"""
        self.prices = pd.read_csv(prices_file, index_col=0)
        # Convert index to DatetimeIndex and remove timezone
        self.prices.index = pd.to_datetime(self.prices.index, utc=True).tz_convert(None)
        self.returns = self.prices.pct_change()
        print(f"Loaded {len(self.prices.columns)} assets, {len(self.prices)} days")
        print(f"Date range: {self.prices.index[0].date()} to {self.prices.index[-1].date()}")

    def calculate_momentum_signals(self, lookback=20, skip=1):
        """Calculate momentum signals with configurable lookback"""
        momentum = self.prices.pct_change(lookback).shift(skip)
        signals = pd.DataFrame(0, index=self.prices.index, columns=self.prices.columns)

        for date in momentum.index:
            if pd.notna(momentum.loc[date]).sum() > 0:
                ranked = momentum.loc[date].rank(pct=True)
                signals.loc[date, ranked >= 0.7] = 1
                signals.loc[date, ranked <= 0.3] = -1

        return signals

    def backtest_strategy(self, signals, transaction_cost=0.0):
        """
        Backtest strategy with transaction costs

        Args:
            signals: Trading signals (-1, 0, 1)
            transaction_cost: Cost per trade (e.g., 0.001 = 10 bps)
        """
        portfolio_returns = pd.Series(0.0, index=self.returns.index)

        for i, date in enumerate(signals.index[1:], 1):
            prev_date = signals.index[i-1]

            # Long positions
            longs = signals.loc[date][signals.loc[date] == 1].index
            if len(longs) > 0:
                daily_ret = self.returns.loc[date, longs].mean()

                # Calculate turnover (positions that changed)
                prev_longs = signals.loc[prev_date][signals.loc[prev_date] == 1].index
                turnover = len(set(longs).symmetric_difference(set(prev_longs))) / max(len(longs), 1)

                # Apply transaction costs
                daily_ret -= turnover * transaction_cost
                portfolio_returns[date] = daily_ret

        return portfolio_returns

    def calculate_metrics(self, returns):
        """Calculate performance metrics"""
        if len(returns) == 0 or returns.std() == 0:
            return {
                'Total Return': 0,
                'Annual Return': 0,
                'Volatility': 0,
                'Sharpe': 0,
                'Max Drawdown': 0,
                'Calmar': 0
            }

        total_return = (1 + returns).prod() - 1
        years = len(returns) / 252
        ann_return = (1 + total_return) ** (1/years) - 1
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0

        cum_returns = (1 + returns).cumprod()
        running_max = cum_returns.expanding().max()
        drawdown = (cum_returns - running_max) / running_max
        max_dd = drawdown.min()

        calmar = ann_return / abs(max_dd) if max_dd != 0 else 0

        return {
            'Total Return': total_return * 100,
            'Annual Return': ann_return * 100,
            'Volatility': ann_vol * 100,
            'Sharpe': sharpe,
            'Max Drawdown': max_dd * 100,
            'Calmar': calmar
        }

    def rolling_sharpe_analysis(self, lookback=20, window_days=252):
        """
        Calculate rolling Sharpe ratio to detect regime changes
        """
        print("\n" + "="*80)
        print("ROLLING SHARPE RATIO ANALYSIS")
        print("="*80)

        signals = self.calculate_momentum_signals(lookback=lookback)
        returns = self.backtest_strategy(signals, transaction_cost=0.0010)

        # Calculate rolling Sharpe
        rolling_sharpe = returns.rolling(window_days).apply(
            lambda x: (x.mean() * 252) / (x.std() * np.sqrt(252)) if x.std() > 0 else 0
        )

        print(f"\nRolling Sharpe Statistics (lookback={lookback}, window={window_days} days):")
        print(f"  Mean:   {rolling_sharpe.mean():.2f}")
        print(f"  Median: {rolling_sharpe.median():.2f}")
        print(f"  Std:    {rolling_sharpe.std():.2f}")
        print(f"  Min:    {rolling_sharpe.min():.2f}")
        print(f"  Max:    {rolling_sharpe.max():.2f}")

        # Find worst periods
        worst_periods = rolling_sharpe.nsmallest(5)
        print(f"\n5 Worst Rolling Sharpe Periods:")
        for date, sharpe in worst_periods.items():
            print(f"  {date.date()}: {sharpe:.2f}")

        return rolling_sharpe, returns

    def regime_analysis(self, lookback=20):
        """
        Analyze strategy performance across different market regimes
        """
        print("\n" + "="*80)
        print("REGIME-BASED PERFORMANCE ANALYSIS")
        print("="*80)

        signals = self.calculate_momentum_signals(lookback=lookback)
        returns = self.backtest_strategy(signals, transaction_cost=0.0010)

        regime_performance = []

        for regime_name, (start, end) in REGIMES.items():
            start_date = pd.to_datetime(start)
            end_date = pd.to_datetime(end)

            # Filter returns for this regime
            regime_returns = returns[(returns.index >= start_date) & (returns.index <= end_date)]

            if len(regime_returns) > 20:  # Minimum sample size
                metrics = self.calculate_metrics(regime_returns)
                regime_performance.append({
                    'Regime': regime_name,
                    'Start': start,
                    'End': end,
                    'Days': len(regime_returns),
                    'Annual Return': f"{metrics['Annual Return']:.1f}%",
                    'Sharpe': f"{metrics['Sharpe']:.2f}",
                    'Max DD': f"{metrics['Max Drawdown']:.1f}%"
                })

        df = pd.DataFrame(regime_performance)
        print("\n" + df.to_string(index=False))

        return df

    def walk_forward_analysis(self, lookback=20, train_years=3, test_years=1):
        """
        Walk-forward analysis: train on N years, test on M years

        This detects overfitting - if in-sample performance >> out-of-sample,
        we have data snooping bias.
        """
        print("\n" + "="*80)
        print("WALK-FORWARD ANALYSIS (Out-of-Sample Testing)")
        print("="*80)
        print(f"Train: {train_years} years, Test: {test_years} years\n")

        results = []

        start_date = self.prices.index[0]
        end_date = self.prices.index[-1]

        current_date = start_date
        window = 0

        while current_date + pd.DateOffset(years=train_years+test_years) <= end_date:
            train_end = current_date + pd.DateOffset(years=train_years)
            test_end = train_end + pd.DateOffset(years=test_years)

            # In-sample (training)
            train_prices = self.prices[(self.prices.index >= current_date) & (self.prices.index < train_end)]
            train_returns = train_prices.pct_change()
            train_signals = pd.DataFrame(0, index=train_prices.index, columns=train_prices.columns)

            momentum_train = train_prices.pct_change(lookback).shift(1)
            for date in momentum_train.index:
                if pd.notna(momentum_train.loc[date]).sum() > 0:
                    ranked = momentum_train.loc[date].rank(pct=True)
                    train_signals.loc[date, ranked >= 0.7] = 1

            train_port_ret = pd.Series(0.0, index=train_returns.index)
            for date in train_signals.index[1:]:
                longs = train_signals.loc[date][train_signals.loc[date] == 1].index
                if len(longs) > 0:
                    train_port_ret[date] = train_returns.loc[date, longs].mean()

            train_metrics = self.calculate_metrics(train_port_ret)

            # Out-of-sample (testing)
            test_prices = self.prices[(self.prices.index >= train_end) & (self.prices.index < test_end)]
            test_returns = test_prices.pct_change()
            test_signals = pd.DataFrame(0, index=test_prices.index, columns=test_prices.columns)

            momentum_test = test_prices.pct_change(lookback).shift(1)
            for date in momentum_test.index:
                if pd.notna(momentum_test.loc[date]).sum() > 0:
                    ranked = momentum_test.loc[date].rank(pct=True)
                    test_signals.loc[date, ranked >= 0.7] = 1

            test_port_ret = pd.Series(0.0, index=test_returns.index)
            for date in test_signals.index[1:]:
                longs = test_signals.loc[date][test_signals.loc[date] == 1].index
                if len(longs) > 0:
                    test_port_ret[date] = test_returns.loc[date, longs].mean()

            test_metrics = self.calculate_metrics(test_port_ret)

            results.append({
                'Window': window + 1,
                'Train Period': f"{current_date.date()} to {train_end.date()}",
                'Test Period': f"{train_end.date()} to {test_end.date()}",
                'In-Sample Sharpe': f"{train_metrics['Sharpe']:.2f}",
                'Out-Sample Sharpe': f"{test_metrics['Sharpe']:.2f}",
                'Degradation': f"{train_metrics['Sharpe'] - test_metrics['Sharpe']:.2f}"
            })

            current_date = train_end
            window += 1

        df = pd.DataFrame(results)
        print(df.to_string(index=False))

        # Calculate average degradation
        degradations = [float(r['Degradation']) for r in results]
        avg_degradation = np.mean(degradations)

        print(f"\n{'='*80}")
        print(f"Average Sharpe Degradation (In-Sample - Out-Sample): {avg_degradation:.2f}")

        if avg_degradation > 1.0:
            print("⚠️  WARNING: Significant overfitting detected!")
            print("   In-sample performance is much better than out-of-sample.")
            print("   Strategy may not generalize to future data.")
        elif avg_degradation > 0.5:
            print("⚠️  CAUTION: Moderate overfitting detected.")
        else:
            print("✅ Strategy appears robust (minimal overfitting)")

        return df

    def parameter_sensitivity(self, lookbacks=[5, 10, 15, 20, 30, 40, 60, 90, 120]):
        """
        Test sensitivity to momentum lookback parameter

        If performance is highly dependent on specific parameter (e.g., only 20-day works),
        this indicates data snooping bias.
        """
        print("\n" + "="*80)
        print("PARAMETER SENSITIVITY ANALYSIS")
        print("="*80)
        print("Testing momentum across different lookback periods...\n")

        results = []

        for lookback in lookbacks:
            signals = self.calculate_momentum_signals(lookback=lookback)
            returns = self.backtest_strategy(signals, transaction_cost=0.0010)
            metrics = self.calculate_metrics(returns)

            results.append({
                'Lookback': f"{lookback}-day",
                'Annual Return': f"{metrics['Annual Return']:.1f}%",
                'Sharpe': f"{metrics['Sharpe']:.2f}",
                'Max DD': f"{metrics['Max Drawdown']:.1f}%",
                'Calmar': f"{metrics['Calmar']:.2f}"
            })

        df = pd.DataFrame(results)
        print(df.to_string(index=False))

        # Check if one parameter dominates
        sharpes = [float(r['Sharpe']) for r in results]
        max_sharpe = max(sharpes)
        avg_sharpe = np.mean(sharpes)
        sharpe_std = np.std(sharpes)

        print(f"\n{'='*80}")
        print(f"Sharpe Statistics Across Parameters:")
        print(f"  Max:  {max_sharpe:.2f}")
        print(f"  Mean: {avg_sharpe:.2f}")
        print(f"  Std:  {sharpe_std:.2f}")

        if max_sharpe > avg_sharpe + 2*sharpe_std:
            print("\n⚠️  WARNING: Performance highly sensitive to specific parameter!")
            print("   This suggests data snooping bias or overfitting.")
            print("   Strategy may not be robust.")
        else:
            print("\n✅ Strategy performance is stable across parameters")

        return df

    def transaction_cost_sensitivity(self, lookback=20, costs=[0, 5, 10, 20, 30, 50]):
        """
        Test impact of transaction costs (in basis points)

        High-frequency strategies can be killed by realistic transaction costs.
        """
        print("\n" + "="*80)
        print("TRANSACTION COST SENSITIVITY")
        print("="*80)
        print("Testing impact of transaction costs on strategy...\n")

        signals = self.calculate_momentum_signals(lookback=lookback)
        results = []

        for cost_bps in costs:
            cost = cost_bps / 10000  # Convert bps to decimal
            returns = self.backtest_strategy(signals, transaction_cost=cost)
            metrics = self.calculate_metrics(returns)

            results.append({
                'Cost (bps)': cost_bps,
                'Annual Return': f"{metrics['Annual Return']:.1f}%",
                'Sharpe': f"{metrics['Sharpe']:.2f}",
                'Max DD': f"{metrics['Max Drawdown']:.1f}%"
            })

        df = pd.DataFrame(results)
        print(df.to_string(index=False))

        # Calculate turnover
        turnover_per_day = 0
        for i in range(1, len(signals)):
            prev_longs = set(signals.iloc[i-1][signals.iloc[i-1] == 1].index)
            curr_longs = set(signals.iloc[i][signals.iloc[i] == 1].index)
            if len(curr_longs) > 0:
                turnover_per_day += len(prev_longs.symmetric_difference(curr_longs)) / len(curr_longs)

        avg_turnover = turnover_per_day / len(signals)

        print(f"\n{'='*80}")
        print(f"Average Daily Turnover: {avg_turnover*100:.1f}%")

        sharpe_0 = float(results[0]['Sharpe'])
        sharpe_20 = float(results[3]['Sharpe']) if len(results) > 3 else 0

        if sharpe_0 - sharpe_20 > 2.0:
            print("\n⚠️  WARNING: Strategy highly sensitive to transaction costs!")
            print("   Real-world performance may be much lower.")
        else:
            print("\n✅ Strategy appears robust to realistic transaction costs")

        return df

    def conditional_parameter_optimization(self, volatility_lookback=60):
        """
        Conditional Parameter Optimization (CPO)

        Adjust momentum lookback based on market volatility:
        - High vol (trending) → Shorter lookback (faster reaction)
        - Low vol (choppy) → Longer lookback (avoid whipsaws)
        """
        print("\n" + "="*80)
        print("CONDITIONAL PARAMETER OPTIMIZATION (CPO)")
        print("="*80)
        print("Adapting momentum lookback to market volatility...\n")

        # Calculate rolling volatility
        rolling_vol = self.returns.mean(axis=1).rolling(volatility_lookback).std() * np.sqrt(252)

        # Adaptive signals
        adaptive_signals = pd.DataFrame(0, index=self.prices.index, columns=self.prices.columns)

        for date in self.prices.index[volatility_lookback:]:
            current_vol = rolling_vol.loc[date]

            # High vol → use 10-day momentum (fast)
            # Medium vol → use 20-day momentum (normal)
            # Low vol → use 60-day momentum (slow)
            if current_vol > rolling_vol.quantile(0.67):
                lookback = 10  # High vol regime
            elif current_vol > rolling_vol.quantile(0.33):
                lookback = 20  # Medium vol regime
            else:
                lookback = 60  # Low vol regime

            # Calculate momentum for this lookback
            if date >= self.prices.index[lookback + 1]:
                momentum = self.prices.loc[date] / self.prices.shift(lookback).loc[date] - 1
                ranked = momentum.rank(pct=True)
                adaptive_signals.loc[date, ranked >= 0.7] = 1
                adaptive_signals.loc[date, ranked <= 0.3] = -1

        # Backtest adaptive strategy
        adaptive_returns = self.backtest_strategy(adaptive_signals, transaction_cost=0.0010)
        adaptive_metrics = self.calculate_metrics(adaptive_returns)

        # Compare to fixed 20-day
        fixed_signals = self.calculate_momentum_signals(lookback=20)
        fixed_returns = self.backtest_strategy(fixed_signals, transaction_cost=0.0010)
        fixed_metrics = self.calculate_metrics(fixed_returns)

        results = pd.DataFrame([
            {
                'Strategy': 'Fixed 20-day',
                'Annual Return': f"{fixed_metrics['Annual Return']:.1f}%",
                'Sharpe': f"{fixed_metrics['Sharpe']:.2f}",
                'Max DD': f"{fixed_metrics['Max Drawdown']:.1f}%",
                'Calmar': f"{fixed_metrics['Calmar']:.2f}"
            },
            {
                'Strategy': 'CPO (Adaptive)',
                'Annual Return': f"{adaptive_metrics['Annual Return']:.1f}%",
                'Sharpe': f"{adaptive_metrics['Sharpe']:.2f}",
                'Max DD': f"{adaptive_metrics['Max Drawdown']:.1f}%",
                'Calmar': f"{adaptive_metrics['Calmar']:.2f}"
            }
        ])

        print(results.to_string(index=False))

        improvement = adaptive_metrics['Sharpe'] - fixed_metrics['Sharpe']

        print(f"\n{'='*80}")
        print(f"Sharpe Improvement: {improvement:+.2f}")

        if improvement > 0.3:
            print("✅ CPO significantly improves performance!")
            print("   Adaptive parameters provide better risk-adjusted returns.")
        elif improvement > 0:
            print("✓ CPO provides modest improvement")
        else:
            print("⚠️  CPO does not improve performance")
            print("   Fixed parameters may be sufficient")

        return results


def main():
    """Run all robustness tests"""
    print("="*80)
    print("BACKTEST ROBUSTNESS ANALYSIS")
    print("Critical evaluation of momentum strategy")
    print("="*80)

    analyzer = BacktestRobustnessAnalyzer('data/raw/equities_prices.csv')

    # 1. Rolling Sharpe Analysis
    rolling_sharpe, returns = analyzer.rolling_sharpe_analysis(lookback=20, window_days=252)

    # 2. Regime-Based Analysis
    regime_results = analyzer.regime_analysis(lookback=20)

    # 3. Walk-Forward Analysis (critical for overfitting detection)
    wf_results = analyzer.walk_forward_analysis(lookback=20, train_years=3, test_years=1)

    # 4. Parameter Sensitivity
    param_results = analyzer.parameter_sensitivity()

    # 5. Transaction Cost Sensitivity
    cost_results = analyzer.transaction_cost_sensitivity(lookback=20)

    # 6. Conditional Parameter Optimization
    cpo_results = analyzer.conditional_parameter_optimization()

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print("\nKey Questions Answered:")
    print("  1. ✓ Does performance vary significantly across time? (Rolling Sharpe)")
    print("  2. ✓ Which regimes does the strategy work/fail in? (Regime Analysis)")
    print("  3. ✓ Is there overfitting? (Walk-Forward)")
    print("  4. ✓ Is performance dependent on specific parameters? (Sensitivity)")
    print("  5. ✓ How do transaction costs impact results? (Cost Analysis)")
    print("  6. ✓ Can adaptive parameters improve robustness? (CPO)")


if __name__ == "__main__":
    main()
