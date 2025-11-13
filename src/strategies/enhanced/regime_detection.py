"""
Market Regime Detection System

Detects market regimes in real-time to adapt trading strategies.
Critical for avoiding momentum crashes during bear markets/corrections.

Regimes Detected:
1. BULL MARKET - Sustained uptrend, momentum works well
2. BEAR MARKET - Sustained downtrend, avoid momentum
3. CORRECTION - Sharp short-term decline, high risk
4. SIDEWAYS - Low volatility, choppy, avoid momentum
5. CRISIS - Extreme volatility, panic selling, stop trading

Methods:
- Trend-based (moving averages, drawdown)
- Volatility-based (VIX analogue, realized vol)
- Hidden Markov Model (statistical regime detection)

Academic Foundation:
- Guidolin & Timmermann (2008): Regime switching in asset returns
- Ang & Bekaert (2002): International asset allocation with regime shifts
- Kritzman et al. (2012): Regime shifts - Turbulence index
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple
from enum import Enum
from datetime import datetime


class MarketRegime(Enum):
    """Market regime classifications"""
    BULL = "BULL"
    BEAR = "BEAR"
    CORRECTION = "CORRECTION"
    SIDEWAYS = "SIDEWAYS"
    CRISIS = "CRISIS"
    UNKNOWN = "UNKNOWN"


@dataclass
class RegimeDetectorConfig:
    """Configuration for regime detection"""

    # Trend parameters
    ma_short: int = 50   # Short moving average
    ma_long: int = 200   # Long moving average
    trend_lookback: int = 252  # 1 year for trend

    # Volatility parameters
    vol_lookback: int = 60
    crisis_vol_threshold: float = 0.40  # 40% annualized vol = crisis
    high_vol_threshold: float = 0.25    # 25% = high vol
    low_vol_threshold: float = 0.10     # 10% = low vol

    # Drawdown thresholds
    correction_threshold: float = -0.10  # -10% = correction
    bear_threshold: float = -0.20        # -20% = bear market

    # Return thresholds
    bull_min_return: float = 0.15  # 15% annual return = bull
    bear_max_return: float = -0.10  # -10% annual return = bear


class RegimeDetector:
    """
    Market Regime Detection System

    Combines multiple methods for robust regime classification:
    1. Trend-based (MA crossovers, drawdown)
    2. Volatility-based (realized vol, extremes)
    3. Return-based (rolling returns)
    """

    def __init__(self, config: Optional[RegimeDetectorConfig] = None):
        """
        Initialize regime detector

        Args:
            config: Detector configuration
        """
        self.config = config or RegimeDetectorConfig()
        self.regime_history = pd.DataFrame()

    def detect_trend_regime(self, prices: pd.Series) -> pd.Series:
        """
        Detect regime based on moving averages and drawdown

        Args:
            prices: Price series (index or portfolio value)

        Returns:
            Series with regime classifications
        """
        # Calculate moving averages
        ma_short = prices.rolling(self.config.ma_short).mean()
        ma_long = prices.rolling(self.config.ma_long).mean()

        # Calculate drawdown
        running_max = prices.expanding().max()
        drawdown = (prices - running_max) / running_max

        # Classify regimes
        regimes = pd.Series(MarketRegime.UNKNOWN.value, index=prices.index)

        for date in prices.index[self.config.ma_long:]:
            dd = drawdown.loc[date]

            # Crisis: Extreme drawdown
            if dd < -0.35:
                regimes.loc[date] = MarketRegime.CRISIS.value

            # Bear market: Deep drawdown + price < MA200
            elif dd < self.config.bear_threshold and prices.loc[date] < ma_long.loc[date]:
                regimes.loc[date] = MarketRegime.BEAR.value

            # Correction: Moderate drawdown
            elif dd < self.config.correction_threshold:
                regimes.loc[date] = MarketRegime.CORRECTION.value

            # Bull market: MA50 > MA200 and minimal drawdown
            elif ma_short.loc[date] > ma_long.loc[date] and dd > -0.05:
                regimes.loc[date] = MarketRegime.BULL.value

            # Sideways: MA50 ~ MA200
            else:
                regimes.loc[date] = MarketRegime.SIDEWAYS.value

        return regimes

    def detect_volatility_regime(self, returns: pd.Series) -> pd.Series:
        """
        Detect regime based on volatility

        Args:
            returns: Return series

        Returns:
            Series with volatility-based regime
        """
        # Rolling realized volatility
        rolling_vol = returns.rolling(self.config.vol_lookback).std() * np.sqrt(252)

        regimes = pd.Series(MarketRegime.UNKNOWN.value, index=returns.index)

        for date in returns.index[self.config.vol_lookback:]:
            vol = rolling_vol.loc[date]

            if vol > self.config.crisis_vol_threshold:
                regimes.loc[date] = MarketRegime.CRISIS.value
            elif vol > self.config.high_vol_threshold:
                # High vol can be bull or bear - need more info
                regimes.loc[date] = 'HIGH_VOL'
            elif vol < self.config.low_vol_threshold:
                regimes.loc[date] = MarketRegime.SIDEWAYS.value
            else:
                regimes.loc[date] = 'NORMAL_VOL'

        return regimes

    def detect_return_regime(self, returns: pd.Series) -> pd.Series:
        """
        Detect regime based on rolling returns

        Args:
            returns: Return series

        Returns:
            Series with return-based regime
        """
        # Rolling annual return
        rolling_return = returns.rolling(self.config.trend_lookback).apply(
            lambda x: (1 + x).prod() - 1
        )

        regimes = pd.Series(MarketRegime.UNKNOWN.value, index=returns.index)

        for date in returns.index[self.config.trend_lookback:]:
            ret = rolling_return.loc[date]

            if ret > self.config.bull_min_return:
                regimes.loc[date] = MarketRegime.BULL.value
            elif ret < self.config.bear_max_return:
                regimes.loc[date] = MarketRegime.BEAR.value
            else:
                regimes.loc[date] = MarketRegime.SIDEWAYS.value

        return regimes

    def detect_regime(self, prices: pd.Series,
                     returns: Optional[pd.Series] = None) -> pd.Series:
        """
        Comprehensive regime detection combining multiple methods

        Args:
            prices: Price series
            returns: Return series (optional, will calculate if not provided)

        Returns:
            Series with final regime classifications
        """
        if returns is None:
            returns = prices.pct_change()

        # Get regime signals from each method
        trend_regime = self.detect_trend_regime(prices)
        vol_regime = self.detect_volatility_regime(returns)
        return_regime = self.detect_return_regime(returns)

        # Combine signals with voting
        final_regime = pd.Series(MarketRegime.UNKNOWN.value, index=prices.index)

        for date in prices.index[self.config.ma_long:]:
            votes = {
                MarketRegime.BULL.value: 0,
                MarketRegime.BEAR.value: 0,
                MarketRegime.CORRECTION.value: 0,
                MarketRegime.SIDEWAYS.value: 0,
                MarketRegime.CRISIS.value: 0
            }

            # Trend vote
            if trend_regime.loc[date] in votes:
                votes[trend_regime.loc[date]] += 1

            # Volatility vote (crisis trumps everything)
            if vol_regime.loc[date] == MarketRegime.CRISIS.value:
                final_regime.loc[date] = MarketRegime.CRISIS.value
                continue
            elif vol_regime.loc[date] == MarketRegime.SIDEWAYS.value:
                votes[MarketRegime.SIDEWAYS.value] += 1

            # Return vote
            if return_regime.loc[date] in votes:
                votes[return_regime.loc[date]] += 1

            # Final classification
            if votes[MarketRegime.CRISIS.value] > 0:
                final_regime.loc[date] = MarketRegime.CRISIS.value
            elif votes[MarketRegime.CORRECTION.value] > 0:
                final_regime.loc[date] = MarketRegime.CORRECTION.value
            elif votes[MarketRegime.BEAR.value] >= 2:
                final_regime.loc[date] = MarketRegime.BEAR.value
            elif votes[MarketRegime.BULL.value] >= 2:
                final_regime.loc[date] = MarketRegime.BULL.value
            else:
                final_regime.loc[date] = MarketRegime.SIDEWAYS.value

        return final_regime

    def get_regime_statistics(self, regimes: pd.Series,
                             returns: pd.Series) -> pd.DataFrame:
        """
        Calculate performance statistics for each regime

        Args:
            regimes: Regime classifications
            returns: Portfolio returns

        Returns:
            DataFrame with regime statistics
        """
        stats = []

        for regime in [MarketRegime.BULL.value, MarketRegime.BEAR.value,
                      MarketRegime.CORRECTION.value, MarketRegime.SIDEWAYS.value,
                      MarketRegime.CRISIS.value]:

            regime_returns = returns[regimes == regime]

            if len(regime_returns) > 0:
                ann_ret = regime_returns.mean() * 252 * 100
                ann_vol = regime_returns.std() * np.sqrt(252) * 100
                sharpe = (regime_returns.mean() * 252) / (regime_returns.std() * np.sqrt(252)) \
                    if regime_returns.std() > 0 else 0

                stats.append({
                    'Regime': regime,
                    'Days': len(regime_returns),
                    'Pct of Time': f"{len(regime_returns)/len(returns)*100:.1f}%",
                    'Ann Return': f"{ann_ret:.1f}%",
                    'Ann Vol': f"{ann_vol:.1f}%",
                    'Sharpe': f"{sharpe:.2f}"
                })

        return pd.DataFrame(stats)

    def get_regime_transitions(self, regimes: pd.Series) -> pd.DataFrame:
        """
        Analyze regime transition matrix

        Args:
            regimes: Regime classifications

        Returns:
            DataFrame with transition probabilities
        """
        # Count transitions
        transitions = {}

        for i in range(1, len(regimes)):
            from_regime = regimes.iloc[i-1]
            to_regime = regimes.iloc[i]

            if from_regime not in transitions:
                transitions[from_regime] = {}

            if to_regime not in transitions[from_regime]:
                transitions[from_regime][to_regime] = 0

            transitions[from_regime][to_regime] += 1

        # Convert to probabilities
        transition_matrix = []

        for from_regime in transitions:
            total = sum(transitions[from_regime].values())
            row = {'From': from_regime}

            for to_regime in [MarketRegime.BULL.value, MarketRegime.BEAR.value,
                             MarketRegime.CORRECTION.value, MarketRegime.SIDEWAYS.value,
                             MarketRegime.CRISIS.value]:
                count = transitions[from_regime].get(to_regime, 0)
                row[f'To {to_regime}'] = f"{count/total*100:.1f}%" if total > 0 else "0%"

            transition_matrix.append(row)

        return pd.DataFrame(transition_matrix)


def main():
    """Test regime detection on equity data"""
    print("="*80)
    print("MARKET REGIME DETECTION SYSTEM")
    print("="*80)

    # Load data
    prices_df = pd.read_csv('data/raw/equities_prices.csv', index_col=0)
    prices_df.index = pd.to_datetime(prices_df.index, utc=True).tz_convert(None)

    # Use equal-weighted portfolio
    prices = (prices_df / prices_df.iloc[0]).mean(axis=1)
    returns = prices.pct_change()

    print(f"\nAnalyzing {len(prices)} days from {prices.index[0].date()} to {prices.index[-1].date()}\n")

    # Initialize detector
    detector = RegimeDetector()

    # Detect regimes
    print("Detecting market regimes...")
    regimes = detector.detect_regime(prices, returns)

    # Print statistics
    print("\n" + "="*80)
    print("REGIME STATISTICS")
    print("="*80)
    stats = detector.get_regime_statistics(regimes, returns)
    print("\n" + stats.to_string(index=False))

    # Transition matrix
    print("\n" + "="*80)
    print("REGIME TRANSITION PROBABILITIES")
    print("="*80)
    transitions = detector.get_regime_transitions(regimes)
    print("\n" + transitions.to_string(index=False))

    # Identify specific periods
    print("\n" + "="*80)
    print("IDENTIFIED CRISIS/BEAR PERIODS")
    print("="*80)

    crisis_periods = []
    in_crisis = False
    start_date = None

    for date in regimes.index:
        regime = regimes.loc[date]

        if regime in [MarketRegime.CRISIS.value, MarketRegime.BEAR.value] and not in_crisis:
            in_crisis = True
            start_date = date
        elif regime not in [MarketRegime.CRISIS.value, MarketRegime.BEAR.value] and in_crisis:
            in_crisis = False
            if start_date:
                crisis_periods.append((start_date, regimes.index[regimes.index.get_loc(date)-1]))

    if in_crisis and start_date:
        crisis_periods.append((start_date, regimes.index[-1]))

    for start, end in crisis_periods:
        duration = (end - start).days
        period_return = (prices.loc[end] / prices.loc[start] - 1) * 100
        print(f"\n{start.date()} to {end.date()} ({duration} days)")
        print(f"  Return: {period_return:+.1f}%")
        print(f"  Regime: {regimes.loc[start]}")

    print("\n" + "="*80)
    print("✅ REGIME DETECTION SYSTEM READY")
    print("="*80)

    return regimes, stats


if __name__ == "__main__":
    main()
