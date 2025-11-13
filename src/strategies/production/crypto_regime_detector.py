"""
Crypto Market Regime Detection

Detects crypto-specific market regimes to avoid momentum crashes during crypto winters.

Crypto-Specific Regimes:
1. BULL MARKET - Strong uptrend, BTC dominance stable, momentum works
2. CRYPTO WINTER - Extended bear market, momentum fails
3. ALTCOIN SEASON - Alts outperform BTC, momentum excellent
4. CRASH - Sudden collapse, stop trading
5. RECOVERY - Bottom formation, momentum starts working

Key Differences from Traditional Markets:
- Faster regime changes (weeks not months)
- BTC dominance is key indicator
- Higher volatility thresholds
- 24/7 trading (no gaps)
- Retail-driven sentiment shifts

Academic Foundation:
- Liu & Tsyvinski (2021): Cryptocurrency momentum
- Hu, Parlour & Rajan (2019): Cryptocurrencies and momentum
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, Tuple
from enum import Enum
from datetime import datetime


class CryptoRegime(Enum):
    """Crypto market regime classifications"""
    BULL = "BULL"
    CRYPTO_WINTER = "CRYPTO_WINTER"
    ALTCOIN_SEASON = "ALTCOIN_SEASON"
    CRASH = "CRASH"
    RECOVERY = "RECOVERY"
    UNKNOWN = "UNKNOWN"


@dataclass
class CryptoRegimeConfig:
    """Configuration for crypto regime detection"""

    # Volatility thresholds (crypto-specific, higher than equities)
    crisis_vol_threshold: float = 0.80  # 80% annualized = crisis
    high_vol_threshold: float = 0.60    # 60% = high vol
    low_vol_threshold: float = 0.30     # 30% = low vol

    # Drawdown thresholds (crypto-specific)
    crash_threshold: float = -0.30      # -30% in short period = crash
    winter_threshold: float = -0.50     # -50% from peak = crypto winter

    # Trend parameters
    ma_short: int = 21   # ~1 month (crypto trades 24/7)
    ma_long: int = 100   # ~3 months

    # Lookback periods
    vol_lookback: int = 30   # 30 days for volatility (crypto is faster)
    trend_lookback: int = 90  # 90 days for trend

    # BTC dominance (if available)
    use_btc_dominance: bool = False  # Set to True if you have dominance data


class CryptoRegimeDetector:
    """
    Crypto-specific regime detection

    Optimized for crypto's unique characteristics:
    - Higher volatility
    - Faster regime changes
    - 24/7 trading
    - Retail-driven
    """

    def __init__(self, config: Optional[CryptoRegimeConfig] = None):
        """Initialize crypto regime detector"""
        self.config = config or CryptoRegimeConfig()

    def detect_crash(self, prices: pd.Series, lookback: int = 7) -> pd.Series:
        """
        Detect rapid crash periods (>20% drop in < 1 week)

        Args:
            prices: Price series
            lookback: Days to look back for crash

        Returns:
            Boolean series indicating crash periods
        """
        # Calculate rolling max and drawdown
        rolling_max = prices.rolling(lookback, min_periods=1).max()
        drawdown = (prices - rolling_max) / rolling_max

        # Crash = rapid >30% decline
        is_crash = drawdown < self.config.crash_threshold

        return is_crash

    def detect_crypto_winter(self, prices: pd.Series) -> pd.Series:
        """
        Detect crypto winter (extended bear market from peak)

        Crypto winter = >50% down from all-time high for extended period

        Args:
            prices: Price series

        Returns:
            Boolean series indicating crypto winter
        """
        # All-time high
        ath = prices.expanding().max()

        # Current drawdown from ATH
        dd_from_ath = (prices - ath) / ath

        # Crypto winter = >50% down from ATH
        is_winter = dd_from_ath < self.config.winter_threshold

        # Must persist for at least 30 days to avoid false signals
        is_winter_smoothed = is_winter.rolling(30, min_periods=1).mean() > 0.7

        return is_winter_smoothed

    def detect_recovery(self, prices: pd.Series, returns: pd.Series) -> pd.Series:
        """
        Detect recovery phase (bottom formation after winter)

        Recovery = price stabilizing after winter, starting to trend up

        Args:
            prices: Price series
            returns: Return series

        Returns:
            Boolean series indicating recovery
        """
        # Calculate moving averages
        ma_short = prices.rolling(self.config.ma_short).mean()
        ma_long = prices.rolling(self.config.ma_long).mean()

        # Volatility declining
        rolling_vol = returns.rolling(self.config.vol_lookback).std() * np.sqrt(365)
        vol_declining = rolling_vol.diff(10) < 0

        # Price above short MA but below long MA (early uptrend)
        early_uptrend = (prices > ma_short) & (prices < ma_long)

        # Recent positive returns (last 2 weeks)
        recent_positive = returns.rolling(14).sum() > 0

        # Recovery = early uptrend + vol declining + recent positive
        is_recovery = early_uptrend & vol_declining & recent_positive

        return is_recovery

    def detect_altcoin_season(self, btc_prices: pd.Series,
                              alt_prices: pd.DataFrame) -> pd.Series:
        """
        Detect altcoin season (alts outperforming BTC)

        Momentum works EXCEPTIONALLY well during altcoin seasons

        Args:
            btc_prices: BTC price series
            alt_prices: Altcoin prices (DataFrame)

        Returns:
            Boolean series indicating altcoin season
        """
        # BTC returns
        btc_returns = btc_prices.pct_change()

        # Alt returns (average)
        alt_returns = alt_prices.pct_change().mean(axis=1)

        # Altcoin season = alts outperforming BTC over 30 days
        btc_30d = btc_returns.rolling(30).sum()
        alt_30d = alt_returns.rolling(30).sum()

        alts_outperforming = alt_30d > btc_30d

        # Must be consistent (>70% of days)
        altcoin_season = alts_outperforming.rolling(30, min_periods=1).mean() > 0.7

        return altcoin_season

    def detect_regime(self, prices: pd.DataFrame) -> pd.Series:
        """
        Comprehensive crypto regime detection

        Args:
            prices: Crypto prices (BTC should be first column)

        Returns:
            Series with regime classifications
        """
        # Use BTC as market proxy (or equal-weighted portfolio)
        if 'BTC-USD' in prices.columns:
            market_prices = prices['BTC-USD']
        else:
            market_prices = prices.mean(axis=1)

        market_returns = market_prices.pct_change()

        # Initialize regime series
        regimes = pd.Series(CryptoRegime.UNKNOWN.value, index=prices.index)

        # Detect each regime type
        is_crash = self.detect_crash(market_prices)
        is_winter = self.detect_crypto_winter(market_prices)
        is_recovery = self.detect_recovery(market_prices, market_returns)

        # Detect altcoin season if we have multiple coins
        if len(prices.columns) > 1:
            btc_prices = prices.iloc[:, 0]  # First column
            alt_prices = prices.iloc[:, 1:]  # Rest
            is_altcoin_season = self.detect_altcoin_season(btc_prices, alt_prices)
        else:
            is_altcoin_season = pd.Series(False, index=prices.index)

        # Calculate trend indicators for bull market
        ma_short = market_prices.rolling(self.config.ma_short).mean()
        ma_long = market_prices.rolling(self.config.ma_long).mean()

        # Classify regimes (priority order matters!)
        for date in prices.index[self.config.ma_long:]:
            # 1. CRASH (highest priority)
            if is_crash.loc[date]:
                regimes.loc[date] = CryptoRegime.CRASH.value

            # 2. CRYPTO WINTER
            elif is_winter.loc[date]:
                regimes.loc[date] = CryptoRegime.CRYPTO_WINTER.value

            # 3. RECOVERY
            elif is_recovery.loc[date]:
                regimes.loc[date] = CryptoRegime.RECOVERY.value

            # 4. ALTCOIN SEASON (momentum works great)
            elif is_altcoin_season.loc[date]:
                regimes.loc[date] = CryptoRegime.ALTCOIN_SEASON.value

            # 5. BULL MARKET (default if uptrend)
            elif ma_short.loc[date] > ma_long.loc[date]:
                regimes.loc[date] = CryptoRegime.BULL.value

            # 6. Unknown/Sideways
            else:
                regimes.loc[date] = CryptoRegime.UNKNOWN.value

        return regimes

    def get_regime_statistics(self, regimes: pd.Series,
                             returns: pd.Series) -> pd.DataFrame:
        """Calculate statistics for each regime"""

        stats = []

        for regime in [CryptoRegime.BULL.value, CryptoRegime.CRYPTO_WINTER.value,
                      CryptoRegime.ALTCOIN_SEASON.value, CryptoRegime.CRASH.value,
                      CryptoRegime.RECOVERY.value]:

            regime_returns = returns[regimes == regime]

            if len(regime_returns) > 0:
                ann_ret = regime_returns.mean() * 365 * 100
                ann_vol = regime_returns.std() * np.sqrt(365) * 100
                sharpe = (regime_returns.mean() * 365) / (regime_returns.std() * np.sqrt(365)) \
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

    def should_trade_momentum(self, current_regime: str) -> Tuple[bool, float]:
        """
        Determine if momentum should be traded in current regime

        Args:
            current_regime: Current regime classification

        Returns:
            (should_trade, leverage_multiplier)
        """
        regime_params = {
            CryptoRegime.BULL.value: (True, 1.0),           # Full size
            CryptoRegime.ALTCOIN_SEASON.value: (True, 1.2), # Increase size!
            CryptoRegime.RECOVERY.value: (True, 0.5),       # Half size (testing)
            CryptoRegime.CRYPTO_WINTER.value: (False, 0.0), # DON'T TRADE
            CryptoRegime.CRASH.value: (False, 0.0),         # DON'T TRADE
            CryptoRegime.UNKNOWN.value: (True, 0.3),        # Small size
        }

        return regime_params.get(current_regime, (True, 0.5))


def main():
    """Test crypto regime detection"""

    print("="*80)
    print("CRYPTO REGIME DETECTION SYSTEM")
    print("="*80)

    # Load crypto data
    prices = pd.read_csv('data/raw/crypto_prices.csv', index_col=0)
    prices.index = pd.to_datetime(prices.index, utc=True).tz_convert(None)

    print(f"\nAnalyzing {len(prices)} days from {prices.index[0].date()} to {prices.index[-1].date()}")
    print(f"Assets: {list(prices.columns)}\n")

    # Initialize detector
    detector = CryptoRegimeDetector()

    # Detect regimes
    print("Detecting crypto market regimes...")
    regimes = detector.detect_regime(prices)

    # Calculate market returns (BTC or equal-weighted)
    if 'BTC-USD' in prices.columns:
        market_returns = prices['BTC-USD'].pct_change()
    else:
        market_returns = prices.mean(axis=1).pct_change()

    # Print statistics
    print("\n" + "="*80)
    print("REGIME STATISTICS")
    print("="*80)
    stats = detector.get_regime_statistics(regimes, market_returns)
    print("\n" + stats.to_string(index=False))

    # Identify crypto winters and crashes
    print("\n" + "="*80)
    print("IDENTIFIED CRYPTO WINTERS & CRASHES")
    print("="*80)

    # Find continuous periods
    winter_periods = []
    in_winter = False
    start_date = None

    for date in regimes.index:
        regime = regimes.loc[date]

        if regime in [CryptoRegime.CRYPTO_WINTER.value, CryptoRegime.CRASH.value] and not in_winter:
            in_winter = True
            start_date = date
            start_regime = regime
        elif regime not in [CryptoRegime.CRYPTO_WINTER.value, CryptoRegime.CRASH.value] and in_winter:
            in_winter = False
            if start_date:
                winter_periods.append((start_date, regimes.index[regimes.index.get_loc(date)-1], start_regime))

    if in_winter and start_date:
        winter_periods.append((start_date, regimes.index[-1], start_regime))

    if len(winter_periods) > 0:
        for start, end, regime_type in winter_periods:
            duration = (end - start).days

            # Calculate return during period
            if 'BTC-USD' in prices.columns:
                period_return = (prices['BTC-USD'].loc[end] / prices['BTC-USD'].loc[start] - 1) * 100
            else:
                period_prices = prices.mean(axis=1)
                period_return = (period_prices.loc[end] / period_prices.loc[start] - 1) * 100

            print(f"\n{regime_type}: {start.date()} to {end.date()} ({duration} days)")
            print(f"  Return: {period_return:+.1f}%")
    else:
        print("\n✅ No major crypto winters detected in this period")

    # Trading recommendations
    print("\n" + "="*80)
    print("MOMENTUM TRADING RECOMMENDATIONS BY REGIME")
    print("="*80)

    print("\nRegime Trading Rules:")
    print("  BULL MARKET:      ✅ Trade normally (100% size)")
    print("  ALTCOIN SEASON:   ✅ INCREASE size (120% - best regime!)")
    print("  RECOVERY:         ⚠️  Trade cautiously (50% size)")
    print("  CRYPTO WINTER:    ❌ STOP TRADING (0% - momentum fails)")
    print("  CRASH:            ❌ STOP TRADING (0% - extreme vol)")
    print("  UNKNOWN:          ⚠️  Trade small (30% size)")

    # Current regime
    current_regime = regimes.iloc[-1]
    should_trade, leverage = detector.should_trade_momentum(current_regime)

    print(f"\n{'='*80}")
    print(f"CURRENT REGIME: {current_regime}")
    print(f"{'='*80}")

    if should_trade:
        print(f"✅ TRADE MOMENTUM at {leverage*100:.0f}% of normal size")
    else:
        print(f"❌ DO NOT TRADE - Wait for regime change")

    print("\n" + "="*80)
    print("✅ CRYPTO REGIME DETECTION READY")
    print("="*80)

    return regimes, stats


if __name__ == "__main__":
    main()
