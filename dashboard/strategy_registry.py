"""
Strategy Registry & Manager

Centralized registry for all trading strategies with automatic discovery,
metadata management, and integration with the dashboard.

Manages:
- Strategy discovery and registration
- Metadata (category, asset class, timeframe, status)
- Performance tracking across all strategies
- Multi-strategy portfolio allocation
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Optional, Type
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import importlib.util
import inspect

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


class StrategyCategory(Enum):
    """Strategy categories"""
    TECHNICAL = "technical"
    MOMENTUM = "momentum"
    MEAN_REVERSION = "mean_reversion"
    FACTOR = "factor"
    PAIRS_TRADING = "pairs_trading"
    CARRY_TRADE = "carry_trade"
    ENSEMBLE = "ensemble"
    ML_BASED = "ml_based"


class AssetClass(Enum):
    """Asset classes"""
    EQUITY = "equity"
    CRYPTO = "crypto"
    FOREX = "forex"
    COMMODITIES = "commodities"
    MULTI_ASSET = "multi_asset"


class StrategyStatus(Enum):
    """Strategy deployment status"""
    RESEARCH = "research"
    BACKTESTED = "backtested"
    VALIDATED = "validated"
    PRODUCTION = "production"
    DEPRECATED = "deprecated"


@dataclass
class StrategyMetadata:
    """Metadata for a trading strategy"""
    name: str
    display_name: str
    category: StrategyCategory
    asset_class: AssetClass
    status: StrategyStatus
    module_path: str
    class_name: str
    description: str = ""
    author: str = ""
    created_date: Optional[datetime] = None

    # Performance expectations
    expected_sharpe: Optional[float] = None
    expected_return: Optional[float] = None
    expected_volatility: Optional[float] = None
    max_drawdown_limit: Optional[float] = None

    # Configuration
    default_params: Dict = field(default_factory=dict)
    required_data: List[str] = field(default_factory=list)  # ['prices', 'volume', etc.]
    timeframe: str = "daily"  # 'intraday', 'daily', 'weekly', etc.

    # Risk management
    use_regime_filter: bool = True
    use_position_sizing: bool = True
    max_leverage: float = 1.0

    # Walk-forward validation results
    walk_forward_passed: Optional[bool] = None
    oos_sharpe: Optional[float] = None
    overfitting_score: Optional[float] = None

    # Tags for filtering
    tags: List[str] = field(default_factory=list)


class StrategyRegistry:
    """
    Central registry for all trading strategies

    Discovers, catalogs, and manages all strategies in the platform.
    Provides easy access for backtesting, optimization, and portfolio allocation.
    """

    def __init__(self, strategies_dir: str = "src/strategies"):
        """Initialize strategy registry"""
        self.strategies_dir = Path(strategies_dir)
        self.strategies: Dict[str, StrategyMetadata] = {}
        self._discover_strategies()

    def _discover_strategies(self):
        """Automatically discover all strategies in the codebase"""
        print(f"\n{'='*80}")
        print("DISCOVERING STRATEGIES")
        print(f"{'='*80}\n")

        # Manually register known strategies with metadata
        self._register_production_strategies()
        self._register_enhanced_strategies()
        self._register_factor_strategies()
        self._register_basic_strategies()

        print(f"\n✅ Discovered {len(self.strategies)} strategies")
        print(f"{'='*80}\n")

    def _register_production_strategies(self):
        """Register production-ready strategies"""
        print("📊 Production Strategies:")

        # Equity Momentum - VALIDATED
        self.register(StrategyMetadata(
            name="equity_momentum_90d",
            display_name="Equity Momentum (90-day)",
            category=StrategyCategory.MOMENTUM,
            asset_class=AssetClass.EQUITY,
            status=StrategyStatus.VALIDATED,
            module_path="src.strategies.production.equity_momentum",
            class_name="EquityMomentumStrategy",
            description="Cross-sectional equity momentum with regime filtering. VALIDATED with OOS Sharpe 1.95.",
            expected_sharpe=1.95,
            expected_return=28.5,
            expected_volatility=14.6,
            max_drawdown_limit=-20.0,
            default_params={'lookback': 90, 'top_n': 10, 'rebalance_freq': 'monthly'},
            required_data=['prices', 'market_cap'],
            use_regime_filter=True,
            walk_forward_passed=True,
            oos_sharpe=1.95,
            overfitting_score=0.02,  # OOS better than in-sample
            tags=['validated', 'low_overfitting', 'regime_filtered']
        ))

        # Crypto Momentum - FAILED (kept for analysis)
        self.register(StrategyMetadata(
            name="crypto_momentum",
            display_name="Crypto Momentum",
            category=StrategyCategory.MOMENTUM,
            asset_class=AssetClass.CRYPTO,
            status=StrategyStatus.DEPRECATED,
            module_path="src.strategies.production.crypto_momentum",
            class_name="CryptoMomentumStrategy",
            description="Crypto momentum strategy. FAILED walk-forward validation - severe overfitting.",
            expected_sharpe=-1.42,
            expected_return=-45.0,
            expected_volatility=55.0,
            walk_forward_passed=False,
            oos_sharpe=-1.42,
            overfitting_score=1.5,  # Catastrophic overfitting
            tags=['failed', 'high_overfitting', 'deprecated']
        ))

        # Combined Momentum Portfolio
        self.register(StrategyMetadata(
            name="combined_momentum_portfolio",
            display_name="Combined Momentum Portfolio",
            category=StrategyCategory.ENSEMBLE,
            asset_class=AssetClass.MULTI_ASSET,
            status=StrategyStatus.PRODUCTION,
            module_path="src.strategies.production.combined_momentum_portfolio",
            class_name="CombinedMomentumPortfolio",
            description="Multi-asset momentum portfolio combining equity and crypto strategies.",
            expected_sharpe=1.68,
            expected_return=25.0,
            default_params={'equity_weight': 0.9, 'crypto_weight': 0.1},
            use_regime_filter=True,
            tags=['multi_asset', 'production']
        ))

        # Multi-Timeframe Ensemble
        self.register(StrategyMetadata(
            name="multi_timeframe_ensemble",
            display_name="Multi-Timeframe Ensemble",
            category=StrategyCategory.ENSEMBLE,
            asset_class=AssetClass.EQUITY,
            status=StrategyStatus.VALIDATED,
            module_path="src.strategies.production.multi_timeframe_ensemble",
            class_name="MultiTimeframeEnsemble",
            description="Ensemble strategy combining signals from multiple timeframes.",
            expected_sharpe=2.1,
            default_params={'timeframes': ['daily', 'weekly', 'monthly']},
            tags=['ensemble', 'multi_timeframe']
        ))

        print("  ✓ equity_momentum_90d (VALIDATED - Sharpe 1.95)")
        print("  ✗ crypto_momentum (FAILED - Sharpe -1.42)")
        print("  ✓ combined_momentum_portfolio (Sharpe 1.68)")
        print("  ✓ multi_timeframe_ensemble (Sharpe 2.1)")

    def _register_enhanced_strategies(self):
        """Register enhanced strategies with advanced features"""
        print("\n⚡ Enhanced Strategies:")

        self.register(StrategyMetadata(
            name="cpo_momentum",
            display_name="CPO Momentum",
            category=StrategyCategory.MOMENTUM,
            asset_class=AssetClass.EQUITY,
            status=StrategyStatus.VALIDATED,
            module_path="src.strategies.enhanced.cpo_momentum",
            class_name="CPOMomentumStrategy",
            description="Conservative Portfolio Optimization with momentum.",
            expected_sharpe=1.8,
            tags=['cpo', 'optimized']
        ))

        self.register(StrategyMetadata(
            name="multi_factor_momentum",
            display_name="Multi-Factor Momentum",
            category=StrategyCategory.FACTOR,
            asset_class=AssetClass.EQUITY,
            status=StrategyStatus.VALIDATED,
            module_path="src.strategies.enhanced.multi_factor_momentum",
            class_name="MultiFactorMomentumStrategy",
            description="Combines momentum with value and quality factors.",
            expected_sharpe=2.0,
            tags=['multi_factor', 'systematic']
        ))

        print("  ✓ cpo_momentum")
        print("  ✓ multi_factor_momentum")

    def _register_factor_strategies(self):
        """Register factor-based strategies"""
        print("\n📈 Factor Strategies:")

        for factor_name, display_name in [
            ('momentum_factor', 'Momentum Factor'),
            ('value_factor', 'Value Factor'),
            ('quality_factor', 'Quality Factor'),
            ('multi_factor', 'Multi-Factor Composite')
        ]:
            self.register(StrategyMetadata(
                name=factor_name,
                display_name=display_name,
                category=StrategyCategory.FACTOR,
                asset_class=AssetClass.EQUITY,
                status=StrategyStatus.BACKTESTED,
                module_path=f"src.strategies.factors.{factor_name}",
                class_name=f"{''.join(w.capitalize() for w in factor_name.split('_'))}",
                description=f"{display_name} based equity selection",
                tags=['factor', 'systematic']
            ))
            print(f"  ✓ {factor_name}")

    def _register_basic_strategies(self):
        """Register basic technical strategies"""
        print("\n🔧 Technical Strategies:")

        technical_strategies = [
            ('rsi_strategy', 'RSI Strategy', StrategyCategory.TECHNICAL),
            ('macd_strategy', 'MACD Strategy', StrategyCategory.TECHNICAL),
            ('bollinger_bands_strategy', 'Bollinger Bands', StrategyCategory.TECHNICAL),
            ('mean_reversion_strategy', 'Mean Reversion', StrategyCategory.MEAN_REVERSION),
            ('pairs_trading_strategy', 'Pairs Trading', StrategyCategory.PAIRS_TRADING),
            ('carry_trade_strategy', 'Carry Trade', StrategyCategory.CARRY_TRADE),
            ('time_series_momentum', 'Time Series Momentum', StrategyCategory.MOMENTUM),
            ('cross_sectional_momentum', 'Cross-Sectional Momentum', StrategyCategory.MOMENTUM),
            ('parabolic_sar_strategy', 'Parabolic SAR', StrategyCategory.TECHNICAL),
            ('heikin_ashi_strategy', 'Heikin Ashi', StrategyCategory.TECHNICAL),
        ]

        for name, display, category in technical_strategies:
            self.register(StrategyMetadata(
                name=name,
                display_name=display,
                category=category,
                asset_class=AssetClass.MULTI_ASSET,
                status=StrategyStatus.RESEARCH,
                module_path=f"src.strategies.{name}",
                class_name=f"{''.join(w.capitalize() for w in name.split('_'))}",
                description=f"{display} technical strategy",
                tags=['technical', 'research']
            ))
            print(f"  ✓ {name}")

    def register(self, metadata: StrategyMetadata):
        """Register a strategy"""
        self.strategies[metadata.name] = metadata

    def get(self, name: str) -> Optional[StrategyMetadata]:
        """Get strategy metadata by name"""
        return self.strategies.get(name)

    def list_all(self) -> List[StrategyMetadata]:
        """List all registered strategies"""
        return list(self.strategies.values())

    def filter_by_status(self, status: StrategyStatus) -> List[StrategyMetadata]:
        """Filter strategies by deployment status"""
        return [s for s in self.strategies.values() if s.status == status]

    def filter_by_category(self, category: StrategyCategory) -> List[StrategyMetadata]:
        """Filter strategies by category"""
        return [s for s in self.strategies.values() if s.category == category]

    def filter_by_asset_class(self, asset_class: AssetClass) -> List[StrategyMetadata]:
        """Filter strategies by asset class"""
        return [s for s in self.strategies.values() if s.asset_class == asset_class]

    def filter_by_tag(self, tag: str) -> List[StrategyMetadata]:
        """Filter strategies by tag"""
        return [s for s in self.strategies.values() if tag in s.tags]

    def get_production_strategies(self) -> List[StrategyMetadata]:
        """Get all production-ready strategies"""
        return [s for s in self.strategies.values()
                if s.status in [StrategyStatus.PRODUCTION, StrategyStatus.VALIDATED]]

    def get_validated_strategies(self) -> List[StrategyMetadata]:
        """Get strategies that passed walk-forward validation"""
        return [s for s in self.strategies.values() if s.walk_forward_passed == True]

    def get_summary(self) -> Dict:
        """Get summary statistics of all strategies"""
        return {
            'total': len(self.strategies),
            'by_status': {
                status.value: len(self.filter_by_status(status))
                for status in StrategyStatus
            },
            'by_category': {
                cat.value: len(self.filter_by_category(cat))
                for cat in StrategyCategory
            },
            'by_asset_class': {
                ac.value: len(self.filter_by_asset_class(ac))
                for ac in AssetClass
            },
            'validated': len(self.get_validated_strategies()),
            'production_ready': len(self.get_production_strategies())
        }

    def print_summary(self):
        """Print formatted summary"""
        summary = self.get_summary()

        print(f"\n{'='*80}")
        print("STRATEGY REGISTRY SUMMARY")
        print(f"{'='*80}\n")

        print(f"Total Strategies: {summary['total']}")

        print("\nBy Status:")
        for status, count in summary['by_status'].items():
            if count > 0:
                print(f"  {status:15s}: {count}")

        print("\nBy Category:")
        for category, count in summary['by_category'].items():
            if count > 0:
                print(f"  {category:20s}: {count}")

        print("\nBy Asset Class:")
        for asset_class, count in summary['by_asset_class'].items():
            if count > 0:
                print(f"  {asset_class:15s}: {count}")

        print(f"\n✅ Validated Strategies: {summary['validated']}")
        print(f"🚀 Production Ready: {summary['production_ready']}")
        print(f"\n{'='*80}\n")


# Global registry instance
_registry = None

def get_registry() -> StrategyRegistry:
    """Get the global strategy registry (singleton)"""
    global _registry
    if _registry is None:
        _registry = StrategyRegistry()
    return _registry


if __name__ == '__main__':
    """Test strategy registry"""

    registry = get_registry()
    registry.print_summary()

    print("\nProduction Strategies:")
    for strategy in registry.get_production_strategies():
        print(f"  - {strategy.display_name} ({strategy.name})")
        print(f"    Expected Sharpe: {strategy.expected_sharpe}")
        print(f"    Status: {strategy.status.value}")
        print()
