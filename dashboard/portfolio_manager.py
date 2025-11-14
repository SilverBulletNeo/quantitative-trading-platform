"""
Multi-Strategy Portfolio Manager

Manages a portfolio of multiple trading strategies with:
- Dynamic allocation based on performance and regime
- Cross-strategy correlation analysis
- Portfolio-level risk management
- Real-time rebalancing
- Performance attribution
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
import scipy.optimize as optimize
from sklearn.covariance import LedoitWolf

from dashboard.strategy_registry import get_registry, StrategyMetadata, StrategyStatus
from dashboard.database import DatabaseManager


@dataclass
class PortfolioAllocation:
    """Portfolio allocation snapshot"""
    timestamp: datetime
    allocations: Dict[str, float]  # strategy_name -> weight
    expected_return: float
    expected_volatility: float
    expected_sharpe: float
    regime: str
    reason: str  # Why this allocation was chosen


class PortfolioOptimizer:
    """
    Optimize allocation across multiple strategies

    Methods:
    - Equal weight
    - Sharpe maximization
    - Minimum variance
    - Risk parity
    - Regime-adaptive
    """

    def __init__(self, lookback_days: int = 252):
        """Initialize optimizer"""
        self.lookback_days = lookback_days

    def optimize_sharpe(self,
                       returns: pd.DataFrame,
                       risk_free_rate: float = 0.0,
                       constraints: Optional[Dict] = None) -> Dict[str, float]:
        """
        Maximize portfolio Sharpe ratio

        Args:
            returns: DataFrame with strategy returns (columns = strategies)
            risk_free_rate: Risk-free rate (annual)
            constraints: Optional constraints (min_weight, max_weight per strategy)

        Returns:
            Dict mapping strategy name to weight
        """
        n_strategies = len(returns.columns)

        # Annualized mean returns and covariance
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        # Robust covariance estimation
        lw = LedoitWolf()
        cov_matrix = pd.DataFrame(
            lw.fit(returns).covariance_ * 252,
            index=returns.columns,
            columns=returns.columns
        )

        def negative_sharpe(weights):
            """Negative Sharpe ratio (for minimization)"""
            portfolio_return = np.dot(weights, mean_returns)
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            if portfolio_vol == 0:
                return -np.inf
            sharpe = (portfolio_return - risk_free_rate) / portfolio_vol
            return -sharpe

        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]

        # Bounds
        if constraints and 'min_weight' in constraints:
            min_weight = constraints['min_weight']
        else:
            min_weight = 0.0

        if constraints and 'max_weight' in constraints:
            max_weight = constraints['max_weight']
        else:
            max_weight = 0.5  # No strategy > 50%

        bounds = tuple((min_weight, max_weight) for _ in range(n_strategies))

        # Initial guess (equal weight)
        x0 = np.array([1.0 / n_strategies] * n_strategies)

        # Optimize
        result = optimize.minimize(
            negative_sharpe,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints_list
        )

        if not result.success:
            print(f"⚠️  Optimization failed: {result.message}")
            # Fall back to equal weight
            return {col: 1.0 / n_strategies for col in returns.columns}

        # Return as dictionary
        return {col: weight for col, weight in zip(returns.columns, result.x)}

    def optimize_min_variance(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Minimum variance portfolio"""
        n_strategies = len(returns.columns)
        cov_matrix = returns.cov() * 252

        # Robust covariance
        lw = LedoitWolf()
        cov_matrix = pd.DataFrame(
            lw.fit(returns).covariance_ * 252,
            index=returns.columns,
            columns=returns.columns
        )

        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0, 0.5) for _ in range(n_strategies))
        x0 = np.array([1.0 / n_strategies] * n_strategies)

        result = optimize.minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            return {col: 1.0 / n_strategies for col in returns.columns}

        return {col: weight for col, weight in zip(returns.columns, result.x)}

    def risk_parity(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Risk parity allocation (equal risk contribution)

        Each strategy contributes equally to portfolio risk
        """
        n_strategies = len(returns.columns)
        cov_matrix = returns.cov() * 252

        # Robust covariance
        lw = LedoitWolf()
        cov_matrix = pd.DataFrame(
            lw.fit(returns).covariance_ * 252,
            index=returns.columns,
            columns=returns.columns
        )

        def risk_parity_objective(weights):
            """Objective: minimize difference in risk contributions"""
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_contrib = np.dot(cov_matrix, weights)
            risk_contrib = weights * marginal_contrib / portfolio_vol

            # We want equal risk contributions
            target_risk = portfolio_vol / n_strategies
            return np.sum((risk_contrib - target_risk) ** 2)

        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        bounds = tuple((0.01, 0.5) for _ in range(n_strategies))
        x0 = np.array([1.0 / n_strategies] * n_strategies)

        result = optimize.minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        if not result.success:
            return {col: 1.0 / n_strategies for col in returns.columns}

        return {col: weight for col, weight in zip(returns.columns, result.x)}

    def regime_adaptive(self,
                       returns: pd.DataFrame,
                       regime: str,
                       strategy_metadata: Dict[str, StrategyMetadata]) -> Dict[str, float]:
        """
        Regime-adaptive allocation

        Adjusts weights based on current market regime and strategy characteristics
        """
        if regime in ['BULL', 'SIDEWAYS']:
            # Bull/Sideways: favor momentum strategies
            weights = {}
            total_sharpe = 0

            for strategy in returns.columns:
                metadata = strategy_metadata.get(strategy)
                if metadata and metadata.expected_sharpe:
                    # Momentum strategies get bonus in bull markets
                    bonus = 1.5 if 'momentum' in strategy.lower() else 1.0
                    weight = metadata.expected_sharpe * bonus
                    weights[strategy] = max(weight, 0)
                    total_sharpe += weights[strategy]
                else:
                    weights[strategy] = 1.0
                    total_sharpe += 1.0

            # Normalize
            return {s: w / total_sharpe for s, w in weights.items()}

        elif regime in ['BEAR', 'CORRECTION', 'CRISIS']:
            # Bear markets: favor defensive strategies, reduce leverage
            # Use minimum variance
            allocation = self.optimize_min_variance(returns)

            # Scale down to 50% in severe downturns
            if regime == 'CRISIS':
                return {s: w * 0.5 for s, w in allocation.items()}
            elif regime == 'BEAR':
                return {s: w * 0.7 for s, w in allocation.items()}
            else:
                return allocation

        else:
            # Unknown regime: equal weight
            return {col: 1.0 / len(returns.columns) for col in returns.columns}


class MultiStrategyPortfolioManager:
    """
    Multi-Strategy Portfolio Manager

    Manages portfolio of multiple strategies with:
    - Dynamic rebalancing
    - Correlation monitoring
    - Risk management
    - Performance tracking
    """

    def __init__(self,
                 db_path: str = 'dashboard/data/dashboard.db',
                 rebalance_frequency: str = 'monthly',
                 optimization_method: str = 'sharpe',
                 use_regime_adaptation: bool = True):
        """
        Initialize portfolio manager

        Args:
            db_path: Path to dashboard database
            rebalance_frequency: 'daily', 'weekly', 'monthly'
            optimization_method: 'equal', 'sharpe', 'min_variance', 'risk_parity', 'regime_adaptive'
            use_regime_adaptation: Use regime-based allocation
        """
        self.db_manager = DatabaseManager(db_path)
        self.registry = get_registry()
        self.optimizer = PortfolioOptimizer()

        self.rebalance_frequency = rebalance_frequency
        self.optimization_method = optimization_method
        self.use_regime_adaptation = use_regime_adaptation

        self.current_allocation: Optional[PortfolioAllocation] = None
        self.allocation_history: List[PortfolioAllocation] = []

    def get_strategy_returns(self,
                            strategies: List[str],
                            lookback_days: int = 252) -> pd.DataFrame:
        """
        Get returns for multiple strategies from database

        Returns: DataFrame with columns = strategy names, index = dates
        """
        session = self.db_manager.get_session()
        try:
            all_returns = {}

            for strategy in strategies:
                from dashboard.database import PerformanceMetric

                # Get returns for this strategy
                metrics = session.query(PerformanceMetric).filter(
                    PerformanceMetric.strategy_name == strategy
                ).order_by(PerformanceMetric.date.desc()).limit(lookback_days).all()

                if metrics:
                    dates = [m.date for m in reversed(metrics)]
                    returns = [m.daily_return for m in reversed(metrics)]
                    all_returns[strategy] = pd.Series(returns, index=dates)

            if not all_returns:
                return pd.DataFrame()

            return pd.DataFrame(all_returns)

        finally:
            session.close()

    def calculate_correlation_matrix(self, strategies: List[str]) -> pd.DataFrame:
        """Calculate correlation matrix between strategies"""
        returns = self.get_strategy_returns(strategies)

        if returns.empty:
            return pd.DataFrame()

        return returns.corr()

    def optimize_allocation(self,
                          strategies: List[str],
                          regime: Optional[str] = None) -> PortfolioAllocation:
        """
        Optimize portfolio allocation

        Args:
            strategies: List of strategy names to include
            regime: Current market regime (if regime_adaptive)

        Returns:
            PortfolioAllocation object
        """
        # Get returns
        returns = self.get_strategy_returns(strategies)

        if returns.empty:
            # Equal weight fallback
            weights = {s: 1.0 / len(strategies) for s in strategies}
            return PortfolioAllocation(
                timestamp=datetime.now(),
                allocations=weights,
                expected_return=0,
                expected_volatility=0,
                expected_sharpe=0,
                regime=regime or 'UNKNOWN',
                reason='No data available - equal weight'
            )

        # Get strategy metadata
        metadata = {s: self.registry.get(s) for s in strategies}

        # Choose optimization method
        if self.use_regime_adaptation and regime:
            method = 'regime_adaptive'
        else:
            method = self.optimization_method

        # Optimize
        if method == 'equal':
            weights = {s: 1.0 / len(strategies) for s in strategies}
            reason = 'Equal weight'

        elif method == 'sharpe':
            weights = self.optimizer.optimize_sharpe(returns)
            reason = 'Sharpe maximization'

        elif method == 'min_variance':
            weights = self.optimizer.optimize_min_variance(returns)
            reason = 'Minimum variance'

        elif method == 'risk_parity':
            weights = self.optimizer.risk_parity(returns)
            reason = 'Risk parity'

        elif method == 'regime_adaptive':
            weights = self.optimizer.regime_adaptive(returns, regime or 'SIDEWAYS', metadata)
            reason = f'Regime-adaptive ({regime})'

        else:
            weights = {s: 1.0 / len(strategies) for s in strategies}
            reason = 'Unknown method - equal weight'

        # Calculate portfolio metrics
        mean_returns = returns.mean() * 252
        cov_matrix = returns.cov() * 252

        weights_array = np.array([weights[s] for s in returns.columns])
        portfolio_return = np.dot(weights_array, mean_returns)
        portfolio_vol = np.sqrt(np.dot(weights_array, np.dot(cov_matrix, weights_array)))
        portfolio_sharpe = portfolio_return / portfolio_vol if portfolio_vol > 0 else 0

        allocation = PortfolioAllocation(
            timestamp=datetime.now(),
            allocations=weights,
            expected_return=portfolio_return,
            expected_volatility=portfolio_vol,
            expected_sharpe=portfolio_sharpe,
            regime=regime or 'UNKNOWN',
            reason=reason
        )

        self.current_allocation = allocation
        self.allocation_history.append(allocation)

        return allocation

    def get_diversification_ratio(self, strategies: List[str]) -> float:
        """
        Calculate diversification ratio

        DR = (sum of individual volatilities * weights) / portfolio volatility
        DR > 1 indicates diversification benefit
        """
        returns = self.get_strategy_returns(strategies)

        if returns.empty or self.current_allocation is None:
            return 1.0

        weights = self.current_allocation.allocations
        weights_array = np.array([weights.get(s, 0) for s in returns.columns])

        # Individual volatilities
        individual_vols = returns.std() * np.sqrt(252)
        weighted_sum_vols = np.dot(weights_array, individual_vols)

        # Portfolio volatility
        portfolio_vol = self.current_allocation.expected_volatility

        if portfolio_vol == 0:
            return 1.0

        return weighted_sum_vols / portfolio_vol

    def rebalance(self, strategies: List[str], regime: Optional[str] = None):
        """Rebalance portfolio"""
        print(f"\n{'='*80}")
        print(f"REBALANCING PORTFOLIO")
        print(f"{'='*80}\n")

        allocation = self.optimize_allocation(strategies, regime)

        print(f"Method: {allocation.reason}")
        print(f"Regime: {allocation.regime}")
        print(f"\nExpected Metrics:")
        print(f"  Return:     {allocation.expected_return*100:.2f}%")
        print(f"  Volatility: {allocation.expected_volatility*100:.2f}%")
        print(f"  Sharpe:     {allocation.expected_sharpe:.2f}")

        print(f"\nAllocations:")
        for strategy, weight in sorted(allocation.allocations.items(), key=lambda x: -x[1]):
            if weight > 0.01:  # Show only meaningful allocations
                print(f"  {strategy:30s}: {weight*100:5.1f}%")

        diversification = self.get_diversification_ratio(strategies)
        print(f"\nDiversification Ratio: {diversification:.2f}")
        print(f"{'='*80}\n")

        return allocation

    def get_portfolio_summary(self, strategies: List[str]) -> Dict:
        """Get comprehensive portfolio summary"""
        returns = self.get_strategy_returns(strategies)
        correlation = self.calculate_correlation_matrix(strategies)

        if self.current_allocation:
            allocation = self.current_allocation
        else:
            allocation = self.optimize_allocation(strategies)

        # Find highly correlated pairs
        high_correlation_pairs = []
        if not correlation.empty:
            for i in range(len(correlation.columns)):
                for j in range(i+1, len(correlation.columns)):
                    corr_val = correlation.iloc[i, j]
                    if abs(corr_val) > 0.7:
                        high_correlation_pairs.append({
                            'strategy1': correlation.columns[i],
                            'strategy2': correlation.columns[j],
                            'correlation': corr_val
                        })

        return {
            'allocation': allocation,
            'correlation_matrix': correlation,
            'diversification_ratio': self.get_diversification_ratio(strategies),
            'high_correlation_pairs': high_correlation_pairs,
            'n_strategies': len(strategies),
            'rebalance_frequency': self.rebalance_frequency,
            'optimization_method': self.optimization_method
        }


if __name__ == '__main__':
    """Test portfolio manager"""

    # Get registry
    registry = get_registry()

    # Get production strategies
    production_strategies = registry.get_production_strategies()
    strategy_names = [s.name for s in production_strategies if s.walk_forward_passed != False]

    print(f"Managing portfolio with {len(strategy_names)} strategies:")
    for name in strategy_names:
        print(f"  - {name}")

    # Create portfolio manager
    manager = MultiStrategyPortfolioManager(
        optimization_method='sharpe',
        use_regime_adaptation=True
    )

    # Rebalance for different regimes
    for regime in ['BULL', 'BEAR', 'SIDEWAYS']:
        manager.rebalance(strategy_names, regime=regime)
