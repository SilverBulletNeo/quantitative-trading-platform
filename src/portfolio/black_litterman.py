"""
Black-Litterman Model

Bayesian portfolio optimization combining market equilibrium with investor views.
Developed by Fischer Black and Robert Litterman at Goldman Sachs (1992).

Key Innovation:
- Mean-variance optimization requires expected returns → hard to estimate
- Black-Litterman starts with market equilibrium (CAPM)
- Allows investors to express views on specific assets
- Combines equilibrium + views using Bayesian statistics
- Results in more stable, diversified portfolios

Academic Foundation:
- Black & Litterman (1992) - "Global Portfolio Optimization"
- Satchell & Scowcroft (2000) - "A demystification of the Black-Litterman model"
- He & Litterman (1999) - "The Intuition Behind Black-Litterman Model Portfolios"

Process:
1. Calculate equilibrium returns (reverse-optimize from market caps)
2. Express views: "I believe asset X will return Y%"
3. Specify confidence in views
4. Combine equilibrium + views → posterior returns
5. Optimize portfolio using posterior returns

Why It Works:
- Avoids extreme portfolios from estimation errors
- Incorporates market wisdom (equilibrium)
- Allows tactical adjustments (views)
- Bayesian framework handles uncertainty
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


@dataclass
class BlackLittermanConfig:
    """Configuration for Black-Litterman Model"""
    risk_free_rate: float = 0.02  # Annual risk-free rate
    risk_aversion: float = 2.5  # Market risk aversion coefficient
    tau: float = 0.025  # Scaling factor for prior uncertainty
    max_position: float = 0.40  # Maximum 40% per asset
    min_position: float = 0.0  # Long-only by default


@dataclass
class View:
    """
    Investor view on asset returns

    Types:
    - Absolute: "Asset X will return 10%"
    - Relative: "Asset X will outperform Asset Y by 5%"
    """
    assets: List[str]  # Assets the view applies to
    returns: float  # Expected return (absolute or relative)
    confidence: float = 1.0  # Confidence level (0-1)
    view_type: str = 'absolute'  # 'absolute' or 'relative'


class BlackLitterman:
    """
    Black-Litterman Portfolio Optimization

    Combines market equilibrium with investor views.
    """

    def __init__(self, config: BlackLittermanConfig = None):
        self.config = config or BlackLittermanConfig()

    def calculate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate annualized covariance matrix"""
        cov_matrix = returns.cov() * 252
        return cov_matrix

    def calculate_market_caps(
        self,
        prices: pd.DataFrame,
        market_caps: Optional[pd.Series] = None
    ) -> pd.Series:
        """
        Calculate or use provided market capitalizations

        If not provided, use equal weights (naïve assumption)
        """
        if market_caps is not None:
            # Normalize to sum to 1
            return market_caps / market_caps.sum()
        else:
            # Equal market caps (naïve assumption)
            n = len(prices.columns)
            return pd.Series(1/n, index=prices.columns)

    def reverse_optimize_returns(
        self,
        cov_matrix: pd.DataFrame,
        market_weights: pd.Series
    ) -> pd.Series:
        """
        Calculate equilibrium returns (CAPM implied)

        Formula: Π = λ * Σ * w_mkt

        Where:
        - Π = Equilibrium returns
        - λ = Risk aversion coefficient
        - Σ = Covariance matrix
        - w_mkt = Market capitalization weights

        This is "reverse optimization" - we assume the market portfolio
        is optimal, and back out what returns would make it optimal.
        """
        risk_aversion = self.config.risk_aversion

        # Calculate equilibrium returns
        equilibrium_returns = risk_aversion * cov_matrix.dot(market_weights)

        return equilibrium_returns

    def build_view_matrices(
        self,
        views: List[View],
        assets: List[str]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build P (picking matrix), Q (view returns), and Ω (view uncertainty)

        P matrix:
        - Each row represents one view
        - Columns represent assets
        - For absolute view on asset i: P[i] = 1.0
        - For relative view (asset i vs j): P[i]=1.0, P[j]=-1.0

        Q vector:
        - Expected returns for each view

        Ω matrix:
        - Diagonal matrix of view uncertainties
        - Ω[i,i] = (1 - confidence) * view_variance
        """
        n_views = len(views)
        n_assets = len(assets)

        P = np.zeros((n_views, n_assets))
        Q = np.zeros(n_views)
        Omega = np.zeros((n_views, n_views))

        for i, view in enumerate(views):
            # Q vector (view returns)
            Q[i] = view.returns

            # P matrix (picking matrix)
            if view.view_type == 'absolute':
                # Absolute view: single asset
                asset_idx = assets.index(view.assets[0])
                P[i, asset_idx] = 1.0

            elif view.view_type == 'relative':
                # Relative view: asset A vs asset B
                asset_a_idx = assets.index(view.assets[0])
                asset_b_idx = assets.index(view.assets[1])
                P[i, asset_a_idx] = 1.0
                P[i, asset_b_idx] = -1.0

            # Omega matrix (view uncertainty)
            # Higher confidence → Lower uncertainty
            # Formula: variance * (1 - confidence)
            view_variance = 0.01  # Base variance (1% squared)
            Omega[i, i] = view_variance * (1 - view.confidence)

        return P, Q, Omega

    def combine_views_with_equilibrium(
        self,
        equilibrium_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        P: np.ndarray,
        Q: np.ndarray,
        Omega: np.ndarray
    ) -> pd.Series:
        """
        Calculate posterior returns (Bayesian combination)

        Formula:
        E[R] = [(τΣ)^-1 + P'Ω^-1P]^-1 [(τΣ)^-1Π + P'Ω^-1Q]

        Components:
        - Prior: τΣ (scaled covariance)
        - Views: P, Q, Ω
        - Posterior: Weighted average of prior and views

        Returns:
        - Expected returns incorporating both equilibrium and views
        """
        tau = self.config.tau

        # Convert to numpy arrays
        Pi = equilibrium_returns.values
        Sigma = cov_matrix.values

        # Calculate components
        tau_Sigma_inv = np.linalg.inv(tau * Sigma)
        Omega_inv = np.linalg.inv(Omega)

        # Posterior precision (inverse covariance)
        posterior_precision = tau_Sigma_inv + P.T @ Omega_inv @ P

        # Posterior covariance
        posterior_cov = np.linalg.inv(posterior_precision)

        # Posterior mean (expected returns)
        posterior_mean = posterior_cov @ (
            tau_Sigma_inv @ Pi + P.T @ Omega_inv @ Q
        )

        # Convert back to Series
        posterior_returns = pd.Series(posterior_mean, index=equilibrium_returns.index)

        return posterior_returns

    def optimize_portfolio(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> pd.Series:
        """
        Optimize portfolio using mean-variance optimization

        Maximize: w' * E[R] - λ/2 * w' * Σ * w

        Subject to:
        - Weights sum to 1
        - Position limits
        """
        n_assets = len(expected_returns)
        risk_aversion = self.config.risk_aversion

        def objective(weights):
            """Minimize negative utility"""
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
            utility = portfolio_return - (risk_aversion / 2) * portfolio_var
            return -utility  # Minimize negative = maximize utility

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]

        # Bounds
        bounds = tuple(
            (self.config.min_position, self.config.max_position)
            for _ in range(n_assets)
        )

        # Initial guess
        init_weights = np.array([1/n_assets] * n_assets)

        # Optimize
        result = minimize(
            objective,
            init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if not result.success:
            print(f"Warning: Optimization did not converge - {result.message}")

        weights = pd.Series(result.x, index=expected_returns.index)

        return weights

    def run_black_litterman(
        self,
        prices: pd.DataFrame,
        views: List[View],
        market_caps: Optional[pd.Series] = None
    ) -> Dict:
        """
        Main Black-Litterman function

        Args:
            prices: Historical prices
            views: List of investor views
            market_caps: Market capitalizations (optional)

        Returns:
            Dict with weights, returns, and intermediate results
        """
        # Calculate returns and covariance
        returns = prices.pct_change().dropna()
        cov_matrix = self.calculate_covariance(returns)

        # 1. Calculate market cap weights
        market_weights = self.calculate_market_caps(prices, market_caps)

        # 2. Reverse-optimize to get equilibrium returns
        equilibrium_returns = self.reverse_optimize_returns(cov_matrix, market_weights)

        # 3. Build view matrices
        assets = list(prices.columns)
        P, Q, Omega = self.build_view_matrices(views, assets)

        # 4. Combine views with equilibrium (Bayesian update)
        posterior_returns = self.combine_views_with_equilibrium(
            equilibrium_returns,
            cov_matrix,
            P, Q, Omega
        )

        # 5. Optimize portfolio using posterior returns
        weights = self.optimize_portfolio(posterior_returns, cov_matrix)

        # Calculate portfolio metrics
        portfolio_return = np.dot(weights, posterior_returns)
        portfolio_var = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_var)
        sharpe = (portfolio_return - self.config.risk_free_rate) / portfolio_vol

        return {
            'weights': weights,
            'equilibrium_returns': equilibrium_returns,
            'posterior_returns': posterior_returns,
            'market_weights': market_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
            'P': P,
            'Q': Q,
            'Omega': Omega
        }


if __name__ == "__main__":
    """Test Black-Litterman Model"""

    print("=" * 80)
    print("BLACK-LITTERMAN MODEL TEST")
    print("=" * 80)
    print()

    # Create synthetic data
    print("Creating synthetic test data...")
    print("- 5 assets with different risk/return profiles")
    print()

    np.random.seed(42)
    n_days = 252 * 3

    # Generate returns
    returns_data = {
        'US Equity': np.random.normal(0.0008, 0.015, n_days),
        'Intl Equity': np.random.normal(0.0006, 0.013, n_days),
        'Bonds': np.random.normal(0.0002, 0.005, n_days),
        'Gold': np.random.normal(0.0004, 0.012, n_days),
        'Crypto': np.random.normal(0.0015, 0.030, n_days)
    }

    # Add some correlation
    returns_data['Intl Equity'] = 0.7 * returns_data['US Equity'] + 0.3 * returns_data['Intl Equity']

    returns_df = pd.DataFrame(returns_data)
    prices = (1 + returns_df).cumprod() * 100

    # Market capitalizations (representing global market)
    market_caps = pd.Series({
        'US Equity': 0.40,  # 40% of global market
        'Intl Equity': 0.30,  # 30%
        'Bonds': 0.20,  # 20%
        'Gold': 0.05,  # 5%
        'Crypto': 0.05  # 5%
    })

    # Initialize Black-Litterman
    bl = BlackLitterman()

    # Scenario 1: No views (pure equilibrium)
    print("=" * 80)
    print("SCENARIO 1: NO VIEWS (Market Equilibrium)")
    print("=" * 80)
    print()

    result_no_views = bl.run_black_litterman(prices, [], market_caps)

    print("Market Weights:")
    for asset, weight in result_no_views['market_weights'].items():
        print(f"  {asset:15s}: {weight:>6.2%}")

    print("\nEquilibrium Returns (annualized):")
    for asset, ret in result_no_views['equilibrium_returns'].items():
        print(f"  {asset:15s}: {ret:>6.2%}")

    print("\nOptimal Weights (no views):")
    for asset, weight in result_no_views['weights'].sort_values(ascending=False).items():
        print(f"  {asset:15s}: {weight:>6.2%}")

    print(f"\nExpected Return: {result_no_views['expected_return']:.2%}")
    print(f"Volatility: {result_no_views['volatility']:.2%}")
    print(f"Sharpe Ratio: {result_no_views['sharpe_ratio']:.2f}")

    # Scenario 2: Bullish on Crypto
    print("\n" + "=" * 80)
    print("SCENARIO 2: BULLISH VIEW ON CRYPTO")
    print("=" * 80)
    print()

    views_bullish = [
        View(
            assets=['Crypto'],
            returns=0.50,  # 50% annual return
            confidence=0.8,  # 80% confident
            view_type='absolute'
        )
    ]

    print("View: 'Crypto will return 50% annually (80% confident)'")
    print()

    result_bullish = bl.run_black_litterman(prices, views_bullish, market_caps)

    print("Posterior Returns (after incorporating view):")
    for asset, ret in result_bullish['posterior_returns'].items():
        change = ret - result_no_views['equilibrium_returns'][asset]
        print(f"  {asset:15s}: {ret:>6.2%}  (change: {change:+.2%})")

    print("\nOptimal Weights:")
    for asset, weight in result_bullish['weights'].sort_values(ascending=False).items():
        change = weight - result_no_views['weights'][asset]
        print(f"  {asset:15s}: {weight:>6.2%}  (change: {change:+.2%})")

    print(f"\nExpected Return: {result_bullish['expected_return']:.2%}")
    print(f"Volatility: {result_bullish['volatility']:.2%}")
    print(f"Sharpe Ratio: {result_bullish['sharpe_ratio']:.2f}")

    # Scenario 3: Relative view (US vs Intl Equity)
    print("\n" + "=" * 80)
    print("SCENARIO 3: RELATIVE VIEW (US vs International Equity)")
    print("=" * 80)
    print()

    views_relative = [
        View(
            assets=['US Equity', 'Intl Equity'],
            returns=0.05,  # US will outperform Intl by 5%
            confidence=0.6,  # 60% confident
            view_type='relative'
        )
    ]

    print("View: 'US Equity will outperform Intl Equity by 5% (60% confident)'")
    print()

    result_relative = bl.run_black_litterman(prices, views_relative, market_caps)

    print("Optimal Weights:")
    for asset, weight in result_relative['weights'].sort_values(ascending=False).items():
        change = weight - result_no_views['weights'][asset]
        print(f"  {asset:15s}: {weight:>6.2%}  (change: {change:+.2%})")

    # Scenario 4: Multiple views
    print("\n" + "=" * 80)
    print("SCENARIO 4: MULTIPLE VIEWS")
    print("=" * 80)
    print()

    views_multiple = [
        View(
            assets=['Crypto'],
            returns=0.30,
            confidence=0.7,
            view_type='absolute'
        ),
        View(
            assets=['US Equity', 'Intl Equity'],
            returns=0.03,
            confidence=0.5,
            view_type='relative'
        ),
        View(
            assets=['Bonds'],
            returns=0.04,
            confidence=0.9,
            view_type='absolute'
        )
    ]

    print("Views:")
    print("  1. Crypto will return 30% (70% confident)")
    print("  2. US Equity will outperform Intl by 3% (50% confident)")
    print("  3. Bonds will return 4% (90% confident)")
    print()

    result_multiple = bl.run_black_litterman(prices, views_multiple, market_caps)

    print("Optimal Weights:")
    for asset, weight in result_multiple['weights'].sort_values(ascending=False).items():
        change = weight - result_no_views['weights'][asset]
        print(f"  {asset:15s}: {weight:>6.2%}  (change: {change:+.2%})")

    print(f"\nExpected Return: {result_multiple['expected_return']:.2%}")
    print(f"Volatility: {result_multiple['volatility']:.2%}")
    print(f"Sharpe Ratio: {result_multiple['sharpe_ratio']:.2f}")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("Black-Litterman Framework:")
    print("  - Starts with market equilibrium (stable starting point)")
    print("  - Allows tactical adjustments via views")
    print("  - Confidence levels determine how much views impact allocation")
    print("  - Bayesian framework handles uncertainty gracefully")
    print()
    print("vs. Traditional Mean-Variance:")
    print("  - Mean-Variance: Very sensitive to return estimates")
    print("  - Black-Litterman: Anchored to market equilibrium")
    print("  - Result: More stable, diversified portfolios")
    print()
    print("=" * 80)
    print("Test complete!")
    print("=" * 80)
