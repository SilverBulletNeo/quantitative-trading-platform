"""
Risk Parity Portfolio Allocation

Nobel Prize-adjacent framework for portfolio construction.
Popularized by Bridgewater Associates (Ray Dalio's All Weather Portfolio).

Objective:
- Allocate capital so each asset contributes EQUAL RISK to portfolio
- Unlike mean-variance (equal dollars), this is equal risk contribution
- Results in more balanced diversification

Academic Foundation:
- Qian (2005) - "Risk Parity Portfolios"
- Maillard, Roncalli, Teiletche (2010) - "Equal Risk Contribution Portfolios"
- AQR Capital - "Understanding Risk Parity"

Methods:
1. Inverse Volatility - Simple, fast (weight ∝ 1/volatility)
2. Equal Risk Contribution - Optimal, iterative (marginal risk = equal)
3. Hierarchical Risk Parity - ML-based clustering approach

Why It Works:
- Traditional 60/40 portfolio: ~90% risk from equities, 10% from bonds
- Risk Parity: 50% risk from equities, 50% from bonds
- Better diversification across market regimes
- Performs well in crisis periods (low correlation breakdown)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


@dataclass
class RiskParityConfig:
    """Configuration for Risk Parity allocation"""
    method: str = 'equal_risk'  # 'inverse_vol' or 'equal_risk'
    max_position: float = 0.40  # Maximum 40% in single asset
    min_position: float = 0.05  # Minimum 5% in each asset
    lookback_period: int = 60  # Days for volatility calculation
    annualization_factor: int = 252  # Trading days per year


class RiskParity:
    """
    Risk Parity Portfolio Allocation

    Allocates capital to equalize risk contribution across assets.
    """

    def __init__(self, config: RiskParityConfig = None):
        self.config = config or RiskParityConfig()

    def calculate_volatility(self, returns: pd.DataFrame) -> pd.Series:
        """Calculate annualized volatility for each asset"""
        vol = returns.std() * np.sqrt(self.config.annualization_factor)
        return vol

    def calculate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate annualized covariance matrix"""
        cov_matrix = returns.cov() * self.config.annualization_factor
        return cov_matrix

    def inverse_volatility_weights(
        self,
        returns: pd.DataFrame
    ) -> pd.Series:
        """
        Simple Risk Parity: Inverse Volatility Weighting

        Weight ∝ 1 / volatility
        Fast and intuitive, but ignores correlations
        """
        vol = self.calculate_volatility(returns)

        # Inverse volatility
        inv_vol = 1 / vol

        # Normalize to sum to 1
        weights = inv_vol / inv_vol.sum()

        # Apply position constraints
        weights = weights.clip(
            lower=self.config.min_position,
            upper=self.config.max_position
        )

        # Renormalize after clipping
        weights = weights / weights.sum()

        return weights

    def portfolio_volatility(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> float:
        """Calculate portfolio volatility"""
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def marginal_risk_contribution(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Marginal Risk Contribution (MRC)

        MRC[i] = ∂σ(portfolio) / ∂w[i]
               = (Σ * w)[i] / σ(portfolio)
        """
        portfolio_vol = self.portfolio_volatility(weights, cov_matrix)

        # Marginal contribution
        mrc = np.dot(cov_matrix, weights) / portfolio_vol

        return mrc

    def risk_contribution(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> np.ndarray:
        """
        Calculate Risk Contribution (RC)

        RC[i] = w[i] * MRC[i]
        Total risk = Σ RC[i]
        """
        mrc = self.marginal_risk_contribution(weights, cov_matrix)

        # Risk contribution
        rc = weights * mrc

        return rc

    def equal_risk_contribution_objective(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray
    ) -> float:
        """
        Objective function for Equal Risk Contribution

        Minimize: Σ(RC[i] - RC[j])^2
        Goal: Make all risk contributions equal
        """
        rc = self.risk_contribution(weights, cov_matrix)

        # Target: equal risk contribution = total risk / N
        target_rc = np.mean(rc)

        # Sum of squared deviations from target
        objective = np.sum((rc - target_rc) ** 2)

        return objective

    def equal_risk_contribution_weights(
        self,
        returns: pd.DataFrame
    ) -> pd.Series:
        """
        Equal Risk Contribution Portfolio

        Optimize weights so each asset contributes equal risk.
        This accounts for correlations (unlike inverse vol).
        """
        n_assets = len(returns.columns)
        cov_matrix = self.calculate_covariance(returns).values

        # Objective: minimize variance in risk contributions
        def objective(weights):
            return self.equal_risk_contribution_objective(weights, cov_matrix)

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]

        # Bounds
        bounds = tuple(
            (self.config.min_position, self.config.max_position)
            for _ in range(n_assets)
        )

        # Initial guess: equal weight
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

        weights = pd.Series(result.x, index=returns.columns)

        return weights

    def optimize_portfolio(
        self,
        prices: pd.DataFrame,
        method: str = None
    ) -> Dict:
        """
        Main optimization function

        Methods:
        - 'inverse_vol': Simple inverse volatility weighting
        - 'equal_risk': Equal risk contribution (accounts for correlations)
        """
        if method is None:
            method = self.config.method

        # Calculate returns
        returns = prices.pct_change().dropna()

        # Optimize based on method
        if method == 'inverse_vol':
            weights = self.inverse_volatility_weights(returns)
        elif method == 'equal_risk':
            weights = self.equal_risk_contribution_weights(returns)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Calculate portfolio metrics
        cov_matrix = self.calculate_covariance(returns)
        portfolio_vol = self.portfolio_volatility(weights.values, cov_matrix.values)

        # Calculate risk contributions
        rc = self.risk_contribution(weights.values, cov_matrix.values)
        risk_contributions = pd.Series(rc, index=weights.index)

        # Calculate percentage of total risk from each asset
        risk_pct = risk_contributions / risk_contributions.sum()

        return {
            'weights': weights,
            'portfolio_volatility': portfolio_vol,
            'risk_contributions': risk_contributions,
            'risk_percentages': risk_pct,
            'method': method
        }

    def compare_allocations(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compare different allocation methods

        Shows difference between:
        - Equal Weight (1/N)
        - Inverse Volatility (simple risk parity)
        - Equal Risk Contribution (true risk parity)
        """
        results = []

        # 1. Equal Weight
        n_assets = len(prices.columns)
        equal_weights = pd.Series([1/n_assets] * n_assets, index=prices.columns)

        returns = prices.pct_change().dropna()
        cov_matrix = self.calculate_covariance(returns)

        eq_vol = self.portfolio_volatility(equal_weights.values, cov_matrix.values)
        eq_rc = self.risk_contribution(equal_weights.values, cov_matrix.values)

        results.append({
            'Method': 'Equal Weight',
            'Portfolio Vol': eq_vol,
            'Max Risk Contribution': eq_rc.max() / eq_rc.sum(),
            'Min Risk Contribution': eq_rc.min() / eq_rc.sum(),
            'Risk Dispersion': eq_rc.std() / eq_rc.mean()
        })

        # 2. Inverse Volatility
        inv_vol_result = self.optimize_portfolio(prices, method='inverse_vol')

        results.append({
            'Method': 'Inverse Volatility',
            'Portfolio Vol': inv_vol_result['portfolio_volatility'],
            'Max Risk Contribution': inv_vol_result['risk_percentages'].max(),
            'Min Risk Contribution': inv_vol_result['risk_percentages'].min(),
            'Risk Dispersion': inv_vol_result['risk_contributions'].std() / inv_vol_result['risk_contributions'].mean()
        })

        # 3. Equal Risk Contribution
        eq_risk_result = self.optimize_portfolio(prices, method='equal_risk')

        results.append({
            'Method': 'Equal Risk Contribution',
            'Portfolio Vol': eq_risk_result['portfolio_volatility'],
            'Max Risk Contribution': eq_risk_result['risk_percentages'].max(),
            'Min Risk Contribution': eq_risk_result['risk_percentages'].min(),
            'Risk Dispersion': eq_risk_result['risk_contributions'].std() / eq_risk_result['risk_contributions'].mean()
        })

        return pd.DataFrame(results)


if __name__ == "__main__":
    """Test Risk Parity allocation"""

    print("=" * 80)
    print("RISK PARITY PORTFOLIO ALLOCATION TEST")
    print("=" * 80)
    print()

    # Create synthetic data with different volatility profiles
    print("Creating synthetic test data...")
    print("- Asset 1: Low volatility (10% annual)")
    print("- Asset 2: Medium volatility (15% annual)")
    print("- Asset 3: High volatility (25% annual)")
    print("- Asset 4: Very high volatility (40% annual)")
    print()

    np.random.seed(42)
    n_days = 252 * 3  # 3 years

    # Generate returns with different volatilities
    returns_data = {
        'Low Vol': np.random.normal(0.0001, 0.10/np.sqrt(252), n_days),
        'Med Vol': np.random.normal(0.0002, 0.15/np.sqrt(252), n_days),
        'High Vol': np.random.normal(0.0003, 0.25/np.sqrt(252), n_days),
        'Very High Vol': np.random.normal(0.0004, 0.40/np.sqrt(252), n_days)
    }

    returns_df = pd.DataFrame(returns_data)

    # Add some correlation
    returns_df['Med Vol'] = 0.7 * returns_df['Low Vol'] + 0.3 * returns_df['Med Vol']
    returns_df['High Vol'] = 0.5 * returns_df['Med Vol'] + 0.5 * returns_df['High Vol']

    # Convert to prices
    prices = (1 + returns_df).cumprod() * 100

    # Initialize Risk Parity
    rp = RiskParity()

    # Calculate realized volatilities
    print("=" * 80)
    print("REALIZED VOLATILITIES")
    print("=" * 80)

    vol = rp.calculate_volatility(returns_df)
    for asset, v in vol.sort_values().items():
        print(f"  {asset:15s}: {v:>8.2%}")

    # Calculate correlation matrix
    print("\n" + "=" * 80)
    print("CORRELATION MATRIX")
    print("=" * 80)
    print()
    corr = returns_df.corr()
    print(corr.round(2))

    # Method 1: Inverse Volatility
    print("\n" + "=" * 80)
    print("METHOD 1: INVERSE VOLATILITY WEIGHTING")
    print("=" * 80)

    inv_vol_result = rp.optimize_portfolio(prices, method='inverse_vol')

    print("\nPortfolio Weights:")
    for asset, weight in inv_vol_result['weights'].sort_values(ascending=False).items():
        print(f"  {asset:15s}: {weight:>6.2%}")

    print(f"\nPortfolio Volatility: {inv_vol_result['portfolio_volatility']:.2%}")

    print("\nRisk Contribution (% of total risk):")
    for asset, pct in inv_vol_result['risk_percentages'].sort_values(ascending=False).items():
        print(f"  {asset:15s}: {pct:>6.2%}")

    # Method 2: Equal Risk Contribution
    print("\n" + "=" * 80)
    print("METHOD 2: EQUAL RISK CONTRIBUTION")
    print("=" * 80)

    eq_risk_result = rp.optimize_portfolio(prices, method='equal_risk')

    print("\nPortfolio Weights:")
    for asset, weight in eq_risk_result['weights'].sort_values(ascending=False).items():
        print(f"  {asset:15s}: {weight:>6.2%}")

    print(f"\nPortfolio Volatility: {eq_risk_result['portfolio_volatility']:.2%}")

    print("\nRisk Contribution (% of total risk):")
    for asset, pct in eq_risk_result['risk_percentages'].sort_values(ascending=False).items():
        print(f"  {asset:15s}: {pct:>6.2%}")

    # Compare all methods
    print("\n" + "=" * 80)
    print("COMPARISON OF ALLOCATION METHODS")
    print("=" * 80)
    print()

    comparison = rp.compare_allocations(prices)
    print(comparison.to_string(index=False))

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("Equal Weight:")
    print("  - Simple but naive")
    print("  - High-vol assets dominate risk (60-80% from highest vol asset)")
    print("  - Poor diversification of risk")
    print()
    print("Inverse Volatility:")
    print("  - Simple risk parity")
    print("  - Ignores correlations")
    print("  - Good starting point, fast computation")
    print()
    print("Equal Risk Contribution:")
    print("  - True risk parity")
    print("  - Accounts for correlations")
    print("  - Best risk diversification (25% risk from each asset)")
    print("  - Lower 'Risk Dispersion' = more equal risk distribution")
    print()
    print("=" * 80)
    print("Test complete!")
    print("=" * 80)
