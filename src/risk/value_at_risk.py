"""
Value at Risk (VaR) Calculations

VaR answers: "What is the maximum loss over a given time period with X% confidence?"

Example: "95% VaR of $1M means we are 95% confident we won't lose more than $1M tomorrow"

Regulatory Importance:
- Required by Basel Committee for banks
- Used by hedge funds, pension funds, insurers
- Risk reporting to investors and regulators
- Position sizing and leverage decisions

Academic Foundation:
- J.P. Morgan (1994) - RiskMetrics methodology
- Jorion (2006) - Value at Risk: The New Benchmark for Managing Financial Risk
- Basel Committee (1996) - Amendment to Capital Accord to Incorporate Market Risks

Methods Implemented:
1. Historical VaR - Non-parametric (empirical distribution)
2. Parametric VaR - Assumes normal distribution (variance-covariance)
3. Monte Carlo VaR - Simulation-based
4. Conditional VaR (CVaR) - Expected Shortfall beyond VaR

Confidence Levels:
- 95% - Standard for internal risk management
- 99% - Standard for regulatory capital (Basel)
- 99.9% - Extreme tail risk

Time Horizons:
- 1 day - Trading desks
- 10 days - Basel regulatory requirement
- 1 month - Strategic portfolios
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


@dataclass
class VaRConfig:
    """Configuration for VaR calculations"""
    confidence_level: float = 0.95  # 95% confidence
    time_horizon: int = 1  # 1-day VaR
    portfolio_value: float = 1_000_000  # $1M portfolio
    lookback_period: int = 252  # 1 year of data
    n_simulations: int = 10_000  # Monte Carlo simulations


class ValueAtRisk:
    """
    Value at Risk (VaR) Calculator

    Implements all major VaR methodologies.
    """

    def __init__(self, config: VaRConfig = None):
        self.config = config or VaRConfig()

    def calculate_portfolio_returns(
        self,
        prices: pd.DataFrame,
        weights: pd.Series
    ) -> pd.Series:
        """
        Calculate portfolio returns from asset prices and weights

        Args:
            prices: DataFrame of asset prices
            weights: Series of portfolio weights

        Returns:
            Series of portfolio returns
        """
        # Calculate asset returns
        asset_returns = prices.pct_change().dropna()

        # Ensure weights match returns columns
        common_assets = asset_returns.columns.intersection(weights.index)
        asset_returns = asset_returns[common_assets]
        weights = weights[common_assets]

        # Normalize weights
        weights = weights / weights.sum()

        # Calculate portfolio returns
        portfolio_returns = (asset_returns * weights).sum(axis=1)

        return portfolio_returns

    def historical_var(
        self,
        returns: pd.Series,
        confidence_level: float = None
    ) -> Dict:
        """
        Historical VaR (Non-parametric)

        Uses empirical distribution of returns.
        No assumptions about distribution shape.

        Steps:
        1. Take historical returns
        2. Sort from worst to best
        3. Find percentile corresponding to confidence level

        Advantages:
        - No distribution assumptions
        - Captures fat tails and skewness
        - Easy to understand

        Disadvantages:
        - Limited by historical data
        - Assumes future like past
        - Requires large dataset
        """
        if confidence_level is None:
            confidence_level = self.config.confidence_level

        # Get recent returns (lookback period)
        recent_returns = returns.iloc[-self.config.lookback_period:]

        # Calculate VaR as percentile of distribution
        # For 95% VaR, we want 5th percentile (worst 5%)
        alpha = 1 - confidence_level
        var_return = np.percentile(recent_returns, alpha * 100)

        # Scale to time horizon (sqrt(T) rule)
        var_return_scaled = var_return * np.sqrt(self.config.time_horizon)

        # Convert to dollar amount
        var_dollar = abs(var_return_scaled * self.config.portfolio_value)

        # Calculate CVaR (Conditional VaR / Expected Shortfall)
        # Average of all returns worse than VaR
        cvar_return = recent_returns[recent_returns <= var_return].mean()
        cvar_return_scaled = cvar_return * np.sqrt(self.config.time_horizon)
        cvar_dollar = abs(cvar_return_scaled * self.config.portfolio_value)

        return {
            'var_return': var_return_scaled,
            'var_dollar': var_dollar,
            'cvar_return': cvar_return_scaled,
            'cvar_dollar': cvar_dollar,
            'method': 'historical',
            'confidence_level': confidence_level,
            'observations': len(recent_returns)
        }

    def parametric_var(
        self,
        returns: pd.Series,
        confidence_level: float = None
    ) -> Dict:
        """
        Parametric VaR (Variance-Covariance Method)

        Assumes returns are normally distributed.

        Formula: VaR = μ + σ * Z_α

        Where:
        - μ = Mean return
        - σ = Standard deviation
        - Z_α = Z-score for confidence level (e.g., -1.65 for 95%)

        Advantages:
        - Fast computation
        - Only needs mean and std dev
        - Smooth estimates

        Disadvantages:
        - Assumes normality (fat tails underestimated)
        - Underestimates risk in crisis periods
        """
        if confidence_level is None:
            confidence_level = self.config.confidence_level

        # Get recent returns
        recent_returns = returns.iloc[-self.config.lookback_period:]

        # Calculate mean and std dev
        mu = recent_returns.mean()
        sigma = recent_returns.std()

        # Z-score for confidence level
        # For 95%, alpha = 0.05, Z = -1.65
        alpha = 1 - confidence_level
        z_score = stats.norm.ppf(alpha)

        # VaR formula
        var_return = mu + sigma * z_score

        # Scale to time horizon
        var_return_scaled = var_return * np.sqrt(self.config.time_horizon)

        # Convert to dollars
        var_dollar = abs(var_return_scaled * self.config.portfolio_value)

        # Calculate CVaR for normal distribution
        # CVaR = μ + σ * φ(Z_α) / α
        # Where φ is the standard normal PDF
        phi_z = stats.norm.pdf(z_score)
        cvar_return = mu + sigma * (phi_z / alpha)
        cvar_return_scaled = cvar_return * np.sqrt(self.config.time_horizon)
        cvar_dollar = abs(cvar_return_scaled * self.config.portfolio_value)

        return {
            'var_return': var_return_scaled,
            'var_dollar': var_dollar,
            'cvar_return': cvar_return_scaled,
            'cvar_dollar': cvar_dollar,
            'method': 'parametric',
            'confidence_level': confidence_level,
            'mean': mu,
            'std_dev': sigma,
            'z_score': z_score
        }

    def monte_carlo_var(
        self,
        returns: pd.Series,
        confidence_level: float = None,
        n_simulations: int = None
    ) -> Dict:
        """
        Monte Carlo VaR (Simulation-based)

        Simulates thousands of possible future scenarios.

        Steps:
        1. Estimate return distribution parameters
        2. Generate random scenarios from distribution
        3. Calculate VaR from simulated returns

        Advantages:
        - Can handle complex portfolios
        - Can incorporate non-linear relationships
        - Flexible distribution assumptions

        Disadvantages:
        - Computationally intensive
        - Results vary (random)
        - Requires careful parameterization
        """
        if confidence_level is None:
            confidence_level = self.config.confidence_level

        if n_simulations is None:
            n_simulations = self.config.n_simulations

        # Get recent returns
        recent_returns = returns.iloc[-self.config.lookback_period:]

        # Estimate parameters (using t-distribution for fat tails)
        mu = recent_returns.mean()
        sigma = recent_returns.std()

        # Fit t-distribution (captures fat tails better than normal)
        params = stats.t.fit(recent_returns)
        df, loc, scale = params

        # Generate simulations
        np.random.seed(42)  # For reproducibility
        simulated_returns = stats.t.rvs(
            df=df,
            loc=loc,
            scale=scale,
            size=n_simulations
        )

        # Scale to time horizon
        simulated_returns_scaled = simulated_returns * np.sqrt(self.config.time_horizon)

        # Calculate VaR as percentile
        alpha = 1 - confidence_level
        var_return = np.percentile(simulated_returns_scaled, alpha * 100)
        var_dollar = abs(var_return * self.config.portfolio_value)

        # Calculate CVaR
        cvar_return = simulated_returns_scaled[simulated_returns_scaled <= var_return].mean()
        cvar_dollar = abs(cvar_return * self.config.portfolio_value)

        return {
            'var_return': var_return,
            'var_dollar': var_dollar,
            'cvar_return': cvar_return,
            'cvar_dollar': cvar_dollar,
            'method': 'monte_carlo',
            'confidence_level': confidence_level,
            'n_simulations': n_simulations,
            't_distribution_df': df
        }

    def calculate_all_methods(
        self,
        returns: pd.Series,
        confidence_level: float = None
    ) -> pd.DataFrame:
        """
        Calculate VaR using all three methods

        Allows comparison of methods
        """
        if confidence_level is None:
            confidence_level = self.config.confidence_level

        results = []

        # Historical VaR
        hist_result = self.historical_var(returns, confidence_level)
        results.append({
            'Method': 'Historical',
            'VaR ($)': hist_result['var_dollar'],
            'VaR (%)': hist_result['var_return'] * 100,
            'CVaR ($)': hist_result['cvar_dollar'],
            'CVaR (%)': hist_result['cvar_return'] * 100
        })

        # Parametric VaR
        param_result = self.parametric_var(returns, confidence_level)
        results.append({
            'Method': 'Parametric',
            'VaR ($)': param_result['var_dollar'],
            'VaR (%)': param_result['var_return'] * 100,
            'CVaR ($)': param_result['cvar_dollar'],
            'CVaR (%)': param_result['cvar_return'] * 100
        })

        # Monte Carlo VaR
        mc_result = self.monte_carlo_var(returns, confidence_level)
        results.append({
            'Method': 'Monte Carlo',
            'VaR ($)': mc_result['var_dollar'],
            'VaR (%)': mc_result['var_return'] * 100,
            'CVaR ($)': mc_result['cvar_dollar'],
            'CVaR (%)': mc_result['cvar_return'] * 100
        })

        return pd.DataFrame(results)

    def backtest_var(
        self,
        returns: pd.Series,
        var_method: str = 'historical'
    ) -> Dict:
        """
        Backtest VaR model

        Tests if actual losses exceed VaR at expected frequency.
        For 95% VaR, we expect 5% of days to exceed VaR.

        Basel traffic light system:
        - Green zone: 0-4 exceptions (model is accurate)
        - Yellow zone: 5-9 exceptions (watch closely)
        - Red zone: 10+ exceptions (model inadequate)
        """
        # Calculate rolling VaR
        var_breaches = 0
        total_days = 0

        # Use expanding window for backtesting
        for i in range(self.config.lookback_period, len(returns)):
            # Historical returns up to day i
            hist_returns = returns.iloc[:i]

            # Calculate VaR
            if var_method == 'historical':
                var_result = self.historical_var(hist_returns)
            elif var_method == 'parametric':
                var_result = self.parametric_var(hist_returns)
            elif var_method == 'monte_carlo':
                var_result = self.monte_carlo_var(hist_returns)
            else:
                raise ValueError(f"Unknown method: {var_method}")

            # Actual return on day i
            actual_return = returns.iloc[i]

            # Check if VaR was breached
            if actual_return < var_result['var_return']:
                var_breaches += 1

            total_days += 1

        # Calculate breach rate
        breach_rate = var_breaches / total_days if total_days > 0 else 0

        # Expected breach rate
        expected_breach_rate = 1 - self.config.confidence_level

        # Basel zone
        if var_breaches <= 4:
            basel_zone = 'green'
        elif var_breaches <= 9:
            basel_zone = 'yellow'
        else:
            basel_zone = 'red'

        return {
            'var_breaches': var_breaches,
            'total_days': total_days,
            'breach_rate': breach_rate,
            'expected_breach_rate': expected_breach_rate,
            'basel_zone': basel_zone,
            'method': var_method
        }


if __name__ == "__main__":
    """Test VaR calculations"""

    print("=" * 80)
    print("VALUE AT RISK (VaR) CALCULATIONS TEST")
    print("=" * 80)
    print()

    # Create synthetic portfolio returns
    print("Creating synthetic test data...")
    print("- Portfolio: 60% Equity / 40% Bonds")
    print("- Historical period: 3 years")
    print()

    np.random.seed(42)
    n_days = 252 * 3

    # Generate equity and bond returns
    equity_returns = np.random.normal(0.0008, 0.015, n_days)  # Higher vol
    bond_returns = np.random.normal(0.0003, 0.005, n_days)    # Lower vol

    # Portfolio returns (60/40)
    portfolio_returns = 0.6 * equity_returns + 0.4 * bond_returns
    portfolio_returns = pd.Series(portfolio_returns)

    # Initialize VaR calculator
    config = VaRConfig(
        confidence_level=0.95,
        time_horizon=1,
        portfolio_value=1_000_000,
        lookback_period=252
    )
    var_calc = ValueAtRisk(config)

    # Test all three methods
    print("=" * 80)
    print("VAR CALCULATIONS (95% Confidence, 1-Day Horizon, $1M Portfolio)")
    print("=" * 80)
    print()

    comparison = var_calc.calculate_all_methods(portfolio_returns, confidence_level=0.95)
    print(comparison.to_string(index=False))

    print("\n" + "=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    print("VaR = 'We are 95% confident we won't lose more than $X tomorrow'")
    print()
    print(f"Historical VaR: ${comparison.iloc[0]['VaR ($)']:,.0f}")
    print("  - Uses actual historical distribution")
    print("  - No assumptions about distribution shape")
    print()
    print(f"Parametric VaR: ${comparison.iloc[1]['VaR ($)']:,.0f}")
    print("  - Assumes normal distribution")
    print("  - Fast but may underestimate tail risk")
    print()
    print(f"Monte Carlo VaR: ${comparison.iloc[2]['VaR ($)']:,.0f}")
    print("  - Simulates 10,000 scenarios")
    print("  - Uses t-distribution (fat tails)")
    print()

    # CVaR explanation
    print("=" * 80)
    print("CONDITIONAL VAR (CVaR / Expected Shortfall)")
    print("=" * 80)
    print()
    print("CVaR = 'Given we exceed VaR, what is the average loss?'")
    print()
    print(f"Historical CVaR: ${comparison.iloc[0]['CVaR ($)']:,.0f}")
    print("  - Average loss in worst 5% of days")
    print("  - Better measure of tail risk than VaR")
    print("  - Preferred by risk managers (captures severity)")
    print()

    # Different confidence levels
    print("=" * 80)
    print("DIFFERENT CONFIDENCE LEVELS")
    print("=" * 80)
    print()

    for confidence in [0.90, 0.95, 0.99, 0.999]:
        result = var_calc.historical_var(portfolio_returns, confidence_level=confidence)
        print(f"{confidence*100:.1f}% VaR: ${result['var_dollar']:>10,.0f}  "
              f"(CVaR: ${result['cvar_dollar']:>10,.0f})")

    print()
    print("Note: Higher confidence → Larger VaR (protecting against more extreme events)")

    # Backtesting
    print("\n" + "=" * 80)
    print("VAR MODEL BACKTESTING")
    print("=" * 80)
    print()

    backtest_results = var_calc.backtest_var(portfolio_returns, var_method='historical')

    print(f"VaR Breaches: {backtest_results['var_breaches']} out of {backtest_results['total_days']} days")
    print(f"Breach Rate: {backtest_results['breach_rate']:.2%}")
    print(f"Expected Rate: {backtest_results['expected_breach_rate']:.2%}")
    print(f"Basel Zone: {backtest_results['basel_zone'].upper()}")
    print()

    if backtest_results['basel_zone'] == 'green':
        print("✅ GREEN ZONE: VaR model is accurate")
    elif backtest_results['basel_zone'] == 'yellow':
        print("⚠️  YELLOW ZONE: Monitor model closely")
    else:
        print("❌ RED ZONE: VaR model inadequate, needs recalibration")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("Why VaR Matters:")
    print("  - Required by Basel Committee for bank capital")
    print("  - Risk reporting to investors")
    print("  - Position sizing and leverage decisions")
    print("  - Early warning system for risk managers")
    print()
    print("Which Method to Use:")
    print("  - Historical: Simple, no assumptions, good for normal markets")
    print("  - Parametric: Fast, smooth, but assumes normality")
    print("  - Monte Carlo: Flexible, handles complexity, best for fat tails")
    print()
    print("VaR Limitations:")
    print("  - Doesn't capture 'black swan' events beyond confidence level")
    print("  - CVaR (Expected Shortfall) is better tail risk measure")
    print("  - Should be supplemented with stress testing")
    print()
    print("=" * 80)
    print("Test complete!")
    print("=" * 80)
