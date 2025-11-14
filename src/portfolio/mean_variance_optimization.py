"""
Mean-Variance Optimization (Markowitz Portfolio Theory)

Nobel Prize-winning framework for portfolio construction.
Developed by Harry Markowitz (1952).

Objective:
- Maximize expected return for given level of risk
- OR minimize risk for given level of expected return
- Constructs "efficient frontier" of optimal portfolios

Academic Foundation:
- Markowitz (1952) - "Portfolio Selection"
- Modern Portfolio Theory (MPT)
- Foundation of quantitative portfolio management

Methods:
1. Maximum Sharpe Ratio - Best risk-adjusted returns
2. Minimum Variance - Lowest risk portfolio
3. Efficient Frontier - Trade-off between risk and return
4. Target Return - Minimize risk for desired return
5. Target Risk - Maximize return for desired risk
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MVOConfig:
    """Configuration for Mean-Variance Optimization"""
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    max_position: float = 0.30  # Maximum 30% in single asset
    min_position: float = 0.0  # Minimum position (0 = long-only)
    target_return: Optional[float] = None  # Target return for min variance
    target_risk: Optional[float] = None  # Target risk for max return


class MeanVarianceOptimization:
    """
    Mean-Variance Optimization (Markowitz)

    Constructs optimal portfolios based on expected returns,
    volatility, and correlations.
    """

    def __init__(self, config: MVOConfig = None):
        self.config = config or MVOConfig()

    def calculate_returns(self, prices: pd.DataFrame) -> pd.Series:
        """Calculate expected returns (mean historical returns)"""
        returns = prices.pct_change().dropna()
        expected_returns = returns.mean() * 252  # Annualize
        return expected_returns

    def calculate_covariance(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Calculate covariance matrix"""
        returns = prices.pct_change().dropna()
        cov_matrix = returns.cov() * 252  # Annualize
        return cov_matrix

    def portfolio_performance(
        self,
        weights: np.ndarray,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Calculate portfolio performance metrics

        Returns: (return, volatility, sharpe_ratio)
        """
        portfolio_return = np.sum(weights * expected_returns)
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.config.risk_free_rate) / portfolio_vol

        return portfolio_return, portfolio_vol, sharpe_ratio

    def maximize_sharpe_ratio(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame
    ) -> Dict:
        """
        Find portfolio with maximum Sharpe ratio

        This is the "tangency portfolio" - optimal risk-adjusted returns
        """
        n_assets = len(expected_returns)

        # Objective: Minimize negative Sharpe ratio
        def objective(weights):
            ret, vol, sharpe = self.portfolio_performance(
                weights,
                expected_returns.values,
                cov_matrix.values
            )
            return -sharpe  # Negative because we minimize

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]

        # Bounds
        bounds = tuple(
            (self.config.min_position, self.config.max_position)
            for _ in range(n_assets)
        )

        # Initial guess (equal weight)
        init_guess = np.array([1/n_assets] * n_assets)

        # Optimize
        result = minimize(
            objective,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if not result.success:
            print(f"Warning: Optimization did not converge - {result.message}")

        # Calculate final metrics
        weights = result.x
        ret, vol, sharpe = self.portfolio_performance(
            weights,
            expected_returns.values,
            cov_matrix.values
        )

        return {
            'weights': pd.Series(weights, index=expected_returns.index),
            'expected_return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'success': result.success
        }

    def minimize_variance(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        target_return: Optional[float] = None
    ) -> Dict:
        """
        Find minimum variance portfolio

        If target_return specified: Minimize variance for that return level
        Else: Find global minimum variance portfolio
        """
        n_assets = len(expected_returns)

        # Objective: Minimize portfolio variance
        def objective(weights):
            return np.dot(weights.T, np.dot(cov_matrix.values, weights))

        # Constraints
        constraints = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]

        # Add target return constraint if specified
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda w: np.sum(w * expected_returns.values) - target_return
            })

        # Bounds
        bounds = tuple(
            (self.config.min_position, self.config.max_position)
            for _ in range(n_assets)
        )

        # Initial guess
        init_guess = np.array([1/n_assets] * n_assets)

        # Optimize
        result = minimize(
            objective,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )

        if not result.success:
            print(f"Warning: Optimization did not converge - {result.message}")

        # Calculate final metrics
        weights = result.x
        ret, vol, sharpe = self.portfolio_performance(
            weights,
            expected_returns.values,
            cov_matrix.values
        )

        return {
            'weights': pd.Series(weights, index=expected_returns.index),
            'expected_return': ret,
            'volatility': vol,
            'sharpe_ratio': sharpe,
            'success': result.success
        }

    def efficient_frontier(
        self,
        expected_returns: pd.Series,
        cov_matrix: pd.DataFrame,
        n_portfolios: int = 50
    ) -> pd.DataFrame:
        """
        Calculate efficient frontier

        Returns DataFrame with portfolios along efficient frontier
        """
        # Find range of returns
        min_var_portfolio = self.minimize_variance(expected_returns, cov_matrix)
        max_sharpe_portfolio = self.maximize_sharpe_ratio(expected_returns, cov_matrix)

        min_return = min_var_portfolio['expected_return']
        max_return = expected_returns.max()

        # Generate target returns
        target_returns = np.linspace(min_return, max_return, n_portfolios)

        results = []
        for target_ret in target_returns:
            try:
                portfolio = self.minimize_variance(
                    expected_returns,
                    cov_matrix,
                    target_return=target_ret
                )

                if portfolio['success']:
                    results.append({
                        'target_return': target_ret,
                        'expected_return': portfolio['expected_return'],
                        'volatility': portfolio['volatility'],
                        'sharpe_ratio': portfolio['sharpe_ratio'],
                        'weights': portfolio['weights']
                    })
            except:
                continue

        return pd.DataFrame(results)

    def optimize_portfolio(
        self,
        prices: pd.DataFrame,
        method: str = 'max_sharpe'
    ) -> Dict:
        """
        Main optimization function

        Methods:
        - 'max_sharpe': Maximum Sharpe ratio (best risk-adjusted)
        - 'min_variance': Minimum variance (lowest risk)
        - 'efficient_frontier': Full efficient frontier
        """
        # Calculate inputs
        expected_returns = self.calculate_returns(prices)
        cov_matrix = self.calculate_covariance(prices)

        # Optimize based on method
        if method == 'max_sharpe':
            return self.maximize_sharpe_ratio(expected_returns, cov_matrix)
        elif method == 'min_variance':
            return self.minimize_variance(expected_returns, cov_matrix)
        elif method == 'efficient_frontier':
            frontier = self.efficient_frontier(expected_returns, cov_matrix)
            return {'frontier': frontier}
        else:
            raise ValueError(f"Unknown method: {method}")


if __name__ == "__main__":
    """Test Mean-Variance Optimization"""
    import yfinance as yf

    print("=" * 80)
    print("MEAN-VARIANCE OPTIMIZATION (MARKOWITZ) TEST")
    print("=" * 80)
    print()

    # Test with diversified portfolio
    symbols = [
        'BTC-USD', 'ETH-USD',  # Crypto
        'SPY', 'QQQ',          # Equity
        'TLT',                 # Bonds
        'GLD',                 # Gold
    ]

    print(f"Fetching data for {len(symbols)} assets...")
    data = yf.download(symbols, start='2022-01-01', end='2024-12-31', progress=False)['Adj Close']

    # Initialize optimizer
    optimizer = MeanVarianceOptimization()

    # Calculate expected returns and covariance
    print("\n" + "=" * 80)
    print("EXPECTED RETURNS (Annualized)")
    print("=" * 80)

    expected_returns = optimizer.calculate_returns(data)
    for symbol, ret in expected_returns.sort_values(ascending=False).items():
        print(f"  {symbol:12s}: {ret:>8.2%}")

    # Calculate covariance
    print("\n" + "=" * 80)
    print("CORRELATION MATRIX")
    print("=" * 80)

    cov_matrix = optimizer.calculate_covariance(data)
    corr_matrix = data.pct_change().corr()
    print(corr_matrix.round(2))

    # Maximum Sharpe Ratio Portfolio
    print("\n" + "=" * 80)
    print("MAXIMUM SHARPE RATIO PORTFOLIO")
    print("=" * 80)

    max_sharpe = optimizer.optimize_portfolio(data, method='max_sharpe')
    print(f"\nExpected Return : {max_sharpe['expected_return']:>8.2%}")
    print(f"Volatility      : {max_sharpe['volatility']:>8.2%}")
    print(f"Sharpe Ratio    : {max_sharpe['sharpe_ratio']:>8.2f}")
    print("\nOptimal Weights:")
    for symbol, weight in max_sharpe['weights'].sort_values(ascending=False).items():
        if weight > 0.01:  # Only show positions >1%
            print(f"  {symbol:12s}: {weight:>6.2%}")

    # Minimum Variance Portfolio
    print("\n" + "=" * 80)
    print("MINIMUM VARIANCE PORTFOLIO")
    print("=" * 80)

    min_var = optimizer.optimize_portfolio(data, method='min_variance')
    print(f"\nExpected Return : {min_var['expected_return']:>8.2%}")
    print(f"Volatility      : {min_var['volatility']:>8.2%}")
    print(f"Sharpe Ratio    : {min_var['sharpe_ratio']:>8.2f}")
    print("\nOptimal Weights:")
    for symbol, weight in min_var['weights'].sort_values(ascending=False).items():
        if weight > 0.01:
            print(f"  {symbol:12s}: {weight:>6.2%}")

    # Comparison
    print("\n" + "=" * 80)
    print("PORTFOLIO COMPARISON")
    print("=" * 80)

    # Equal weight for comparison
    equal_weights = np.array([1/len(symbols)] * len(symbols))
    eq_ret, eq_vol, eq_sharpe = optimizer.portfolio_performance(
        equal_weights,
        expected_returns.values,
        cov_matrix.values
    )

    comparison = pd.DataFrame({
        'Portfolio': ['Equal Weight', 'Max Sharpe', 'Min Variance'],
        'Return': [eq_ret, max_sharpe['expected_return'], min_var['expected_return']],
        'Volatility': [eq_vol, max_sharpe['volatility'], min_var['volatility']],
        'Sharpe': [eq_sharpe, max_sharpe['sharpe_ratio'], min_var['sharpe_ratio']]
    })

    print("\n", comparison.to_string(index=False))

    print("\n" + "=" * 80)
    print("Test complete!")
    print("=" * 80)
    print("\nKey Insights:")
    print("- Max Sharpe: Best risk-adjusted returns (recommended)")
    print("- Min Variance: Lowest risk, but may sacrifice returns")
    print("- Diversification reduces risk without sacrificing returns")
