"""
Fama-French Factor Analysis

Implements factor models for portfolio risk decomposition:
- Fama-French 3-Factor Model (Market, Size, Value)
- Fama-French 5-Factor Model (+ Profitability, Investment)
- Carhart 4-Factor Model (+ Momentum)
- Custom Multi-Factor Models

Provides institutional-grade factor exposure analysis and attribution.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats


@dataclass
class FactorExposure:
    """Factor exposure results"""
    factor_loadings: Dict[str, float]  # beta coefficients
    factor_returns: Dict[str, float]  # factor contributions
    alpha: float  # Jensen's alpha (strategy excess return)
    r_squared: float  # model fit
    residual_vol: float  # unexplained volatility
    t_stats: Dict[str, float]  # statistical significance
    p_values: Dict[str, float]


@dataclass
class FactorAttribution:
    """Performance attribution to factors"""
    total_return: float
    factor_contributions: Dict[str, float]
    alpha_contribution: float
    residual_contribution: float
    percentage_explained: float


class FamaFrenchFactorModel:
    """
    Fama-French Factor Models

    Decomposes portfolio returns into systematic factor exposures:
    - Market Risk Premium (Mkt-RF)
    - Size (SMB): Small Minus Big
    - Value (HML): High Minus Low book-to-market
    - Profitability (RMW): Robust Minus Weak
    - Investment (CMA): Conservative Minus Aggressive

    Reference: Fama & French (1993, 2015)
    """

    def __init__(self, model_type: str = 'ff5'):
        """
        Initialize factor model

        Args:
            model_type: 'ff3' (3-factor), 'ff5' (5-factor), 'carhart' (4-factor with momentum)
        """
        self.model_type = model_type
        self.factor_data = None
        self.fitted = False

    def load_factor_data(self,
                        factor_file: Optional[str] = None,
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load Fama-French factor data

        Args:
            factor_file: Path to CSV with factor returns
            start_date: Start date for analysis
            end_date: End date for analysis

        Returns:
            DataFrame with factor returns

        Note: You can download official Fama-French data from:
        https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
        """
        if factor_file:
            # Load from file
            self.factor_data = pd.read_csv(factor_file, index_col=0, parse_dates=True)
        else:
            # Generate synthetic factor data for demonstration
            print("⚠️  Using synthetic factor data. Download real data from Ken French's website.")
            self.factor_data = self._generate_synthetic_factors(start_date, end_date)

        # Filter by date range
        if start_date:
            self.factor_data = self.factor_data[self.factor_data.index >= start_date]
        if end_date:
            self.factor_data = self.factor_data[self.factor_data.index <= end_date]

        return self.factor_data

    def _generate_synthetic_factors(self,
                                   start_date: Optional[datetime] = None,
                                   end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Generate synthetic factor returns for testing

        Based on historical factor characteristics
        """
        if start_date is None:
            start_date = datetime(2020, 1, 1)
        if end_date is None:
            end_date = datetime.now()

        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        n = len(dates)

        np.random.seed(42)

        # Realistic factor parameters (annualized)
        factor_params = {
            'Mkt-RF': {'mean': 0.08/252, 'vol': 0.18/np.sqrt(252)},  # Market premium
            'SMB': {'mean': 0.02/252, 'vol': 0.12/np.sqrt(252)},      # Size
            'HML': {'mean': 0.03/252, 'vol': 0.14/np.sqrt(252)},      # Value
            'RMW': {'mean': 0.025/252, 'vol': 0.10/np.sqrt(252)},     # Profitability
            'CMA': {'mean': 0.02/252, 'vol': 0.09/np.sqrt(252)},      # Investment
            'MOM': {'mean': 0.06/252, 'vol': 0.16/np.sqrt(252)},      # Momentum
            'RF': {'mean': 0.02/252, 'vol': 0.001/np.sqrt(252)}       # Risk-free rate
        }

        # Generate correlated factor returns
        factors = pd.DataFrame(index=dates)

        for factor_name, params in factor_params.items():
            factors[factor_name] = np.random.normal(
                params['mean'],
                params['vol'],
                n
            )

        # Add some correlation structure
        # Market and Size tend to be correlated
        factors['SMB'] = factors['SMB'] * 0.7 + factors['Mkt-RF'] * 0.3

        # Value and Profitability are negatively correlated
        factors['RMW'] = factors['RMW'] - factors['HML'] * 0.2

        return factors

    def calculate_factor_exposure(self,
                                  strategy_returns: pd.Series,
                                  risk_free_rate: Optional[pd.Series] = None) -> FactorExposure:
        """
        Calculate factor exposures using regression

        Args:
            strategy_returns: Daily strategy returns
            risk_free_rate: Daily risk-free rate (optional, will use factor data if available)

        Returns:
            FactorExposure object with betas and statistics
        """
        if self.factor_data is None:
            raise ValueError("Factor data not loaded. Call load_factor_data() first.")

        # Align dates
        common_dates = strategy_returns.index.intersection(self.factor_data.index)
        y = strategy_returns.loc[common_dates].values

        # Get risk-free rate
        if risk_free_rate is not None:
            rf = risk_free_rate.loc[common_dates].values
        elif 'RF' in self.factor_data.columns:
            rf = self.factor_data.loc[common_dates, 'RF'].values
        else:
            rf = np.zeros(len(common_dates))

        # Excess returns
        excess_returns = y - rf

        # Select factors based on model type
        if self.model_type == 'ff3':
            factor_names = ['Mkt-RF', 'SMB', 'HML']
        elif self.model_type == 'ff5':
            factor_names = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        elif self.model_type == 'carhart':
            factor_names = ['Mkt-RF', 'SMB', 'HML', 'MOM']
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Check if all factors are available
        available_factors = [f for f in factor_names if f in self.factor_data.columns]
        if len(available_factors) < len(factor_names):
            missing = set(factor_names) - set(available_factors)
            print(f"⚠️  Missing factors: {missing}. Using available: {available_factors}")
            factor_names = available_factors

        X = self.factor_data.loc[common_dates, factor_names].values

        # Run regression
        model = LinearRegression()
        model.fit(X, excess_returns)

        # Get predictions and residuals
        predictions = model.predict(X)
        residuals = excess_returns - predictions

        # Calculate statistics
        n = len(y)
        k = len(factor_names)

        # R-squared
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((excess_returns - excess_returns.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Residual volatility (annualized)
        residual_vol = np.std(residuals) * np.sqrt(252)

        # Alpha (annualized)
        alpha = model.intercept_ * 252

        # Factor loadings (betas)
        factor_loadings = {name: beta for name, beta in zip(factor_names, model.coef_)}

        # Calculate t-statistics and p-values
        # Standard errors
        mse = ss_res / (n - k - 1)
        X_with_intercept = np.column_stack([np.ones(n), X])
        var_coef = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept).diagonal()
        se_coef = np.sqrt(var_coef)

        # T-statistics
        coefficients = np.concatenate([[model.intercept_], model.coef_])
        t_stats_values = coefficients / se_coef

        # P-values (two-tailed)
        p_values_array = 2 * (1 - stats.t.cdf(np.abs(t_stats_values), n - k - 1))

        t_stats = {'alpha': t_stats_values[0]}
        p_values = {'alpha': p_values_array[0]}

        for i, name in enumerate(factor_names):
            t_stats[name] = t_stats_values[i + 1]
            p_values[name] = p_values_array[i + 1]

        # Calculate factor returns (contribution to portfolio return)
        factor_returns = {}
        for name, beta in factor_loadings.items():
            avg_factor_return = self.factor_data.loc[common_dates, name].mean() * 252
            factor_returns[name] = beta * avg_factor_return

        self.fitted = True

        return FactorExposure(
            factor_loadings=factor_loadings,
            factor_returns=factor_returns,
            alpha=alpha,
            r_squared=r_squared,
            residual_vol=residual_vol,
            t_stats=t_stats,
            p_values=p_values
        )

    def attribute_performance(self,
                            strategy_returns: pd.Series,
                            risk_free_rate: Optional[pd.Series] = None) -> FactorAttribution:
        """
        Attribute performance to factors

        Decomposes total return into factor contributions
        """
        exposure = self.calculate_factor_exposure(strategy_returns, risk_free_rate)

        # Total return
        total_return = (1 + strategy_returns).prod() - 1

        # Factor contributions
        factor_contributions = {}
        total_factor_contribution = 0

        for factor, loading in exposure.factor_loadings.items():
            contribution = exposure.factor_returns[factor]
            factor_contributions[factor] = contribution
            total_factor_contribution += contribution

        # Alpha contribution
        alpha_contribution = exposure.alpha

        # Residual
        residual_contribution = total_return - total_factor_contribution - alpha_contribution

        # Percentage explained by model
        percentage_explained = (total_factor_contribution + alpha_contribution) / total_return if total_return != 0 else 0

        return FactorAttribution(
            total_return=total_return,
            factor_contributions=factor_contributions,
            alpha_contribution=alpha_contribution,
            residual_contribution=residual_contribution,
            percentage_explained=percentage_explained
        )

    def rolling_factor_exposure(self,
                               strategy_returns: pd.Series,
                               window: int = 252,
                               risk_free_rate: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Calculate rolling factor exposures over time

        Args:
            strategy_returns: Daily returns
            window: Rolling window size (default 252 = 1 year)
            risk_free_rate: Risk-free rate

        Returns:
            DataFrame with rolling factor loadings and alpha
        """
        results = []

        for i in range(window, len(strategy_returns)):
            window_returns = strategy_returns.iloc[i-window:i]

            try:
                exposure = self.calculate_factor_exposure(window_returns, risk_free_rate)

                result = {'date': strategy_returns.index[i], 'alpha': exposure.alpha}
                result.update(exposure.factor_loadings)
                result['r_squared'] = exposure.r_squared

                results.append(result)
            except:
                # Skip if not enough data
                continue

        return pd.DataFrame(results).set_index('date')

    def print_factor_report(self, exposure: FactorExposure):
        """Print formatted factor analysis report"""
        print(f"\n{'='*80}")
        print(f"FACTOR ANALYSIS REPORT - {self.model_type.upper()}")
        print(f"{'='*80}\n")

        print("Factor Exposures (Betas):")
        print("-" * 60)
        for factor, beta in exposure.factor_loadings.items():
            t_stat = exposure.t_stats[factor]
            p_val = exposure.p_values[factor]
            sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
            print(f"  {factor:10s}: {beta:7.3f}  (t={t_stat:6.2f}, p={p_val:.4f}) {sig}")

        print(f"\nAlpha (Annualized):")
        print("-" * 60)
        t_stat = exposure.t_stats['alpha']
        p_val = exposure.p_values['alpha']
        sig = "***" if p_val < 0.01 else "**" if p_val < 0.05 else "*" if p_val < 0.10 else ""
        print(f"  Alpha: {exposure.alpha*100:6.2f}%  (t={t_stat:6.2f}, p={p_val:.4f}) {sig}")

        print(f"\nModel Fit:")
        print("-" * 60)
        print(f"  R-squared:      {exposure.r_squared:.4f} ({exposure.r_squared*100:.1f}% explained)")
        print(f"  Residual Vol:   {exposure.residual_vol*100:.2f}% (unexplained risk)")

        print(f"\nFactor Return Contributions (Annualized):")
        print("-" * 60)
        for factor, ret in exposure.factor_returns.items():
            print(f"  {factor:10s}: {ret*100:6.2f}%")

        print(f"\n{'='*80}\n")


class CustomFactorModel:
    """
    Custom factor model for specific strategies

    Allows defining custom factors beyond Fama-French
    """

    def __init__(self):
        self.factors = {}

    def add_factor(self, name: str, returns: pd.Series):
        """Add a custom factor"""
        self.factors[name] = returns

    def calculate_exposure(self, strategy_returns: pd.Series) -> Dict[str, float]:
        """Calculate exposure to custom factors"""
        # Align dates
        common_dates = strategy_returns.index
        for factor_name, factor_returns in self.factors.items():
            common_dates = common_dates.intersection(factor_returns.index)

        y = strategy_returns.loc[common_dates].values
        X = pd.DataFrame({name: self.factors[name].loc[common_dates]
                         for name in self.factors.keys()}).values

        # Regression
        model = LinearRegression()
        model.fit(X, y)

        return {name: beta for name, beta in zip(self.factors.keys(), model.coef_)}


if __name__ == '__main__':
    """Test factor analysis"""

    print("Fama-French Factor Analysis - Demo\n")

    # Generate sample strategy returns
    np.random.seed(42)
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')

    # Simulate a momentum strategy (positive exposure to Mkt, SMB, MOM)
    strategy_returns = pd.Series(
        np.random.normal(0.0008, 0.015, len(dates)),
        index=dates
    )

    # Initialize factor model
    ff5 = FamaFrenchFactorModel(model_type='ff5')

    # Load (synthetic) factor data
    ff5.load_factor_data(start_date=dates[0], end_date=dates[-1])

    # Calculate factor exposure
    exposure = ff5.calculate_factor_exposure(strategy_returns)

    # Print report
    ff5.print_factor_report(exposure)

    # Attribution
    attribution = ff5.attribute_performance(strategy_returns)

    print("Performance Attribution:")
    print(f"  Total Return: {attribution.total_return*100:.2f}%")
    print(f"  Alpha:        {attribution.alpha_contribution*100:.2f}%")
    for factor, contrib in attribution.factor_contributions.items():
        print(f"  {factor:10s}: {contrib*100:.2f}%")
    print(f"  Residual:     {attribution.residual_contribution*100:.2f}%")
    print(f"\n  Explained by model: {attribution.percentage_explained*100:.1f}%")
