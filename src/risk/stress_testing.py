"""
Stress Testing and Scenario Analysis

Stress testing answers: "How would my portfolio perform in extreme market conditions?"

Unlike VaR (which uses historical distributions), stress testing:
- Tests specific adverse scenarios (2008 crisis, COVID crash, etc.)
- Applies shocks to risk factors (rates up 200bps, equity down 30%, etc.)
- Identifies portfolio vulnerabilities
- Required by regulators (Dodd-Frank, CCAR, Basel III)

Academic Foundation:
- Kupiec (1998) - "Stress Testing in a Value at Risk Framework"
- Berkowitz (1999) - "A Coherent Framework for Stress-Testing"
- Basel Committee (2009) - "Principles for Sound Stress Testing"
- Dodd-Frank Act (2010) - Comprehensive Capital Analysis and Review (CCAR)

Types of Stress Tests:
1. Historical Scenarios - Replay past crises
2. Hypothetical Scenarios - "What if" analysis
3. Sensitivity Analysis - Shock individual factors
4. Reverse Stress Testing - Find breaking point

Key Scenarios:
- 2008 Financial Crisis: -50% equity, credit spreads +400bps
- COVID-19 Crash (March 2020): -35% equity in 1 month
- Dot-com Bubble (2000-2002): -50% tech stocks
- 1987 Black Monday: -22% equity in 1 day
- Taper Tantrum (2013): Rates +100bps, EM -15%
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


@dataclass
class StressScenario:
    """
    Single stress scenario definition

    Examples:
    - "2008 Crisis": equity_shock=-0.50, credit_spread_shock=0.04
    - "COVID Crash": equity_shock=-0.35, volatility_spike=2.0
    - "Rate Shock": rate_shock=0.02 (200bps)
    """
    name: str
    equity_shock: float = 0.0  # % change (e.g., -0.30 = -30%)
    rate_shock: float = 0.0  # Absolute change (e.g., 0.02 = +200bps)
    credit_spread_shock: float = 0.0  # Absolute change in spreads
    fx_shock: float = 0.0  # % change in FX
    commodity_shock: float = 0.0  # % change in commodities
    volatility_spike: float = 1.0  # Multiplier (e.g., 2.0 = double vol)
    correlation_shock: float = 0.0  # Change in correlation (e.g., +0.3)
    description: str = ""


@dataclass
class StressTestConfig:
    """Configuration for stress testing"""
    portfolio_value: float = 1_000_000  # $1M portfolio
    time_horizon: int = 1  # Days to project
    use_correlation_shock: bool = True  # Apply correlation changes


class StressTesting:
    """
    Stress Testing and Scenario Analysis

    Tests portfolio resilience to extreme market moves.
    """

    def __init__(self, config: StressTestConfig = None):
        self.config = config or StressTestConfig()

        # Pre-defined historical scenarios
        self.historical_scenarios = self._define_historical_scenarios()

    def _define_historical_scenarios(self) -> Dict[str, StressScenario]:
        """
        Define major historical crisis scenarios

        Based on actual market moves during crises
        """
        scenarios = {
            '2008_financial_crisis': StressScenario(
                name='2008 Financial Crisis',
                equity_shock=-0.50,  # -50% equity
                credit_spread_shock=0.04,  # +400bps credit spreads
                rate_shock=-0.03,  # -300bps (flight to safety)
                volatility_spike=3.0,  # VIX spiked 3x
                correlation_shock=0.4,  # Correlations went to 1.0
                description="Lehman Brothers collapse, global credit freeze"
            ),

            'covid_crash_2020': StressScenario(
                name='COVID-19 Crash (March 2020)',
                equity_shock=-0.35,  # -35% in 1 month
                rate_shock=-0.015,  # -150bps (emergency cuts)
                commodity_shock=-0.25,  # -25% oil
                volatility_spike=2.5,  # VIX to 80+
                correlation_shock=0.3,
                description="Pandemic lockdowns, global recession fears"
            ),

            'dotcom_bubble_2000': StressScenario(
                name='Dot-com Bubble Burst (2000-2002)',
                equity_shock=-0.50,  # Tech -50%, overall -30%
                rate_shock=-0.02,  # -200bps
                volatility_spike=2.0,
                description="Internet bubble collapse, Nasdaq -78% peak-to-trough"
            ),

            'black_monday_1987': StressScenario(
                name='Black Monday (Oct 1987)',
                equity_shock=-0.22,  # -22% in one day
                volatility_spike=5.0,  # Unprecedented vol spike
                correlation_shock=0.5,  # Everything moved together
                description="Largest one-day percentage decline in history"
            ),

            'taper_tantrum_2013': StressScenario(
                name='Taper Tantrum (2013)',
                equity_shock=-0.05,  # -5% equity
                rate_shock=0.01,  # +100bps rates
                fx_shock=0.15,  # +15% USD (EM crisis)
                volatility_spike=1.5,
                description="Fed hints at QE tapering, EM selloff"
            ),

            'china_devaluation_2015': StressScenario(
                name='China Devaluation (Aug 2015)',
                equity_shock=-0.12,  # -12% equity
                fx_shock=-0.05,  # -5% CNY
                commodity_shock=-0.15,  # -15% commodities
                volatility_spike=2.0,
                description="China devalues yuan, global growth fears"
            ),

            'european_debt_crisis_2011': StressScenario(
                name='European Debt Crisis (2011)',
                equity_shock=-0.20,  # -20% equity
                credit_spread_shock=0.03,  # +300bps spreads
                rate_shock=-0.01,  # -100bps (ECB cuts)
                volatility_spike=2.5,
                description="Greek default fears, eurozone breakup risk"
            )
        }

        return scenarios

    def apply_shock_to_returns(
        self,
        returns: pd.DataFrame,
        scenario: StressScenario,
        asset_types: Dict[str, str] = None
    ) -> pd.DataFrame:
        """
        Apply scenario shocks to asset returns

        Args:
            returns: Historical returns DataFrame
            scenario: Stress scenario to apply
            asset_types: Dict mapping asset names to types
                        ('equity', 'bond', 'commodity', 'fx')

        Returns:
            Shocked returns DataFrame
        """
        shocked_returns = returns.copy()

        if asset_types is None:
            # Default: assume all equity for simplicity
            asset_types = {col: 'equity' for col in returns.columns}

        # Apply shocks based on asset type
        for asset, asset_type in asset_types.items():
            if asset not in shocked_returns.columns:
                continue

            if asset_type == 'equity':
                # Apply equity shock
                shocked_returns[asset] += scenario.equity_shock / 252  # Daily shock

            elif asset_type == 'bond':
                # Apply rate shock (inverse relationship: rates up → bonds down)
                shocked_returns[asset] -= scenario.rate_shock / 252

            elif asset_type == 'commodity':
                # Apply commodity shock
                shocked_returns[asset] += scenario.commodity_shock / 252

            elif asset_type == 'fx':
                # Apply FX shock
                shocked_returns[asset] += scenario.fx_shock / 252

        # Apply volatility spike
        if scenario.volatility_spike != 1.0:
            shocked_returns = shocked_returns * scenario.volatility_spike

        return shocked_returns

    def calculate_stressed_var(
        self,
        returns: pd.Series,
        scenario: StressScenario
    ) -> Dict:
        """
        Calculate VaR under stressed conditions

        Combines normal VaR with scenario shock
        """
        # Baseline VaR (95%)
        baseline_var = np.percentile(returns, 5)

        # Apply scenario shock
        stressed_returns = returns + scenario.equity_shock

        # Stressed VaR
        stressed_var = np.percentile(stressed_returns, 5)

        # VaR change
        var_change = stressed_var - baseline_var

        return {
            'baseline_var': baseline_var,
            'stressed_var': stressed_var,
            'var_change': var_change,
            'var_change_dollar': var_change * self.config.portfolio_value
        }

    def run_scenario(
        self,
        portfolio_returns: pd.Series,
        scenario: StressScenario
    ) -> Dict:
        """
        Run a single stress scenario

        Calculates portfolio P&L under scenario
        """
        # Calculate portfolio statistics
        mean_return = portfolio_returns.mean() * 252  # Annualized
        vol = portfolio_returns.std() * np.sqrt(252)

        # Apply scenario shock
        scenario_return = scenario.equity_shock  # Simplified: assume all equity
        scenario_pnl = scenario_return * self.config.portfolio_value

        # Calculate drawdown from current level
        current_value = self.config.portfolio_value
        stressed_value = current_value * (1 + scenario_return)
        drawdown = (stressed_value - current_value) / current_value

        # Calculate VaR impact
        var_impact = self.calculate_stressed_var(portfolio_returns, scenario)

        return {
            'scenario_name': scenario.name,
            'scenario_return': scenario_return,
            'scenario_pnl': scenario_pnl,
            'current_value': current_value,
            'stressed_value': stressed_value,
            'drawdown': drawdown,
            'baseline_var': var_impact['baseline_var'],
            'stressed_var': var_impact['stressed_var'],
            'var_deterioration': var_impact['var_change'],
            'description': scenario.description
        }

    def run_all_historical_scenarios(
        self,
        portfolio_returns: pd.Series
    ) -> pd.DataFrame:
        """
        Run all pre-defined historical scenarios

        Returns DataFrame with results
        """
        results = []

        for scenario_key, scenario in self.historical_scenarios.items():
            result = self.run_scenario(portfolio_returns, scenario)
            results.append(result)

        df = pd.DataFrame(results)

        # Sort by severity (worst drawdown first)
        df = df.sort_values('drawdown')

        return df

    def sensitivity_analysis(
        self,
        portfolio_returns: pd.Series,
        factor: str = 'equity',
        shock_range: Tuple[float, float] = (-0.50, 0.30),
        n_steps: int = 20
    ) -> pd.DataFrame:
        """
        Sensitivity Analysis: Vary one factor, measure impact

        Args:
            portfolio_returns: Portfolio returns series
            factor: Which factor to shock ('equity', 'rates', 'fx', etc.)
            shock_range: Range of shocks (min, max)
            n_steps: Number of steps in range

        Returns:
            DataFrame with shock levels and resulting P&L
        """
        shocks = np.linspace(shock_range[0], shock_range[1], n_steps)
        results = []

        for shock in shocks:
            # Create scenario with single factor shock
            if factor == 'equity':
                scenario = StressScenario(
                    name=f'Equity {shock:+.0%}',
                    equity_shock=shock
                )
            elif factor == 'rates':
                scenario = StressScenario(
                    name=f'Rates {shock:+.2%}',
                    rate_shock=shock
                )
            elif factor == 'fx':
                scenario = StressScenario(
                    name=f'FX {shock:+.0%}',
                    fx_shock=shock
                )
            else:
                scenario = StressScenario(name=f'{factor} {shock:+.0%}')

            # Run scenario
            result = self.run_scenario(portfolio_returns, scenario)
            results.append({
                'shock_level': shock,
                'scenario_pnl': result['scenario_pnl'],
                'stressed_value': result['stressed_value'],
                'drawdown': result['drawdown']
            })

        return pd.DataFrame(results)

    def reverse_stress_test(
        self,
        portfolio_returns: pd.Series,
        max_loss_threshold: float = -0.20  # -20% max acceptable loss
    ) -> Dict:
        """
        Reverse Stress Testing: Find what shock causes unacceptable loss

        "What market move would cause a 20% portfolio loss?"

        This is required by Basel III and best practice.
        """
        # Binary search for breaking point
        low, high = -1.0, 0.0
        tolerance = 0.001

        while high - low > tolerance:
            mid = (low + high) / 2

            scenario = StressScenario(
                name=f'Equity {mid:.1%}',
                equity_shock=mid
            )

            result = self.run_scenario(portfolio_returns, scenario)

            if result['drawdown'] < max_loss_threshold:
                # Too severe, search higher
                low = mid
            else:
                # Not severe enough, search lower
                high = mid

        # Final scenario
        breaking_scenario = StressScenario(
            name='Breaking Point',
            equity_shock=mid,
            description=f"Equity shock that causes {max_loss_threshold:.0%} loss"
        )

        result = self.run_scenario(portfolio_returns, breaking_scenario)

        return {
            'breaking_shock': mid,
            'threshold_loss': max_loss_threshold,
            'scenario': breaking_scenario,
            'result': result
        }

    def correlation_stress_test(
        self,
        returns: pd.DataFrame,
        weights: pd.Series,
        correlation_shock: float = 0.5  # Increase all correlations by 0.5
    ) -> Dict:
        """
        Test impact of correlation breakdown

        During crises, correlations spike (diversification fails)
        """
        # Baseline correlation
        baseline_corr = returns.corr()
        baseline_cov = returns.cov() * 252

        # Shocked correlation (add correlation_shock to all off-diagonal)
        shocked_corr = baseline_corr.copy()
        n = len(shocked_corr)
        for i in range(n):
            for j in range(n):
                if i != j:
                    shocked_corr.iloc[i, j] = min(shocked_corr.iloc[i, j] + correlation_shock, 0.99)

        # Convert to covariance (preserve individual volatilities)
        std_devs = returns.std() * np.sqrt(252)
        shocked_cov = shocked_corr.copy()
        for i in range(n):
            for j in range(n):
                shocked_cov.iloc[i, j] = shocked_corr.iloc[i, j] * std_devs.iloc[i] * std_devs.iloc[j]

        # Calculate portfolio volatility
        baseline_vol = np.sqrt(np.dot(weights, np.dot(baseline_cov, weights)))
        shocked_vol = np.sqrt(np.dot(weights, np.dot(shocked_cov, weights)))

        vol_increase = shocked_vol - baseline_vol

        return {
            'baseline_correlation': baseline_corr.values[np.triu_indices_from(baseline_corr.values, k=1)].mean(),
            'shocked_correlation': shocked_corr.values[np.triu_indices_from(shocked_corr.values, k=1)].mean(),
            'baseline_volatility': baseline_vol,
            'shocked_volatility': shocked_vol,
            'volatility_increase': vol_increase,
            'diversification_loss': vol_increase / baseline_vol
        }


if __name__ == "__main__":
    """Test Stress Testing Framework"""

    print("=" * 80)
    print("STRESS TESTING & SCENARIO ANALYSIS")
    print("=" * 80)
    print()

    # Create synthetic portfolio returns
    print("Creating synthetic portfolio (60/40 Equity/Bond)...")
    print()

    np.random.seed(42)
    n_days = 252 * 5  # 5 years

    # Normal market conditions
    equity_returns = np.random.normal(0.0008, 0.015, n_days)
    bond_returns = np.random.normal(0.0003, 0.005, n_days)
    portfolio_returns = 0.6 * equity_returns + 0.4 * bond_returns
    portfolio_returns = pd.Series(portfolio_returns)

    # Initialize stress tester
    config = StressTestConfig(portfolio_value=1_000_000)
    stress_tester = StressTesting(config)

    # Run all historical scenarios
    print("=" * 80)
    print("HISTORICAL CRISIS SCENARIOS")
    print("=" * 80)
    print()

    historical_results = stress_tester.run_all_historical_scenarios(portfolio_returns)

    print("Portfolio Impact Ranking (Worst to Best):")
    print()
    for idx, row in historical_results.iterrows():
        print(f"{row['scenario_name']}")
        print(f"  Drawdown: {row['drawdown']:>8.2%}")
        print(f"  P&L: ${row['scenario_pnl']:>12,.0f}")
        print(f"  {row['description']}")
        print()

    # Sensitivity analysis
    print("=" * 80)
    print("SENSITIVITY ANALYSIS: Equity Market Shock")
    print("=" * 80)
    print()

    sensitivity = stress_tester.sensitivity_analysis(
        portfolio_returns,
        factor='equity',
        shock_range=(-0.50, 0.30),
        n_steps=10
    )

    print("Impact of Equity Market Moves:")
    for idx, row in sensitivity.iterrows():
        print(f"  Equity {row['shock_level']:+6.0%}  →  P&L: ${row['scenario_pnl']:>12,.0f}  "
              f"(Portfolio: ${row['stressed_value']:>10,.0f})")

    # Reverse stress test
    print("\n" + "=" * 80)
    print("REVERSE STRESS TEST: Breaking Point Analysis")
    print("=" * 80)
    print()

    reverse_result = stress_tester.reverse_stress_test(
        portfolio_returns,
        max_loss_threshold=-0.20  # -20% max loss
    )

    print(f"Question: What equity shock causes -20% portfolio loss?")
    print(f"Answer: {reverse_result['breaking_shock']:.1%} equity decline")
    print()
    print("This is your portfolio's 'breaking point'")
    print("Regulators require knowing this for risk management")

    # Correlation stress test
    print("\n" + "=" * 80)
    print("CORRELATION STRESS TEST: Diversification Breakdown")
    print("=" * 80)
    print()

    # Create multi-asset returns for correlation test
    returns_df = pd.DataFrame({
        'US Equity': np.random.normal(0.0008, 0.015, n_days),
        'Intl Equity': np.random.normal(0.0006, 0.013, n_days),
        'Bonds': np.random.normal(0.0003, 0.005, n_days),
        'Commodities': np.random.normal(0.0004, 0.018, n_days)
    })

    # Add some correlation
    returns_df['Intl Equity'] = 0.7 * returns_df['US Equity'] + 0.3 * returns_df['Intl Equity']

    weights = pd.Series([0.40, 0.20, 0.30, 0.10], index=returns_df.columns)

    corr_result = stress_tester.correlation_stress_test(returns_df, weights, correlation_shock=0.5)

    print(f"Baseline Avg Correlation: {corr_result['baseline_correlation']:.2f}")
    print(f"Stressed Avg Correlation: {corr_result['shocked_correlation']:.2f}")
    print()
    print(f"Baseline Portfolio Vol: {corr_result['baseline_volatility']:.2%}")
    print(f"Stressed Portfolio Vol: {corr_result['shocked_volatility']:.2%}")
    print(f"Volatility Increase: {corr_result['volatility_increase']:.2%}")
    print(f"Diversification Loss: {corr_result['diversification_loss']:.1%}")
    print()
    print("During crises, correlations spike → diversification fails")

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("Stress Testing vs. VaR:")
    print("  - VaR: 'What's the normal range of losses?'")
    print("  - Stress: 'What happens in a crisis?'")
    print("  - Both are required for complete risk picture")
    print()
    print("Historical Scenarios:")
    print("  - Test portfolio against past crises")
    print("  - Identifies vulnerabilities")
    print("  - Required by regulators (CCAR, Basel III)")
    print()
    print("Reverse Stress Testing:")
    print("  - Find your portfolio's breaking point")
    print("  - 'What market move would wipe us out?'")
    print("  - Best practice for risk management")
    print()
    print("Correlation Stress:")
    print("  - Diversification fails in crises")
    print("  - Correlations spike to 1.0")
    print("  - Must account for this in risk models")
    print()
    print("=" * 80)
    print("Test complete!")
    print("=" * 80)
