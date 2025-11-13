"""
Performance Calculation Utilities

Borrows concepts from QIS (QuantInvestStrats) and pyfolio for
battle-tested performance metrics calculations.

Key metrics:
- Sharpe, Sortino, Calmar ratios
- Drawdown analysis
- Rolling performance
- Risk-adjusted returns
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
from scipy import stats


def calculate_returns(prices: pd.Series) -> pd.Series:
    """Calculate simple returns from prices"""
    return prices.pct_change().fillna(0)


def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
    """Calculate cumulative returns"""
    return (1 + returns).cumprod() - 1


def calculate_sharpe_ratio(returns: pd.Series,
                          risk_free_rate: float = 0.0,
                          periods_per_year: int = 252) -> float:
    """
    Calculate annualized Sharpe ratio

    Args:
        returns: Period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) < 2 or returns.std() == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe = excess_returns.mean() / returns.std() * np.sqrt(periods_per_year)

    return sharpe


def calculate_sortino_ratio(returns: pd.Series,
                            risk_free_rate: float = 0.0,
                            periods_per_year: int = 252) -> float:
    """
    Calculate Sortino ratio (uses downside deviation instead of total volatility)

    Args:
        returns: Period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) < 2:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)

    # Downside deviation (only negative returns)
    downside_returns = returns[returns < 0]
    if len(downside_returns) == 0:
        return np.inf

    downside_std = downside_returns.std()
    if downside_std == 0:
        return 0.0

    sortino = excess_returns.mean() / downside_std * np.sqrt(periods_per_year)

    return sortino


def calculate_max_drawdown(returns: pd.Series) -> Tuple[float, pd.Series]:
    """
    Calculate maximum drawdown and drawdown series

    Args:
        returns: Period returns

    Returns:
        (max_drawdown, drawdown_series)
    """
    cum_returns = calculate_cumulative_returns(returns)
    running_max = (1 + cum_returns).expanding().max()
    drawdown = (1 + cum_returns) / running_max - 1

    max_dd = drawdown.min()

    return max_dd, drawdown


def calculate_calmar_ratio(returns: pd.Series,
                          periods_per_year: int = 252) -> float:
    """
    Calculate Calmar ratio (annual return / max drawdown)

    Args:
        returns: Period returns
        periods_per_year: Trading periods per year

    Returns:
        Calmar ratio
    """
    if len(returns) < 2:
        return 0.0

    annual_return = returns.mean() * periods_per_year
    max_dd, _ = calculate_max_drawdown(returns)

    if max_dd == 0:
        return 0.0

    calmar = annual_return / abs(max_dd)

    return calmar


def calculate_var(returns: pd.Series,
                 confidence: float = 0.95) -> float:
    """
    Calculate Value at Risk (VaR) at given confidence level

    Args:
        returns: Period returns
        confidence: Confidence level (0.95 = 95%)

    Returns:
        VaR (negative value represents loss)
    """
    if len(returns) < 2:
        return 0.0

    var = returns.quantile(1 - confidence)

    return var


def calculate_cvar(returns: pd.Series,
                  confidence: float = 0.95) -> float:
    """
    Calculate Conditional Value at Risk (CVaR / Expected Shortfall)

    Average loss in worst (1-confidence)% of cases

    Args:
        returns: Period returns
        confidence: Confidence level

    Returns:
        CVaR (negative value represents loss)
    """
    if len(returns) < 2:
        return 0.0

    var = calculate_var(returns, confidence)
    cvar = returns[returns <= var].mean()

    return cvar


def calculate_win_rate(returns: pd.Series) -> float:
    """
    Calculate win rate (fraction of positive returns)

    Args:
        returns: Period returns

    Returns:
        Win rate (0 to 1)
    """
    trading_returns = returns[returns != 0]

    if len(trading_returns) == 0:
        return 0.0

    win_rate = (trading_returns > 0).sum() / len(trading_returns)

    return win_rate


def calculate_payoff_ratio(returns: pd.Series) -> float:
    """
    Calculate payoff ratio (average win / average loss)

    Args:
        returns: Period returns

    Returns:
        Payoff ratio
    """
    wins = returns[returns > 0]
    losses = returns[returns < 0]

    if len(wins) == 0 or len(losses) == 0:
        return 0.0

    avg_win = wins.mean()
    avg_loss = abs(losses.mean())

    if avg_loss == 0:
        return 0.0

    payoff = avg_win / avg_loss

    return payoff


def calculate_volatility(returns: pd.Series,
                        periods_per_year: int = 252) -> float:
    """
    Calculate annualized volatility

    Args:
        returns: Period returns
        periods_per_year: Trading periods per year

    Returns:
        Annualized volatility
    """
    if len(returns) < 2:
        return 0.0

    vol = returns.std() * np.sqrt(periods_per_year)

    return vol


def calculate_skewness(returns: pd.Series) -> float:
    """Calculate skewness of returns distribution"""
    if len(returns) < 3:
        return 0.0

    return stats.skew(returns.dropna())


def calculate_kurtosis(returns: pd.Series) -> float:
    """Calculate excess kurtosis of returns distribution"""
    if len(returns) < 4:
        return 0.0

    return stats.kurtosis(returns.dropna())


def calculate_rolling_sharpe(returns: pd.Series,
                             window: int = 252,
                             periods_per_year: int = 252) -> pd.Series:
    """
    Calculate rolling Sharpe ratio

    Args:
        returns: Period returns
        window: Rolling window size
        periods_per_year: Trading periods per year

    Returns:
        Rolling Sharpe ratio series
    """
    rolling_mean = returns.rolling(window).mean()
    rolling_std = returns.rolling(window).std()

    rolling_sharpe = (rolling_mean / rolling_std) * np.sqrt(periods_per_year)

    return rolling_sharpe.fillna(0)


def calculate_underwater(returns: pd.Series) -> pd.Series:
    """
    Calculate underwater plot (distance from peak)

    Args:
        returns: Period returns

    Returns:
        Underwater series (always <= 0)
    """
    _, drawdown = calculate_max_drawdown(returns)

    return drawdown


def calculate_recovery_time(returns: pd.Series) -> Optional[int]:
    """
    Calculate time to recover from maximum drawdown

    Args:
        returns: Period returns

    Returns:
        Number of periods to recover, or None if not recovered
    """
    max_dd, drawdown = calculate_max_drawdown(returns)

    if max_dd == 0:
        return 0

    # Find when max DD occurred
    max_dd_idx = drawdown.idxmin()
    max_dd_loc = drawdown.index.get_loc(max_dd_idx)

    # Look for recovery (drawdown back to 0)
    recovery_drawdown = drawdown.iloc[max_dd_loc:]
    recovered = recovery_drawdown[recovery_drawdown >= -0.001]  # Within 0.1%

    if len(recovered) > 0:
        recovery_idx = recovered.index[0]
        recovery_loc = drawdown.index.get_loc(recovery_idx)
        recovery_time = recovery_loc - max_dd_loc
        return recovery_time
    else:
        return None  # Not yet recovered


def calculate_comprehensive_metrics(returns: pd.Series,
                                   risk_free_rate: float = 0.0,
                                   periods_per_year: int = 252) -> Dict:
    """
    Calculate comprehensive performance metrics

    Args:
        returns: Period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Dictionary with all metrics
    """
    if len(returns) < 2:
        return {}

    # Basic returns
    total_return = calculate_cumulative_returns(returns).iloc[-1]
    years = len(returns) / periods_per_year
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0

    # Risk metrics
    volatility = calculate_volatility(returns, periods_per_year)
    sharpe = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    sortino = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)

    # Drawdown
    max_dd, _ = calculate_max_drawdown(returns)
    calmar = calculate_calmar_ratio(returns, periods_per_year)

    # Risk measures
    var_95 = calculate_var(returns, 0.95)
    cvar_95 = calculate_cvar(returns, 0.95)

    # Trading statistics
    win_rate = calculate_win_rate(returns)
    payoff_ratio = calculate_payoff_ratio(returns)

    # Distribution
    skew = calculate_skewness(returns)
    kurt = calculate_kurtosis(returns)

    # Recovery
    recovery = calculate_recovery_time(returns)

    metrics = {
        'total_return': total_return * 100,
        'annual_return': annual_return * 100,
        'volatility': volatility * 100,
        'sharpe_ratio': sharpe,
        'sortino_ratio': sortino,
        'calmar_ratio': calmar,
        'max_drawdown': max_dd * 100,
        'var_95': var_95 * 100,
        'cvar_95': cvar_95 * 100,
        'win_rate': win_rate * 100,
        'payoff_ratio': payoff_ratio,
        'skewness': skew,
        'kurtosis': kurt,
        'recovery_days': recovery,
        'num_periods': len(returns)
    }

    return metrics


def calculate_monthly_returns_table(returns: pd.Series) -> pd.DataFrame:
    """
    Calculate monthly returns table (pyfolio-style)

    Args:
        returns: Daily returns

    Returns:
        DataFrame with years as rows, months as columns
    """
    if len(returns) < 1:
        return pd.DataFrame()

    # Ensure datetime index
    if not isinstance(returns.index, pd.DatetimeIndex):
        returns.index = pd.to_datetime(returns.index)

    # Resample to monthly
    monthly = (1 + returns).resample('M').prod() - 1

    # Create table
    table = monthly.to_frame('return')
    table['year'] = table.index.year
    table['month'] = table.index.month

    # Pivot to year x month format
    pivot = table.pivot(index='year', columns='month', values='return')

    # Rename columns to month names
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot.columns = [month_names[m-1] for m in pivot.columns]

    # Add YTD column
    pivot['YTD'] = table.groupby('year')['return'].apply(
        lambda x: (1 + x).prod() - 1
    )

    return pivot * 100  # Convert to percentage


if __name__ == '__main__':
    """Test performance calculations"""

    # Generate sample returns
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', periods=1000, freq='D')
    returns = pd.Series(
        np.random.normal(0.0005, 0.01, 1000),
        index=dates
    )

    print("Testing Performance Calculations")
    print("="*80)

    metrics = calculate_comprehensive_metrics(returns)

    print("\nComprehensive Metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"  {key:20s}: {value:>10.2f}")
        else:
            print(f"  {key:20s}: {value}")

    print("\n" + "="*80)
    print("✅ Performance calculations ready!")
