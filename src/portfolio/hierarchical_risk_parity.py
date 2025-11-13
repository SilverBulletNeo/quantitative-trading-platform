"""
Hierarchical Risk Parity (HRP)

Machine learning-based portfolio allocation using hierarchical clustering.
Developed by Marcos López de Prado (2016).

Key Innovation:
- Traditional mean-variance requires matrix inversion → unstable
- HRP uses hierarchical clustering → no inversion needed
- Based on graph theory and information theory
- More robust to estimation errors

Academic Foundation:
- López de Prado (2016) - "Building Diversified Portfolios that Outperform Out-of-Sample"
- Published in Journal of Portfolio Management
- Used by quant hedge funds (AQR, Two Sigma, etc.)

Algorithm Steps:
1. Distance Matrix - Convert correlation to distance
2. Hierarchical Clustering - Build dendrogram (tree structure)
3. Quasi-Diagonalization - Sort covariance matrix by clusters
4. Recursive Bisection - Allocate weights top-down

Why HRP Works:
- Discovers natural asset groupings (clusters)
- Allocates within and between clusters optimally
- Avoids numerical instability of matrix inversion
- Better out-of-sample performance than mean-variance
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy.cluster.hierarchy import dendrogram, linkage, leaves_list
from scipy.spatial.distance import squareform
import warnings
warnings.filterwarnings('ignore')


@dataclass
class HRPConfig:
    """Configuration for Hierarchical Risk Parity"""
    linkage_method: str = 'single'  # 'single', 'complete', 'average', 'ward'
    distance_metric: str = 'correlation'  # 'correlation' or 'euclidean'
    max_clusters: int = None  # Maximum number of clusters (None = auto)
    min_cluster_size: int = 2  # Minimum assets per cluster


class HierarchicalRiskParity:
    """
    Hierarchical Risk Parity Portfolio Allocation

    Uses machine learning (hierarchical clustering) to build optimal portfolios.
    """

    def __init__(self, config: HRPConfig = None):
        self.config = config or HRPConfig()

    def calculate_covariance(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate annualized covariance matrix"""
        cov_matrix = returns.cov() * 252  # Annualize
        return cov_matrix

    def calculate_correlation(self, returns: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix"""
        return returns.corr()

    def correlation_to_distance(self, corr_matrix: pd.DataFrame) -> np.ndarray:
        """
        Convert correlation matrix to distance matrix

        Distance = sqrt(0.5 * (1 - correlation))

        Why this formula?
        - Correlation of 1.0 → Distance of 0 (perfectly similar)
        - Correlation of 0.0 → Distance of 0.707 (independent)
        - Correlation of -1.0 → Distance of 1.0 (perfectly opposite)
        """
        dist_matrix = np.sqrt(0.5 * (1 - corr_matrix))
        return dist_matrix

    def cluster_assets(
        self,
        dist_matrix: np.ndarray,
        method: str = None
    ) -> np.ndarray:
        """
        Hierarchical clustering of assets

        Args:
            dist_matrix: Distance matrix (square)
            method: Linkage method ('single', 'complete', 'average', 'ward')

        Returns:
            Linkage matrix (tree structure)
        """
        if method is None:
            method = self.config.linkage_method

        # Convert to condensed form (required by scipy)
        dist_condensed = squareform(dist_matrix, checks=False)

        # Perform hierarchical clustering
        link_matrix = linkage(dist_condensed, method=method)

        return link_matrix

    def quasi_diagonalize(
        self,
        link_matrix: np.ndarray,
        asset_names: List[str]
    ) -> List[str]:
        """
        Quasi-diagonalize: Sort assets by hierarchical clustering

        This orders assets so similar assets are adjacent.
        Result: Covariance matrix becomes quasi-diagonal (block structure).

        Args:
            link_matrix: Hierarchical clustering linkage matrix
            asset_names: List of asset names

        Returns:
            Sorted list of asset names
        """
        # Get leaf order from dendrogram
        sort_idx = leaves_list(link_matrix)

        # Sort asset names
        sorted_assets = [asset_names[i] for i in sort_idx]

        return sorted_assets

    def get_cluster_variance(
        self,
        cov_matrix: pd.DataFrame,
        cluster_assets: List[str]
    ) -> float:
        """
        Calculate variance of a cluster

        Uses inverse-variance weighting within the cluster.
        """
        # Get cluster covariance sub-matrix
        cluster_cov = cov_matrix.loc[cluster_assets, cluster_assets]

        # Calculate inverse-variance weights
        inv_diag = 1 / np.diag(cluster_cov)
        weights = inv_diag / inv_diag.sum()

        # Cluster variance
        cluster_var = np.dot(weights, np.dot(cluster_cov, weights))

        return cluster_var

    def recursive_bisection(
        self,
        sorted_assets: List[str],
        cov_matrix: pd.DataFrame
    ) -> pd.Series:
        """
        Recursive Bisection: Core HRP algorithm

        Recursively split asset tree and allocate weights:
        1. Start with all assets (weight = 1.0)
        2. Split into two clusters (left and right)
        3. Allocate weight between clusters using inverse-variance
        4. Recurse on each cluster

        Args:
            sorted_assets: Assets sorted by hierarchical clustering
            cov_matrix: Covariance matrix

        Returns:
            Series of weights (sum to 1.0)
        """
        # Initialize weights
        weights = pd.Series(1.0, index=sorted_assets)

        # List of clusters to process (start with all assets)
        clusters = [sorted_assets]

        while len(clusters) > 0:
            # Process each cluster
            new_clusters = []

            for cluster in clusters:
                if len(cluster) > 1:
                    # Split cluster in half
                    mid = len(cluster) // 2
                    left_cluster = cluster[:mid]
                    right_cluster = cluster[mid:]

                    # Calculate variance of each sub-cluster
                    left_var = self.get_cluster_variance(cov_matrix, left_cluster)
                    right_var = self.get_cluster_variance(cov_matrix, right_cluster)

                    # Allocate weight using inverse-variance
                    # Lower variance → Higher weight (risk parity logic)
                    alpha = 1 - left_var / (left_var + right_var)

                    # Current cluster weight
                    cluster_weight = weights[cluster[0]]  # All assets in cluster have same weight at this level

                    # Update weights
                    for asset in left_cluster:
                        weights[asset] *= alpha

                    for asset in right_cluster:
                        weights[asset] *= (1 - alpha)

                    # Add sub-clusters for further processing
                    new_clusters.append(left_cluster)
                    new_clusters.append(right_cluster)

            clusters = new_clusters

        return weights

    def optimize_portfolio(
        self,
        prices: pd.DataFrame,
        method: str = None
    ) -> Dict:
        """
        Main HRP optimization function

        Args:
            prices: DataFrame of asset prices
            method: Linkage method (overrides config)

        Returns:
            Dict with weights, linkage, sorted_assets, etc.
        """
        if method is None:
            method = self.config.linkage_method

        # Calculate returns
        returns = prices.pct_change().dropna()

        # 1. Calculate correlation and covariance
        corr_matrix = self.calculate_correlation(returns)
        cov_matrix = self.calculate_covariance(returns)

        # 2. Convert correlation to distance
        dist_matrix = self.correlation_to_distance(corr_matrix)

        # 3. Hierarchical clustering
        link_matrix = self.cluster_assets(dist_matrix, method=method)

        # 4. Quasi-diagonalize (sort assets by clusters)
        sorted_assets = self.quasi_diagonalize(link_matrix, list(prices.columns))

        # 5. Recursive bisection to allocate weights
        weights = self.recursive_bisection(sorted_assets, cov_matrix)

        # Calculate portfolio metrics
        portfolio_var = np.dot(weights.values, np.dot(cov_matrix, weights.values))
        portfolio_vol = np.sqrt(portfolio_var)

        return {
            'weights': weights,
            'sorted_assets': sorted_assets,
            'linkage_matrix': link_matrix,
            'correlation_matrix': corr_matrix,
            'covariance_matrix': cov_matrix,
            'distance_matrix': dist_matrix,
            'portfolio_volatility': portfolio_vol,
            'method': method
        }

    def plot_dendrogram(
        self,
        link_matrix: np.ndarray,
        asset_names: List[str]
    ):
        """
        Plot dendrogram (hierarchical tree)

        Visualizes asset clustering structure.
        """
        try:
            import matplotlib.pyplot as plt

            plt.figure(figsize=(12, 6))
            dendrogram(
                link_matrix,
                labels=asset_names,
                leaf_rotation=90,
                leaf_font_size=10
            )
            plt.title('Hierarchical Clustering Dendrogram')
            plt.xlabel('Assets')
            plt.ylabel('Distance')
            plt.tight_layout()
            plt.show()

        except ImportError:
            print("Matplotlib not available. Install with: pip install matplotlib")

    def compare_methods(self, prices: pd.DataFrame) -> pd.DataFrame:
        """
        Compare different linkage methods

        Tests: single, complete, average, ward
        """
        methods = ['single', 'complete', 'average', 'ward']
        results = []

        for method in methods:
            try:
                result = self.optimize_portfolio(prices, method=method)

                # Calculate weight dispersion (how concentrated are weights?)
                weight_std = result['weights'].std()
                weight_max = result['weights'].max()
                weight_min = result['weights'].min()

                results.append({
                    'Method': method,
                    'Portfolio Vol': result['portfolio_volatility'],
                    'Max Weight': weight_max,
                    'Min Weight': weight_min,
                    'Weight Std': weight_std,
                    'Weight Concentration': weight_max / weight_min
                })

            except Exception as e:
                print(f"Method {method} failed: {e}")
                continue

        return pd.DataFrame(results)


if __name__ == "__main__":
    """Test Hierarchical Risk Parity"""

    print("=" * 80)
    print("HIERARCHICAL RISK PARITY (HRP) TEST")
    print("=" * 80)
    print()

    # Create synthetic data with cluster structure
    print("Creating synthetic test data...")
    print("- Cluster 1: Tech stocks (high correlation)")
    print("- Cluster 2: Bonds (high correlation)")
    print("- Cluster 3: Commodities (high correlation)")
    print()

    np.random.seed(42)
    n_days = 252 * 3  # 3 years

    # Cluster 1: Tech stocks (correlated)
    base_tech = np.random.normal(0.001, 0.02, n_days)
    tech_1 = base_tech + np.random.normal(0, 0.005, n_days)
    tech_2 = base_tech + np.random.normal(0, 0.005, n_days)
    tech_3 = base_tech + np.random.normal(0, 0.005, n_days)

    # Cluster 2: Bonds (correlated, low vol)
    base_bond = np.random.normal(0.0003, 0.005, n_days)
    bond_1 = base_bond + np.random.normal(0, 0.001, n_days)
    bond_2 = base_bond + np.random.normal(0, 0.001, n_days)

    # Cluster 3: Commodities (correlated, medium vol)
    base_commodity = np.random.normal(0.0005, 0.015, n_days)
    commodity_1 = base_commodity + np.random.normal(0, 0.003, n_days)
    commodity_2 = base_commodity + np.random.normal(0, 0.003, n_days)

    # Create price series
    returns_data = {
        'AAPL': tech_1,
        'MSFT': tech_2,
        'GOOGL': tech_3,
        'TLT': bond_1,
        'IEF': bond_2,
        'GLD': commodity_1,
        'DBC': commodity_2
    }

    returns_df = pd.DataFrame(returns_data)
    prices = (1 + returns_df).cumprod() * 100

    # Initialize HRP
    hrp = HierarchicalRiskParity()

    # Calculate correlation matrix
    print("=" * 80)
    print("CORRELATION MATRIX")
    print("=" * 80)
    print()

    corr = returns_df.corr()
    print(corr.round(2))

    # Run HRP optimization
    print("\n" + "=" * 80)
    print("HRP OPTIMIZATION (Single Linkage)")
    print("=" * 80)

    result = hrp.optimize_portfolio(prices, method='single')

    print("\nAsset Clustering Order:")
    print("(Similar assets are grouped together)")
    for i, asset in enumerate(result['sorted_assets']):
        print(f"  {i+1}. {asset}")

    print("\n" + "-" * 80)
    print("Portfolio Weights:")
    print("-" * 80)

    for asset, weight in result['weights'].sort_values(ascending=False).items():
        print(f"  {asset:10s}: {weight:>6.2%}")

    print(f"\nPortfolio Volatility: {result['portfolio_volatility']:.2%}")

    # Compare with equal weight
    print("\n" + "=" * 80)
    print("COMPARISON: HRP vs Equal Weight")
    print("=" * 80)

    # Equal weight
    n_assets = len(prices.columns)
    equal_weights = np.array([1/n_assets] * n_assets)
    cov_matrix = returns_df.cov() * 252
    equal_vol = np.sqrt(np.dot(equal_weights, np.dot(cov_matrix, equal_weights)))

    comparison = pd.DataFrame({
        'Method': ['Equal Weight', 'HRP'],
        'Portfolio Vol': [equal_vol, result['portfolio_volatility']],
        'Max Weight': [1/n_assets, result['weights'].max()],
        'Min Weight': [1/n_assets, result['weights'].min()],
        'Weight Dispersion': [0.0, result['weights'].std()]
    })

    print()
    print(comparison.to_string(index=False))

    # Compare all linkage methods
    print("\n" + "=" * 80)
    print("COMPARING LINKAGE METHODS")
    print("=" * 80)
    print()

    method_comparison = hrp.compare_methods(prices)
    print(method_comparison.round(4).to_string(index=False))

    print("\n" + "=" * 80)
    print("KEY INSIGHTS")
    print("=" * 80)
    print()
    print("HRP Advantages:")
    print("  - No matrix inversion (numerically stable)")
    print("  - Discovers asset clusters automatically")
    print("  - More diversified than equal weight")
    print("  - Better out-of-sample performance than mean-variance")
    print()
    print("Clustering Structure:")
    print("  - Tech stocks grouped together (high correlation)")
    print("  - Bonds grouped together (low volatility)")
    print("  - Commodities grouped separately")
    print("  - Allocates between clusters using inverse-variance")
    print()
    print("Linkage Methods:")
    print("  - Single: Fast, sensitive to outliers")
    print("  - Complete: Compact clusters")
    print("  - Average: Balanced approach (recommended)")
    print("  - Ward: Minimizes within-cluster variance")
    print()
    print("=" * 80)
    print("Test complete!")
    print("=" * 80)
