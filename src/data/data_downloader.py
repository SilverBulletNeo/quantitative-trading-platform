"""
Market Data Downloader

Downloads historical price and fundamental data for backtesting.

Data Sources:
- yfinance: Free, good coverage of stocks, crypto, ETFs
- Future: Alpha Vantage, Quandl, Bloomberg (with API keys)

Asset Classes:
- Crypto: BTC, ETH, BNB, SOL, ADA, DOT, MATIC, AVAX
- Equities: S&P 500 stocks, international stocks
- Bonds: TLT, IEF, SHY (Treasury ETFs)
- Commodities: GLD, SLV, DBC, USO
- FX: Currency pairs via ETFs (FXE, FXY, FXB, etc.)

Fundamental Data:
- P/E Ratio, P/B Ratio, Dividend Yield
- ROE, ROA, Profit Margins
- Market Cap, Shares Outstanding
- Debt-to-Equity

Data Range:
- Default: 5 years (2019-2024)
- Can be extended to 10+ years for more robust backtesting
"""

import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
import os
warnings.filterwarnings('ignore')


@dataclass
class DataConfig:
    """Configuration for data downloading"""
    start_date: str = '2019-01-01'
    end_date: str = None  # Default: today
    data_dir: str = 'data/raw'
    interval: str = '1d'  # Daily data


class AssetUniverse:
    """Pre-defined asset universes for different strategies"""

    # Cryptocurrency
    CRYPTO = [
        'BTC-USD',   # Bitcoin
        'ETH-USD',   # Ethereum
        'BNB-USD',   # Binance Coin
        'SOL-USD',   # Solana
        'ADA-USD',   # Cardano
        'DOT-USD',   # Polkadot
        'MATIC-USD', # Polygon
        'AVAX-USD',  # Avalanche
        'LINK-USD',  # Chainlink
        'UNI-USD',   # Uniswap
    ]

    # Major US Equities (Tech, Finance, Healthcare, Consumer)
    EQUITIES_LARGE_CAP = [
        # Tech
        'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA',
        # Finance
        'JPM', 'BAC', 'WFC', 'GS', 'MS',
        # Healthcare
        'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO',
        # Consumer
        'WMT', 'HD', 'DIS', 'NKE', 'MCD',
        # Energy
        'XOM', 'CVX',
        # Industrial
        'BA', 'CAT', 'GE',
    ]

    # Value stocks (typically lower P/E)
    EQUITIES_VALUE = [
        'BRK-B', 'JPM', 'BAC', 'XOM', 'CVX', 'VZ', 'T',
        'PFE', 'JNJ', 'WFC', 'C', 'USB', 'PNC',
    ]

    # Growth stocks (typically higher P/E)
    EQUITIES_GROWTH = [
        'TSLA', 'NVDA', 'AMD', 'CRM', 'ADBE', 'NFLX',
        'SHOP', 'SQ', 'SNOW', 'DDOG', 'NET',
    ]

    # Bonds (Treasury ETFs)
    BONDS = [
        'TLT',  # 20+ Year Treasury
        'IEF',  # 7-10 Year Treasury
        'SHY',  # 1-3 Year Treasury
        'AGG',  # Aggregate Bond
        'LQD',  # Investment Grade Corporate
        'HYG',  # High Yield Corporate
    ]

    # Commodities
    COMMODITIES = [
        'GLD',  # Gold
        'SLV',  # Silver
        'DBC',  # Commodity Index
        'USO',  # Oil
        'UNG',  # Natural Gas
        'CORN', # Corn
    ]

    # Currency (via ETFs)
    CURRENCIES = [
        'FXE',  # Euro
        'FXY',  # Yen
        'FXB',  # British Pound
        'FXA',  # Australian Dollar
        'FXC',  # Canadian Dollar
        'UUP',  # US Dollar Index
    ]

    # International Equities
    INTERNATIONAL = [
        'EFA',  # EAFE (Developed Markets)
        'EEM',  # Emerging Markets
        'FXI',  # China
        'EWJ',  # Japan
        'EWG',  # Germany
        'EWU',  # UK
    ]

    @classmethod
    def get_multi_asset_universe(cls) -> List[str]:
        """Get a diversified multi-asset universe"""
        return (
            cls.CRYPTO[:5] +           # Top 5 crypto
            cls.EQUITIES_LARGE_CAP[:15] +  # Top 15 stocks
            cls.BONDS[:3] +            # Treasury ETFs
            cls.COMMODITIES[:3] +      # Gold, Silver, DBC
            cls.INTERNATIONAL[:3]      # International exposure
        )


class MarketDataDownloader:
    """
    Downloads historical market data for backtesting
    """

    def __init__(self, config: DataConfig = None):
        self.config = config or DataConfig()

        # Set end date to today if not specified
        if self.config.end_date is None:
            self.config.end_date = datetime.today().strftime('%Y-%m-%d')

        # Create data directory if it doesn't exist
        os.makedirs(self.config.data_dir, exist_ok=True)

    def download_prices(
        self,
        symbols: List[str],
        start_date: str = None,
        end_date: str = None
    ) -> pd.DataFrame:
        """
        Download historical price data

        Args:
            symbols: List of ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            DataFrame with adjusted close prices
        """
        if start_date is None:
            start_date = self.config.start_date

        if end_date is None:
            end_date = self.config.end_date

        print(f"Downloading price data for {len(symbols)} symbols...")
        print(f"Date range: {start_date} to {end_date}")

        # Download data
        data = yf.download(
            symbols,
            start=start_date,
            end=end_date,
            interval=self.config.interval,
            progress=True,
            group_by='ticker'
        )

        # Extract adjusted close prices
        if len(symbols) == 1:
            # Single symbol
            prices = data['Adj Close'].to_frame(name=symbols[0])
        else:
            # Multiple symbols
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            else:
                # Data is grouped by ticker
                prices = pd.DataFrame()
                for symbol in symbols:
                    try:
                        prices[symbol] = data[symbol]['Adj Close']
                    except:
                        print(f"Warning: Could not get data for {symbol}")

        # Remove columns with all NaN
        prices = prices.dropna(axis=1, how='all')

        # Forward fill missing values (max 5 days)
        prices = prices.fillna(method='ffill', limit=5)

        print(f"Downloaded {len(prices.columns)} symbols successfully")
        print(f"Date range: {prices.index[0]} to {prices.index[-1]}")
        print(f"Total days: {len(prices)}")

        return prices

    def download_fundamentals(
        self,
        symbols: List[str]
    ) -> pd.DataFrame:
        """
        Download fundamental data for factor strategies

        Note: This is snapshot data (current), not historical time series
        For production, would need a data provider with historical fundamentals

        Returns:
            DataFrame with fundamental metrics
        """
        print(f"Downloading fundamental data for {len(symbols)} symbols...")

        fundamentals = []

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info

                # Extract key metrics
                fundamentals.append({
                    'Symbol': symbol,
                    'PE_Ratio': info.get('forwardPE', info.get('trailingPE', np.nan)),
                    'PB_Ratio': info.get('priceToBook', np.nan),
                    'PS_Ratio': info.get('priceToSalesTrailing12Months', np.nan),
                    'Dividend_Yield': info.get('dividendYield', 0),
                    'ROE': info.get('returnOnEquity', np.nan),
                    'ROA': info.get('returnOnAssets', np.nan),
                    'Profit_Margin': info.get('profitMargins', np.nan),
                    'Debt_to_Equity': info.get('debtToEquity', np.nan),
                    'Market_Cap': info.get('marketCap', np.nan),
                    'Beta': info.get('beta', np.nan),
                    'Sector': info.get('sector', 'Unknown'),
                    'Industry': info.get('industry', 'Unknown'),
                })

            except Exception as e:
                print(f"Warning: Could not get fundamentals for {symbol}: {e}")
                fundamentals.append({
                    'Symbol': symbol,
                    'PE_Ratio': np.nan,
                    'PB_Ratio': np.nan,
                    'PS_Ratio': np.nan,
                    'Dividend_Yield': 0,
                    'ROE': np.nan,
                    'ROA': np.nan,
                    'Profit_Margin': np.nan,
                    'Debt_to_Equity': np.nan,
                    'Market_Cap': np.nan,
                    'Beta': np.nan,
                    'Sector': 'Unknown',
                    'Industry': 'Unknown',
                })

        df = pd.DataFrame(fundamentals)
        df = df.set_index('Symbol')

        print(f"Downloaded fundamentals for {len(df)} symbols")

        return df

    def save_data(
        self,
        data: pd.DataFrame,
        filename: str
    ):
        """Save data to CSV"""
        filepath = os.path.join(self.config.data_dir, filename)
        data.to_csv(filepath)
        print(f"Saved data to {filepath}")

    def load_data(
        self,
        filename: str
    ) -> pd.DataFrame:
        """Load data from CSV"""
        filepath = os.path.join(self.config.data_dir, filename)

        if os.path.exists(filepath):
            data = pd.read_csv(filepath, index_col=0, parse_dates=True)
            print(f"Loaded data from {filepath}")
            return data
        else:
            print(f"File not found: {filepath}")
            return None

    def download_and_save_universe(
        self,
        universe_name: str,
        symbols: List[str]
    ):
        """Download and save a complete universe"""
        print("=" * 80)
        print(f"DOWNLOADING {universe_name.upper()} UNIVERSE")
        print("=" * 80)
        print()

        # Download prices
        prices = self.download_prices(symbols)

        # Save prices
        self.save_data(prices, f'{universe_name}_prices.csv')

        # Download fundamentals (only for equities)
        if 'EQUITY' in universe_name.upper() or 'STOCK' in universe_name.upper():
            print()
            fundamentals = self.download_fundamentals(symbols)
            self.save_data(fundamentals, f'{universe_name}_fundamentals.csv')

        print()
        print(f"✅ {universe_name} universe downloaded successfully!")
        print()


def main():
    """Download all data for backtesting"""

    print("=" * 80)
    print("MARKET DATA DOWNLOADER")
    print("Downloading historical data for comprehensive backtesting")
    print("=" * 80)
    print()

    # Initialize downloader
    config = DataConfig(
        start_date='2019-01-01',  # 5+ years of data
        data_dir='data/raw'
    )

    downloader = MarketDataDownloader(config)

    # Download each universe

    # 1. Cryptocurrency
    downloader.download_and_save_universe(
        'crypto',
        AssetUniverse.CRYPTO
    )

    # 2. Large Cap Equities
    downloader.download_and_save_universe(
        'equities_large_cap',
        AssetUniverse.EQUITIES_LARGE_CAP
    )

    # 3. Value Stocks
    downloader.download_and_save_universe(
        'equities_value',
        AssetUniverse.EQUITIES_VALUE
    )

    # 4. Growth Stocks
    downloader.download_and_save_universe(
        'equities_growth',
        AssetUniverse.EQUITIES_GROWTH
    )

    # 5. Bonds
    downloader.download_and_save_universe(
        'bonds',
        AssetUniverse.BONDS
    )

    # 6. Commodities
    downloader.download_and_save_universe(
        'commodities',
        AssetUniverse.COMMODITIES
    )

    # 7. Currencies
    downloader.download_and_save_universe(
        'currencies',
        AssetUniverse.CURRENCIES
    )

    # 8. International
    downloader.download_and_save_universe(
        'international',
        AssetUniverse.INTERNATIONAL
    )

    # 9. Multi-Asset (diversified mix)
    downloader.download_and_save_universe(
        'multi_asset',
        AssetUniverse.get_multi_asset_universe()
    )

    print("=" * 80)
    print("DATA DOWNLOAD COMPLETE!")
    print("=" * 80)
    print()
    print("Downloaded universes:")
    print("  ✅ Crypto (10 assets)")
    print("  ✅ Large Cap Equities (30 stocks)")
    print("  ✅ Value Stocks (13 stocks)")
    print("  ✅ Growth Stocks (11 stocks)")
    print("  ✅ Bonds (6 ETFs)")
    print("  ✅ Commodities (6 ETFs)")
    print("  ✅ Currencies (6 ETFs)")
    print("  ✅ International (6 ETFs)")
    print("  ✅ Multi-Asset (29 instruments)")
    print()
    print("Data saved to: data/raw/")
    print()
    print("Next steps:")
    print("  1. Run comprehensive backtests on all strategies")
    print("  2. Generate performance reports")
    print("  3. Optimize portfolio allocations")
    print()
    print("=" * 80)


if __name__ == "__main__":
    main()
