"""
Simple Market Data Downloader

Downloads data one symbol at a time for better error handling.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import os
import time

# Asset universes
CRYPTO = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'SOL-USD', 'ADA-USD']
EQUITIES = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'JPM', 'BAC', 'WFC']
BONDS = ['TLT', 'IEF', 'SHY']
COMMODITIES = ['GLD', 'SLV', 'DBC']

def download_single_symbol(symbol, start_date='2020-01-01', end_date=None):
    """Download data for a single symbol"""
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')

    try:
        print(f"Downloading {symbol}...", end=' ')
        ticker = yf.Ticker(symbol)
        data = ticker.history(start=start_date, end=end_date)

        if len(data) > 0:
            print(f"✓ ({len(data)} days)")
            return data['Close']
        else:
            print("✗ No data")
            return None
    except Exception as e:
        print(f"✗ Error: {e}")
        return None

def download_universe(symbols, universe_name):
    """Download data for a universe of symbols"""
    print(f"\n{'='*80}")
    print(f"Downloading {universe_name}")
    print(f"{'='*80}\n")

    prices_dict = {}

    for symbol in symbols:
        prices = download_single_symbol(symbol)
        if prices is not None and len(prices) > 0:
            prices_dict[symbol] = prices
        time.sleep(0.5)  # Rate limiting

    if len(prices_dict) == 0:
        print(f"\n✗ No data downloaded for {universe_name}")
        return None

    # Combine into DataFrame
    prices_df = pd.DataFrame(prices_dict)

    # Forward fill missing values
    prices_df = prices_df.ffill(limit=5)

    # Drop rows with any NaN
    prices_df = prices_df.dropna()

    print(f"\n✅ Downloaded {len(prices_df.columns)} symbols")
    print(f"Date range: {prices_df.index[0].date()} to {prices_df.index[-1].date()}")
    print(f"Total days: {len(prices_df)}")

    # Save to CSV
    os.makedirs('data/raw', exist_ok=True)
    filename = f'data/raw/{universe_name}_prices.csv'
    prices_df.to_csv(filename)
    print(f"Saved to: {filename}")

    return prices_df

def main():
    """Download all data"""
    print("="*80)
    print("SIMPLE DATA DOWNLOADER")
    print("="*80)

    # Download each universe
    crypto_data = download_universe(CRYPTO, 'crypto')
    equities_data = download_universe(EQUITIES, 'equities')
    bonds_data = download_universe(BONDS, 'bonds')
    commodities_data = download_universe(COMMODITIES, 'commodities')

    # Create a multi-asset universe
    if crypto_data is not None and equities_data is not None:
        print(f"\n{'='*80}")
        print("Creating multi-asset universe")
        print(f"{'='*80}\n")

        # Combine all assets
        all_assets = []
        if crypto_data is not None:
            all_assets.append(crypto_data)
        if equities_data is not None:
            all_assets.append(equities_data)
        if bonds_data is not None:
            all_assets.append(bonds_data)
        if commodities_data is not None:
            all_assets.append(commodities_data)

        multi_asset = pd.concat(all_assets, axis=1)
        multi_asset = multi_asset.dropna()

        filename = 'data/raw/multi_asset_prices.csv'
        multi_asset.to_csv(filename)
        print(f"✅ Multi-asset universe: {len(multi_asset.columns)} assets")
        print(f"Saved to: {filename}")

    print(f"\n{'='*80}")
    print("DOWNLOAD COMPLETE!")
    print(f"{'='*80}\n")
    print("Data saved to data/raw/")
    print("\nNext: Run backtests with:")
    print("  python src/backtest/comprehensive_backtester.py --universe crypto")
    print("  python src/backtest/comprehensive_backtester.py --universe equities")
    print()

if __name__ == "__main__":
    main()
