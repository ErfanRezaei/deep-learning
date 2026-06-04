"""Download daily OHLC stock data with yfinance for Part 1.

Period: January 2020 -- December 2025 (as required by the assignment).
Saves one CSV per ticker under ``data/`` next to this file.
"""
import argparse
import os

import pandas as pd
import yfinance as yf

DEFAULT_TICKERS = ["AAPL", "MSFT", "JPM"]

START_DATE = "2020-01-01"
END_DATE = "2025-12-31"

FEATURES = ["Open", "High", "Low", "Close"]


def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download daily OHLC data for a single ticker and return a clean frame."""
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=True,
        progress=False,
    )
    if df.empty:
        raise RuntimeError(f"No data returned for {ticker}.")

    # yfinance may return a MultiIndex column frame for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df[FEATURES].copy()
    df.dropna(inplace=True)
    df.index.name = "Date"
    return df


def main():
    parser = argparse.ArgumentParser(description="Download OHLC stock data.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start", default=START_DATE)
    parser.add_argument("--end", default=END_DATE)
    parser.add_argument(
        "--out-dir",
        default=os.path.join(os.path.dirname(__file__), "data"),
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    for ticker in args.tickers:
        df = download_ticker(ticker, args.start, args.end)
        path = os.path.join(args.out_dir, f"{ticker}.csv")
        df.to_csv(path)
        print(f"{ticker}: {len(df)} rows  ({df.index.min().date()} -> "
              f"{df.index.max().date()})  saved to {path}")


if __name__ == "__main__":
    main()
