import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

FACTOR_PROXIES = {
    "Rates/Duration": "TLT",
    "Oil/Commodities": "USO",
    "USD Strength": "UUP",
    "Risk Appetite": "HYG",
    "Small Cap / Cyclicality": "IWM",
}

# Style factors: each is (long leg, benchmark). Daily return = long - benchmark,
# giving a market-neutral factor return for each style tilt.
STYLE_FACTOR_PROXIES = {
    "Growth": ("IWF", "SPY"),
    "Value": ("IWD", "SPY"),
    "Momentum": ("MTUM", "SPY"),
    "Small-Cap": ("IJR", "SPY"),
    "Low Volatility": ("USMV", "SPY"),
    "High Dividend Yield": ("VYM", "SPY"),
}

SECTOR_ETFS = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLE": "Energy",
    "XLV": "Healthcare",
    "XLI": "Industrials",
    "XLY": "Consumer Disc.",
    "XLP": "Consumer Staples",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLB": "Materials",
    "XLC": "Comm. Services",
}


@st.cache_data(ttl=3600)
def fetch_prices(tickers: list[str], period_days: int = 365) -> pd.DataFrame:
    style_etfs = [etf for pair in STYLE_FACTOR_PROXIES.values() for etf in pair]
    all_tickers = list(set(tickers + list(FACTOR_PROXIES.values()) + list(SECTOR_ETFS.keys()) + style_etfs + ["SPY"]))
    start = datetime.today() - timedelta(days=period_days)
    raw = yf.download(all_tickers, start=start.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = all_tickers
    prices = prices.dropna(how="all")
    return prices


def compute_returns(prices: pd.DataFrame, tickers: list[str]) -> pd.DataFrame:
    periods = {"1D": 1, "1W": 5, "1M": 21, "3M": 63}
    rows = []
    for ticker in tickers:
        if ticker not in prices.columns:
            continue
        row = {"Ticker": ticker}
        series = prices[ticker].dropna()
        for label, days in periods.items():
            if len(series) > days:
                ret = (series.iloc[-1] / series.iloc[-1 - days] - 1) * 100
                row[label] = round(ret, 2)
            else:
                row[label] = None
        rows.append(row)
    return pd.DataFrame(rows).set_index("Ticker")


@st.cache_data(ttl=86400)
def get_sector_info(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        try:
            info = yf.Ticker(ticker).info
            rows.append({
                "Ticker": ticker,
                "Sector": info.get("sector", "Unknown"),
                "Industry": info.get("industry", "Unknown"),
                "Market Cap": info.get("marketCap"),
            })
        except Exception:
            rows.append({"Ticker": ticker, "Sector": "Unknown", "Industry": "Unknown", "Market Cap": None})
    return pd.DataFrame(rows).set_index("Ticker")
