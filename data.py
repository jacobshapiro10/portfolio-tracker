import streamlit as st
import requests
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from urllib.parse import quote

FACTORSTODAY_BASE = "https://www.factorstoday.com/api"


@st.cache_data(ttl=3600)
def fetch_prices(tickers: list[str], period_days: int = 365) -> pd.DataFrame:
    all_tickers = list(set(tickers + ["SPY"]))
    start = datetime.today() - timedelta(days=period_days)
    raw = yf.download(all_tickers, start=start.strftime("%Y-%m-%d"), auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = all_tickers
    return prices.dropna(how="all")


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


@st.cache_data(ttl=3600)
def fetch_portfolio_exposures(holdings_tuple: tuple, model: str = "Base + Sector") -> dict:
    holdings = [{"ticker": t, "shares": s} for t, s in holdings_tuple]
    resp = requests.post(
        f"{FACTORSTODAY_BASE}/portfolio/analyze",
        params={"model": model},
        json={"holdings": holdings},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=3600)
def fetch_factor_returns() -> dict:
    resp = requests.get(f"{FACTORSTODAY_BASE}/factor-returns/historic", timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=60)
def fetch_factor_returns_intraday() -> dict:
    resp = requests.get(f"{FACTORSTODAY_BASE}/factor-returns/intraday", timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=3600)
def fetch_factor_covariance(window: int = 252) -> dict:
    resp = requests.get(
        f"{FACTORSTODAY_BASE}/factor-covariance",
        params={"window": window, "annualized": "true"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=86400)
def fetch_stock_specific_vol(ticker: str) -> dict:
    resp = requests.get(f"{FACTORSTODAY_BASE}/stock-specific-vol/{ticker}", timeout=15)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=3600)
def fetch_factor_history(factor_id: str, days: int = 252) -> list:
    resp = requests.get(
        f"{FACTORSTODAY_BASE}/factor-history/{quote(factor_id)}",
        params={"days": days},
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()
