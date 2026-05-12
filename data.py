import streamlit as st
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from urllib.parse import quote

FACTORSTODAY_BASE = "https://www.factorstoday.com/api"


@st.cache_data(ttl=3600)
def fetch_prices(tickers: list[str], start_date: str | None = None, period_days: int = 365) -> pd.DataFrame:
    all_tickers = list(set(tickers + ["SPY"]))
    if start_date:
        start = pd.to_datetime(start_date) - timedelta(days=5)
    else:
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


def compute_trade_metrics(prices: pd.DataFrame, trades: list[dict]) -> pd.DataFrame:
    """
    trades: [{"ticker": str, "shares": float, "entry_date": str | None}]
    Negative shares = short position. Returns per-trade P&L metrics.
    """
    rows = []
    for trade in trades:
        ticker = trade["ticker"]
        shares = trade["shares"]
        entry_date_str = trade.get("entry_date")

        if ticker not in prices.columns:
            continue

        series = prices[ticker].dropna()
        if series.empty:
            continue

        direction = "Long" if shares >= 0 else "Short"
        abs_shares = abs(shares)

        if entry_date_str:
            entry_dt = pd.to_datetime(entry_date_str)
            valid = series[series.index >= entry_dt]
            if valid.empty:
                rows.append({
                    "Ticker": ticker, "Direction": direction, "Shares": abs_shares,
                    "Entry Date": entry_date_str, "Entry Price": None,
                    "Current Price": round(float(series.iloc[-1]), 2),
                    "P&L ($)": None, "Return (%)": None,
                    "Ann. Return (%)": None, "Days Held": None,
                    "Max Adverse Move (%)": None,
                })
                continue
            entry_price = float(valid.iloc[0])
            actual_entry_dt = valid.index[0]
        else:
            entry_price = float(series.iloc[0])
            actual_entry_dt = series.index[0]

        trade_series = series[series.index >= actual_entry_dt]
        current_price = float(series.iloc[-1])
        days_held = max((series.index[-1] - actual_entry_dt).days, 1)

        pnl_total = shares * (current_price - entry_price)
        cost_basis = abs_shares * entry_price
        position_return = pnl_total / cost_basis * 100
        ann_return = ((1 + position_return / 100) ** (365.25 / days_held) - 1) * 100

        if direction == "Long":
            worst_price = float(trade_series.min())
            max_adverse = (worst_price - entry_price) / entry_price * 100
        else:
            worst_price = float(trade_series.max())
            max_adverse = -((worst_price - entry_price) / entry_price * 100)

        rows.append({
            "Ticker": ticker,
            "Direction": direction,
            "Shares": abs_shares,
            "Entry Date": actual_entry_dt.strftime("%Y-%m-%d"),
            "Entry Price": round(entry_price, 2),
            "Current Price": round(current_price, 2),
            "P&L ($)": round(pnl_total, 2),
            "Return (%)": round(position_return, 2),
            "Ann. Return (%)": round(ann_return, 1),
            "Days Held": days_held,
            "Max Adverse Move (%)": round(max_adverse, 2),
        })

    return pd.DataFrame(rows)


def compute_portfolio_pnl_series(prices: pd.DataFrame, trades: list[dict]) -> pd.DataFrame:
    """
    Returns a DataFrame of daily dollar P&L per position and a 'Total' column,
    starting from the earliest entry date found in trades.
    Only includes trades that have an entry_date.
    """
    dated = [t for t in trades if t.get("entry_date")]
    if not dated:
        return pd.DataFrame()

    earliest = min(pd.to_datetime(t["entry_date"]) for t in dated)
    date_index = prices.index[prices.index >= earliest]
    if date_index.empty:
        return pd.DataFrame()

    result = pd.DataFrame(index=date_index)

    for trade in dated:
        ticker = trade["ticker"]
        shares = trade["shares"]
        entry_date_str = trade["entry_date"]

        if ticker not in prices.columns:
            continue

        series = prices[ticker].dropna()
        entry_dt = pd.to_datetime(entry_date_str)
        valid = series[series.index >= entry_dt]
        if valid.empty:
            continue

        entry_price = float(valid.iloc[0])
        actual_entry_dt = valid.index[0]
        active = series[series.index >= actual_entry_dt]
        pnl = shares * (active - entry_price)
        result[ticker] = pnl

    result = result.fillna(0.0)
    result["Total"] = result.sum(axis=1)
    # Zero out each position before its entry date (already handled by fillna, but make explicit)
    for trade in dated:
        ticker = trade["ticker"]
        if ticker not in result.columns:
            continue
        entry_dt = pd.to_datetime(trade["entry_date"])
        result.loc[result.index < entry_dt, ticker] = 0.0
    result["Total"] = result[[c for c in result.columns if c != "Total"]].sum(axis=1)

    return result
