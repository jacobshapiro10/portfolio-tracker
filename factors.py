import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
from data import FACTOR_PROXIES, STYLE_FACTOR_PROXIES


def _build_factor_returns(daily: pd.DataFrame, window: int) -> pd.DataFrame:
    macro_names = {v: k for k, v in FACTOR_PROXIES.items()}
    macro_cols = [f for f in FACTOR_PROXIES.values() if f in daily.columns]
    factor_df = daily[macro_cols].rename(columns=macro_names).copy()

    for name, (long_etf, short_etf) in STYLE_FACTOR_PROXIES.items():
        if long_etf in daily.columns and short_etf in daily.columns:
            factor_df[name] = daily[long_etf] - daily[short_etf]

    return factor_df.tail(window)


@st.cache_data
def compute_factor_betas(
    prices: pd.DataFrame, tickers: list[str], window: int = 60
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    OLS of each stock's daily returns vs all factor returns over trailing `window` days.
    Returns (raw_betas, standardized_betas).
    Standardized beta = raw_beta × (σ_factor / σ_stock): effect of a 1 SD factor move
    in stock SD units, so betas are comparable across factors.
    """
    daily = prices.pct_change().dropna()
    factor_df = _build_factor_returns(daily, window)
    factor_df = factor_df.dropna(how="all")
    factor_names = list(factor_df.columns)
    factor_stds = factor_df.std()

    rows_raw, rows_std = [], []
    for ticker in tickers:
        if ticker not in daily.columns:
            continue
        stock_ret = daily[ticker].tail(window)
        common_idx = stock_ret.dropna().index.intersection(factor_df.dropna(how="all").index)
        if len(common_idx) < 20:
            continue
        y = stock_ret.loc[common_idx].values
        X = factor_df.loc[common_idx].values
        X_with_const = sm.add_constant(X)
        try:
            model = sm.OLS(y, X_with_const).fit()
            betas = model.params[1:]
            stock_std = stock_ret.loc[common_idx].std()
            row_raw = {"Ticker": ticker}
            row_std = {"Ticker": ticker}
            for i, fname in enumerate(factor_names):
                row_raw[fname] = round(betas[i], 3)
                row_std[fname] = round(betas[i] * factor_stds[fname] / stock_std, 3) if stock_std > 0 else 0.0
            row_raw["R²"] = round(model.rsquared, 3)
            row_std["R²"] = round(model.rsquared, 3)
            rows_raw.append(row_raw)
            rows_std.append(row_std)
        except Exception:
            continue

    if not rows_raw:
        return pd.DataFrame(), pd.DataFrame()
    return (
        pd.DataFrame(rows_raw).set_index("Ticker"),
        pd.DataFrame(rows_std).set_index("Ticker"),
    )


def detect_dislocations(prices: pd.DataFrame, tickers: list[str], threshold: float = 0.03) -> pd.DataFrame:
    """
    Finds stocks where a factor has moved significantly (>threshold cumulative over 5 days)
    but the stock return diverges from what beta would predict.
    """
    daily = prices.pct_change().dropna()
    factor_df = _build_factor_returns(daily, 60).dropna(how="all")
    factor_names = list(factor_df.columns)

    raw_betas, _ = compute_factor_betas(prices, tickers)
    if raw_betas.empty:
        return pd.DataFrame()

    factor_5d = factor_df.tail(5).sum()

    rows = []
    for ticker in tickers:
        if ticker not in daily.columns or ticker not in raw_betas.index:
            continue
        stock_5d = daily[ticker].tail(5).sum()
        for fname in factor_names:
            if fname not in raw_betas.columns:
                continue
            beta = raw_betas.loc[ticker, fname]
            factor_move = factor_5d.get(fname, 0)
            if abs(factor_move) < threshold:
                continue
            expected_contribution = beta * factor_move
            gap = expected_contribution - stock_5d
            if abs(gap) > threshold:
                rows.append({
                    "Ticker": ticker,
                    "Factor": fname,
                    "Factor 5D Move": round(factor_move * 100, 2),
                    "Stock 5D Return": round(stock_5d * 100, 2),
                    "Beta": round(beta, 3),
                    "Expected Contribution": round(expected_contribution * 100, 2),
                    "Gap (%)": round(gap * 100, 2),
                    "Signal": "Long" if gap > 0 else "Short",
                })

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("Gap (%)", key=abs, ascending=False).reset_index(drop=True)
