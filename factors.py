import pandas as pd
import numpy as np
import statsmodels.api as sm
from data import FACTOR_PROXIES


def compute_factor_betas(prices: pd.DataFrame, tickers: list[str], window: int = 60) -> pd.DataFrame:
    """
    For each stock, run OLS of daily returns against factor proxy returns
    over the trailing `window` trading days to get factor betas.
    """
    daily = prices.pct_change().dropna()
    factor_cols = list(FACTOR_PROXIES.values())

    available_factors = [f for f in factor_cols if f in daily.columns]
    factor_returns = daily[available_factors].tail(window)
    factor_names = {v: k for k, v in FACTOR_PROXIES.items()}

    rows = []
    for ticker in tickers:
        if ticker not in daily.columns:
            continue
        stock_ret = daily[ticker].tail(window)
        common_idx = stock_ret.dropna().index.intersection(factor_returns.dropna(how="all").index)
        if len(common_idx) < 20:
            continue
        y = stock_ret.loc[common_idx].values
        X = factor_returns.loc[common_idx].values
        X_with_const = sm.add_constant(X)
        try:
            model = sm.OLS(y, X_with_const).fit()
            betas = model.params[1:]
            row = {"Ticker": ticker}
            for i, col in enumerate(available_factors):
                row[factor_names.get(col, col)] = round(betas[i], 3)
            row["R²"] = round(model.rsquared, 3)
            rows.append(row)
        except Exception:
            continue

    return pd.DataFrame(rows).set_index("Ticker") if rows else pd.DataFrame()


def detect_dislocations(prices: pd.DataFrame, tickers: list[str], threshold: float = 0.03) -> pd.DataFrame:
    """
    Finds stocks where a factor has moved significantly (>threshold in 5 days)
    but the stock return is in the opposite direction or lagging badly.
    Returns dislocation candidates with a brief description.
    """
    daily = prices.pct_change().dropna()
    factor_cols = list(FACTOR_PROXIES.values())
    factor_names = {v: k for k, v in FACTOR_PROXIES.items()}

    betas_df = compute_factor_betas(prices, tickers)
    if betas_df.empty:
        return pd.DataFrame()

    recent = daily.tail(5)
    factor_5d = recent[factor_cols].sum()

    rows = []
    for ticker in tickers:
        if ticker not in daily.columns or ticker not in betas_df.index:
            continue
        stock_5d = daily[ticker].tail(5).sum()
        for fcol in factor_cols:
            fname = factor_names.get(fcol, fcol)
            if fname not in betas_df.columns:
                continue
            beta = betas_df.loc[ticker, fname]
            factor_move = factor_5d.get(fcol, 0)
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
