import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from data import (
    fetch_prices,
    compute_returns,
    get_sector_info,
    fetch_portfolio_exposures,
    fetch_factor_returns,
    fetch_factor_returns_intraday,
    fetch_factor_covariance,
    fetch_stock_specific_vol,
    fetch_factor_history,
)

st.set_page_config(page_title="Portfolio Attribution", layout="wide")
st.title("Portfolio Return Attribution")
st.caption("Understand what's driving your portfolio — by sector, factor, and day.")

# --- Sidebar ---
with st.sidebar:
    st.header("Portfolio")
    st.caption("Format: TICKER or TICKER:SHARES (default 100 shares)")
    default_input = "AAPL:100, MSFT:50, XOM:75, JPM:60, LYB:80, CF:40, DOW:90"
    raw_input = st.text_area("Holdings", value=default_input, height=140)

    holdings: dict[str, float] = {}
    for item in raw_input.split(","):
        item = item.strip().upper()
        if not item:
            continue
        if ":" in item:
            ticker, _, qty = item.partition(":")
            try:
                shares = float(qty.strip())
            except ValueError:
                shares = 100.0
        else:
            ticker, shares = item, 100.0
        ticker = ticker.strip()
        if ticker:
            holdings[ticker] = shares

    tickers = list(holdings.keys())

    if st.button("Run Analysis", type="primary", use_container_width=True):
        st.session_state["run"] = True

if not st.session_state.get("run"):
    st.info("Enter your holdings in the sidebar and click **Run Analysis**.")
    st.stop()

# --- Fetch price data ---
with st.spinner("Fetching price data..."):
    prices = fetch_prices(tickers)
    sector_info = get_sector_info(tickers)

if prices.empty:
    st.error("Could not fetch price data. Check your tickers.")
    st.stop()

available = [t for t in tickers if t in prices.columns]
if not available:
    st.error("None of your tickers returned data.")
    st.stop()

holdings_tuple = tuple((t, holdings[t]) for t in available)

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Returns", "Sector Breakdown", "Factor Exposures", "Today's Drivers", "Risk Attribution"
])

# ============================================================
# TAB 1: Returns
# ============================================================
with tab1:
    st.subheader("Return Summary")
    returns_df = compute_returns(prices, available)
    spy_returns = compute_returns(prices, ["SPY"])

    def color_returns(val):
        if pd.isna(val):
            return ""
        return f"color: {'#1a9641' if val > 0 else '#d7191c'}; font-weight: bold"

    st.dataframe(
        returns_df.style.map(color_returns, subset=["1D", "1W", "1M", "3M"]).format("{:.2f}%", na_rep="—"),
        use_container_width=True,
    )
    st.markdown("**Benchmark (SPY)**")
    st.dataframe(
        spy_returns.style.map(color_returns, subset=["1D", "1W", "1M", "3M"]).format("{:.2f}%", na_rep="—"),
        use_container_width=True,
    )

    st.subheader("Cumulative Performance (3M)")
    price_subset = prices[available + ["SPY"]].dropna(how="all").tail(63)
    normalized = price_subset / price_subset.iloc[0] * 100
    fig = px.line(normalized, labels={"value": "Indexed (base=100)", "variable": "Ticker"})
    fig.update_layout(hovermode="x unified", height=420)
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 2: Sector Breakdown
# ============================================================
with tab2:
    st.subheader("Sector Composition")
    merged = returns_df.join(sector_info[["Sector"]])
    sector_counts = merged["Sector"].value_counts().reset_index()
    sector_counts.columns = ["Sector", "Count"]
    fig_pie = px.pie(sector_counts, names="Sector", values="Count", title="Portfolio by Sector")
    st.plotly_chart(fig_pie, use_container_width=True)

    st.subheader("1-Month Returns by Sector")
    merged["Sector"] = merged["Sector"].fillna("Unknown")
    fig_bar = px.bar(
        merged.reset_index(),
        x="Ticker", y="1M", color="Sector",
        labels={"1M": "1M Return (%)"},
        title="1M Return per Name, colored by Sector",
    )
    fig_bar.add_hline(y=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Full Details")
    st.dataframe(
        merged.style.format("{:.2f}%", subset=["1D", "1W", "1M", "3M"], na_rep="—"),
        use_container_width=True,
    )

# ============================================================
# TAB 3: Factor Exposures
# ============================================================
with tab3:
    st.subheader("Portfolio Factor Exposures vs Benchmark")
    with st.spinner("Loading factor exposures..."):
        try:
            port_data = fetch_portfolio_exposures(holdings_tuple)
        except Exception as e:
            st.error(f"Could not load factor exposures: {e}")
            st.stop()

    r2 = port_data.get("portfolioRSquared", 0)
    total_val = port_data.get("totalValue", 0)

    c1, c2, c3 = st.columns(3)
    c1, c2 = st.columns(2)
    c1.metric("Portfolio Value", f"${total_val:,.0f}")
    c2.metric("Model R²", f"{r2:.1%}")

    exposures = port_data.get("exposures", [])
    if not exposures:
        st.warning("No exposure data returned.")
    else:
        df_exp = pd.DataFrame(exposures)
        df_exp = df_exp[df_exp["tilt"].abs() > 0.01].sort_values("tilt", key=abs, ascending=False)

        display = df_exp[["factor", "exposure", "benchmarkExposure", "tilt", "dollarValueAtRisk"]].copy()
        display.columns = ["Factor", "Portfolio", "Benchmark", "Tilt", "$ at Risk"]
        display = display.set_index("Factor")

        def color_tilt(val):
            if pd.isna(val):
                return ""
            return f"color: {'#1a9641' if val > 0 else '#d7191c'}; font-weight: bold"

        st.dataframe(
            display.style.map(color_tilt, subset=["Tilt"]).format({
                "Portfolio": "{:.3f}",
                "Benchmark": "{:.3f}",
                "Tilt": "{:+.3f}",
                "$ at Risk": "${:,.0f}",
            }),
            use_container_width=True,
        )

        st.subheader("Factor Tilts (overweight / underweight vs benchmark)")
        top_tilts = df_exp.head(20)
        fig_tilts = px.bar(
            top_tilts, x="tilt", y="factor", orientation="h",
            color="tilt",
            color_continuous_scale=["#d7191c", "#f7f7f7", "#1a9641"],
            color_continuous_midpoint=0,
            labels={"tilt": "Tilt vs Benchmark", "factor": "Factor"},
        )
        fig_tilts.update_layout(
            showlegend=False, coloraxis_showscale=False,
            height=max(400, 28 * len(top_tilts)),
        )
        st.plotly_chart(fig_tilts, use_container_width=True)

        # Historical performance of top 3 most tilted factors
        st.subheader("Historical Performance of Top Tilted Factors (1Y)")
        top3 = df_exp.head(3)["factor"].tolist()
        hist_series = {}
        for f in top3:
            try:
                raw = fetch_factor_history(f, days=252)
                if raw:
                    s = pd.DataFrame(raw).set_index("date")["close"]
                    s.index = pd.to_datetime(s.index)
                    hist_series[f] = s / s.iloc[0] * 100
            except Exception:
                pass
        if hist_series:
            fig_hist = px.line(
                pd.DataFrame(hist_series),
                labels={"value": "Cumulative Return (base=100)", "variable": "Factor"},
            )
            fig_hist.update_layout(hovermode="x unified", height=350)
            st.plotly_chart(fig_hist, use_container_width=True)

# ============================================================
# TAB 4: Today's Drivers
# ============================================================
with tab4:
    st.subheader("Today's Factor Drivers")

    with st.spinner("Loading factor returns..."):
        try:
            intraday = fetch_factor_returns_intraday()
            port_data2 = fetch_portfolio_exposures(holdings_tuple)
        except Exception as e:
            st.error(f"Could not load factor returns: {e}")
            st.stop()

    # Use intraday if market is open, fall back to previous session's 1d close
    market_open = any((v.get("value") or 0) != 0 for v in intraday.values())
    if market_open:
        ret_map = {f: (v.get("value") or 0) for f, v in intraday.items()}
        source_label = "Intraday (live)"
    else:
        try:
            historic = fetch_factor_returns()
            ret_map = {f: ((p.get("1d") or {}).get("value") or 0) for f, p in historic.items()}
            source_label = "Previous Session (1D close)"
        except Exception as e:
            st.error(f"Could not load historic returns: {e}")
            st.stop()

    st.caption(
        f"Source: {source_label}. Contribution = portfolio exposure × factor return. "
        "Sum shows how much of the move is explained by systematic factors."
    )

    exp_map2 = {e["factor"]: e["exposure"] for e in port_data2.get("exposures", [])}

    rows = []
    for factor, ret_1d in ret_map.items():
        if factor not in exp_map2:
            continue
        exposure = exp_map2[factor]
        rows.append({
            "Factor": factor,
            "Factor Return (%)": round(ret_1d * 100, 3),
            "Exposure": round(exposure, 3),
            "Contribution (%)": round(exposure * ret_1d * 100, 4),
        })

    if not rows:
        st.warning("No attribution data available.")
    else:
        df_attr = pd.DataFrame(rows).sort_values("Contribution (%)", key=abs, ascending=False)
        total = df_attr["Contribution (%)"].sum()
        st.metric("Total Factor-Attributed Return", f"{total:+.2f}%")

        top15 = df_attr.head(15)
        fig_attr = px.bar(
            top15, x="Contribution (%)", y="Factor", orientation="h",
            color="Contribution (%)",
            color_continuous_scale=["#d7191c", "#f7f7f7", "#1a9641"],
            color_continuous_midpoint=0,
            title="Top 15 Factor Contributions",
        )
        fig_attr.add_vline(x=0, line_dash="dot", line_color="gray")
        fig_attr.update_layout(
            showlegend=False, coloraxis_showscale=False,
            height=max(400, 28 * len(top15)),
        )
        st.plotly_chart(fig_attr, use_container_width=True)

        def color_contribution(val):
            if pd.isna(val) or val == 0:
                return ""
            return f"color: {'#1a9641' if val > 0 else '#d7191c'}; font-weight: bold"

        st.dataframe(
            df_attr.style.map(color_contribution, subset=["Contribution (%)"]).format({
                "Factor Return (%)": "{:+.3f}%",
                "Exposure": "{:.3f}",
                "Contribution (%)": "{:+.4f}%",
            }),
            use_container_width=True,
        )

# ============================================================
# TAB 5: Risk Attribution
# ============================================================
with tab5:
    st.subheader("Portfolio Risk Attribution")
    st.caption(
        "Decomposes annual portfolio volatility into systematic (factor) risk and "
        "idiosyncratic (stock-specific) risk using a full factor covariance matrix."
    )

    with st.spinner("Computing risk attribution..."):
        try:
            cov_data = fetch_factor_covariance()
            port_data3 = fetch_portfolio_exposures(holdings_tuple)
        except Exception as e:
            st.error(f"Could not load risk data: {e}")
            st.stop()

    cov_factors = cov_data["factors"]
    cov_matrix_raw = cov_data["matrix"]
    exp_map3 = {e["factor"]: e["exposure"] for e in port_data3.get("exposures", [])}
    holdings_list = port_data3.get("holdings", [])

    # Factor variance: e' Σ e (annualized)
    active = [f for f in cov_factors if f in exp_map3]
    if not active:
        st.warning("No overlapping factors between covariance matrix and portfolio exposures.")
        st.stop()

    e_vec = np.array([exp_map3[f] for f in active])
    cov_sub = np.array([[cov_matrix_raw[f1][f2] for f2 in active] for f1 in active])
    sigma_e = cov_sub @ e_vec
    factor_variance = float(e_vec @ sigma_e)
    factor_contribs = {f: float(e_vec[i] * sigma_e[i]) for i, f in enumerate(active)}

    # Idiosyncratic variance: Σ_k w_k² × σ²_k (annualized, uncorrelated residuals)
    specific_variance = 0.0
    spec_rows = []
    for h in holdings_list:
        ticker = h["ticker"]
        weight = h.get("weight", 0)
        try:
            vol_data = fetch_stock_specific_vol(ticker)
            spec_vol = vol_data.get("specific_vol_annual") or 0
            r2_stock = vol_data.get("r_squared", 0)
            contrib = weight ** 2 * spec_vol ** 2
            specific_variance += contrib
            spec_rows.append({
                "Ticker": ticker,
                "Weight": weight,
                "Specific Vol (Ann.)": spec_vol,
                "R² (factor model)": r2_stock,
                "Variance Contribution": contrib,
            })
        except Exception:
            spec_rows.append({
                "Ticker": ticker,
                "Weight": weight,
                "Specific Vol (Ann.)": None,
                "R² (factor model)": None,
                "Variance Contribution": None,
            })

    total_variance = factor_variance + specific_variance
    if total_variance <= 0:
        st.warning("Could not compute portfolio variance.")
        st.stop()

    port_vol = np.sqrt(total_variance)
    factor_vol = np.sqrt(factor_variance)
    specific_vol_total = np.sqrt(specific_variance)

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated Annual Vol", f"{port_vol:.1%}")
    c2.metric("Systematic Risk", f"{factor_vol:.1%}",
              f"{factor_variance / total_variance:.0%} of variance")
    c3.metric("Idiosyncratic Risk", f"{specific_vol_total:.1%}",
              f"{specific_variance / total_variance:.0%} of variance")

    col_left, col_right = st.columns(2)

    with col_left:
        fig_decomp = px.pie(
            values=[factor_variance, specific_variance],
            names=["Systematic (Factor)", "Idiosyncratic"],
            title="Variance Decomposition",
            color_discrete_sequence=["#2196F3", "#FF9800"],
        )
        st.plotly_chart(fig_decomp, use_container_width=True)

    with col_right:
        # Top factor contributions to factor variance
        df_fcontrib = pd.DataFrame([
            {"Factor": f, "% of Factor Variance": v / factor_variance * 100}
            for f, v in factor_contribs.items()
        ]).sort_values("% of Factor Variance", key=abs, ascending=False).head(10)

        fig_fcontrib = px.bar(
            df_fcontrib, x="% of Factor Variance", y="Factor", orientation="h",
            color="% of Factor Variance",
            color_continuous_scale=["#d7191c", "#f7f7f7", "#1a9641"],
            color_continuous_midpoint=0,
            title="Top 10 Factors Driving Systematic Risk",
        )
        fig_fcontrib.update_layout(showlegend=False, coloraxis_showscale=False, height=380)
        st.plotly_chart(fig_fcontrib, use_container_width=True)

    st.subheader("Idiosyncratic Risk by Holding")
    if spec_rows:
        df_spec = pd.DataFrame(spec_rows).sort_values("Variance Contribution", ascending=False, na_position="last")
        df_spec["% of Specific Variance"] = df_spec["Variance Contribution"] / specific_variance * 100
        st.dataframe(
            df_spec.style.format({
                "Weight": "{:.1%}",
                "Specific Vol (Ann.)": "{:.1%}",
                "R² (factor model)": "{:.1%}",
                "Variance Contribution": "{:.6f}",
                "% of Specific Variance": "{:.1f}%",
            }, na_rep="—"),
            use_container_width=True,
        )
