import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from data import fetch_prices, compute_returns, get_sector_info, FACTOR_PROXIES
from factors import compute_factor_betas, detect_dislocations

st.set_page_config(page_title="Portfolio Attribution", layout="wide")
st.title("Portfolio Return Attribution")
st.caption("Identify which names, sectors, and factors are driving your portfolio returns.")

# --- Sidebar: Portfolio Input ---
with st.sidebar:
    st.header("Portfolio")
    default_tickers = "AAPL, MSFT, XOM, JPM, MATV, LYB, CF, DOW"
    raw_input = st.text_area("Enter tickers (comma-separated)", value=default_tickers, height=120)
    tickers = [t.strip().upper() for t in raw_input.split(",") if t.strip()]

    factor_window = st.slider("Factor regression window (trading days)", 20, 120, 60)
    dislocation_threshold = st.slider("Dislocation threshold (%)", 1, 10, 3) / 100

    run = st.button("Run Analysis", type="primary", use_container_width=True)

if not run:
    st.info("Enter your tickers in the sidebar and click **Run Analysis**.")
    st.stop()

# --- Fetch Data ---
with st.spinner("Fetching market data..."):
    prices = fetch_prices(tickers)
    sector_info = get_sector_info(tickers)

if prices.empty:
    st.error("Could not fetch price data. Check your tickers and try again.")
    st.stop()

available = [t for t in tickers if t in prices.columns]
if not available:
    st.error("None of your tickers returned data.")
    st.stop()

# ============================================================
# TAB LAYOUT
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["Returns", "Sector Breakdown", "Factor Exposures", "Dislocations"])

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
        color = "#1a9641" if val > 0 else "#d7191c"
        return f"color: {color}; font-weight: bold"

    styled = returns_df.style.applymap(color_returns, subset=["1D", "1W", "1M", "3M"]).format("{:.2f}%", na_rep="—")
    st.dataframe(styled, use_container_width=True)

    st.markdown("**Benchmark (SPY)**")
    st.dataframe(
        spy_returns.style.applymap(color_returns, subset=["1D", "1W", "1M", "3M"]).format("{:.2f}%", na_rep="—"),
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
    fig_box = px.bar(
        merged.reset_index(),
        x="Ticker",
        y="1M",
        color="Sector",
        labels={"1M": "1M Return (%)"},
        title="1M Return per Name, colored by Sector",
    )
    fig_box.add_hline(y=0, line_dash="dot", line_color="gray")
    st.plotly_chart(fig_box, use_container_width=True)

    st.subheader("Full Details")
    st.dataframe(merged.style.format("{:.2f}%", subset=["1D", "1W", "1M", "3M"], na_rep="—"), use_container_width=True)

# ============================================================
# TAB 3: Factor Exposures
# ============================================================
with tab3:
    st.subheader("Factor Betas")
    st.caption(
        f"OLS regression of daily returns vs factor proxies over trailing {factor_window} trading days. "
        "Factor proxies: " + ", ".join(f"{k} ({v})" for k, v in FACTOR_PROXIES.items())
    )
    with st.spinner("Running regressions..."):
        betas_df = compute_factor_betas(prices, available, window=factor_window)

    if betas_df.empty:
        st.warning("Not enough data to compute factor betas.")
    else:
        factor_cols = [c for c in betas_df.columns if c != "R²"]
        styled_betas = betas_df.style.background_gradient(subset=factor_cols, cmap="RdYlGn", vmin=-1.5, vmax=1.5).format("{:.3f}")
        st.dataframe(styled_betas, use_container_width=True)

        st.subheader("Heatmap")
        fig_heat = go.Figure(data=go.Heatmap(
            z=betas_df[factor_cols].values,
            x=factor_cols,
            y=betas_df.index.tolist(),
            colorscale="RdYlGn",
            zmid=0,
            text=betas_df[factor_cols].round(2).values,
            texttemplate="%{text}",
        ))
        fig_heat.update_layout(height=max(300, 60 * len(betas_df)), margin=dict(l=80, r=20, t=40, b=40))
        st.plotly_chart(fig_heat, use_container_width=True)

        st.subheader("Factor Proxy Performance")
        factor_tickers = list(FACTOR_PROXIES.values())
        factor_returns_df = compute_returns(prices, factor_tickers)
        factor_returns_df.index = [f"{FACTOR_PROXIES.get(t, t)} ({t})" for t in factor_returns_df.index]
        st.dataframe(
            factor_returns_df.style.applymap(
                lambda v: f"color: {'#1a9641' if v > 0 else '#d7191c'}; font-weight: bold" if pd.notna(v) else "",
                subset=["1D", "1W", "1M", "3M"],
            ).format("{:.2f}%", na_rep="—"),
            use_container_width=True,
        )

# ============================================================
# TAB 4: Dislocations
# ============================================================
with tab4:
    st.subheader("Factor Dislocation Signals")
    st.caption(
        "Stocks where a factor has moved meaningfully in the past 5 days but the stock hasn't responded as beta would predict. "
        "These are potential long/short setups."
    )
    with st.spinner("Scanning for dislocations..."):
        dislocations = detect_dislocations(prices, available, threshold=dislocation_threshold)

    if dislocations.empty:
        st.success("No significant dislocations detected at current threshold. Try lowering the threshold in the sidebar.")
    else:
        def highlight_signal(val):
            if val == "Long":
                return "background-color: #d4edda; color: #155724; font-weight: bold"
            elif val == "Short":
                return "background-color: #f8d7da; color: #721c24; font-weight: bold"
            return ""

        styled_disl = dislocations.style.applymap(highlight_signal, subset=["Signal"]).format(
            {
                "Factor 5D Move": "{:.2f}%",
                "Stock 5D Return": "{:.2f}%",
                "Expected Contribution": "{:.2f}%",
                "Gap (%)": "{:.2f}%",
                "Beta": "{:.3f}",
            }
        )
        st.dataframe(styled_disl, use_container_width=True)

        st.markdown("**How to read this:** If `Gap (%)` is positive and signal is `Long`, the factor move implies the stock should be higher than it is. Negative gap → `Short`.")
