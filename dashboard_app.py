from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"
LOGO_PATH = PROJECT_ROOT / "blackfin_logo.png"

PAGES = [
    "Executive Overview",
    "Inherited Fund Review",
    "Candidate Stock Research",
    "Revised Portfolio Construction",
    "Backtest Comparison",
    "Risk and Diagnostics",
    "Scenario / Stress Test",
    "Final Recommendation",
]

st.set_page_config(
    page_title="BlackFin Inc. — Fund Management Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — tighten cards and accent colours
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Metric card border */
    [data-testid="metric-container"] {
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 12px 16px;
        background-color: #161b22;
    }
    /* Section dividers */
    hr { border-color: #30363d; }
    /* Dataframe header */
    thead tr th { background-color: #1f2937 !important; }
    /* Callout boxes */
    .callout {
        background-color: #1a2332;
        border-left: 4px solid #58a6ff;
        border-radius: 4px;
        padding: 12px 16px;
        margin: 8px 0 16px 0;
        font-size: 0.93rem;
        line-height: 1.55;
    }
    .callout-green {
        background-color: #0d2318;
        border-left: 4px solid #3fb950;
    }
    .callout-amber {
        background-color: #251d00;
        border-left: 4px solid #d29922;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def safe_read_csv(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()


def read_table(filename: str) -> pd.DataFrame:
    return safe_read_csv(str(TABLES_DIR / filename))


@st.cache_data(show_spinner=False)
def load_kv(filename: str) -> dict[str, str]:
    df = safe_read_csv(str(TABLES_DIR / filename))
    if df.empty or not {"Parameter", "Value"}.issubset(df.columns):
        return {}
    return {str(r["Parameter"]).strip(): str(r["Value"]).strip() for _, r in df.iterrows() if str(r["Parameter"]).strip()}


def normalize(v: object) -> str:
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    return str(v).strip()


def fmt_pct(v: object, digits: int = 1) -> str:
    try:
        x = float(v)
    except Exception:
        return normalize(v)
    return "" if np.isnan(x) else f"{x:.{digits}%}"


def fmt_ratio(v: object, digits: int = 2) -> str:
    try:
        x = float(v)
    except Exception:
        return normalize(v)
    return "" if np.isnan(x) else f"{x:.{digits}f}"


def fmt_dollar(v: object) -> str:
    try:
        x = float(v)
    except Exception:
        return normalize(v)
    return "" if np.isnan(x) else f"${x:,.0f}"


def coerce_numeric(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


COLUMN_LABELS: dict[str, str] = {
    "portfolio": "Portfolio",
    "ending_value": "Terminal Value",
    "ann_return": "Ann. Return",
    "ann_vol": "Ann. Volatility",
    "sharpe": "Sharpe",
    "sortino": "Sortino",
    "max_drawdown": "Max Drawdown",
    "alpha_ann": "Alpha (Ann.)",
    "beta": "Beta",
    "beta_mkt": "Beta (Mkt)",
    "beta_smb": "Beta (SMB)",
    "beta_hml": "Beta (HML)",
    "r_squared": "R²",
    "alpha_tstat": "Alpha T-Stat",
    "alpha_pvalue": "Alpha P-Value",
    "decision_date": "Decision Date",
    "effective_date": "Effective Date",
    "rule_used": "Rule Used",
    "turnover": "Turnover",
    "transaction_cost_rate": "Transaction Cost",
    "ticker": "Ticker",
    "legacy_ticker": "Ticker",
    "candidate_ticker": "Ticker",
    "company_name": "Company",
    "sector": "Sector",
    "sector_theme": "Sector",
    "keep_in_revised": "Decision",
    "decision_2020": "Decision",
    "target_weight_2020": "Target Weight",
    "static_weight": "Static Weight",
    "latest_active_weight": "Active Weight",
    "source": "Source",
    "add_decision": "Decision",
    "selected_for_final": "Selected",
    "ann_return_pre2020": "Ann. Return",
    "ann_vol_pre2020": "Ann. Volatility",
    "sharpe_pre2020": "Sharpe",
    "max_drawdown_pre2020": "Max Drawdown",
    "n_obs": "Observations",
    "n_missing": "Missing",
    "first_date": "First Date",
    "last_date": "Last Date",
    "scenario": "Scenario",
    "benchmark_return": "Benchmark Return",
    "legacy_fund_impact": "Legacy Fund Impact",
    "static_portfolio_impact": "Static Impact",
    "active_portfolio_impact": "Active Impact",
    "mean_terminal_return": "Mean Return",
    "median_terminal_return": "Median Return",
    "var_95": "VaR (95%)",
    "cvar_95": "CVaR (95%)",
    "prob_loss": "P(Loss)",
}


def show_table(
    df: pd.DataFrame,
    *,
    pct_cols: Iterable[str] = (),
    ratio_cols: Iterable[str] = (),
    dollar_cols: Iterable[str] = (),
    height: int | None = None,
) -> None:
    if df.empty:
        st.info("No data available.")
        return
    disp = df.copy()
    for c in pct_cols:
        if c in disp.columns:
            disp[c] = disp[c].map(fmt_pct)
    for c in ratio_cols:
        if c in disp.columns:
            disp[c] = disp[c].map(fmt_ratio)
    for c in dollar_cols:
        if c in disp.columns:
            disp[c] = disp[c].map(fmt_dollar)
    disp = disp.rename(columns=lambda c: COLUMN_LABELS.get(c, c.replace("_", " ").title()))
    kwargs: dict = {"width": "stretch", "hide_index": True}
    if height:
        kwargs["height"] = height
    st.dataframe(disp, **kwargs)


def show_perf_table(df: pd.DataFrame) -> None:
    """Performance summary table with Sharpe and Sortino highlighted; Sortino shown at 4dp."""
    if df.empty:
        st.info("No data available.")
        return
    cols = [c for c in ["portfolio", "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown"] if c in df.columns]
    disp = df[cols].copy()
    for c in ["ann_return", "ann_vol", "max_drawdown"]:
        if c in disp.columns:
            disp[c] = disp[c].map(fmt_pct)
    if "sharpe" in disp.columns:
        disp["sharpe"] = disp["sharpe"].map(lambda v: fmt_ratio(v, 3))
    if "sortino" in disp.columns:
        disp["sortino"] = disp["sortino"].map(lambda v: fmt_ratio(v, 4))
    disp = disp.rename(columns=lambda c: COLUMN_LABELS.get(c, c.replace("_", " ").title()))

    def _hl(col: pd.Series) -> list[str]:
        if col.name in ("Sharpe", "Sortino"):
            return ["background-color: #162d1a; font-weight: bold"] * len(col)
        return [""] * len(col)

    styled = disp.style.apply(_hl, axis=0).hide(axis="index")
    st.dataframe(styled, width="stretch")


def img(filename: str, caption: str | None = None, width: int | None = None) -> bool:
    p = FIGURES_DIR / filename
    if not p.exists():
        return False
    kw: dict = {"width": "stretch"}
    if width:
        kw = {"width": width}
    st.image(str(p), caption=caption or filename, **kw)
    return True


def callout(text: str, kind: str = "info") -> None:
    css = {"info": "callout", "success": "callout callout-green", "warning": "callout callout-amber"}.get(kind, "callout")
    st.markdown(f'<div class="{css}">{text}</div>', unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Keep / Drop rationale — our actual reasons
# ---------------------------------------------------------------------------
KEEP_DROP_RATIONALE: dict[str, str] = {
    "AAPL": "Kept — highest in-sample Sharpe (1.06) and second-highest alpha (13.3%). Brand moat and ecosystem switching costs make it our top long-run compounder alongside AMZN.",
    "AMZN": "Kept — highest annualized alpha in the inherited fund (14.9%) and the strongest compounding story. AWS + marketplace network effects are durable over a 30–40 year horizon.",
    "CMCSA": "Kept — infrastructure moat (last-mile broadband) and meaningful switching costs. Alpha of 7.9% over 10 years justifies retention despite lower growth profile.",
    "GOOG": "Kept — network effect moat on search and digital advertising. Alpha 3.3% in-sample; lower than AAPL/AMZN but the secular digital-ad tailwind remains intact at the decision date.",
    "MSFT": "Kept — switching-cost moat (enterprise software) with cloud transition well underway by 2020. Alpha 6.8%, Sharpe 0.91 in-sample.",
    "WFC": "Dropped — failed our character screen. Wells Fargo was under active consent orders from the 2016 fake-accounts scandal with ongoing regulatory penalties. We will not hold a name under active regulatory sanction in a long-horizon retirement fund.",
    "KO": "Dropped — failed our growth screen. Flat to declining real revenue over the decade, no identifiable secular tailwind, and a Sharpe of 0.69 with alpha of only 3.2%. A defensive name with no compelling growth story for a 30–40 year horizon.",
    "ORCL": "Dropped — failed our moat screen. Oracle had the lowest Sharpe in the portfolio (0.40) and produced negative alpha (−3.0%). More importantly, it was visibly losing the cloud transition race to AWS and Azure by 2020 — a structural, not cyclical, problem.",
    "BRK-B": "Dropped — redundant. Berkshire Hathaway is a holding company that itself holds substantial positions in AAPL, KO, and financial sector names already in our portfolio. Adding BRK-B layered indirect exposure without independent sector diversification.",
    "XOM": "Dropped — CVX preferred on balance sheet strength. Both are integrated energy majors, but Chevron's lower debt-to-equity ratio and stronger 5-year EPS CAGR made it the best-in-class pick in the Energy sector under our screening rule.",
}

CANDIDATE_RATIONALE: dict[str, str] = {
    "V": "Visa: Near-monopoly global payments network with ~50%+ operating margins. Secular cashless payment trend is multi-decade. Best-in-class in Financials on 5yr EPS CAGR with A+ credit rating.",
    "MA": "Mastercard: Strong network but Visa selected as best-in-class by EPS CAGR and network scale at the decision date.",
    "JPM": "JPMorgan Chase: Solid bank but carries more interest-rate and credit-cycle sensitivity than the pure payment networks. Less aligned with long-horizon mandate.",
    "WMT": "Walmart: Largest physical retail footprint with a credible e-commerce pivot underway by 2020. Recession-proof consumer staples anchor with 40+ year dividend growth record. Best-in-class in Consumer Staples.",
    "COST": "Costco: High-quality business but lower EPS CAGR than WMT at the 2020 decision date. Both passed all screens; WMT selected as best-in-class.",
    "TGT": "Target: Passed size and profitability screens but lower EPS growth and weaker moat depth than WMT.",
    "CVX": "Chevron: Best-in-class integrated energy major. Lower D/E ratio and stronger EPS CAGR than XOM. Regulatory license + scale moat. Selected over XOM as the Energy sector representative.",
    "COP": "ConocoPhillips: Strong E&P operator but less diversified than CVX's integrated model. CVX selected as best-in-class.",
    "NEE": "NextEra Energy: World's largest wind and solar operator by 2020. 5-year EPS CAGR of ~8–10% vs. 2–3% for utility peers (DUK, SO). Florida monopoly utility + renewable secular trend. Best-in-class in Utilities by a wide margin.",
    "DUK": "Duke Energy: Passed screens but heavy coal exposure and 2–3% EPS CAGR vs. NEE's 8–10%. Not best-in-class.",
    "SO": "Southern Company: Similar profile to DUK. Coal-heavy with slower growth; NEE dominates on best-in-class criteria.",
    "UNH": "UnitedHealth Group: Largest U.S. health insurer combined with Optum data/pharmacy. Aging demographics secular tailwind. Consistent 15%+ EPS growth in-sample. Best-in-class in Healthcare.",
    "JNJ": "Johnson & Johnson: Diversified healthcare but lower EPS growth than UNH. More pharmaceutical/consumer mix than pure insurance/managed care.",
    "CVS": "CVS Health: Pharmacy + insurance pivot underway but less proven than UNH's integrated model at the decision date.",
}

# Full name and sector for every ticker in the fund universe
TICKER_INFO: dict[str, tuple[str, str]] = {
    "AAPL": ("Apple Inc.", "Technology"),
    "AMZN": ("Amazon.com, Inc.", "Consumer Discretionary"),
    "CMCSA": ("Comcast Corp.", "Communication Services"),
    "GOOG": ("Alphabet Inc.", "Communication Services"),
    "MSFT": ("Microsoft Corp.", "Technology"),
    "WFC": ("Wells Fargo & Co.", "Financials"),
    "KO": ("Coca-Cola Co.", "Consumer Staples"),
    "ORCL": ("Oracle Corp.", "Technology"),
    "BRK-B": ("Berkshire Hathaway", "Financials"),
    "XOM": ("Exxon Mobil Corp.", "Energy"),
    "V": ("Visa Inc.", "Financials"),
    "WMT": ("Walmart Inc.", "Consumer Staples"),
    "CVX": ("Chevron Corp.", "Energy"),
    "NEE": ("NextEra Energy, Inc.", "Utilities"),
    "UNH": ("UnitedHealth Group", "Healthcare"),
    "MA": ("Mastercard Inc.", "Financials"),
    "JPM": ("JPMorgan Chase", "Financials"),
    "COST": ("Costco Wholesale", "Consumer Staples"),
    "TGT": ("Target Corp.", "Consumer Discretionary"),
    "COP": ("ConocoPhillips", "Energy"),
    "DUK": ("Duke Energy", "Utilities"),
    "SO": ("Southern Company", "Utilities"),
    "JNJ": ("Johnson & Johnson", "Healthcare"),
    "CVS": ("CVS Health", "Healthcare"),
}


def ticker_company(t: str) -> str:
    return TICKER_INFO.get(t.upper(), (t, ""))[0] or t


def ticker_sector(t: str) -> str:
    return TICKER_INFO.get(t.upper(), ("", ""))[1]


def get_rationale(ticker: str, decision: str) -> str:
    t = normalize(ticker).upper()
    return KEEP_DROP_RATIONALE.get(t, CANDIDATE_RATIONALE.get(t, f"{'Kept' if 'keep' in normalize(decision).lower() else 'Dropped'} based on screening framework."))


# ---------------------------------------------------------------------------
# Terminal weights — computed from existing CSVs
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def compute_terminal_weights() -> pd.DataFrame:
    tickers_revised = ["AAPL", "AMZN", "CMCSA", "GOOG", "MSFT", "V", "WMT", "CVX", "NEE", "UNH"]
    tickers_legacy = ["AAPL", "AMZN", "BRK-B", "CMCSA", "GOOG", "KO", "MSFT", "ORCL", "WFC", "XOM"]

    static_df = safe_read_csv(str(TABLES_DIR / "tbl_revised_static_weights_daily.csv"))
    static_terminal: dict[str, float] = {}
    if not static_df.empty:
        last = static_df.iloc[-1]
        for t in tickers_revised:
            if t in static_df.columns:
                static_terminal[t] = float(last[t])

    active_snap = safe_read_csv(str(TABLES_DIR / "tbl_revised_active_weights_snapshot.csv"))
    active_terminal: dict[str, float] = {}
    if not active_snap.empty and "ticker" in active_snap.columns and "latest_active_weight" in active_snap.columns:
        for _, row in active_snap.iterrows():
            active_terminal[normalize(row["ticker"]).upper()] = float(pd.to_numeric(row["latest_active_weight"], errors="coerce"))

    legacy_df = safe_read_csv(str(TABLES_DIR / "tbl_legacy_weights_daily.csv"))
    legacy_terminal: dict[str, float] = {}
    if not legacy_df.empty:
        last = legacy_df.iloc[-1]
        for t in tickers_legacy:
            if t in legacy_df.columns:
                legacy_terminal[t] = float(last[t])

    static_initial = {
        "AAPL": 0.15, "AMZN": 0.15, "CMCSA": 0.10, "GOOG": 0.10, "MSFT": 0.10,
        "V": 0.08, "WMT": 0.08, "CVX": 0.08, "NEE": 0.08, "UNH": 0.08,
    }

    rows = []
    for t in tickers_revised:
        rows.append({
            "Ticker": t,
            "Initial Weight (Jan 2020)": static_initial.get(t, np.nan),
            "Static Terminal (Dec 2025)": static_terminal.get(t, np.nan),
            "Active Terminal (Dec 2025)": active_terminal.get(t, np.nan),
            "Static Drift": static_terminal.get(t, np.nan) - static_initial.get(t, np.nan),
        })
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def compute_legacy_stress_impacts() -> dict[str, float | None]:
    """
    Compute Legacy Fund returns for each historical stress scenario.
    Returns a dict keyed by scenario name. Synthetic scenarios return None.
    """
    legacy_df = safe_read_csv(str(TABLES_DIR / "tbl_legacy_fund_daily.csv"))
    if legacy_df.empty or "date" not in legacy_df.columns or "legacy_fund_return" not in legacy_df.columns:
        return {}
    legacy_df["date"] = pd.to_datetime(legacy_df["date"])
    legacy_df = legacy_df.sort_values("date").reset_index(drop=True)
    oos = legacy_df[legacy_df["date"] >= "2020-01-01"].reset_index(drop=True)

    def single_day(target: str) -> float | None:
        row = oos[oos["date"] == pd.Timestamp(target)]
        if row.empty:
            return None
        return float(row.iloc[0]["legacy_fund_return"])

    def window_21(end_date: str) -> float | None:
        end_idx_list = oos.index[oos["date"] == pd.Timestamp(end_date)].tolist()
        if not end_idx_list:
            return None
        end_idx = end_idx_list[0]
        start_idx = max(0, end_idx - 20)
        window = oos.iloc[start_idx : end_idx + 1]
        return float((1 + window["legacy_fund_return"]).prod() - 1)

    return {
        "COVID Crash Day": single_day("2020-03-16"),
        "Worst OOS Day": single_day("2020-03-16"),
        "Worst 21-Day Window": window_21("2020-03-23"),
    }


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def render_sidebar(inputs: dict[str, str], constraints: dict[str, str]) -> str:
    if LOGO_PATH.exists():
        st.sidebar.image(str(LOGO_PATH), width="stretch")
    st.sidebar.markdown("---")
    st.sidebar.title("BlackFin Inc.")
    st.sidebar.caption("BFIN 491 | Fund Management Dashboard")

    default = inputs.get("dashboard_default_page", PAGES[0])
    idx = PAGES.index(default) if default in PAGES else 0
    page = st.sidebar.radio("Navigate", PAGES, index=idx)

    st.sidebar.markdown("---")
    st.sidebar.write(f"**Team:** Barthuly · Binando · Bogard")
    st.sidebar.write(f"**Decision date:** {inputs.get('decision_date', '2020-01-01')}")
    st.sidebar.write(f"**OOS window:** {inputs.get('oos_start', '2020-01-01')} → {inputs.get('oos_end', '2025-12-31')}")
    st.sidebar.write(f"**Benchmark:** {inputs.get('market_benchmark', 'SPY')}")
    st.sidebar.write(f"**Active rule:** Sharpe-tilt (monthly)")
    return page


# ---------------------------------------------------------------------------
# Page 1 — Executive Overview
# ---------------------------------------------------------------------------
def render_overview(inputs: dict[str, str]) -> None:
    st.title("Executive Overview")
    st.caption("BlackFin Inc. — BFIN 491 Fund Management Dashboard Capstone")

    callout(
        "<strong>Final recommendation:</strong> Replace the inherited fund with the Revised 10-Stock Portfolio "
        "and implement it using monthly Sharpe-tilt active rebalancing. The active fund delivers the best "
        "risk-adjusted return of all four portfolios — Sharpe 0.793 and Sortino 1.004 — while reducing "
        "maximum drawdown by 5.9 pp versus the benchmark. The legacy fund's higher raw return (18.5% vs. 16.9%) "
        "was a concentrated AMZN/AAPL bet that happened to pay off; the revised fund wins on risk per unit of return.",
        "success",
    )

    perf = coerce_numeric(
        read_table("tbl_performance_summary.csv"),
        ["ending_value", "total_return", "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown"],
    )

    def _val(name: str, col: str) -> str:
        if perf.empty:
            return "N/A"
        row = perf.loc[perf["portfolio"].eq(name)]
        if row.empty or pd.isna(row.iloc[0].get(col, np.nan)):
            return "N/A"
        v = float(row.iloc[0][col])
        return fmt_dollar(v) if col == "ending_value" else fmt_pct(v)

    c1, c2, c3 = st.columns(3)
    c1.metric("Static terminal value", _val("Revised Static Fund", "ending_value"))
    c2.metric("Active terminal value", _val("Revised Active Fund", "ending_value"))
    c3.metric("SPY (Benchmark) terminal value", _val("Benchmark", "ending_value"))

    st.markdown("---")

    inherited_df = read_table("tbl_inherited_fund.csv")
    candidates_df = read_table("tbl_candidates.csv")

    keeps, drops = [], []
    if not inherited_df.empty and "legacy_ticker" in inherited_df.columns and "keep_in_revised" in inherited_df.columns:
        flag = inherited_df["keep_in_revised"].map(normalize).str.lower()
        keeps = inherited_df.loc[flag.isin({"yes", "keep"}), "legacy_ticker"].map(normalize).tolist()
        drops = inherited_df.loc[flag.eq("no"), "legacy_ticker"].map(normalize).tolist()

    adds = []
    if not candidates_df.empty and "candidate_ticker" in candidates_df.columns and "selected_for_final" in candidates_df.columns:
        adds = candidates_df.loc[candidates_df["selected_for_final"].map(normalize).str.lower().eq("yes"), "candidate_ticker"].map(normalize).tolist()

    st.markdown("### Keep / Drop / Add")
    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown("**Kept from inherited fund**")
        for t in keeps:
            st.write(f"- {ticker_company(t)} ({t})")
    with k2:
        st.markdown("**Dropped from inherited fund**")
        for t in drops:
            st.write(f"- {ticker_company(t)} ({t})")
    with k3:
        st.markdown("**New additions**")
        for t in adds:
            st.write(f"- {ticker_company(t)} ({t})")

    st.markdown("---")
    st.markdown("### Risk-adjusted performance (Jan 2020 – Dec 2025)")

    # Table first — Sharpe and Sortino tell the real story before showing the chart
    if not perf.empty:
        show_perf_table(
            perf[[c for c in ["portfolio", "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown"] if c in perf.columns]]
        )

    if not img("fig_legacy_static_active_benchmark.png"):
        st.info("Run the pipeline to generate the performance comparison chart.")


# ---------------------------------------------------------------------------
# Page 2 — Inherited Fund Review
# ---------------------------------------------------------------------------
def render_inherited_review() -> None:
    st.title("Inherited Fund Review")
    st.write("Audit of the 10-stock inherited fund as received on January 1, 2020. All analysis uses only information available through December 31, 2019.")

    snapshot = coerce_numeric(read_table("tbl_legacy_weights_snapshot.csv"), ["target_weight_2020"])
    inherited_df = read_table("tbl_inherited_fund.csv")
    perf = coerce_numeric(read_table("tbl_performance_summary.csv"), ["ann_return", "ann_vol", "sharpe", "max_drawdown"])

    if not snapshot.empty and "target_weight_2020" in snapshot.columns:
        snap = snapshot.sort_values("target_weight_2020", ascending=False).copy()
        top1 = snap.iloc[0]
        top2_sum = snap.head(2)["target_weight_2020"].sum()
        legacy_row = perf.loc[perf["portfolio"].eq("Legacy Fund")] if not perf.empty else pd.DataFrame()

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Largest single position", f"{normalize(top1.get('legacy_ticker', top1.get('ticker', '?')))}: {fmt_pct(top1['target_weight_2020'])}")
        m2.metric("Top-2 concentration", fmt_pct(top2_sum))
        if not legacy_row.empty:
            m3.metric("In-sample annualized return", fmt_pct(legacy_row.iloc[0].get("ann_return", np.nan)))
            m4.metric("In-sample Sharpe", fmt_ratio(legacy_row.iloc[0].get("sharpe", np.nan)))

    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        img("fig_inherited_fund_overview.png", "Inherited fund value path (2010–2019)")
    with c2:
        img("fig_inherited_drawdown.png", "Inherited fund drawdown (2010–2019)")

    # Constrain the third figure to the same width as the side-by-side pair above
    col_fig, col_spacer = st.columns([1, 1])
    with col_fig:
        img("fig_inherited_weights_snapshot.png", "Weight drift — inception vs. decision date (Dec 2019)")

    callout(
        "<strong>Key finding:</strong> The fund was never rebalanced. By December 2019, AMZN had grown from 10% to 25.4% "
        "and AAPL from 10% to 20.3% — two names controlled 45.7% of a ten-stock portfolio. Three names (ORCL, WFC, XOM) "
        "produced negative alpha over the decade. The portfolio beta was 1.065 with 0.908 correlation to SPY — "
        "essentially a slightly leveraged index with uncompensated single-name concentration on top.",
    )

    if not inherited_df.empty:
        st.markdown("### Legacy holding decisions and rationale")

        # Summary table first
        df = inherited_df.copy()
        out = pd.DataFrame({
            "Ticker": df.get("legacy_ticker", pd.Series(dtype=str)).map(normalize),
            "Company": df.get("company_name", pd.Series(dtype=str)).map(normalize),
            "Sector": df.get("sector", pd.Series(dtype=str)).map(normalize),
            "Decision": df.get("decision_2020", pd.Series(dtype=str)).map(normalize),
            "Target Weight": df.get("target_weight_2020", pd.Series(dtype=float)).map(fmt_pct),
        })
        show_table(out)

        # Expandable rationale per ticker
        st.markdown("#### Rationale by holding")
        for _, row in df.iterrows():
            t = normalize(row.get("legacy_ticker", ""))
            dec = normalize(row.get("decision_2020", ""))
            label = f"{t} — {ticker_company(t)} | {dec}"
            with st.expander(label):
                st.write(get_rationale(t, row.get("keep_in_revised", "")))

    st.markdown("---")
    c3, c4 = st.columns(2)
    with c3:
        img("fig_audit_risk_return.png", "Per-stock risk vs. return (2010–2019)")
    with c4:
        img("fig_audit_correlation_heatmap.png", "Pairwise correlation heatmap (2010–2019)")


# ---------------------------------------------------------------------------
# Page 3 — Candidate Stock Research
# ---------------------------------------------------------------------------
def render_candidate_research() -> None:
    st.title("Candidate Stock Research")

    st.markdown("**Screening framework — four required criteria:**")
    st.markdown(
        "- Market cap > $50B\n"
        "- 3+ consecutive years of positive net income\n"
        "- Identifiable competitive moat (brand, network effect, cost advantage, switching cost, or regulatory license)\n"
        "- Max 2–3 holdings per GICS sector\n\n"
        "**Best-in-class sub-rule:** within each sector, select the name with the highest 5-year EPS CAGR "
        "that also carries an investment-grade credit rating."
    )

    candidates_df = read_table("tbl_candidates.csv")
    screen_df = coerce_numeric(
        read_table("tbl_candidate_screen.csv"),
        ["ann_return_pre2020", "ann_vol_pre2020", "sharpe_pre2020", "max_drawdown_pre2020"],
    )

    if not candidates_df.empty:
        final_series = candidates_df.get("selected_for_final", pd.Series(dtype=str)).map(normalize).str.lower()
        m1, m2, m3 = st.columns(3)
        m1.metric("Sectors covered", "5")
        m2.metric("Candidates studied", len(candidates_df))
        m3.metric("Selected additions", int(final_series.eq("yes").sum()))

    if not candidates_df.empty:
        st.markdown("### Candidate decision table")
        if not screen_df.empty:
            disp = screen_df[[c for c in ["candidate_ticker", "company_name", "sector_theme", "add_decision", "selected_for_final", "ann_return_pre2020", "ann_vol_pre2020", "sharpe_pre2020"] if c in screen_df.columns]].copy()
            show_table(
                disp,
                pct_cols=["ann_return_pre2020", "ann_vol_pre2020"],
                ratio_cols=["sharpe_pre2020"],
            )
        else:
            disp = pd.DataFrame({
                "Ticker": candidates_df.get("candidate_ticker", pd.Series(dtype=str)).map(normalize),
                "Company": candidates_df.get("company_name", pd.Series(dtype=str)).map(normalize),
                "Sector / Theme": candidates_df.get("sector_theme", pd.Series(dtype=str)).map(normalize),
                "Decision": candidates_df.get("add_decision", pd.Series(dtype=str)).map(normalize),
                "Selected?": candidates_df.get("selected_for_final", pd.Series(dtype=str)).map(normalize),
            })
            show_table(disp)

    c1, c2 = st.columns(2)
    with c1:
        img("fig_candidate_risk_return.png", "Candidate risk vs. return (through Dec 2019)")
    with c2:
        img("fig_candidate_recent_return.png", "Candidate 1-year return into the decision date")

    if not candidates_df.empty:
        st.markdown("### Candidate profiles")
        selected = candidates_df.loc[candidates_df.get("selected_for_final", pd.Series(dtype=str)).map(normalize).str.lower().eq("yes")]
        others = candidates_df.loc[~candidates_df.index.isin(selected.index)]

        if not selected.empty:
            st.markdown("**Final additions**")
            for _, row in selected.iterrows():
                t = normalize(row.get("candidate_ticker", ""))
                label = f"{normalize(row.get('company_name', ''))} ({t}) — Selected"
                with st.expander(label):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Sector:** {normalize(row.get('sector_theme', ''))}")
                        st.write(f"**Target weight:** {fmt_pct(row.get('target_weight_2020', np.nan))}")
                    with col2:
                        st.write(f"**Investment thesis:** {normalize(row.get('thesis_1line', ''))}")
                        st.write(f"**Screening note:** {normalize(row.get('screening_note', ''))}")
                    st.write(f"**Why selected:** {CANDIDATE_RATIONALE.get(t, 'Best-in-class in sector.')}")

        if not others.empty:
            st.markdown("**Other names studied**")
            for _, row in others.iterrows():
                t = normalize(row.get("candidate_ticker", ""))
                label = f"{normalize(row.get('company_name', ''))} ({t}) — Not selected"
                with st.expander(label):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**Sector:** {normalize(row.get('sector_theme', ''))}")
                        st.write(f"**Decision:** {normalize(row.get('add_decision', ''))}")
                    with col2:
                        st.write(f"**Investment thesis:** {normalize(row.get('thesis_1line', ''))}")
                        st.write(f"**Screening note:** {normalize(row.get('screening_note', ''))}")
                    st.write(f"**Why not selected:** {CANDIDATE_RATIONALE.get(t, 'Did not win best-in-class comparison within sector.')}")


# ---------------------------------------------------------------------------
# Page 4 — Revised Portfolio Construction
# ---------------------------------------------------------------------------
def render_portfolio_construction(constraints: dict[str, str]) -> None:
    st.title("Revised Portfolio Construction")
    st.write(
        "The revised fund initializes on January 1, 2020 with target weights set by conviction tier. "
        "The static version holds those dollar positions unchanged. The active version rebalances monthly."
    )

    portfolio_df = coerce_numeric(read_table("tbl_portfolio_summary.csv"), ["static_weight", "latest_active_weight"])
    term_df = compute_terminal_weights()

    callout(
        "<strong>Why these specific weights (not equal weight)?</strong><br>"
        "AAPL and AMZN receive <strong>15% each</strong> — the highest conviction tier — because they generated "
        "the two strongest annualized alphas in the inherited fund over 2010–2019 (AMZN +14.9%, AAPL +13.3%). "
        "That is a decade of evidence. Equal-weighting them at 10% would deliberately underweight names we have "
        "the most reason to believe in.<br><br>"
        "CMCSA, GOOG, and MSFT receive <strong>10% each</strong> — meaningful but a step below the top two, "
        "reflecting solid but slightly lower conviction (GOOG's alpha was the weakest of the five kept names at 3.3%).<br><br>"
        "V, WMT, CVX, NEE, and UNH receive <strong>8% each</strong> — equal weight among themselves because "
        "they have no in-sample performance history in this portfolio. Their lower weight reflects the difference "
        "between conviction earned over 10 years and conviction based on pre-2020 screening alone."
    )

    st.markdown("---")

    if not portfolio_df.empty:
        st.markdown("### Portfolio holdings")
        out = pd.DataFrame({
            "Ticker": portfolio_df.get("ticker", pd.Series(dtype=str)).map(normalize),
            "Company": portfolio_df.get("ticker", pd.Series(dtype=str)).map(normalize).map(ticker_company),
            "Sector": portfolio_df.get("ticker", pd.Series(dtype=str)).map(normalize).map(ticker_sector),
            "Source": portfolio_df.get("source", pd.Series(dtype=str)).map(normalize),
            "Decision": portfolio_df.get("decision", pd.Series(dtype=str)).map(normalize),
            "Initial Weight (Jan 2020)": portfolio_df.get("static_weight", pd.Series(dtype=float)).map(fmt_pct),
            "Latest Active Weight (Dec 2025)": portfolio_df.get("latest_active_weight", pd.Series(dtype=float)).map(fmt_pct),
        })
        show_table(out)

    st.markdown("---")
    st.markdown("### Terminal weights — Initial vs. Dec 2025")
    st.write(
        "The static fund starts at target weights but is never rebalanced — prices drift it over time. "
        "By December 2025, the tech winners have again dominated. The active fund's monthly rebalancing "
        "keeps weights close to the original targets."
    )

    if not term_df.empty:
        disp = term_df.copy()
        st.dataframe(
            disp.style.format({
                "Initial Weight (Jan 2020)": "{:.1%}",
                "Static Terminal (Dec 2025)": "{:.1%}",
                "Active Terminal (Dec 2025)": "{:.1%}",
                "Static Drift": lambda x: f"+{x:.1%}" if x > 0 else f"{x:.1%}",
            }).background_gradient(subset=["Static Drift"], cmap="RdYlGn_r", vmin=-0.10, vmax=0.10)
            .hide(axis="index"),
            width="stretch",
        )

        callout(
            "<strong>Static drift story:</strong> AAPL started at 15% and drifted to 22.2% by Dec 2025. "
            "GOOG started at 10% and drifted to 18.2%. CMCSA started at 10% and shrank to just 3.1%. "
            "Without rebalancing, the static fund repeats the same concentration problem as the inherited fund — "
            "just with a different set of winners. The active fund's monthly Sharpe-tilt keeps all names "
            "within a few percentage points of their targets.",
            "warning",
        )

    st.markdown("---")
    st.markdown("### Active weight evolution (2020–2025)")
    img("fig_revised_active_weights.png", "Monthly Sharpe-tilt weight evolution")

    callout(
        "<strong>Why Sharpe-tilt rebalancing, not equal-weight rebalancing?</strong><br>"
        "Our control workbook specifies an 'optimizer' rule with objective 'max_sharpe'. The pipeline implements "
        "this as a <strong>Sharpe-tilt</strong>: at each month-end, it computes a trailing 36-month mean/volatility "
        "signal for each holding and scales the static target weights proportionally toward names with "
        "higher risk-adjusted momentum. Names that are compounding well on a risk-adjusted basis receive "
        "more weight; names that are deteriorating receive less.<br><br>"
        "A <em>true</em> max-Sharpe optimizer (quadratic programming) was not used because it is highly sensitive "
        "to noisy short-window return estimates, frequently produces corner solutions that concentrate the portfolio "
        "in one or two names, and would undermine our diversification mandate. The tilt achieves the same "
        "directional intent — tilt toward quality momentum — while remaining diversified and respecting all "
        "constraints (25% single-name cap, 20% turnover cap, 10 bps transaction cost).<br><br>"
        "Equal-weight rebalancing was rejected because it would systematically sell our best performers to buy "
        "laggards every month — the opposite of a quality-momentum thesis."
    )

    st.markdown("### Active implementation settings")
    s1, s2, s3, s4, s5 = st.columns(5)
    s1.metric("Active rule", "Sharpe-tilt")
    s2.metric("Max weight", fmt_pct(constraints.get("max_weight")))
    s3.metric("Min weight", fmt_pct(constraints.get("min_weight", "0")))
    s4.metric("Turnover cap", fmt_pct(constraints.get("turnover_cap")))
    s5.metric("Txn cost", f"{constraints.get('transaction_cost_bps', '10')} bps")


# ---------------------------------------------------------------------------
# Page 5 — Backtest Comparison
# ---------------------------------------------------------------------------
def render_backtest() -> None:
    st.title("Backtest Comparison")
    st.write("Out-of-sample evaluation: January 1, 2020 – December 31, 2025. All three BlackFin portfolios beat SPY on both return and risk-adjusted terms.")

    perf = coerce_numeric(
        read_table("tbl_performance_summary.csv"),
        ["ending_value", "total_return", "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown"],
    )

    # Table first — Sharpe and Sortino carry the argument; chart follows
    if not perf.empty:
        st.markdown("### Risk-adjusted summary (OOS: 2020–2025)")
        show_perf_table(
            perf[[c for c in ["portfolio", "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown"] if c in perf.columns]]
        )

    callout(
        "<strong>The risk-adjusted case:</strong> The Legacy Fund earned the highest raw return (18.5%) "
        "driven by concentrated AMZN and AAPL exposure that happened to work well in the 2020–2025 environment. "
        "But it carried the highest volatility (24.3%) and lowest Sharpe (0.759) of the three BlackFin portfolios. "
        "The <strong>Revised Active Fund achieves the best Sharpe (0.793) and Sortino (1.004)</strong> of all four "
        "portfolios — more return per unit of risk — while reducing maximum drawdown by 3.6 pp vs. Legacy "
        "and 5.9 pp vs. SPY. The raw return gap (~1.6 pp) is the deliberate cost of proper diversification."
    )

    img("fig_legacy_static_active_benchmark.png", "Growth of $1 — all portfolios vs. benchmark (OOS)")

    st.markdown("---")
    st.markdown("### Rebalance log summary")
    log = coerce_numeric(read_table("tbl_revised_active_rebalance_log.csv"), ["turnover"])
    if not log.empty:
        avg_to = log["turnover"].mean() if "turnover" in log.columns else np.nan
        n_reb = len(log)
        rule_counts = log["rule_used"].value_counts().to_dict() if "rule_used" in log.columns else {}
        r1, r2, r3 = st.columns(3)
        r1.metric("Total rebalances", str(n_reb))
        r2.metric("Avg monthly turnover", fmt_pct(avg_to))
        r3.metric("Sharpe-tilt months", str(rule_counts.get("sharpe_tilt", "N/A")))

        # Turnover time series instead of a raw table
        if "decision_date" in log.columns and "turnover" in log.columns:
            to_chart = log[["decision_date", "turnover"]].copy()
            to_chart["decision_date"] = pd.to_datetime(to_chart["decision_date"], errors="coerce")
            to_chart = to_chart.dropna(subset=["decision_date"]).set_index("decision_date").sort_index()
            to_chart.columns = ["Monthly Turnover"]
            st.line_chart(to_chart)


# ---------------------------------------------------------------------------
# Page 6 — Risk and Diagnostics
# ---------------------------------------------------------------------------
def render_risk_diagnostics() -> None:
    st.title("Risk and Diagnostics")
    st.write("CAPM and Fama-French 3-factor regressions plus rolling beta diagnostics over the 2020–2025 out-of-sample period.")

    tab_capm, tab_ff3 = st.tabs(["CAPM", "Fama-French 3-Factor"])

    with tab_capm:
        capm = coerce_numeric(
            read_table("tbl_factor_capm_summary.csv"),
            ["alpha_ann", "beta", "r_squared", "alpha_tstat", "alpha_pvalue"],
        )

        if not capm.empty and "status" not in capm.columns:
            st.markdown("### CAPM summary (OOS: 2020–2025)")
            show_table(
                capm[[c for c in ["portfolio", "alpha_ann", "beta", "r_squared", "alpha_tstat", "alpha_pvalue"] if c in capm.columns]],
                pct_cols=["alpha_ann"],
                ratio_cols=["beta", "r_squared", "alpha_tstat", "alpha_pvalue"],
            )

        callout(
            "<strong>CAPM interpretation:</strong> The Legacy Fund's beta of 1.065 confirms it was essentially a "
            "leveraged market bet — investors took on more volatility without statistical evidence of skill. "
            "The revised funds bring beta below 1.0 (0.966–0.987), meaning they move less than the market in "
            "both directions. All alpha estimates are positive but none are statistically significant (p > 0.40), "
            "which is expected from a 6-year sample — uncertainty bands are wide."
        )

        c1, c2 = st.columns(2)
        with c1:
            img("fig_factor_alpha_beta.png", "Annualized alpha vs. beta — all portfolios")
        with c2:
            img("fig_factor_rolling_beta.png", "Rolling 63-day CAPM beta (2020–2025)")

    with tab_ff3:
        ff3 = coerce_numeric(
            read_table("tbl_factor_ff3_summary.csv"),
            ["alpha_ann", "alpha_tstat", "alpha_pvalue", "beta_mkt", "beta_smb", "beta_hml", "r_squared"],
        )

        if not ff3.empty:
            st.markdown("### Fama-French 3-Factor summary (OOS: 2020–2025)")
            show_table(
                ff3[[c for c in ["portfolio", "alpha_ann", "alpha_tstat", "alpha_pvalue", "beta_mkt", "beta_smb", "beta_hml", "r_squared"] if c in ff3.columns]],
                pct_cols=["alpha_ann"],
                ratio_cols=["alpha_tstat", "alpha_pvalue", "beta_mkt", "beta_smb", "beta_hml", "r_squared"],
            )
            callout(
                "<strong>FF3 interpretation:</strong> Adding SMB (size) and HML (value) factors reveals the style tilts "
                "embedded in each portfolio. Negative SMB loadings confirm all three funds are large-cap biased — "
                "consistent with our screening rule (market cap > $50B). Negative HML loadings confirm a growth tilt "
                "(AAPL, AMZN, GOOG, MSFT all trade at high price-to-book). The key question is whether alpha survives "
                "after stripping out these passive factor exposures. Alphas that remain positive after FF3 adjustment "
                "represent genuine stock-selection value beyond size and style."
            )
        else:
            st.info("FF3 outputs not available — run the pipeline to generate them.")

        img("fig_factor_ff3.png", "FF3 factor loadings and annualized alpha by portfolio")

    st.markdown("---")
    price_q = read_table("tbl_price_quality_summary.csv")
    if not price_q.empty:
        st.markdown("### Data quality")
        show_table(price_q[[c for c in ["ticker", "n_obs", "n_missing", "first_date", "last_date"] if c in price_q.columns]])


# ---------------------------------------------------------------------------
# Page 7 — Scenario / Stress Test
# ---------------------------------------------------------------------------
def render_scenario_stress() -> None:
    st.title("Scenario / Stress Test")
    st.write("Historical stress events and Monte Carlo simulation for the redesigned fund.")

    tab1, tab2 = st.tabs(["Historical Stress Events", "Monte Carlo Simulation"])

    with tab1:
        stress = coerce_numeric(
            read_table("tbl_scenario_stress_summary.csv"),
            ["benchmark_return", "static_portfolio_impact", "active_portfolio_impact"],
        )

        # Add Legacy Fund column from daily returns data
        legacy_impacts = compute_legacy_stress_impacts()
        if not stress.empty and legacy_impacts:
            stress = stress.copy()
            stress["legacy_fund_impact"] = stress["scenario"].map(
                lambda s: legacy_impacts.get(normalize(s), np.nan)
            )
            stress["legacy_fund_impact"] = pd.to_numeric(stress["legacy_fund_impact"], errors="coerce")

        if not stress.empty:
            display_cols = [c for c in [
                "scenario", "benchmark_return", "legacy_fund_impact",
                "static_portfolio_impact", "active_portfolio_impact"
            ] if c in stress.columns]
            show_table(
                stress[display_cols],
                pct_cols=["benchmark_return", "legacy_fund_impact", "static_portfolio_impact", "active_portfolio_impact"],
            )

        img("fig_scenario_stress_impacts.png", "Scenario and stress test impacts")

        callout(
            "<strong>Stress finding:</strong> On the worst single day of the evaluation period "
            "(COVID crash, March 16, 2020), SPY fell 11.5%. Both revised portfolios held up marginally better "
            "(Active −11.1%, Static −11.2%), consistent with their lower beta profile. Over the worst 21-day window, "
            "SPY fell 27.9% vs. Active −25.8% and Static −26.1%. The revised fund does not protect against "
            "broad equity selloffs — it was designed for long-term growth — but its lower beta provides "
            "modest cushion in market stress."
        )

    with tab2:
        mc = coerce_numeric(
            read_table("tbl_scenario_monte_carlo_summary.csv"),
            ["mean_terminal_return", "median_terminal_return", "var_95", "cvar_95", "prob_loss"],
        )
        img("fig_scenario_monte_carlo_distribution.png", "Monte Carlo 1-year terminal return distribution (2,000 simulations)")
        if not mc.empty:
            show_table(
                mc[[c for c in ["portfolio", "mean_terminal_return", "median_terminal_return", "var_95", "cvar_95", "prob_loss"] if c in mc.columns]],
                pct_cols=["mean_terminal_return", "median_terminal_return", "var_95", "cvar_95", "prob_loss"],
            )
        callout(
            "<strong>Monte Carlo (1-year, 2,000 simulations):</strong> Active Fund mean return 20.0%, "
            "probability of loss 21.6%, CVaR-95 −23.2%. Static Fund mean 19.2%, probability of loss 23.0%, "
            "CVaR-95 −24.2%. Both revised funds show modestly better downside profiles. These are 1-year "
            "simulations bootstrapped from OOS daily returns — they capture realized volatility structure "
            "but assume stationarity."
        )


# ---------------------------------------------------------------------------
# Page 8 — Final Recommendation
# ---------------------------------------------------------------------------
def render_final_recommendation(constraints: dict[str, str]) -> None:
    st.title("Final Recommendation")

    callout(
        "<strong>BlackFin recommends:</strong> Replace the inherited fund with the Revised 10-Stock Portfolio "
        "and implement it using <strong>monthly Sharpe-tilt active rebalancing</strong>. "
        "The active fund delivers Sharpe 0.793 and Sortino 1.004 — the best risk-adjusted profile of all four portfolios.",
        "success",
    )

    st.markdown("### Fund mandate")
    st.write(
        "**A long-horizon retirement equity portfolio that holds large-cap, moat-quality businesses "
        "across diversified sectors, governed by a monthly rebalancing rule that prevents unmanaged concentration drift.**"
    )

    inherited_df = read_table("tbl_inherited_fund.csv")
    candidates_df = read_table("tbl_candidates.csv")

    keeps, drops, adds = [], [], []
    if not inherited_df.empty:
        flag = inherited_df.get("keep_in_revised", pd.Series(dtype=str)).map(normalize).str.lower()
        keeps = inherited_df.loc[flag.isin({"yes", "keep"}), "legacy_ticker"].map(normalize).tolist()
        drops = inherited_df.loc[flag.eq("no"), "legacy_ticker"].map(normalize).tolist()
    if not candidates_df.empty:
        adds = candidates_df.loc[candidates_df.get("selected_for_final", pd.Series(dtype=str)).map(normalize).str.lower().eq("yes"), "candidate_ticker"].map(normalize).tolist()

    k1, k2, k3 = st.columns(3)
    with k1:
        st.markdown("**Keep**")
        for t in keeps:
            st.write(f"- **{t}** — {ticker_company(t)} | {ticker_sector(t)}")
    with k2:
        st.markdown("**Drop**")
        for t in drops:
            st.write(f"- **{t}** — {ticker_company(t)} | {ticker_sector(t)}")
    with k3:
        st.markdown("**Add**")
        for t in adds:
            st.write(f"- **{t}** — {ticker_company(t)} | {ticker_sector(t)}")

    st.markdown("---")

    perf = coerce_numeric(
        read_table("tbl_performance_summary.csv"),
        ["ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown"],
    )

    if not perf.empty:
        st.markdown("### Why the active fund over the static fund?")
        show_perf_table(
            perf[[c for c in ["portfolio", "ann_return", "ann_vol", "sharpe", "sortino", "max_drawdown"] if c in perf.columns]]
        )
        callout(
            "The static fund generates similar returns but allows concentration to re-form — by December 2025, "
            "AAPL had drifted from 15% to 22.2% and GOOG from 10% to 18.2% without a single active decision. "
            "The active rebalancing rule earns its cost (avg ~8–12% monthly turnover, 10 bps/trade) by preventing "
            "the same structural problem we were hired to fix from silently recurring."
        )

    st.markdown("---")
    st.markdown("### Tradeoffs accepted")
    st.write(
        "The revised fund gives up ~1.6 pp of annual return versus the Legacy Fund (16.9% vs. 18.5%). "
        "That gap reflects the Legacy's concentrated AMZN and AAPL exposure, which happened to work well "
        "in the 2020–2025 growth-stock environment. A manager cannot know in advance that the concentrated "
        "bet will work. The revised fund accepts a more deliberate structure in exchange for lower volatility, "
        "better Sharpe, and a governance rule that a retirement investor can actually defend to clients."
    )
    st.write(
        "The portfolio remains fully invested in equities — it provides no protection in a sustained value "
        "rotation or broad technology de-rating beyond the modest diversification benefit of the new sector "
        "additions. That is a known and accepted risk for a long-horizon growth mandate."
    )

    st.markdown("---")
    st.markdown("### Manager conclusion")
    callout(
        "The revised portfolio better matches the mandate of a long-horizon retirement investor. It preserves the "
        "five strongest inherited compounders, eliminates three negative-alpha names and two redundant/failing holdings, "
        "adds four previously unrepresented sectors, and introduces a transparent monthly governance rule that "
        "prevents unmanaged concentration from recurring. It beats both the Legacy Fund and SPY on risk-adjusted "
        "terms over the entire out-of-sample evaluation window.",
        "success",
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    inputs = load_kv("tbl_inputs.csv")
    constraints = load_kv("tbl_constraints.csv")

    page = render_sidebar(inputs, constraints)

    if page == "Executive Overview":
        render_overview(inputs)
    elif page == "Inherited Fund Review":
        render_inherited_review()
    elif page == "Candidate Stock Research":
        render_candidate_research()
    elif page == "Revised Portfolio Construction":
        render_portfolio_construction(constraints)
    elif page == "Backtest Comparison":
        render_backtest()
    elif page == "Risk and Diagnostics":
        render_risk_diagnostics()
    elif page == "Scenario / Stress Test":
        render_scenario_stress()
    elif page == "Final Recommendation":
        render_final_recommendation(constraints)


if __name__ == "__main__":
    main()
