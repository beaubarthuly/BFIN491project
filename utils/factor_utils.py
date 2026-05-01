
from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.data_utils import normalize_text

try:
    import statsmodels.api as sm
except Exception:  # pragma: no cover - defensive fallback
    sm = None


PORTFOLIO_LABELS = {
    "legacy_return": "Legacy Fund",
    "revised_static_return": "Revised Static Fund",
    "revised_active_return": "Revised Active Fund",
}


def _standardize_date_column(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "date" in out.columns:
        return out
    for candidate in ["Date", "index"]:
        if candidate in out.columns:
            return out.rename(columns={candidate: "date"})
    return out


def build_portfolio_return_panel(
    legacy_fund_daily: pd.DataFrame,
    static_daily: pd.DataFrame | None,
    active_daily: pd.DataFrame | None,
    benchmark_daily: pd.DataFrame,
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
) -> pd.DataFrame:
    """Assemble a clean daily return panel for factor/regression work."""
    frames: list[pd.DataFrame] = []

    legacy = _standardize_date_column(legacy_fund_daily)
    legacy["date"] = pd.to_datetime(legacy["date"], errors="coerce")
    legacy = legacy.dropna(subset=["date"]).set_index("date").sort_index()
    if "legacy_fund_return" in legacy.columns:
        frames.append(legacy[["legacy_fund_return"]].rename(columns={"legacy_fund_return": "legacy_return"}))

    if static_daily is not None and not static_daily.empty:
        static = _standardize_date_column(static_daily)
        static["date"] = pd.to_datetime(static["date"], errors="coerce")
        static = static.dropna(subset=["date"]).set_index("date").sort_index()
        if "revised_static_return" in static.columns:
            frames.append(static[["revised_static_return"]])

    if active_daily is not None and not active_daily.empty:
        active = _standardize_date_column(active_daily)
        active["date"] = pd.to_datetime(active["date"], errors="coerce")
        active = active.dropna(subset=["date"]).set_index("date").sort_index()
        if "revised_active_return" in active.columns:
            frames.append(active[["revised_active_return"]])

    benchmark = _standardize_date_column(benchmark_daily)
    benchmark["date"] = pd.to_datetime(benchmark["date"], errors="coerce")
    benchmark = benchmark.dropna(subset=["date"]).set_index("date").sort_index()
    if "benchmark_return" in benchmark.columns:
        frames.append(benchmark[["benchmark_return"]])

    if not frames:
        return pd.DataFrame()

    panel = pd.concat(frames, axis=1, join="outer").sort_index()
    panel = panel.loc[(panel.index >= pd.Timestamp(oos_start)) & (panel.index <= pd.Timestamp(oos_end))].copy()
    return panel


def _simple_ols(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    """Fallback OLS if statsmodels is unavailable."""
    X = np.column_stack([np.ones(len(x)), x])
    beta_hat, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta_hat
    resid = y - y_hat
    n = len(y)
    k = X.shape[1]
    sse = float((resid**2).sum())
    sst = float(((y - y.mean())**2).sum())
    r2 = np.nan if sst == 0 else 1.0 - sse / sst
    sigma2 = np.nan if n <= k else sse / (n - k)
    try:
        var_beta = sigma2 * np.linalg.inv(X.T @ X)
        se = np.sqrt(np.diag(var_beta))
        tvals = beta_hat / se
    except Exception:
        tvals = np.array([np.nan, np.nan])

    return {
        "alpha_daily": float(beta_hat[0]),
        "beta": float(beta_hat[1]),
        "alpha_tstat": float(tvals[0]) if np.isfinite(tvals[0]) else np.nan,
        "beta_tstat": float(tvals[1]) if np.isfinite(tvals[1]) else np.nan,
        "alpha_pvalue": np.nan,
        "beta_pvalue": np.nan,
        "r_squared": float(r2) if np.isfinite(r2) else np.nan,
    }


def run_capm_regression(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    portfolio_name: str,
    risk_free_daily: float = 0.0,
) -> dict[str, Any]:
    aligned = pd.concat(
        [
            pd.to_numeric(portfolio_returns, errors="coerce").rename("portfolio"),
            pd.to_numeric(benchmark_returns, errors="coerce").rename("benchmark"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    if aligned.shape[0] < 20:
        return {
            "portfolio": portfolio_name,
            "n_obs": int(aligned.shape[0]),
            "alpha_daily": np.nan,
            "alpha_ann": np.nan,
            "beta": np.nan,
            "alpha_tstat": np.nan,
            "alpha_pvalue": np.nan,
            "beta_tstat": np.nan,
            "beta_pvalue": np.nan,
            "r_squared": np.nan,
            "corr_to_benchmark": np.nan,
            "market_proxy": "benchmark_return",
            "risk_free_assumption": risk_free_daily,
            "notes": "Not enough aligned daily observations for CAPM regression.",
        }

    y = aligned["portfolio"].astype(float) - float(risk_free_daily)
    x = aligned["benchmark"].astype(float) - float(risk_free_daily)

    if sm is not None:
        X = sm.add_constant(x.values, has_constant="add")
        model = sm.OLS(y.values, X).fit()
        alpha_daily = float(model.params[0])
        beta = float(model.params[1]) if len(model.params) > 1 else np.nan
        alpha_tstat = float(model.tvalues[0]) if len(model.tvalues) > 0 else np.nan
        alpha_pvalue = float(model.pvalues[0]) if len(model.pvalues) > 0 else np.nan
        beta_tstat = float(model.tvalues[1]) if len(model.tvalues) > 1 else np.nan
        beta_pvalue = float(model.pvalues[1]) if len(model.pvalues) > 1 else np.nan
        r_squared = float(model.rsquared)
    else:
        fallback = _simple_ols(y.values, x.values)
        alpha_daily = fallback["alpha_daily"]
        beta = fallback["beta"]
        alpha_tstat = fallback["alpha_tstat"]
        alpha_pvalue = fallback["alpha_pvalue"]
        beta_tstat = fallback["beta_tstat"]
        beta_pvalue = fallback["beta_pvalue"]
        r_squared = fallback["r_squared"]

    alpha_ann = alpha_daily * 252.0
    corr = float(aligned["portfolio"].corr(aligned["benchmark"]))

    return {
        "portfolio": portfolio_name,
        "n_obs": int(aligned.shape[0]),
        "alpha_daily": alpha_daily,
        "alpha_ann": alpha_ann,
        "beta": beta,
        "alpha_tstat": alpha_tstat,
        "alpha_pvalue": alpha_pvalue,
        "beta_tstat": beta_tstat,
        "beta_pvalue": beta_pvalue,
        "r_squared": r_squared,
        "corr_to_benchmark": corr,
        "market_proxy": "benchmark_return",
        "risk_free_assumption": risk_free_daily,
        "notes": "Starter CAPM uses the benchmark daily return as the market proxy and assumes daily risk-free rate = 0.",
    }


def build_capm_summary(
    return_panel: pd.DataFrame,
    benchmark_col: str = "benchmark_return",
    risk_free_daily: float = 0.0,
) -> pd.DataFrame:
    if return_panel.empty or benchmark_col not in return_panel.columns:
        return pd.DataFrame(
            columns=[
                "portfolio",
                "n_obs",
                "alpha_daily",
                "alpha_ann",
                "beta",
                "alpha_tstat",
                "alpha_pvalue",
                "beta_tstat",
                "beta_pvalue",
                "r_squared",
                "corr_to_benchmark",
                "market_proxy",
                "risk_free_assumption",
                "notes",
            ]
        )

    rows: list[dict[str, Any]] = []
    for col in [c for c in return_panel.columns if c != benchmark_col]:
        label = PORTFOLIO_LABELS.get(col, normalize_text(col))
        rows.append(
            run_capm_regression(
                portfolio_returns=return_panel[col],
                benchmark_returns=return_panel[benchmark_col],
                portfolio_name=label,
                risk_free_daily=risk_free_daily,
            )
        )

    return pd.DataFrame(rows)


def build_rolling_beta_table(
    return_panel: pd.DataFrame,
    benchmark_col: str = "benchmark_return",
    window: int = 63,
) -> pd.DataFrame:
    if return_panel.empty or benchmark_col not in return_panel.columns:
        return pd.DataFrame(columns=["date", "portfolio", "rolling_beta", "window"])

    benchmark = pd.to_numeric(return_panel[benchmark_col], errors="coerce")
    rows: list[pd.DataFrame] = []

    for col in [c for c in return_panel.columns if c != benchmark_col]:
        port = pd.to_numeric(return_panel[col], errors="coerce")
        beta = port.rolling(window).cov(benchmark) / benchmark.rolling(window).var()
        tmp = pd.DataFrame(
            {
                "date": beta.index,
                "portfolio": PORTFOLIO_LABELS.get(col, normalize_text(col)),
                "rolling_beta": beta.values,
                "window": int(window),
            }
        )
        rows.append(tmp)

    if not rows:
        return pd.DataFrame(columns=["date", "portfolio", "rolling_beta", "window"])

    out = pd.concat(rows, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    return out.dropna(subset=["date"]).reset_index(drop=True)


def plot_factor_alpha_beta(capm_summary: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    valid = capm_summary.dropna(subset=["beta", "alpha_ann"]).copy() if not capm_summary.empty else pd.DataFrame()

    if valid.empty:
        ax.text(0.5, 0.5, "Factor/regression outputs are not ready yet.", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.scatter(valid["beta"], valid["alpha_ann"])
    for _, row in valid.iterrows():
        ax.annotate(
            normalize_text(row.get("portfolio")),
            (row["beta"], row["alpha_ann"]),
            fontsize=8,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.axvline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_title("CAPM Summary: Beta vs Annualized Alpha")
    ax.set_xlabel("Benchmark beta")
    ax.set_ylabel("Annualized alpha (daily alpha × 252)")
    ax.grid(alpha=0.25)


# ---------------------------------------------------------------------------
# Fama-French 3-Factor functions
# ---------------------------------------------------------------------------

def load_ff3_factors(
    ff3_path: "str | Path",
    oos_start: pd.Timestamp,
    oos_end: pd.Timestamp,
) -> pd.DataFrame:
    """Load saved FF3 daily factors CSV and trim to the OOS window."""
    from pathlib import Path
    p = Path(str(ff3_path))
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["mkt_rf", "smb", "hml", "rf"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date"]).set_index("date").sort_index()
    return df.loc[(df.index >= oos_start) & (df.index <= oos_end)].copy()


def run_ff3_regression(
    portfolio_returns: pd.Series,
    ff3_factors: pd.DataFrame,
    portfolio_name: str,
) -> dict[str, Any]:
    """OLS regression of excess portfolio returns on Mkt-RF, SMB, HML."""
    required = {"mkt_rf", "smb", "hml", "rf"}
    if ff3_factors.empty or not required.issubset(ff3_factors.columns):
        return {"portfolio": portfolio_name, "notes": "FF3 factor data unavailable."}

    port = pd.to_numeric(portfolio_returns, errors="coerce").rename("portfolio")
    aligned = pd.concat([port, ff3_factors[["mkt_rf", "smb", "hml", "rf"]]], axis=1, join="inner").dropna()

    if len(aligned) < 20:
        return {
            "portfolio": portfolio_name,
            "n_obs": len(aligned),
            "alpha_daily": np.nan,
            "alpha_ann": np.nan,
            "alpha_tstat": np.nan,
            "alpha_pvalue": np.nan,
            "beta_mkt": np.nan,
            "beta_smb": np.nan,
            "beta_hml": np.nan,
            "r_squared": np.nan,
            "notes": "Insufficient observations.",
        }

    y = aligned["portfolio"] - aligned["rf"]
    X_vals = aligned[["mkt_rf", "smb", "hml"]].values

    if sm is not None:
        X = sm.add_constant(X_vals, has_constant="add")
        model = sm.OLS(y.values, X).fit()
        alpha_d   = float(model.params[0])
        beta_mkt  = float(model.params[1])
        beta_smb  = float(model.params[2])
        beta_hml  = float(model.params[3])
        a_tstat   = float(model.tvalues[0])
        a_pvalue  = float(model.pvalues[0])
        r2        = float(model.rsquared)
    else:
        # Fallback: numpy lstsq
        X = np.column_stack([np.ones(len(X_vals)), X_vals])
        params, *_ = np.linalg.lstsq(X, y.values, rcond=None)
        alpha_d, beta_mkt, beta_smb, beta_hml = params
        resid = y.values - X @ params
        n, k = len(y), X.shape[1]
        sse = float((resid**2).sum())
        sst = float(((y.values - y.values.mean())**2).sum())
        r2 = 1.0 - sse / sst if sst > 0 else np.nan
        sigma2 = sse / (n - k) if n > k else np.nan
        try:
            se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
            a_tstat = float(params[0] / se[0])
        except Exception:
            a_tstat = np.nan
        a_pvalue = np.nan

    return {
        "portfolio":    portfolio_name,
        "n_obs":        len(aligned),
        "alpha_daily":  float(alpha_d),
        "alpha_ann":    float(alpha_d) * 252.0,
        "alpha_tstat":  float(a_tstat),
        "alpha_pvalue": float(a_pvalue) if not np.isnan(a_pvalue) else np.nan,
        "beta_mkt":     float(beta_mkt),
        "beta_smb":     float(beta_smb),
        "beta_hml":     float(beta_hml),
        "r_squared":    float(r2),
        "notes":        "FF3 OLS: Ri - Rf = alpha + beta_mkt*(Rmkt-Rf) + beta_smb*SMB + beta_hml*HML",
    }


def build_ff3_summary(
    return_panel: pd.DataFrame,
    ff3_factors: pd.DataFrame,
    benchmark_col: str = "benchmark_return",
) -> pd.DataFrame:
    if return_panel.empty or ff3_factors.empty:
        return pd.DataFrame()
    rows = []
    for col in [c for c in return_panel.columns if c != benchmark_col]:
        label = PORTFOLIO_LABELS.get(col, normalize_text(col))
        rows.append(run_ff3_regression(return_panel[col], ff3_factors, label))
    return pd.DataFrame(rows)


def plot_ff3_exposures(ff3_summary: pd.DataFrame) -> None:
    """Grouped bar chart: Mkt-RF, SMB, HML betas per portfolio, plus alpha annotation."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if ff3_summary.empty or "beta_mkt" not in ff3_summary.columns:
        for ax in axes:
            ax.text(0.5, 0.5, "FF3 outputs not available.", ha="center", va="center")
            ax.set_axis_off()
        return

    df = ff3_summary.dropna(subset=["beta_mkt", "beta_smb", "beta_hml"]).copy()
    portfolios = df["portfolio"].tolist()
    x = np.arange(len(portfolios))
    width = 0.25

    ax1 = axes[0]
    ax1.bar(x - width,  df["beta_mkt"].values, width, label="Mkt-RF (Market)", color="#1f77b4")
    ax1.bar(x,          df["beta_smb"].values, width, label="SMB (Size)",       color="#ff7f0e")
    ax1.bar(x + width,  df["beta_hml"].values, width, label="HML (Value)",      color="#2ca02c")
    ax1.axhline(0.0, color="black", linewidth=0.8)
    ax1.set_title("FF3 Factor Loadings")
    ax1.set_ylabel("Factor beta")
    ax1.set_xticks(x)
    ax1.set_xticklabels([p.replace(" Fund", "\nFund") for p in portfolios], fontsize=8)
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.25)

    ax2 = axes[1]
    alpha_vals = df["alpha_ann"].values
    colors = ["#3fb950" if v >= 0 else "#f85149" for v in alpha_vals]
    bars = ax2.bar(x, alpha_vals, 0.5, color=colors)
    for bar, tstat in zip(bars, df["alpha_tstat"].fillna(0).values):
        sig = "**" if abs(tstat) > 2 else ("*" if abs(tstat) > 1.65 else "")
        if sig:
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + (0.002 if bar.get_height() >= 0 else -0.008),
                     sig, ha="center", va="bottom", fontsize=10)
    ax2.axhline(0.0, color="black", linewidth=0.8)
    ax2.set_title("FF3 Annualized Alpha")
    ax2.set_ylabel("Annualized alpha")
    ax2.set_xticks(x)
    ax2.set_xticklabels([p.replace(" Fund", "\nFund") for p in portfolios], fontsize=8)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.1%}"))
    ax2.grid(axis="y", alpha=0.25)

    fig.suptitle("Fama-French 3-Factor Analysis (OOS: 2020–2025)", fontsize=12, fontweight="bold")
    fig.tight_layout()


def plot_rolling_beta(rolling_beta_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    if rolling_beta_df.empty:
        ax.text(0.5, 0.5, "Rolling-beta table is not ready yet.", ha="center", va="center")
        ax.set_axis_off()
        return

    chart_df = rolling_beta_df.copy()
    chart_df["date"] = pd.to_datetime(chart_df["date"], errors="coerce")
    chart_df = chart_df.dropna(subset=["date"])
    if chart_df.empty:
        ax.text(0.5, 0.5, "Rolling-beta dates could not be parsed.", ha="center", va="center")
        ax.set_axis_off()
        return

    pivot = chart_df.pivot(index="date", columns="portfolio", values="rolling_beta").sort_index()
    if pivot.empty:
        ax.text(0.5, 0.5, "Rolling-beta data is empty after pivoting.", ha="center", va="center")
        ax.set_axis_off()
        return

    for col in pivot.columns:
        ax.plot(pivot.index, pivot[col], label=normalize_text(col), linewidth=1.5)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_title("Rolling 63-Day Benchmark Beta")
    ax.set_xlabel("Date")
    ax.set_ylabel("Rolling beta")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
