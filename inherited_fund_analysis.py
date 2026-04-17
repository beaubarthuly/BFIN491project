from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from utils.data_utils import compute_performance_metrics
from utils.factor_utils import run_capm_regression

# ── Constants ──────────────────────────────────────────────────────────────────
TICKERS = ["AAPL", "AMZN", "BRK-B", "CMCSA", "GOOG", "KO", "MSFT", "ORCL", "WFC", "XOM"]
IS_START = "2010-01-04"
IS_END = "2019-12-31"

PRICES_PATH = ROOT / "data" / "clean" / "prices_adjclose_daily.csv"
WEIGHTS_PATH = ROOT / "outputs" / "tables" / "tbl_legacy_weights_snapshot.csv"
TABLES_DIR = ROOT / "outputs" / "tables"
FIGURES_DIR = ROOT / "outputs" / "figures"


# ── Data loading ───────────────────────────────────────────────────────────────
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prices_full = pd.read_csv(PRICES_PATH, index_col=0, parse_dates=True)
    prices_full.index = pd.to_datetime(prices_full.index)
    prices = prices_full.loc[IS_START:IS_END].copy()
    returns = prices.pct_change().dropna()
    weights_snapshot = pd.read_csv(WEIGHTS_PATH)
    return prices, returns, weights_snapshot


# ── Per-stock statistics ───────────────────────────────────────────────────────
def compute_stock_stats(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    weights_snapshot: pd.DataFrame,
) -> pd.DataFrame:
    spy_returns = returns["SPY"]
    corr_matrix = returns[TICKERS].corr()
    end_ts = pd.Timestamp(IS_END)

    rows = []
    for ticker in TICKERS:
        r = returns[ticker].dropna()
        p = prices[ticker].dropna()

        metrics = compute_performance_metrics(r, turnover=0.0)
        capm = run_capm_regression(r, spy_returns, ticker, risk_free_daily=0.0)

        # 1Y and 3Y returns to decision date
        p_1y = p.loc[end_ts - pd.DateOffset(years=1) : end_ts]
        p_3y = p.loc[end_ts - pd.DateOffset(years=3) : end_ts]
        return_1y = float(p_1y.iloc[-1] / p_1y.iloc[0] - 1) if len(p_1y) > 1 else np.nan
        return_3y = float(p_3y.iloc[-1] / p_3y.iloc[0] - 1) if len(p_3y) > 1 else np.nan

        # Weight drift
        w_end_row = weights_snapshot.loc[weights_snapshot["ticker"] == ticker, "weight"]
        weight_end = float(w_end_row.iloc[0]) if not w_end_row.empty else np.nan

        # Average pairwise correlation to the other 9 holdings
        others = [t for t in TICKERS if t != ticker]
        avg_pairwise_corr = float(corr_matrix.loc[ticker, others].mean())

        rows.append({
            "ticker": ticker,
            "total_return": metrics["total_return"],
            "ann_return": metrics["ann_return"],
            "ann_vol": metrics["ann_vol"],
            "sharpe": metrics["sharpe"],
            "sortino": metrics["sortino"],
            "max_drawdown": metrics["max_drawdown"],
            "alpha_ann": capm["alpha_ann"],
            "beta": capm["beta"],
            "r_squared": capm["r_squared"],
            "corr_to_spy": capm["corr_to_benchmark"],
            "avg_pairwise_corr": avg_pairwise_corr,
            "return_1y": return_1y,
            "return_3y": return_3y,
            "weight_start": 0.10,
            "weight_end": weight_end,
        })

    return pd.DataFrame(rows)


# ── Figure 1: 2×5 return grid ─────────────────────────────────────────────────
def plot_return_grid(prices: pd.DataFrame, stats: pd.DataFrame) -> None:
    spy_growth = prices["SPY"] / prices["SPY"].iloc[0]

    fig, axes = plt.subplots(2, 5, figsize=(20, 7))
    fig.suptitle(
        "Inherited Fund: Individual Stock Performance vs SPY  (2010–2019, In-Sample)",
        fontsize=13, fontweight="bold",
    )

    for idx, ticker in enumerate(TICKERS):
        ax = axes[idx // 5][idx % 5]
        stock_growth = prices[ticker] / prices[ticker].iloc[0]
        total_ret = stats.loc[stats["ticker"] == ticker, "total_return"].iloc[0]

        ax.plot(stock_growth.index, stock_growth.values, color="#1f77b4", linewidth=1.5, label=ticker)
        ax.plot(spy_growth.index, spy_growth.values, color="#aaaaaa", linewidth=1.0, linestyle="--", label="SPY")
        ax.set_title(f"{ticker}  ({total_ret:+.0%})", fontsize=10, fontweight="bold")
        ax.set_ylabel("Growth of $1", fontsize=8)
        ax.tick_params(axis="x", labelrotation=30, labelsize=7)
        ax.tick_params(axis="y", labelsize=7)
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_audit_grid_returns.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── Figure 2: Risk/return scatter ─────────────────────────────────────────────
def plot_risk_return(
    stats: pd.DataFrame,
    spy_ann_return: float,
    spy_ann_vol: float,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))

    for _, row in stats.iterrows():
        ax.scatter(row["ann_vol"], row["ann_return"], s=80, color="#1f77b4", zorder=3)
        ax.annotate(
            row["ticker"],
            (row["ann_vol"], row["ann_return"]),
            textcoords="offset points", xytext=(6, 4), fontsize=9,
        )

    ax.scatter(
        spy_ann_vol, spy_ann_return,
        s=140, color="#ff7f0e", marker="*", zorder=4, label="SPY (benchmark)",
    )
    ax.annotate(
        "SPY", (spy_ann_vol, spy_ann_return),
        textcoords="offset points", xytext=(6, 4), fontsize=9, color="#ff7f0e",
    )

    ax.set_xlabel("Annualized Volatility", fontsize=11)
    ax.set_ylabel("Annualized Return", fontsize=11)
    ax.set_title("Inherited Fund: Risk vs Return by Stock (2010–2019)", fontsize=12, fontweight="bold")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_audit_risk_return.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── Figure 3: Weight drift ────────────────────────────────────────────────────
def plot_weight_drift(stats: pd.DataFrame) -> None:
    x = np.arange(len(TICKERS))
    width = 0.35

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width / 2, stats["weight_start"] * 100, width, label="Start (2010-01-01)", color="#aaaaaa", edgecolor="white")
    bars_end = ax.bar(x + width / 2, stats["weight_end"] * 100, width, label="End (2019-12-31)", color="#1f77b4", edgecolor="white")

    ax.axhline(10.0, color="gray", linewidth=0.8, linestyle="--", alpha=0.6, label="Equal weight (10%)")
    ax.set_xticks(x)
    ax.set_xticklabels(TICKERS, fontsize=10)
    ax.set_ylabel("Portfolio Weight (%)", fontsize=11)
    ax.set_title(
        "Inherited Fund: Weight Drift — Inception vs Decision Date (2019-12-31)",
        fontsize=12, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    for bar in bars_end:
        h = bar.get_height()
        ax.annotate(
            f"{h:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, h),
            xytext=(0, 3), textcoords="offset points",
            ha="center", fontsize=8,
        )

    plt.tight_layout()
    out = FIGURES_DIR / "fig_audit_weight_drift.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── Figure 4: Correlation heatmap ─────────────────────────────────────────────
def plot_correlation_heatmap(returns: pd.DataFrame) -> None:
    corr = returns[TICKERS].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdYlGn_r", vmin=0.0, vmax=1.0, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Pearson Correlation")

    ax.set_xticks(range(len(TICKERS)))
    ax.set_yticks(range(len(TICKERS)))
    ax.set_xticklabels(TICKERS, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(TICKERS, fontsize=10)
    ax.set_title(
        "Inherited Fund: Pairwise Return Correlations (2010–2019)\n"
        "Red = highly correlated (less diversification benefit)",
        fontsize=12, fontweight="bold",
    )

    for i in range(len(TICKERS)):
        for j in range(len(TICKERS)):
            val = corr.iloc[i, j]
            text_color = "white" if val > 0.80 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=text_color)

    plt.tight_layout()
    out = FIGURES_DIR / "fig_audit_correlation_heatmap.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


# ── Console summary ────────────────────────────────────────────────────────────
def print_summary(stats: pd.DataFrame) -> None:
    display = stats.copy()
    pct_cols = ["total_return", "ann_return", "ann_vol", "max_drawdown", "return_1y", "return_3y", "weight_start", "weight_end"]
    for col in pct_cols:
        if col in display.columns:
            display[col] = display[col].map(lambda v: f"{v:.1%}" if pd.notna(v) else "N/A")
    for col in ["sharpe", "sortino", "alpha_ann", "beta", "r_squared", "corr_to_spy", "avg_pairwise_corr"]:
        if col in display.columns:
            display[col] = display[col].map(lambda v: f"{v:.2f}" if pd.notna(v) else "N/A")

    print("\n" + "=" * 130)
    print("  INHERITED FUND AUDIT — Per-Stock Statistics  (In-Sample: 2010–2019)")
    print("=" * 130)
    print(display.to_string(index=False))
    print("=" * 130)


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    print("Loading data...")
    prices, returns, weights_snapshot = load_data()
    print(f"  Price panel: {prices.index[0].date()} → {prices.index[-1].date()}  ({len(prices)} trading days)")

    print("Computing per-stock statistics...")
    stats = compute_stock_stats(prices, returns, weights_snapshot)

    out_csv = TABLES_DIR / "tbl_audit_stock_stats.csv"
    stats.to_csv(out_csv, index=False)
    print(f"  Saved: {out_csv.name}")

    spy_metrics = compute_performance_metrics(returns["SPY"].dropna(), turnover=0.0)

    print("Generating charts...")
    plot_return_grid(prices, stats)
    plot_risk_return(stats, spy_metrics["ann_return"], spy_metrics["ann_vol"])
    plot_weight_drift(stats)
    plot_correlation_heatmap(returns)

    print_summary(stats)
    print("\nDone. All outputs written to outputs/figures/ and outputs/tables/")


if __name__ == "__main__":
    main()
