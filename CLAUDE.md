# BlackFin Inc. — BFIN 491 Capstone Project

**Team:** Beau Barthuly · Bailey Binando · Kjellen Bogard  
**Showcase:** May 4, 2026 — 102 Jabs Hall

---

## Environment Setup

**Always use this venv — do NOT use the one inside the Desktop/BFIN_491 folder (it is iCloud-synced and freezes on import):**

```bash
source ~/bfin_capstone_venv/bin/activate
```

---

## Running the Full Pipeline

This regenerates all tables, figures, and outputs from scratch (downloads fresh price data):

```bash
source ~/bfin_capstone_venv/bin/activate
python run_pipeline.py
```

To skip the Yahoo Finance download and use cached price data (much faster, use this normally):

```bash
python run_pipeline.py --no-download
```

The pipeline produces everything in `outputs/tables/` and `outputs/figures/`. After running, the dashboard will automatically reflect the new outputs — no dashboard restart needed if it is already running.

---

## Regenerating Individual Figures (without re-running the full pipeline)

If you only need to update specific figures without touching the pipeline, run these standalone scripts from the project root:

**All key figures at once:**
```bash
source ~/bfin_capstone_venv/bin/activate
python - <<'EOF'
import sys; sys.path.insert(0, ".")
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from utils.portfolio_utils import plot_legacy_static_active_vs_benchmark, plot_revised_active_weights
from utils.risk_utils import plot_stress_scenarios
from utils.factor_utils import load_ff3_factors, build_ff3_summary, plot_ff3_exposures, build_portfolio_return_panel

T = Path("outputs/tables")
F = Path("outputs/figures")

# Main return chart (with NBER recession shading)
compare = pd.read_csv(T / "tbl_backtest_legacy_static_active_benchmark.csv")
plot_legacy_static_active_vs_benchmark(compare, benchmark_ticker="SPY")
plt.savefig(F / "fig_legacy_static_active_benchmark.png", dpi=150, bbox_inches="tight"); plt.close()

# Active weights chart (legend below chart)
weights = pd.read_csv(T / "tbl_revised_active_weights_daily.csv")
plot_revised_active_weights(weights)
plt.savefig(F / "fig_revised_active_weights.png", dpi=150, bbox_inches="tight"); plt.close()

# Stress chart (all 4 portfolios)
stress = pd.read_csv(T / "tbl_scenario_stress_summary.csv")
legacy = pd.read_csv(T / "tbl_legacy_fund_daily.csv")
plot_stress_scenarios(stress, legacy_daily=legacy)
plt.savefig(F / "fig_scenario_stress_impacts.png", dpi=150, bbox_inches="tight"); plt.close()

# FF3 figure
panel = build_portfolio_return_panel(
    pd.read_csv(T/"tbl_legacy_fund_daily.csv"), pd.read_csv(T/"tbl_revised_static_daily.csv"),
    pd.read_csv(T/"tbl_revised_active_daily.csv"), pd.read_csv(T/"tbl_benchmark_daily.csv"),
    pd.Timestamp("2020-01-01"), pd.Timestamp("2025-12-31"),
)
ff3 = load_ff3_factors("data/ff3_factors_daily.csv", pd.Timestamp("2020-01-01"), pd.Timestamp("2025-12-31"))
summary = build_ff3_summary(panel, ff3)
summary.to_csv(T / "tbl_factor_ff3_summary.csv", index=False)
plot_ff3_exposures(summary)
plt.savefig(F / "fig_factor_ff3.png", dpi=150, bbox_inches="tight"); plt.close()

print("All figures regenerated.")
EOF
```

---

## Launching the Dashboard

```bash
source ~/bfin_capstone_venv/bin/activate
streamlit run dashboard_app.py
```

Opens at `http://localhost:8501`. The dashboard reads live from `outputs/tables/` and `outputs/figures/` — re-running the pipeline updates it automatically without restarting.

---

## Key Files

| File | Purpose |
|------|---------|
| `run_pipeline.py` | Master pipeline — runs all layers end to end |
| `dashboard_app.py` | Streamlit dashboard (8 pages) |
| `data/ff3_factors_daily.csv` | Fama-French 3-factor daily data (downloaded from FRED, covers 1926–2026) |
| `data/fred_usrec.csv` | NBER recession indicator (from FRED USREC, used for grey shading on charts) |
| `utils/factor_utils.py` | CAPM + FF3 regression functions |
| `utils/portfolio_utils.py` | Portfolio construction, backtest, and figure generation |
| `utils/risk_utils.py` | Stress test, Monte Carlo, and stress figure generation |
| `outputs/tables/` | All pipeline-generated CSV tables |
| `outputs/figures/` | All pipeline-generated PNG figures |
| `notebooks/capstone_report.ipynb` | Report notebook (displays figures and tables inline) |
| `.streamlit/config.toml` | Dark mode config — do not delete |

---

## Project Summary

**Decision date:** January 1, 2020  
**OOS evaluation:** 2020–2025 (6 years)  
**Benchmark:** SPY

**Revised 10-stock portfolio** (replaced the inherited fund):

| Ticker | Company | Weight | Decision |
|--------|---------|--------|---------|
| AAPL | Apple Inc. | 15% | Kept |
| AMZN | Amazon.com | 15% | Kept |
| CMCSA | Comcast | 10% | Kept |
| GOOG | Alphabet | 10% | Kept |
| MSFT | Microsoft | 10% | Kept |
| V | Visa | 8% | Added |
| WMT | Walmart | 8% | Added |
| CVX | Chevron | 8% | Added |
| NEE | NextEra Energy | 8% | Added |
| UNH | UnitedHealth | 8% | Added |

**OOS results:**

| Portfolio | Ann. Return | Sharpe | Sortino | Max Drawdown |
|-----------|-------------|--------|---------|--------------|
| Legacy Fund | 18.5% | 0.759 | 1.002 | -31.4% |
| Revised Active | 16.9% | **0.793** | **1.004** | -27.8% |
| Revised Static | 16.8% | 0.769 | 0.982 | -27.4% |
| SPY | 14.9% | 0.716 | 0.880 | -33.7% |

**Active rebalancing rule:** Sharpe-tilt (monthly). Trailing 36-month Sharpe signal scales target weights toward higher risk-adjusted momentum names. This is labeled `sharpe_tilt` in the rebalance log — NOT a quadratic max-Sharpe optimizer.

---

## Important Notes for Teammates

- **Do not change the venv.** The Desktop venv is iCloud-synced and will hang. Always use `~/bfin_capstone_venv`.
- **Do not delete `data/ff3_factors_daily.csv` or `data/fred_usrec.csv`.** These are downloaded reference datasets used by the pipeline and figure scripts. Re-downloading requires internet access.
- **The dashboard uses dark mode** set in `.streamlit/config.toml`. Do not remove or modify that file.
- **The active rule is Sharpe-tilt**, not equal-weight or max-Sharpe optimizer. The control workbook says `optimizer / max_sharpe` but the actual implementation is a proportional tilt. Use "Sharpe-tilt" in all presentation and report language.
- **FF3 alpha is NOT statistically significant** (t-stats ~0.5–0.8). This is expected and normal for a 6-year sample. The point estimates are positive for all three portfolios — present them as directional evidence, not proof of skill.
