"""
models_survival.py
==================
Survival analysis extension for:
  "Predicting Short-Horizon Liquidation Risk in Aave
   Using Explainable Machine Learning"
  Runming Huang & Leran Li — DOTE 6635, Spring 2026

Methods
-------
  PART 1  — Build per-account survival dataset
              (duration, event) from the panel dataset
  PART 2  — Kaplan-Meier curves (overall + stratified by risk quartile)
  PART 3  — Log-rank tests between risk groups
  PART 4  — Cox Proportional Hazards model
  PART 5  — Schoenfeld residual test (proportional-hazard assumption)
  PART 6  — Export survival metrics

Connection to proposal
----------------------
  "As a robustness extension, we may … estimate a simple survival-style
   specification to connect the classification exercise to the broader
   literature on time-to-default modeling."
  — Runming Huang & Leran Li, Final Project Proposal

Usage
-----
  Run model.py first (produces results/metrics_summary.csv and risk scores).
  Then:  python models_survival.py
"""

import os
import warnings
import numpy as np
import pandas as pd
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")

DATA_DIR    = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_STATE = 42
RISK_SCORE_FILE_7D  = os.path.join(RESULTS_DIR, "risk_scores_7d.csv")
RISK_SCORE_FILE_14D = os.path.join(RESULTS_DIR, "risk_scores_14d.csv")

print("=" * 65)
print("  Survival Analysis — Kaplan-Meier + Cox PHM")
print("=" * 65)


# ============================================================
# PART 1 — Build per-account survival dataset
# ============================================================
# The panel dataset has multiple rows per account (one per active day).
# For survival analysis we need one row per account:
#
#   duration = days from account's first observation to first
#              liquidation event (if any) or last observation (censored)
#   event    = 1 if the account was ever liquidated in the study window
#
# We also attach baseline features from the account's first observation
# so the Cox model has covariates to work with.

print("\n[PART 1] Building per-account survival dataset...")

dataset_path = os.path.join(DATA_DIR, "dataset.csv")
if not os.path.exists(dataset_path):
    print("  ERROR: dataset.csv not found. Run fetch_data.py then model.py first.")
    raise SystemExit(1)

panel = pd.read_csv(dataset_path, parse_dates=["obs_date"])
print(f"  Loaded panel: {len(panel):,} rows, {panel['account'].nunique():,} unique accounts")

# ── Liquidation dates from raw borrow events ──────────────────
liq_events = pd.DataFrame(columns=["account", "liq_date"])
raw_borrow_path = os.path.join(DATA_DIR, "raw_borrow_events.csv")
if os.path.exists(raw_borrow_path):
    raw_b = pd.read_csv(raw_borrow_path)
    raw_b["block_time"] = pd.to_datetime(raw_b["block_time"], utc=True).dt.tz_localize(None)
    raw_b["liq_date"]   = raw_b["block_time"].dt.floor("D")
    raw_b["account"]    = raw_b["account"].str.lower()
    liq_events = (
        raw_b[raw_b["transaction_type"] == "borrow_liquidation"]
        .groupby("account")["liq_date"]
        .min()
        .reset_index()
    )
    print(f"  Liquidation events from raw data: {len(liq_events):,} accounts")

# ── Entry / exit dates per account ───────────────────────────
entry = panel.groupby("account")["obs_date"].min().rename("entry_date")
exit_ = panel.groupby("account")["obs_date"].max().rename("exit_date")
surv  = pd.concat([entry, exit_], axis=1).reset_index()

surv = surv.merge(liq_events, on="account", how="left")
surv["event"]    = surv["liq_date"].notna().astype(int)
surv["liq_date"] = surv["liq_date"].fillna(surv["exit_date"])

# Duration: days from entry to event/censoring (min 1)
surv["duration"] = (surv["liq_date"] - surv["entry_date"]).dt.days.clip(lower=1)

# ── Baseline features: first observation per account ─────────
BASELINE_FEATURES = [
    "approx_health_factor", "distance_to_liq", "debt_to_collateral",
    "collateral_value_usd", "total_debt_usd",
    "stablecoin_debt_share", "n_collateral_assets",
    "realized_vol_7d", "eth_return_7d",
    "repay_borrow_ratio", "activity_intensity", "account_age_days",
]

# Derive distance_to_liq and derived ratios if missing
panel["distance_to_liq"]    = np.clip(panel["approx_health_factor"] - 1.0, 0, 10)
panel["debt_to_collateral"]  = panel["total_debt_usd"] / (panel["collateral_value_usd"] + 1e-9)
panel["repay_borrow_ratio"]  = panel["n_repays_30d"]   / (panel["n_borrows_30d"]   + 1)
panel["activity_intensity"]  = panel["total_tx_30d"]   / (panel["account_age_days"] + 1)

first_obs = (
    panel.sort_values("obs_date")
    .groupby("account")
    .first()
    .reset_index()
)
avail_features = [f for f in BASELINE_FEATURES if f in first_obs.columns]
surv = surv.merge(first_obs[["account"] + avail_features], on="account", how="left")
surv = surv.dropna(subset=["duration"] + avail_features).reset_index(drop=True)

print(f"  Survival dataset: {len(surv):,} accounts  |  events: {surv['event'].sum()} ({surv['event'].mean()*100:.1f}%)")
print(f"  Median duration (all): {surv['duration'].median():.0f} days")
print(f"  Median duration (event=1): {surv[surv['event']==1]['duration'].median():.0f} days")

# ── Attach XGBoost risk score for stratification ──────────────
risk_scores = {}
for h, fpath in [(7, RISK_SCORE_FILE_7D), (14, RISK_SCORE_FILE_14D)]:
    if os.path.exists(fpath):
        rs = pd.read_csv(fpath, parse_dates=["obs_date"])
        rs["account"] = rs["account"].str.lower()
        # Use the highest risk score observed per account
        rs_agg = rs.groupby("account")[f"risk_score_{h}d"].max().reset_index()
        surv = surv.merge(rs_agg, on="account", how="left")
        risk_scores[h] = f"risk_score_{h}d"
        print(f"  Attached XGBoost risk scores ({h}d) to {rs_agg['account'].isin(surv['account']).sum()} accounts")


# ============================================================
# PART 2 — Kaplan-Meier Curves
# ============================================================
# We plot KM curves for the overall cohort, then stratify by
# XGBoost risk quartile to show that model-predicted risk aligns
# with observed time-to-liquidation.

print("\n[PART 2] Kaplan-Meier curves...")

PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"]

kmf_overall = KaplanMeierFitter()
kmf_overall.fit(surv["duration"], event_observed=surv["event"], label="All accounts")

for horizon, score_col in risk_scores.items():
    surv_h = surv.dropna(subset=[score_col]).copy()
    if len(surv_h) < 10:
        continue

    surv_h["risk_quartile"] = pd.qcut(
        surv_h[score_col], q=4,
        labels=["Q1 (low risk)", "Q2", "Q3", "Q4 (high risk)"],
        duplicates="drop",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: overall KM
    ax_all = axes[0]
    kmf_overall.plot_survival_function(ax=ax_all, color="steelblue", lw=2)
    ax_all.set_title("KM Survival Curve — All Accounts")
    ax_all.set_xlabel("Days since first observation")
    ax_all.set_ylabel("Survival probability (not yet liquidated)")
    ax_all.set_ylim(0.9, 1.01)
    ax_all.grid(alpha=0.3)

    # Right: stratified by risk quartile
    ax_strat = axes[1]
    for i, grp in enumerate(["Q1 (low risk)", "Q2", "Q3", "Q4 (high risk)"]):
        mask = surv_h["risk_quartile"] == grp
        if mask.sum() < 5:
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(surv_h.loc[mask, "duration"],
                event_observed=surv_h.loc[mask, "event"],
                label=grp)
        kmf.plot_survival_function(ax=ax_strat, color=PALETTE[i], lw=2)
    ax_strat.set_title(f"KM by XGBoost Risk Quartile ({horizon}-day score)")
    ax_strat.set_xlabel("Days since first observation")
    ax_strat.set_ylabel("Survival probability")
    ax_strat.set_ylim(0.9, 1.01)
    ax_strat.grid(alpha=0.3)

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"km_curves_{horizon}d.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved KM curves ({horizon}d) → {path}")


# ============================================================
# PART 3 — Log-rank Tests
# ============================================================
# Test whether survival curves differ significantly across
# risk quartiles.  A low p-value confirms the XGBoost score
# separates accounts with meaningfully different survival.

print("\n[PART 3] Log-rank tests...")

for horizon, score_col in risk_scores.items():
    surv_h = surv.dropna(subset=[score_col]).copy()
    if len(surv_h) < 10 or surv_h["event"].sum() < 4:
        print(f"  [{horizon}d] Not enough events for log-rank test — skipping")
        continue

    surv_h["risk_quartile"] = pd.qcut(
        surv_h[score_col], q=4,
        labels=[0, 1, 2, 3], duplicates="drop",
    )

    # Q1 vs Q4 pairwise
    q1 = surv_h[surv_h["risk_quartile"] == 0]
    q4 = surv_h[surv_h["risk_quartile"] == 3]
    if len(q1) >= 3 and len(q4) >= 3 and (q1["event"].sum() + q4["event"].sum()) >= 2:
        lr = logrank_test(
            q1["duration"], q4["duration"],
            event_observed_A=q1["event"], event_observed_B=q4["event"],
        )
        print(f"  [{horizon}d] Log-rank Q1 vs Q4: χ²={lr.test_statistic:.3f}  p={lr.p_value:.4f}")

    # Multivariate across all quartiles
    valid = surv_h.dropna(subset=["risk_quartile"])
    if valid["event"].sum() >= 4:
        mlr = multivariate_logrank_test(
            valid["duration"], valid["risk_quartile"], valid["event"]
        )
        print(f"  [{horizon}d] Multivariate log-rank (all quartiles): p={mlr.p_value:.4f}")


# ============================================================
# PART 4 — Cox Proportional Hazards Model
# ============================================================
# The Cox PHM estimates the hazard ratio associated with each
# feature, allowing us to answer: "holding other features constant,
# how does a one-unit change in health factor affect the rate of
# liquidation?"  This complements the XGBoost classifier by
# providing interpretable, statistically-tested effect sizes.

print("\n[PART 4] Cox Proportional Hazards model...")

# Variables that violated PH assumption (from Schoenfeld test in prior run):
#   realized_vol_7d, repay_borrow_ratio, stablecoin_debt_share, eth_return_7d
# Strategy: stratify the two worst offenders (realized_vol_7d, repay_borrow_ratio)
# so the baseline hazard is estimated separately per stratum, then include
# stablecoin_debt_share and eth_return_7d as regular covariates.
PH_STRATA = ["realized_vol_7d", "repay_borrow_ratio"]
COX_FEATURES = [
    "approx_health_factor", "distance_to_liq", "debt_to_collateral",
    "stablecoin_debt_share", "eth_return_7d", "activity_intensity",
]
cox_features_avail = [f for f in COX_FEATURES if f in surv.columns]
strata_avail = [f for f in PH_STRATA if f in surv.columns]

cox_df = surv[["duration", "event"] + cox_features_avail + strata_avail].dropna().copy()

# Bin strata variables into quartiles for stratification
for col in strata_avail:
    cox_df[f"{col}_stratum"] = pd.qcut(cox_df[col], q=4, labels=False, duplicates="drop")
strata_cols = [f"{col}_stratum" for col in strata_avail]

# Standardise continuous covariates so hazard ratios are comparable
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
cox_df[cox_features_avail] = scaler.fit_transform(cox_df[cox_features_avail])

cox_results = {}
if len(cox_df) >= 50 and cox_df["event"].sum() >= 5:
    # Stratified Cox: baseline hazard varies by vol × repay strata,
    # removing time-varying PH violations for those two variables.
    cph = CoxPHFitter(penalizer=0.1)
    try:
        fit_kwargs = {}
        if strata_cols:
            fit_kwargs["strata"] = strata_cols
            print(f"  Using stratified Cox (strata: {strata_avail})")
        cph.fit(cox_df, duration_col="duration", event_col="event", **fit_kwargs)
        print("\n  Cox PHM Summary (standardised covariates):")
        cph.print_summary(decimals=4)
        cox_results["model"] = cph

        # Save coefficient plot
        fig, ax = plt.subplots(figsize=(8, 5))
        cph.plot(ax=ax)
        ax.set_title("Stratified Cox PHM — Hazard Ratios (95% CI)")
        ax.axvline(0, color="black", lw=0.8, ls="--")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, "cox_hazard_ratios.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Saved Cox hazard-ratio plot → {path}")

        # C-index (survival AUC equivalent)
        c_idx = concordance_index(cox_df["duration"], -cph.predict_partial_hazard(cox_df), cox_df["event"])
        print(f"  Cox C-index (concordance): {c_idx:.4f}")
        cox_results["c_index"] = c_idx

    except Exception as e:
        print(f"  Cox PHM fit failed: {e}")
else:
    print(f"  Not enough data for Cox PHM (n={len(cox_df)}, events={cox_df['event'].sum()}) — skipping")


# ============================================================
# PART 5 — Proportional Hazard Assumption Check
# ============================================================
# The Schoenfeld residual test checks whether the hazard ratio
# is constant over time (PH assumption).  A significant p-value
# signals time-varying effects that would require an extended Cox
# or a stratified model.

print("\n[PART 5] Schoenfeld residual test (PH assumption)...")

if "model" in cox_results:
    cph = cox_results["model"]
    try:
        ph_test = cph.check_assumptions(cox_df, p_value_threshold=0.05, show_plots=False)
    except Exception as e:
        print(f"  PH test skipped: {e}")


# ============================================================
# PART 6 — Export survival metrics
# ============================================================

print("\n[PART 6] Exporting survival metrics...")

surv_export = surv[["account", "entry_date", "liq_date", "duration", "event"]].copy()
path = os.path.join(RESULTS_DIR, "survival_dataset.csv")
surv_export.to_csv(path, index=False)
print(f"  Survival dataset → {path}  ({len(surv_export):,} rows)")

summary_rows = []
if "model" in cox_results:
    cph = cox_results["model"]
    coef_df = cph.params_.reset_index()
    coef_df.columns = ["feature", "log_hazard_ratio"]
    coef_df["hazard_ratio"] = np.exp(coef_df["log_hazard_ratio"])
    hr_path = os.path.join(RESULTS_DIR, "cox_hazard_ratios.csv")
    coef_df.to_csv(hr_path, index=False)
    print(f"  Cox coefficients → {hr_path}")
    summary_rows.append({"metric": "Cox C-index", "value": round(cox_results.get("c_index", np.nan), 4)})

summary_rows.append({"metric": "N accounts", "value": len(surv)})
summary_rows.append({"metric": "N events (liquidated)", "value": int(surv["event"].sum())})
summary_rows.append({"metric": "Event rate (%)", "value": round(surv["event"].mean() * 100, 2)})
summary_rows.append({"metric": "Median duration (days)", "value": int(surv["duration"].median())})

pd.DataFrame(summary_rows).to_csv(
    os.path.join(RESULTS_DIR, "survival_metrics_summary.csv"), index=False
)
print(f"  Survival summary → results/survival_metrics_summary.csv")

print("\n" + "=" * 65)
print("  models_survival.py complete.")
print("  Outputs → ./results/")
print("=" * 65)
