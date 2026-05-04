"""
model.py
========
Full ML pipeline for:
  "Predicting Short-Horizon Liquidation Risk in Aave
   Using Explainable Machine Learning"
  Runming Huang & Leran Li — DOTE 6635, Spring 2026

Run after fetch_data.py has populated ./data/.
If no real data is present, a synthetic dataset is auto-generated
so you can verify the pipeline end-to-end immediately.

Usage
-----
  pip install pandas numpy scikit-learn xgboost shap matplotlib imbalanced-learn
  python model.py

Sections
--------
  PART 0  — Imports & Global Config
  PART 1  — Data Loading & Synthetic Fallback
  PART 2  — Feature Engineering
  PART 3  — Train / Validation / Test Split (time-aware)
  PART 4  — Class-Imbalance Handling  (SMOTE + class weights)
  PART 5  — Baseline Model  (Logistic Regression)
  PART 6  — Main Model      (XGBoost)
  PART 7  — Model Evaluation (ROC-AUC, PR-AUC, calibration, recall@precision)
  PART 8  — SHAP Explainability (global + per-account)
  PART 9  — Regime-Based Analysis (low-vol vs high-vol periods)
  PART 10 — Results Summary & Export
"""

# ============================================================
# PART 0 — Imports & Global Config
# ============================================================
import os
import warnings
import numpy as np
import pandas as pd
os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib"))
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import matplotlib
matplotlib.use("Agg")          # headless — saves PNG files instead of showing windows
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from datetime import datetime, timedelta

# Scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    brier_score_loss,
    classification_report, confusion_matrix,
)
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay

# XGBoost
import xgboost as xgb

# SHAP
import shap

# Imbalanced-learn
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

# ── TEST MODE ───────────────────────────────────────────────
# Set TEST_MODE=1 to run on a tiny subset for quick validation.
# Usage:  TEST_MODE=1 python model.py
TEST_MODE = os.getenv("TEST_MODE", "0") == "1"

# ── Global config ──────────────────────────────────────────
DATA_DIR    = "data"
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

PREDICTION_HORIZONS = [7, 14]
RANDOM_STATE        = 42
CV_FOLDS            = 2 if TEST_MODE else 5        # fewer folds in test mode
TARGET_PRECISION    = 0.10   # lowered from 0.30 — more realistic for 1% base rate
VOLATILITY_PERCENTILE = 75

# XGBoost grid: tiny in test mode, full in production
PARAM_GRID = {
    "n_estimators":     [50]          if TEST_MODE else [200, 400],
    "max_depth":        [3]           if TEST_MODE else [4, 6],
    "learning_rate":    [0.10]        if TEST_MODE else [0.05, 0.10],
    "subsample":        [0.8],
    "colsample_bytree": [0.8],
    "min_child_weight": [5],
}

print("=" * 65)
print("  Aave Liquidation Risk — Full ML Pipeline")
if TEST_MODE:
    print("  ⚡ TEST MODE — small data, fast settings")
print("=" * 65)


# ============================================================
# PART 1 — Data Loading & Synthetic Fallback
# ============================================================
# Goal: load dataset.csv produced by fetch_data.py.
# If the file is missing we generate a synthetic dataset so the
# entire pipeline can be tested immediately without real data.
# The synthetic dataset mimics the distribution described in the
# proposal (heavy class imbalance, health-factor-driven risk, etc.)

def make_synthetic_dataset(n: int = 8_000, seed: int = RANDOM_STATE) -> pd.DataFrame:
    """
    Generate a synthetic account-date dataset that mirrors the
    feature schema described in the proposal.
    Used only when ./data/dataset.csv is absent.
    """
    rng = np.random.default_rng(seed)

    base_date = pd.Timestamp("2022-06-01")
    dates      = [base_date + timedelta(days=int(d)) for d in rng.integers(0, 550, n)]
    accounts   = [f"0x{rng.integers(0, 2000):04x}" for _ in range(n)]

    # Position features
    collateral = rng.lognormal(mean=10, sigma=1.5, size=n)   # USD
    debt       = collateral / rng.uniform(1.1, 4.0, size=n)
    hf         = collateral / (debt + 1e-9)

    # Portfolio composition
    n_collateral  = rng.integers(1, 5, size=n)
    stbl_share    = rng.beta(1, 4, size=n)

    # Behavioral (30-day tx counts)
    n_dep  = rng.integers(0, 20, size=n)
    n_bor  = rng.integers(0, 15, size=n)
    n_rep  = rng.integers(0, 15, size=n)
    n_wd   = rng.integers(0, 10, size=n)
    acct_age = rng.integers(1, 700, size=n)

    # Market state
    eth_price  = rng.lognormal(mean=7.5, sigma=0.4, size=n)
    eth_ret7   = rng.normal(0, 0.10, size=n)
    vol7       = np.abs(rng.normal(0.6, 0.3, size=n))

    # Label: logistic function of health factor + volatility + noise
    log_odds   = -4 + (-3.5 * np.log(np.clip(hf, 0.5, 10)))  \
                     + (2.0 * vol7)                             \
                     + (0.8 * np.abs(eth_ret7))                 \
                     + rng.normal(0, 0.5, size=n)
    prob       = 1 / (1 + np.exp(-log_odds))
    label_7d   = (rng.uniform(size=n) < prob).astype(int)
    label_14d  = np.where(
        label_7d == 1, 1,
        (rng.uniform(size=n) < prob * 1.4).astype(int)
    )

    df = pd.DataFrame({
        "account":               accounts,
        "obs_date":              dates,
        # Position
        "collateral_value_usd":  collateral,
        "total_debt_usd":        debt,
        "collateralization_ratio": hf,
        "approx_health_factor":  hf,
        "n_collateral_assets":   n_collateral,
        "n_debt_assets":         rng.integers(1, 4, size=n),
        # Portfolio composition
        "stablecoin_debt_share": stbl_share,
        # Behavioral
        "n_deposits_30d":        n_dep,
        "n_borrows_30d":         n_bor,
        "n_repays_30d":          n_rep,
        "n_withdraws_30d":       n_wd,
        "total_tx_30d":          n_dep + n_bor + n_rep + n_wd,
        "account_age_days":      acct_age,
        # Market
        "eth_price_usd":         eth_price,
        "eth_return_7d":         eth_ret7,
        "realized_vol_7d":       vol7,
        # Labels
        "label_7d":              label_7d,
        "label_14d":             label_14d,
    })
    return df


print("\n[PART 1] Loading data...")
dataset_path = os.path.join(DATA_DIR, "dataset.csv")

if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path, parse_dates=["obs_date"])
    print(f"  Loaded real data: {len(df):,} rows from {dataset_path}")
else:
    print("  dataset.csv not found — generating synthetic data for pipeline testing.")
    df = make_synthetic_dataset()
    print(f"  Synthetic dataset: {len(df):,} rows")

# ── TEST MODE: subsample ────────────────────────────────────
if TEST_MODE:
    df = df.sample(n=min(2000, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  ⚡ TEST MODE: subsampled to {len(df):,} rows")

print(f"  Columns: {list(df.columns)}")
print(f"  Date range: {df['obs_date'].min().date()} → {df['obs_date'].max().date()}")
for h in PREDICTION_HORIZONS:
    col = f"label_{h}d"
    if col in df.columns:
        pos = df[col].sum()
        print(f"  label_{h}d  positives: {pos:,} / {len(df):,}  ({pos/len(df)*100:.1f}%)")


# ============================================================
# PART 2 — Feature Engineering
# ============================================================
# We derive additional features that are not already in dataset.csv
# to enrich the feature set described in the proposal.
#
# New features created:
#   debt_to_collateral  — inverse of collateralization ratio (more intuitive)
#   distance_to_liq     — how far HF is above the critical threshold of 1.0
#   repay_borrow_ratio  — proxy for account deleveraging behavior
#   activity_intensity  — total tx / account age (activity rate)
#   high_vol_flag       — binary indicator for market stress regime
#   eth_return_sign     — direction of recent ETH price move

print("\n[PART 2] Feature engineering...")

df = df.copy()
df["obs_date"] = pd.to_datetime(df["obs_date"])

# Position-derived
df["debt_to_collateral"]  = df["total_debt_usd"] / (df["collateral_value_usd"] + 1e-9)
df["distance_to_liq"]     = np.clip(df["approx_health_factor"] - 1.0, 0, 10)
df["log_collateral"]      = np.log1p(df["collateral_value_usd"])
df["log_debt"]            = np.log1p(df["total_debt_usd"])

# Behavioral-derived
df["repay_borrow_ratio"]  = df["n_repays_30d"]  / (df["n_borrows_30d"] + 1)
df["withdraw_dep_ratio"]  = df["n_withdraws_30d"] / (df["n_deposits_30d"] + 1)
df["activity_intensity"]  = df["total_tx_30d"]   / (df.get("account_age_days", pd.Series(np.ones(len(df)) * 365)) + 1)

# Market-derived
df["eth_return_sign"]     = np.sign(df["eth_return_7d"])
df["vol_times_debt"]      = df["realized_vol_7d"] * df["debt_to_collateral"]  # interaction term

# Regime flag (used in Part 9)
vol_thresh = df["realized_vol_7d"].quantile(VOLATILITY_PERCENTILE / 100)
df["high_vol_regime"]     = (df["realized_vol_7d"] > vol_thresh).astype(int)

print(f"  High-vol regime threshold (p{VOLATILITY_PERCENTILE}): {vol_thresh:.3f}")
print(f"  High-vol observations: {df['high_vol_regime'].sum():,}")

# ── Define feature sets matching proposal categories ────────
FEATURES = {
    "position": [
        "collateral_value_usd", "total_debt_usd",
        # collateralization_ratio removed: it is col/debt, near-identical to
        # approx_health_factor, which we keep as the canonical HF proxy.
        "approx_health_factor", "debt_to_collateral", "distance_to_liq",
        "log_collateral", "log_debt",
    ],
    "portfolio": [
        "n_collateral_assets", "n_debt_assets", "stablecoin_debt_share",
    ],
    "behavioral": [
        "n_deposits_30d", "n_borrows_30d", "n_repays_30d", "n_withdraws_30d",
        "total_tx_30d", "repay_borrow_ratio", "withdraw_dep_ratio",
        "activity_intensity",
    ],
    "market": [
        "eth_price_usd", "eth_return_7d", "realized_vol_7d",
        "eth_return_sign", "vol_times_debt",
    ],
}

ALL_FEATURES = [f for grp in FEATURES.values() for f in grp]
# Keep only columns that exist in the DataFrame
ALL_FEATURES = [f for f in ALL_FEATURES if f in df.columns]
print(f"  Total features used: {len(ALL_FEATURES)}")

# VIF check — flag features with VIF > 10 (severe multicollinearity)
try:
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    _vif_df = df[ALL_FEATURES].dropna()
    if len(_vif_df) > len(ALL_FEATURES):
        _X_vif = _vif_df.values.astype(float)
        _vifs   = [variance_inflation_factor(_X_vif, i) for i in range(_X_vif.shape[1])]
        _high   = [(ALL_FEATURES[i], round(v, 1)) for i, v in enumerate(_vifs) if v > 10]
        if _high:
            print(f"  ⚠ High VIF features (>10): {_high}")
        else:
            print("  VIF check passed — no severe multicollinearity detected")
except ImportError:
    pass   # statsmodels optional

# Drop rows with NaN in any feature or label column
label_cols = [f"label_{h}d" for h in PREDICTION_HORIZONS if f"label_{h}d" in df.columns]
df = df.dropna(subset=ALL_FEATURES + label_cols).reset_index(drop=True)
print(f"  Rows after dropping NaN: {len(df):,}")


# ============================================================
# PART 3 — Train / Validation / Test Split  (time-aware)
# ============================================================
# We sort by date and use the last 20% of observations as the
# held-out test set. This prevents look-ahead bias — the model
# is never trained on data from the future relative to its test
# observations.
#
# Within the training portion we use 5-fold cross-validation
# for hyperparameter evaluation.

print("\n[PART 3] Time-aware train/test split...")

df = df.sort_values("obs_date").reset_index(drop=True)
n_test   = int(len(df) * 0.20)
train_df = df.iloc[:-n_test].copy()
test_df  = df.iloc[-n_test:].copy()

# TEST MODE safety: if test set has no positives for any label,
# force at least 2 positives into the test set from the train set.
if TEST_MODE:
    for lc in label_cols:
        if test_df[lc].sum() == 0:
            pos_in_train = train_df[train_df[lc] == 1]
            if len(pos_in_train) >= 2:
                move = pos_in_train.sample(min(2, len(pos_in_train)),
                                           random_state=RANDOM_STATE)
                train_df = train_df.drop(move.index)
                test_df  = pd.concat([test_df, move], ignore_index=True)
                print(f"  ⚡ TEST MODE: moved {len(move)} positives ({lc}) into test set")

print(f"  Train: {len(train_df):,} rows  ({train_df['obs_date'].min().date()} → {train_df['obs_date'].max().date()})")
print(f"  Test : {len(test_df):,}  rows  ({test_df['obs_date'].min().date()} → {test_df['obs_date'].max().date()})")
for lc in label_cols:
    print(f"  {lc} in test: {test_df[lc].sum()} positives")

X_train = train_df[ALL_FEATURES].values
X_test  = test_df[ALL_FEATURES].values


# ============================================================
# PART 4 — Class-Imbalance Handling
# ============================================================
# Liquidation is a rare event (proposal explicitly notes this).
# We address imbalance two ways:
#   (a) SMOTE — synthetic oversampling of the minority class
#       applied only to training data (never test data)
#   (b) class_weight / scale_pos_weight parameters in the models
#       so the loss function down-weights the majority class
#
# Both approaches are used together for best results.

print("\n[PART 4] Class-imbalance strategy: SMOTE + class weights")

def get_class_weight(y: np.ndarray) -> float:
    """Return scale_pos_weight = n_neg / n_pos (used by XGBoost)."""
    n_pos = y.sum()
    n_neg = len(y) - n_pos
    w = n_neg / (n_pos + 1e-9)
    print(f"    scale_pos_weight = {w:.1f}  (neg/pos = {n_neg}/{int(n_pos)})")
    return w


# ============================================================
# PART 5 — Baseline Model: Logistic Regression
# ============================================================
# Logistic Regression is the interpretable baseline described
# in the proposal. We wrap it in a pipeline:
#   StandardScaler → SMOTE → LogisticRegression
#
# This baseline is important because it tells us the "floor"
# of prediction quality achievable with a linear, coefficient-
# interpretable model — the kind used in traditional credit risk.

print("\n[PART 5] Training Logistic Regression baselines...")

lr_results = {}   # store evaluation metrics per horizon

for horizon in PREDICTION_HORIZONS:
    label_col = f"label_{horizon}d"
    if label_col not in df.columns:
        continue

    y_train = train_df[label_col].values
    y_test  = test_df[label_col].values

    print(f"\n  ── Horizon: {horizon}d ──")

    n_pos = int(y_train.sum())
    k_nn  = max(1, min(5, n_pos - 1))   # k_neighbors must be >= 1 and < n_positives
    smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_nn)

    lr_pipe = ImbPipeline([
        ("scaler", StandardScaler()),
        ("smote",  smote),
        ("model",  LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            C=0.1,
            random_state=RANDOM_STATE,
        )),
    ])

    # Fit on first 80% of train, calibrate on last 20% (time-aware split)
    n_cal     = max(50, int(len(X_train) * 0.20))
    X_base, X_cal = X_train[:-n_cal], X_train[-n_cal:]
    y_base, y_cal = y_train[:-n_cal], y_train[-n_cal:]

    lr_pipe.fit(X_base, y_base)

    # Platt scaling: fit a sigmoid layer on the held-out calibration set
    lr_calibrated = CalibratedClassifierCV(lr_pipe, method="sigmoid", cv="prefit")
    lr_calibrated.fit(X_cal, y_cal)

    prob_lr = lr_calibrated.predict_proba(X_test)[:, 1]

    if len(np.unique(y_test)) < 2:
        print(f"    ⚠ Only one class in test set — skipping metrics for {horizon}d")
        continue

    auc_roc = roc_auc_score(y_test, prob_lr)
    auc_pr  = average_precision_score(y_test, prob_lr)
    brier   = brier_score_loss(y_test, prob_lr)

    lr_results[horizon] = {
        "model": lr_calibrated,
        "prob_test": prob_lr,
        "y_test": y_test,
        "roc_auc": auc_roc,
        "pr_auc":  auc_pr,
        "brier":   brier,
    }

    print(f"    ROC-AUC : {auc_roc:.4f}")
    print(f"    PR-AUC  : {auc_pr:.4f}")
    print(f"    Brier   : {brier:.4f}")


# ============================================================
# PART 6 — Main Model: XGBoost
# ============================================================
# XGBoost is the primary predictive model chosen in the proposal
# because it handles:
#   • Non-linear feature interactions (e.g., HF × volatility)
#   • Tabular data with mixed scales
#   • Class imbalance via scale_pos_weight
#
# We tune a small grid of key hyperparameters using StratifiedKFold
# cross-validation on the training set to avoid overfitting.

print("\n[PART 6] Training XGBoost models...")

from itertools import product as iterproduct

def simple_cv_xgb(X, y, param_grid: dict, cv: int = CV_FOLDS) -> dict:
    """
    Grid search using TimeSeriesSplit to respect chronological order.
    Avoids look-ahead leakage that inflates StratifiedKFold AUC on
    time-series data (training fold must always precede validation fold).
    """
    tscv  = TimeSeriesSplit(n_splits=cv)
    keys   = list(param_grid.keys())
    combos = list(iterproduct(*param_grid.values()))

    best_score, best_params = -1, {}
    for combo in combos:
        params = dict(zip(keys, combo))
        scores = []
        for tr_idx, val_idx in tscv.split(X):
            Xtr, Xval = X[tr_idx], X[val_idx]
            ytr, yval = y[tr_idx], y[val_idx]
            if ytr.sum() < 2 or yval.sum() < 1:
                continue   # skip fold if too few positives
            spw = (len(ytr) - ytr.sum()) / (ytr.sum() + 1e-9)
            clf = xgb.XGBClassifier(
                **params,
                scale_pos_weight=spw,
                eval_metric="auc",
                random_state=RANDOM_STATE,
                verbosity=0,
            )
            clf.fit(Xtr, ytr, eval_set=[(Xval, yval)], verbose=False)
            prob = clf.predict_proba(Xval)[:, 1]
            if len(np.unique(yval)) > 1:
                scores.append(roc_auc_score(yval, prob))
        if not scores:
            continue
        mean_score = np.mean(scores)
        if mean_score > best_score:
            best_score  = mean_score
            best_params = params
    print(f"    Best CV ROC-AUC (TimeSeriesSplit): {best_score:.4f}  params: {best_params}")
    return best_params


# PARAM_GRID defined at top of file (TEST_MODE-aware)

xgb_results = {}

for horizon in PREDICTION_HORIZONS:
    label_col = f"label_{horizon}d"
    if label_col not in df.columns:
        continue

    y_train = train_df[label_col].values
    y_test  = test_df[label_col].values
    spw     = get_class_weight(y_train)

    print(f"\n  ── Horizon: {horizon}d ──")
    print("    Running cross-validated grid search...")

    best_params = simple_cv_xgb(X_train, y_train, PARAM_GRID)

    # Train final model on all training data with best params
    xgb_model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=spw,
        eval_metric="auc",
        random_state=RANDOM_STATE,
        verbosity=0,
    )
    xgb_model.fit(X_train, y_train)

    prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    if len(np.unique(y_test)) < 2:
        print(f"    ⚠ Only one class in test set — skipping metrics for {horizon}d")
        continue

    auc_roc = roc_auc_score(y_test, prob_xgb)
    auc_pr  = average_precision_score(y_test, prob_xgb)
    brier   = brier_score_loss(y_test, prob_xgb)

    xgb_results[horizon] = {
        "model": xgb_model,
        "prob_test": prob_xgb,
        "y_test": y_test,
        "roc_auc": auc_roc,
        "pr_auc":  auc_pr,
        "brier":   brier,
        "best_params": best_params,
    }

    print(f"    ROC-AUC : {auc_roc:.4f}")
    print(f"    PR-AUC  : {auc_pr:.4f}")
    print(f"    Brier   : {brier:.4f}")


# ============================================================
# PART 7 — Model Evaluation
# ============================================================
# We report four evaluation dimensions described in the proposal:
#
#   1. ROC-AUC  — overall discrimination ability
#   2. PR-AUC   — precision-recall, better for imbalanced data
#   3. Recall @ fixed precision — operationally most useful:
#      "of all accounts truly about to be liquidated, how many
#       do we catch if we tolerate 30% false-alarm rate?"
#   4. Brier Score — calibration / probabilistic accuracy
#
# We also plot ROC curves and PR curves side by side for both
# models and both horizons.

print("\n[PART 7] Evaluating and plotting...")

def recall_at_precision(y_true, y_prob, target_precision=TARGET_PRECISION):
    """Return recall at the threshold where precision ≈ target_precision."""
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)
    idx = np.where(prec >= target_precision)[0]
    if len(idx) == 0:
        return 0.0
    return float(rec[idx[0]])

def f1_at_optimal_threshold(y_true, y_prob):
    """Return the best F1 score across all thresholds."""
    prec, rec, thresh = precision_recall_curve(y_true, y_prob)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    return float(np.max(f1))


fig, axes = plt.subplots(
    nrows=len(PREDICTION_HORIZONS), ncols=2,
    figsize=(14, 5 * len(PREDICTION_HORIZONS))
)
if len(PREDICTION_HORIZONS) == 1:
    axes = [axes]

for row, horizon in enumerate(PREDICTION_HORIZONS):
    if horizon not in xgb_results:
        continue

    lr_r  = lr_results[horizon]
    xg_r  = xgb_results[horizon]
    y_t   = xg_r["y_test"]

    # — ROC curve —
    ax_roc = axes[row][0]
    for label, prob, color in [
        (f"Logistic Reg (AUC={lr_r['roc_auc']:.3f})", lr_r["prob_test"], "steelblue"),
        (f"XGBoost      (AUC={xg_r['roc_auc']:.3f})", xg_r["prob_test"], "darkorange"),
    ]:
        fpr, tpr, _ = roc_curve(y_t, prob)
        ax_roc.plot(fpr, tpr, label=label, color=color, lw=2)
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
    ax_roc.set_title(f"ROC Curve — {horizon}-day horizon")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.legend(fontsize=9)
    ax_roc.grid(alpha=0.3)

    # — PR curve —
    ax_pr = axes[row][1]
    for label, prob, color in [
        (f"Logistic Reg (AP={lr_r['pr_auc']:.3f})", lr_r["prob_test"], "steelblue"),
        (f"XGBoost      (AP={xg_r['pr_auc']:.3f})", xg_r["prob_test"], "darkorange"),
    ]:
        prec, rec, _ = precision_recall_curve(y_t, prob)
        ax_pr.plot(rec, prec, label=label, color=color, lw=2)
    baseline = y_t.mean()
    ax_pr.axhline(baseline, color="gray", lw=1, linestyle="--", label=f"Baseline (prevalence={baseline:.3f})")
    ax_pr.set_title(f"Precision-Recall Curve — {horizon}-day horizon")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.legend(fontsize=9)
    ax_pr.grid(alpha=0.3)

plt.tight_layout()
roc_path = os.path.join(RESULTS_DIR, "roc_pr_curves.png")
plt.savefig(roc_path, dpi=150)
plt.close()
print(f"  Saved ROC/PR curves → {roc_path}")

# — Print full metrics table —
print("\n  ── Metrics Summary ──")
print(f"  {'Model':<20} {'Horizon':>8} {'ROC-AUC':>9} {'PR-AUC':>9} {'Rec@P10':>9} {'F1@opt':>8} {'Brier':>8}")
print("  " + "-" * 74)
for horizon in PREDICTION_HORIZONS:
    if horizon not in xgb_results:
        continue
    for tag, res in [("LogisticReg", lr_results[horizon]), ("XGBoost", xgb_results[horizon])]:
        rap = recall_at_precision(res["y_test"], res["prob_test"], TARGET_PRECISION)
        f1  = f1_at_optimal_threshold(res["y_test"], res["prob_test"])
        print(f"  {tag:<20} {horizon:>6}d  {res['roc_auc']:>9.4f} {res['pr_auc']:>9.4f} {rap:>9.4f} {f1:>8.4f} {res['brier']:>8.4f}")

# — Calibration curves —
# A well-calibrated model returns P(liq)=0.3 for accounts that actually
# liquidate ~30% of the time.  The reliability diagram shows how close
# each model is to the diagonal ideal.
print("\n  Plotting calibration curves...")
n_cal_horizons = len([h for h in PREDICTION_HORIZONS if h in xgb_results])
if n_cal_horizons > 0:
    fig, axes_cal = plt.subplots(1, n_cal_horizons,
                                 figsize=(6 * n_cal_horizons, 5))
    if n_cal_horizons == 1:
        axes_cal = [axes_cal]
    cal_idx = 0
    for horizon in PREDICTION_HORIZONS:
        if horizon not in xgb_results:
            continue
        ax_c = axes_cal[cal_idx]; cal_idx += 1
        for tag, res, color in [
            ("Logistic Reg", lr_results[horizon],  "steelblue"),
            ("XGBoost",      xgb_results[horizon], "darkorange"),
        ]:
            CalibrationDisplay.from_predictions(
                res["y_test"], res["prob_test"],
                n_bins=min(10, max(5, int(res["y_test"].sum()))),
                ax=ax_c, name=tag, color=color,
            )
        ax_c.set_title(f"Calibration — {horizon}-day horizon")
        ax_c.grid(alpha=0.3)
    plt.tight_layout()
    cal_path = os.path.join(RESULTS_DIR, "calibration_curves.png")
    plt.savefig(cal_path, dpi=150)
    plt.close()
    print(f"  Saved calibration curves → {cal_path}")


# ============================================================
# PART 8 — SHAP Explainability
# ============================================================
# SHAP (SHapley Additive exPlanations) is the core interpretability
# tool specified in the proposal. It decomposes each prediction
# into the contribution of every feature for that specific
# observation, satisfying desirable axiomatic fairness properties.
#
# We produce three SHAP outputs:
#
#   (a) Global summary plot — which features matter most overall
#       (bar chart of mean |SHAP| across all test observations)
#
#   (b) Beeswarm plot — shows direction of each feature's effect
#       (e.g., high HF → low risk, high volatility → high risk)
#
#   (c) Per-account waterfall — shows the top-5 highest-risk
#       accounts with a breakdown of why each is flagged

print("\n[PART 8] SHAP explainability analysis...")

for horizon in PREDICTION_HORIZONS:
    if horizon not in xgb_results:
        continue

    model = xgb_results[horizon]["model"]

    # Use TreeExplainer — exact, fast for tree-based models
    explainer    = shap.TreeExplainer(model)
    shap_values  = explainer.shap_values(X_test)
    X_test_df    = pd.DataFrame(X_test, columns=ALL_FEATURES)

    # (a) Global bar chart — mean |SHAP|
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    feature_importance = pd.Series(mean_abs_shap, index=ALL_FEATURES)
    fi_sorted = feature_importance.sort_values(ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(fi_sorted.index[::-1], fi_sorted.values[::-1], color="steelblue", edgecolor="white")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(f"Global Feature Importance (XGBoost, {horizon}-day horizon)")
    ax.grid(axis="x", alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"shap_global_{horizon}d.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved SHAP global bar chart → {path}")

    # Print top-10 features
    print(f"\n  Top-10 features ({horizon}d horizon):")
    for feat, val in fi_sorted.head(10).items():
        print(f"    {feat:<35}  mean|SHAP| = {val:.4f}")

    # (b) Beeswarm summary plot
    plt.figure(figsize=(10, 7))
    shap.summary_plot(shap_values, X_test_df, show=False, max_display=15)
    plt.title(f"SHAP Beeswarm — {horizon}-day horizon")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"shap_beeswarm_{horizon}d.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP beeswarm → {path}")

    # (c) Waterfall plots for top-5 riskiest accounts
    prob_test   = xgb_results[horizon]["prob_test"]
    top5_idx    = np.argsort(prob_test)[-5:][::-1]

    fig, axes2 = plt.subplots(1, 5, figsize=(20, 6))
    for plot_i, idx in enumerate(top5_idx):
        ax = axes2[plot_i]
        # Manual waterfall: sorted features by |SHAP| for this observation
        sv   = shap_values[idx]
        order = np.argsort(np.abs(sv))[::-1][:8]
        feats = [ALL_FEATURES[j] for j in order]
        vals  = sv[order]
        colors = ["tomato" if v > 0 else "steelblue" for v in vals]
        ax.barh(range(len(feats)), vals[::-1], color=colors[::-1])
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats[::-1], fontsize=7)
        ax.set_title(f"Account #{plot_i+1}\nP(liq)={prob_test[idx]:.2f}", fontsize=9)
        ax.axvline(0, color="black", lw=0.8)
        ax.grid(axis="x", alpha=0.3)
    plt.suptitle(f"SHAP Waterfall — Top-5 Riskiest Accounts ({horizon}-day horizon)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"shap_waterfall_{horizon}d.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved SHAP waterfall (top 5 accounts) → {path}")


# ============================================================
# PART 8.5 — Partial Dependence Plots
# ============================================================
# Partial Dependence Plots (PDPs) show the marginal effect of one
# (or two) features on the predicted liquidation probability, averaging
# out all other features.  This complements SHAP by showing the shape
# of each feature's relationship — not just which features matter, but
# HOW they matter (monotone? threshold? interaction?).
#
# We produce:
#   (a) 1-D PDPs for the 6 most important features
#   (b) 2-D PDP for Health Factor × realized volatility — the core
#       interaction the proposal hypothesises as state-dependent risk.

print("\n[PART 8.5] Partial Dependence Plots...")

from sklearn.inspection import PartialDependenceDisplay

PDP_FEATURES_1D = [
    "approx_health_factor", "distance_to_liq", "realized_vol_7d",
    "debt_to_collateral",   "repay_borrow_ratio", "activity_intensity",
]

for horizon in PREDICTION_HORIZONS:
    if horizon not in xgb_results:
        continue

    model      = xgb_results[horizon]["model"]
    X_test_df  = pd.DataFrame(X_test, columns=ALL_FEATURES)
    feats_1d   = [f for f in PDP_FEATURES_1D if f in ALL_FEATURES]

    # (a) 1-D PDP grid
    n_cols = 3
    n_rows = (len(feats_1d) + n_cols - 1) // n_cols
    fig, axes_pdp = plt.subplots(n_rows, n_cols,
                                  figsize=(5 * n_cols, 4 * n_rows))
    axes_flat = np.array(axes_pdp).flatten()
    try:
        PartialDependenceDisplay.from_estimator(
            model, X_test_df, feats_1d,
            feature_names=ALL_FEATURES,
            ax=axes_flat[:len(feats_1d)],
            grid_resolution=50,
            percentiles=(0.05, 0.95),
        )
    except Exception as e:
        print(f"  1-D PDP error ({horizon}d): {e}")
    for ax in axes_flat[len(feats_1d):]:
        ax.set_visible(False)
    fig.suptitle(f"Partial Dependence Plots — {horizon}-day horizon", fontsize=12)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"pdp_1d_{horizon}d.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved 1-D PDP → {path}")

    # (b) 2-D PDP: Health Factor × Volatility
    if "approx_health_factor" in ALL_FEATURES and "realized_vol_7d" in ALL_FEATURES:
        try:
            fig, ax2d = plt.subplots(figsize=(7, 5))
            PartialDependenceDisplay.from_estimator(
                model, X_test_df,
                [("approx_health_factor", "realized_vol_7d")],
                feature_names=ALL_FEATURES,
                ax=[ax2d],
                grid_resolution=30,
                percentiles=(0.05, 0.95),
            )
            fig.suptitle(
                f"2-D PDP: Health Factor × Volatility ({horizon}-day)", fontsize=11)
            plt.tight_layout()
            path = os.path.join(RESULTS_DIR, f"pdp_2d_hf_vol_{horizon}d.png")
            plt.savefig(path, dpi=150, bbox_inches="tight")
            plt.close()
            print(f"  Saved 2-D PDP (HF × Vol) → {path}")
        except Exception as e:
            print(f"  2-D PDP skipped: {e}")


# ============================================================
# PART 9 — Regime-Based Analysis
# ============================================================
# The proposal's secondary research question: do the determinants
# of liquidation risk differ across market regimes?
#
# We split the TEST set into "low-vol" and "high-vol" periods
# using the 75th percentile of realized_vol_7d as the threshold
# (defined in Part 2).
#
# For each regime we report:
#   • ROC-AUC and PR-AUC
#   • Top-5 SHAP features (to see if importance shifts)
#
# This allows us to answer whether XGBoost picks up on the
# state-dependent nature of liquidation risk described in the proposal.

print("\n[PART 9] Regime-based analysis (low-vol vs high-vol)...")

for horizon in PREDICTION_HORIZONS:
    if horizon not in xgb_results:
        continue

    model    = xgb_results[horizon]["model"]
    prob_all = xgb_results[horizon]["prob_test"]
    y_all    = xgb_results[horizon]["y_test"]

    regime_col = test_df["high_vol_regime"].values
    explainer2 = shap.TreeExplainer(model)

    print(f"\n  ── Horizon: {horizon}d ──")
    regime_metrics = []
    for regime_val, regime_name in [(0, "Low-Vol"), (1, "High-Vol")]:
        mask     = (regime_col == regime_val)
        y_r      = y_all[mask]
        prob_r   = prob_all[mask]
        Xr       = X_test[mask]

        if len(y_r) < 20 or y_r.sum() < 2:
            print(f"    {regime_name}: insufficient samples — skipping")
            continue

        roc_r  = roc_auc_score(y_r, prob_r)
        pr_r   = average_precision_score(y_r, prob_r)
        shap_r = explainer2.shap_values(Xr)
        top5_r = pd.Series(
            np.abs(shap_r).mean(axis=0), index=ALL_FEATURES
        ).sort_values(ascending=False).head(5)

        print(f"    {regime_name}  (n={mask.sum():,}, pos={y_r.sum()})")
        print(f"      ROC-AUC: {roc_r:.4f}  |  PR-AUC: {pr_r:.4f}")
        print(f"      Top-5 features: {list(top5_r.index)}")
        regime_metrics.append({"regime": regime_name, "n": mask.sum(),
                                "roc_auc": roc_r, "pr_auc": pr_r,
                                "top5": list(top5_r.index)})

    # — Bar chart comparing ROC-AUC across regimes —
    if len(regime_metrics) == 2:
        fig, ax = plt.subplots(figsize=(6, 4))
        labels = [r["regime"] for r in regime_metrics]
        rocs   = [r["roc_auc"] for r in regime_metrics]
        prs    = [r["pr_auc"]  for r in regime_metrics]
        x = np.arange(2)
        ax.bar(x - 0.2, rocs, 0.35, label="ROC-AUC", color="steelblue")
        ax.bar(x + 0.2, prs,  0.35, label="PR-AUC",  color="darkorange")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylim(0, 1)
        ax.set_title(f"XGBoost Performance by Volatility Regime ({horizon}d)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"regime_comparison_{horizon}d.png")
        plt.savefig(path, dpi=150)
        plt.close()
        print(f"  Saved regime comparison → {path}")


# ============================================================
# PART 10 — Results Summary & Export
# ============================================================
# Collect all metrics into a single CSV table so results can be
# easily pasted into the final paper. Also export per-account
# risk scores for the test set (useful as an appendix exhibit).

print("\n[PART 10] Exporting results...")

# — Metrics table —
rows = []
for horizon in PREDICTION_HORIZONS:
    if horizon not in xgb_results:
        continue
    for tag, res in [("Logistic Regression", lr_results[horizon]),
                     ("XGBoost",             xgb_results[horizon])]:
        rap = recall_at_precision(res["y_test"], res["prob_test"], TARGET_PRECISION)
        f1  = f1_at_optimal_threshold(res["y_test"], res["prob_test"])
        rows.append({
            "Model":    tag,
            "Horizon":  f"{horizon}d",
            "ROC_AUC":  round(res["roc_auc"], 4),
            "PR_AUC":   round(res["pr_auc"],  4),
            f"Recall_at_P{int(TARGET_PRECISION*100)}": round(rap, 4),
            "F1_optimal": round(f1, 4),
            "Brier":    round(res["brier"],   4),
        })

metrics_df = pd.DataFrame(rows)
metrics_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
metrics_df.to_csv(metrics_path, index=False)
print(f"  Metrics table → {metrics_path}")
print(metrics_df.to_string(index=False))

# — Risk score table —
for horizon in PREDICTION_HORIZONS:
    if horizon not in xgb_results:
        continue
    scores_df = test_df[["account", "obs_date"]].copy().reset_index(drop=True)
    scores_df[f"risk_score_{horizon}d"] = xgb_results[horizon]["prob_test"]
    scores_df[f"label_{horizon}d"]      = xgb_results[horizon]["y_test"]
    scores_df = scores_df.sort_values(f"risk_score_{horizon}d", ascending=False)
    path = os.path.join(RESULTS_DIR, f"risk_scores_{horizon}d.csv")
    scores_df.to_csv(path, index=False)
    print(f"  Risk scores ({horizon}d) → {path}")

print("\n" + "=" * 65)
print("  Pipeline complete. All outputs saved to ./results/")
print("=" * 65)
