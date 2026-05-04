"""
models_advanced.py  (v2 — no mandatory PyTorch)
================================================
Advanced models for:
  "Predicting Short-Horizon Liquidation Risk in Aave
   Using Explainable Machine Learning"
  Runming Huang & Leran Li — DOTE 6635, Spring 2026

Dependencies
------------
  Required (all in Anaconda base):
    numpy, pandas, scikit-learn, matplotlib

  Optional:
    torch          → enables LSTM model
    anthropic      → enables LLM Zero-shot + Agent (Claude)
    openai         → enables LLM Zero-shot + Agent (GPT-4)

Usage
-----
  # Minimum (MLP always runs):
  python models_advanced.py

  # With LSTM:
  pip install torch && python models_advanced.py

  # With LLM models:
  export ANTHROPIC_API_KEY="sk-ant-..."
  python models_advanced.py

Course connection
-----------------
  MLP (sklearn)  → Session 2  (Deep Learning Introduction)
  LSTM (PyTorch) → Session 2  (sequence modeling with RNNs)
  LLM Zero-shot  → Session 8  (LLM Reasoning)
  LLM Agent      → Sessions 9-13 (Agentic AI)
"""

# ============================================================
# PART 0 — Imports & Config
# ============================================================
import os, re, json, time, warnings, textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datetime import timedelta
from itertools import product as iterproduct

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve,
    brier_score_loss,
)

def f1_at_optimal(y_true, y_prob):
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    f1 = np.where((prec + rec) > 0, 2 * prec * rec / (prec + rec), 0.0)
    return float(np.max(f1))

# ── Optional: PyTorch (for LSTM) ───────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)
except ImportError:
    TORCH_AVAILABLE = False
    DEVICE = "cpu"

# ── Optional: LLM APIs ─────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")

warnings.filterwarnings("ignore")
np.random.seed(42)

# ── TEST MODE ───────────────────────────────────────────────
# Usage:  TEST_MODE=1 python models_advanced.py
TEST_MODE = os.getenv("TEST_MODE", "0") == "1"

DATA_DIR          = "data"
RESULTS_DIR       = "results"
RANDOM_STATE      = 42
PREDICTION_HORIZONS = [7, 14]
TARGET_PRECISION  = 0.30
LLM_SAMPLE_SIZE   = 5  if TEST_MODE else 1000  # Zero-shot: 1000 accounts per horizon
AGENT_SAMPLE_SIZE = 3  if TEST_MODE else 1000  # Agent: 1000 accounts per horizon

os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 65)
print("  Advanced Models — MLP | LSTM | LLM Zero-shot | LLM Agent")
print(f"  PyTorch available : {TORCH_AVAILABLE}")
print(f"  Anthropic key set : {bool(ANTHROPIC_API_KEY)}")
print(f"  OpenAI key set    : {bool(OPENAI_API_KEY)}")
print("=" * 65)


# ============================================================
# PART 1 — Load Dataset
# ============================================================
print("\n[PART 1] Loading dataset...")

def make_synthetic_dataset(n=10_000, seed=RANDOM_STATE):
    rng  = np.random.default_rng(seed)
    from datetime import datetime, timedelta
    base = pd.Timestamp("2022-01-01")
    dates = [base + timedelta(days=int(d)) for d in rng.integers(0, 700, n)]
    col   = rng.lognormal(10, 1.5, n)
    debt  = col / rng.uniform(1.1, 5.0, n)
    hf    = col / (debt + 1e-9)
    vol7  = np.abs(rng.normal(0.6, 0.3, n))
    ret7  = rng.normal(0, 0.10, n)
    lo    = -4 + (-3.5*np.log(np.clip(hf,0.5,10))) + 2.0*vol7 + 0.8*np.abs(ret7) + rng.normal(0,0.5,n)
    prob  = 1/(1+np.exp(-lo))
    l7    = (rng.uniform(size=n) < prob).astype(int)
    l14   = np.where(l7==1, 1, (rng.uniform(size=n) < prob*1.4).astype(int))
    return pd.DataFrame({
        "account": [f"0x{rng.integers(0,5000):04x}" for _ in range(n)],
        "obs_date": dates,
        "collateral_value_usd": col, "total_debt_usd": debt,
        "collateralization_ratio": hf, "approx_health_factor": hf,
        "n_collateral_assets": rng.integers(1,5,n), "n_debt_assets": rng.integers(1,4,n),
        "stablecoin_debt_share": rng.beta(1,4,n),
        "n_deposits_30d": rng.integers(0,20,n), "n_borrows_30d": rng.integers(0,15,n),
        "n_repays_30d": rng.integers(0,15,n), "n_withdraws_30d": rng.integers(0,10,n),
        "total_tx_30d": rng.integers(0,40,n), "account_age_days": rng.integers(1,700,n),
        "eth_price_usd": rng.lognormal(7.5,0.4,n),
        "eth_return_7d": ret7, "realized_vol_7d": vol7,
        "label_7d": l7, "label_14d": l14,
    })

dataset_path = os.path.join(DATA_DIR, "dataset.csv")
if os.path.exists(dataset_path):
    df = pd.read_csv(dataset_path, parse_dates=["obs_date"])
    print(f"  Loaded: {len(df):,} rows from dataset.csv")
else:
    print("  dataset.csv not found — generating synthetic data.")
    df = make_synthetic_dataset()

# ── TEST MODE: subsample ────────────────────────────────────
if TEST_MODE:
    df = df.sample(n=min(2000, len(df)), random_state=RANDOM_STATE).reset_index(drop=True)
    print(f"  ⚡ TEST MODE: subsampled to {len(df):,} rows")

# Feature engineering (mirrors model.py)
df["debt_to_collateral"] = df["total_debt_usd"] / (df["collateral_value_usd"] + 1e-9)
df["distance_to_liq"]    = np.clip(df["approx_health_factor"] - 1.0, 0, 10)
df["log_collateral"]     = np.log1p(df["collateral_value_usd"])
df["log_debt"]           = np.log1p(df["total_debt_usd"])
df["repay_borrow_ratio"] = df["n_repays_30d"]   / (df["n_borrows_30d"]   + 1)
df["withdraw_dep_ratio"] = df["n_withdraws_30d"] / (df["n_deposits_30d"] + 1)
df["activity_intensity"] = df["total_tx_30d"]    / (df.get("account_age_days",
                            pd.Series(np.ones(len(df))*365)) + 1)
df["eth_return_sign"]    = np.sign(df["eth_return_7d"])
df["vol_times_debt"]     = df["realized_vol_7d"] * df["debt_to_collateral"]

ALL_FEATURES = [
    "collateral_value_usd","total_debt_usd","collateralization_ratio",
    "approx_health_factor","debt_to_collateral","distance_to_liq",
    "log_collateral","log_debt","n_collateral_assets","n_debt_assets",
    "stablecoin_debt_share","n_deposits_30d","n_borrows_30d","n_repays_30d",
    "n_withdraws_30d","total_tx_30d","repay_borrow_ratio","withdraw_dep_ratio",
    "activity_intensity","eth_price_usd","eth_return_7d","realized_vol_7d",
    "eth_return_sign","vol_times_debt",
]
ALL_FEATURES = [f for f in ALL_FEATURES if f in df.columns]
label_cols   = [f"label_{h}d" for h in PREDICTION_HORIZONS if f"label_{h}d" in df.columns]
df = df.dropna(subset=ALL_FEATURES + label_cols).reset_index(drop=True)

# Time-aware split
df = df.sort_values("obs_date").reset_index(drop=True)
n_test     = int(len(df) * 0.20)
train_df   = df.iloc[:-n_test].copy()
test_df    = df.iloc[-n_test:].copy()

# TEST MODE safety: ensure at least 2 positives in test set per label
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

scaler   = StandardScaler()
X_train  = scaler.fit_transform(train_df[ALL_FEATURES].values)
X_test   = scaler.transform(test_df[ALL_FEATURES].values)
val_split = int(len(X_train) * 0.85)

print(f"  Train: {len(train_df):,}  |  Test: {len(test_df):,}  |  Features: {len(ALL_FEATURES)}")


# ============================================================
# MODEL 3 — MLP  (sklearn MLPClassifier — no PyTorch needed)
# ============================================================
# MLPClassifier is a fully-connected feedforward neural network
# built into scikit-learn. It uses Adam optimiser and supports
# early stopping, making it easy to run without any GPU setup.
#
# Architecture: Input → 256 → 128 → 64 → sigmoid output
# This is conceptually identical to the PyTorch MLP but uses
# sklearn's battle-tested implementation.
#
# Course connection: Session 2 (Deep Learning Introduction)

print("\n[MODEL 3] Training MLP (sklearn MLPClassifier)...")
mlp_results = {}

for horizon in PREDICTION_HORIZONS:
    lc = f"label_{horizon}d"
    if lc not in df.columns:
        continue

    y_train_h = train_df[lc].values
    y_test_h  = test_df[lc].values
    pos_rate  = y_train_h.mean()

    print(f"\n  ── Horizon: {horizon}d  (pos rate: {pos_rate:.1%}) ──")

    # Handle class imbalance via random oversampling of minority class
    # (MLPClassifier doesn't support sample_weight)
    rng      = np.random.default_rng(RANDOM_STATE)
    pos_idx  = np.where(y_train_h == 1)[0]
    neg_idx  = np.where(y_train_h == 0)[0]
    # Oversample positives to ~10% of negatives (keeps training tractable)
    target_pos = min(len(neg_idx) // 10, len(pos_idx) * 20)
    over_idx   = rng.choice(pos_idx, size=target_pos, replace=True)
    all_idx    = np.concatenate([neg_idx, over_idx])
    rng.shuffle(all_idx)
    X_tr_bal   = X_train[all_idx]
    y_tr_bal   = y_train_h[all_idx]
    print(f"    Oversampled: {len(neg_idx)} neg + {len(over_idx)} pos → {len(all_idx)} total")

    mlp = MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        batch_size=256,
        learning_rate_init=1e-3,
        max_iter=50 if TEST_MODE else 200,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=RANDOM_STATE,
        verbose=False,
    )
    mlp.fit(X_tr_bal, y_tr_bal)

    prob_mlp = mlp.predict_proba(X_test)[:, 1]

    if len(np.unique(y_test_h)) < 2:
        print(f"    ⚠ Only one class in test set — skipping metrics for {horizon}d")
        continue

    roc = roc_auc_score(y_test_h, prob_mlp)
    pr  = average_precision_score(y_test_h, prob_mlp)
    bs  = brier_score_loss(y_test_h, prob_mlp)

    mlp_results[horizon] = {
        "prob_test": prob_mlp, "y_test": y_test_h,
        "roc_auc": roc, "pr_auc": pr, "brier": bs,
    }
    print(f"    ROC-AUC : {roc:.4f}")
    print(f"    PR-AUC  : {pr:.4f}")
    print(f"    Brier   : {bs:.4f}")
    print(f"    Stopped at iteration: {mlp.n_iter_}")


# ============================================================
# MODEL 4 — LSTM  (PyTorch, optional)
# ============================================================
# A bidirectional LSTM over each account's transaction sequence.
# Skipped gracefully if PyTorch is not installed.
#
# Course connection: Session 2 (sequence modeling with RNNs)

SEQ_LEN     = 30
TX_FEATURES = 6

def build_sequences(df_main, tx_path, seq_len=SEQ_LEN):
    n    = len(df_main)
    seqs = np.zeros((n, seq_len, TX_FEATURES), dtype=np.float32)

    # Detect which column name is used for tx type and amount
    if os.path.exists(tx_path):
        tx = pd.read_csv(tx_path, parse_dates=["date"])

        # Flexible column name detection
        type_col   = next((c for c in tx.columns if c in ("tx_type","transaction_type","type")), None)
        amount_col = next((c for c in tx.columns if c in ("amount_usd","amount")), None)

        if type_col is None:
            print("    ⚠ No tx-type column found — falling back to simulated sequences.")
            tx = None
        else:
            type_map = {"deposit": 0, "borrow": 1, "repay": 2, "withdraw": 3}
            tx["type_id"] = tx[type_col].map(type_map).fillna(0).astype(int)

    if not os.path.exists(tx_path):
        tx = None

    if tx is not None:
        # Build account lookup index once (much faster than iterrows lookup)
        grp = {acct: grp_df for acct, grp_df in tx.groupby("account")}
        print(f"    Building sequences from {len(tx):,} real transactions...")
        for i, row in df_main.reset_index(drop=True).iterrows():
            acct = row["account"]
            if acct not in grp:
                continue
            acct_tx = grp[acct].sort_values("date").tail(seq_len)
            t = len(acct_tx)
            for j, (_, r) in enumerate(acct_tx.iterrows()):
                slot = seq_len - t + j
                seqs[i, slot, int(r["type_id"])] = 1.0   # one-hot tx type
                seqs[i, slot, 4] = 1.0                    # "has event" flag
                amt = abs(r[amount_col]) if amount_col else 0
                seqs[i, slot, 5] = np.log1p(amt)
        # Normalise amount column
        max_a = seqs[:, :, 5].max()
        if max_a > 0:
            seqs[:, :, 5] /= max_a
    else:
        # Simulate sequences from 30-day aggregate counts
        print("    Simulating sequences from aggregate behavioral features...")
        rng = np.random.default_rng(RANDOM_STATE)
        for i, row in df_main.reset_index(drop=True).iterrows():
            counts = [
                row.get("n_deposits_30d", 0), row.get("n_borrows_30d", 0),
                row.get("n_repays_30d", 0),   row.get("n_withdraws_30d", 0),
            ]
            total = max(int(sum(counts)), 1)
            probs = np.array(counts, dtype=float)
            probs = probs / (probs.sum() + 1e-9)
            events = rng.choice(4, size=min(total, seq_len), p=probs)
            for j, ev in enumerate(events):
                slot = seq_len - len(events) + j
                seqs[i, slot, ev]  = 1.0
                seqs[i, slot, 4]   = 1.0
                seqs[i, slot, 5]   = rng.exponential(2.0)
        max_a = seqs[:, :, 5].max()
        if max_a > 0:
            seqs[:, :, 5] /= max_a
    return seqs

lstm_results = {}

if not TORCH_AVAILABLE:
    print("\n[MODEL 4] LSTM — SKIPPED (PyTorch not installed)")
    print("  Run:  pip install torch   then rerun this script.")
else:
    print("\n[MODEL 4] Training LSTM (PyTorch bidirectional)...")

    class LSTMRiskModel(nn.Module):
        def __init__(self, input_size, hidden=64, layers=2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden, num_layers=layers,
                                batch_first=True, bidirectional=True,
                                dropout=0.3 if layers > 1 else 0)
            self.head = nn.Sequential(
                nn.Linear(hidden*2, 32), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(32, 1),
            )
        def forward(self, x):
            out, _ = self.lstm(x)
            return torch.sigmoid(self.head(out[:, -1, :])).squeeze(-1)

    def focal_loss(pred, target, alpha=0.75, gamma=2.0):
        bce = nn.functional.binary_cross_entropy(pred, target, reduction="none")
        pt  = torch.where(target == 1, pred, 1 - pred)
        return (alpha * ((1 - pt) ** gamma) * bce).mean()

    tx_path = os.path.join(DATA_DIR, "raw_transactions.csv")
    # Rebuild sequences aligned to the (possibly TEST_MODE-adjusted) split.
    # train_df / test_df may differ from df[:-n_test] / df[-n_test:] after
    # positives are moved in TEST_MODE, so we concatenate them in order.
    df_for_seq = pd.concat([train_df, test_df], ignore_index=True)
    seqs_all   = build_sequences(df_for_seq, tx_path)
    seqs_train = seqs_all[:len(train_df)]
    seqs_test  = seqs_all[len(train_df):]

    for horizon in PREDICTION_HORIZONS:
        lc = f"label_{horizon}d"
        if lc not in df.columns:
            continue

        y_tr = train_df[lc].values
        y_te = test_df[lc].values
        print(f"\n  ── Horizon: {horizon}d ──")

        # Train/val split
        Xtr_s, Xvl_s = seqs_train[:val_split], seqs_train[val_split:]
        ytr, yvl     = y_tr[:val_split], y_tr[val_split:]

        ds     = TensorDataset(torch.tensor(Xtr_s),
                               torch.tensor(ytr, dtype=torch.float32))
        loader = DataLoader(ds, batch_size=128, shuffle=True)
        model  = LSTMRiskModel(TX_FEATURES).to(DEVICE)
        opt    = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-4)

        best_auc  = 0
        best_state = {k: v.clone() for k, v in model.state_dict().items()}  # init with epoch-0 weights
        for epoch in range(5 if TEST_MODE else 40):
            model.train()
            for Xb, yb in loader:
                Xb, yb = Xb.to(DEVICE), yb.to(DEVICE)
                opt.zero_grad()
                focal_loss(model(Xb), yb).backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                pv = model(torch.tensor(Xvl_s).to(DEVICE)).cpu().numpy()
            try:
                auc = roc_auc_score(yvl, pv)
                if auc > best_auc:
                    best_auc   = auc
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}
            except Exception:
                pass

        model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            prob_lstm_raw = model(torch.tensor(seqs_test).to(DEVICE)).cpu().numpy()

        # Temperature scaling: learn a single scalar T on the val set so
        # that prob = sigmoid(logit / T) is better calibrated.
        # Optimise T by minimising NLL on validation predictions.
        from scipy.optimize import minimize_scalar
        import scipy.special as sp
        val_logits = np.log(np.clip(pv, 1e-7, 1-1e-7)) - np.log(np.clip(1-pv, 1e-7, 1-1e-7))
        def nll(T):
            scaled = sp.expit(val_logits / max(T, 0.01))
            return -np.mean(yvl * np.log(scaled + 1e-9) + (1-yvl) * np.log(1-scaled + 1e-9))
        res_T  = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        T_opt  = res_T.x
        test_logits = np.log(np.clip(prob_lstm_raw, 1e-7, 1-1e-7)) - np.log(np.clip(1-prob_lstm_raw, 1e-7, 1-1e-7))
        prob_lstm = sp.expit(test_logits / T_opt)
        print(f"    Temperature scaling T = {T_opt:.3f}")

        roc = roc_auc_score(y_te, prob_lstm)
        pr  = average_precision_score(y_te, prob_lstm)
        bs  = brier_score_loss(y_te, prob_lstm)
        lstm_results[horizon] = {
            "prob_test": prob_lstm, "y_test": y_te,
            "roc_auc": roc, "pr_auc": pr, "brier": bs,
        }
        print(f"    Best val AUC : {best_auc:.4f}")
        print(f"    Test ROC-AUC : {roc:.4f}")
        print(f"    Test PR-AUC  : {pr:.4f}")


# ============================================================
# MODEL 5 — LLM Zero-shot
# ============================================================
# Sends each account's features as natural language to Claude/GPT-4
# and asks for a liquidation probability — no training, pure reasoning.
#
# Course connection: Session 8 (LLM Reasoning)

SYSTEM_PROMPT_ZS = textwrap.dedent("""
You are a DeFi risk analyst for Aave. Estimate the probability that
a borrower account will be LIQUIDATED within {horizon} days.
Liquidation happens when Health Factor < 1.0.
Respond ONLY with JSON: {{"liquidation_prob": <0-1>, "reasoning": "<one sentence>"}}
""")

USER_PROMPT_ZS = """
Account snapshot:
- Health Factor:        {hf:.3f}
- Collateral (USD):     {col:.0f}
- Total Debt (USD):     {debt:.0f}
- Debt/Collateral:      {dtc:.3f}
- Distance to liq:      {dtl:.3f}
- Borrows (30d):        {bor}
- Repays  (30d):        {rep}
- ETH 7d return:        {ret:.2%}
- 7d realized vol:      {vol:.3f}

Probability of liquidation within {horizon} days?
"""

def call_llm(row, horizon, provider):
    msg = USER_PROMPT_ZS.format(
        hf=row.get("approx_health_factor",1.5),
        col=row.get("collateral_value_usd",0),
        debt=row.get("total_debt_usd",0),
        dtc=row.get("debt_to_collateral",0),
        dtl=row.get("distance_to_liq",0),
        bor=int(row.get("n_borrows_30d",0)),
        rep=int(row.get("n_repays_30d",0)),
        ret=row.get("eth_return_7d",0),
        vol=row.get("realized_vol_7d",0),
        horizon=horizon,
    )
    sys = SYSTEM_PROMPT_ZS.format(horizon=horizon)
    try:
        if provider == "anthropic":
            import anthropic
            c = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            r = c.messages.create(
                model="claude-haiku-4-5-20251001", max_tokens=128,
                system=sys, messages=[{"role":"user","content":msg}],
            )
            raw = r.content[0].text.strip()
        else:
            from openai import OpenAI
            c = OpenAI(api_key=OPENAI_API_KEY)
            r = c.chat.completions.create(
                model="gpt-4o-mini", max_tokens=128,
                messages=[{"role":"system","content":sys},
                          {"role":"user","content":msg}],
            )
            raw = r.choices[0].message.content.strip()
        # Strip markdown code fences (model sometimes wraps JSON in ```json ... ```)
        raw = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`").strip()
        return float(json.loads(raw)["liquidation_prob"])
    except Exception:
        return None

llm_zeroshot_results = {}
print("\n[MODEL 5] LLM Zero-shot...")
provider = "anthropic" if ANTHROPIC_API_KEY else ("openai" if OPENAI_API_KEY else None)

for horizon in PREDICTION_HORIZONS:
    lc = f"label_{horizon}d"
    if lc not in df.columns:
        continue
    print(f"\n  ── Horizon: {horizon}d ──")
    if provider is None:
        print("  SKIPPED — set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable.")
        llm_zeroshot_results[horizon] = None
        continue

    # Stratified sample: guarantee at least 1/3 positives so ROC metrics are
    # meaningful even with a very low base rate (~0.4 %).
    _pos = test_df[test_df[lc] == 1]
    _neg = test_df[test_df[lc] == 0]
    _n_pos = min(len(_pos), max(2, LLM_SAMPLE_SIZE // 3))
    _n_neg = min(len(_neg), LLM_SAMPLE_SIZE - _n_pos)
    sample = pd.concat([
        _pos.sample(_n_pos, random_state=RANDOM_STATE,
                    replace=len(_pos) < _n_pos),
        _neg.sample(_n_neg, random_state=RANDOM_STATE),
    ]).reset_index(drop=True)
    print(f"    Sample: {_n_pos} positives + {_n_neg} negatives")
    probs, labels = [], []
    for i, (_, row) in enumerate(sample.iterrows()):
        p = call_llm(row, horizon, provider)
        if p is not None:
            probs.append(p)
            labels.append(int(row[lc]))
        if (i+1) % 10 == 0:
            print(f"    {i+1}/{len(sample)} accounts processed...")
        time.sleep(0.05)

    if len(probs) >= 5 and sum(labels) >= 1:
        roc = roc_auc_score(labels, probs)
        pr  = average_precision_score(labels, probs)
        bs  = brier_score_loss(labels, probs)
        llm_zeroshot_results[horizon] = {
            "prob_test": np.array(probs), "y_test": np.array(labels),
            "roc_auc": roc, "pr_auc": pr, "brier": bs,
        }
        print(f"    ROC-AUC: {roc:.4f}  |  PR-AUC: {pr:.4f}  (n={len(probs)})")
    else:
        print("    Not enough positive samples — skipping metrics.")
        llm_zeroshot_results[horizon] = None


# ============================================================
# MODEL 6 — LLM Agent (ReAct)
# ============================================================
# A ReAct-style agent that calls tools before making its risk
# assessment: check HF, get market context, find similar accounts,
# simulate price shock. The reasoning trace is saved to file.
#
# Course connection: Sessions 9-13 (Agentic AI)

AGENT_SYS = textwrap.dedent("""
You are a DeFi risk analyst agent. You have tools:
  - get_health_factor: check current HF
  - market_context: ETH trend and volatility
  - similar_accounts: historical liquidation rate for similar accounts
  - price_shock: simulate ETH price drop impact

Use Thought/Action/Observation format. After ≤4 tool calls, output:
FINAL: {"liquidation_prob": <0-1>, "reasoning": "<one sentence>"}
""")

def tool_hf(row):
    hf = row.get("approx_health_factor", 1.5)
    risk = "HIGH" if hf < 1.2 else ("MEDIUM" if hf < 1.5 else "LOW")
    return f"Health Factor: {hf:.3f} ({risk} risk)"

def tool_market(row):
    ret = row.get("eth_return_7d", 0)
    vol = row.get("realized_vol_7d", 0.5)
    trend = "bullish" if ret > 0.02 else ("bearish" if ret < -0.02 else "neutral")
    level = "high" if vol > 0.8 else ("moderate" if vol > 0.4 else "low")
    return f"ETH 7d return: {ret:.1%} ({trend}), vol: {vol:.3f} ({level})"

def tool_similar(hf, vol, lc):
    mask = (
        df["approx_health_factor"].between(hf*0.9, hf*1.1) &
        df["realized_vol_7d"].between(vol*0.8, vol*1.2)
    )
    cohort = df[mask]
    if len(cohort) < 5:
        return "Insufficient historical data."
    rate = cohort[lc].mean()
    return f"{len(cohort)} similar accounts, liquidation rate: {rate:.1%}"

def tool_shock(hf, shock=-0.20):
    new_hf = max(hf * (1 + shock), 0)
    status = "LIQUIDATED" if new_hf < 1.0 else f"HF = {new_hf:.3f} (safe)"
    return f"After {shock:.0%} ETH move: {status}"

def run_agent(row, horizon, provider):
    lc  = f"label_{horizon}d"
    hf  = row.get("approx_health_factor", 1.5)
    vol = row.get("realized_vol_7d", 0.5)

    init = (f"Account HF≈{hf:.2f}, debt/col={row.get('debt_to_collateral',0):.2f}, "
            f"vol={vol:.3f}, borrows(30d)={int(row.get('n_borrows_30d',0))}, "
            f"repays(30d)={int(row.get('n_repays_30d',0))}. "
            f"Task: liquidation probability within {horizon} days?")

    messages = [{"role": "user", "content": init}]
    trace    = [f"=== Account snippet ===\n{init}\n"]
    prob     = 0.1

    try:
        for step in range(5):
            if provider == "anthropic":
                import anthropic
                c = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
                r = c.messages.create(
                    model="claude-haiku-4-5-20251001", max_tokens=300,
                    system=AGENT_SYS,
                    messages=messages,
                )
                reply = r.content[0].text.strip()
            else:
                from openai import OpenAI
                c = OpenAI(api_key=OPENAI_API_KEY)
                r = c.chat.completions.create(
                    model="gpt-4o-mini", max_tokens=300,
                    messages=[{"role":"system","content":AGENT_SYS}]+messages,
                )
                reply = r.choices[0].message.content.strip()

            messages.append({"role": "assistant", "content": reply})
            trace.append(f"[Step {step+1}]\n{reply}\n")

            obs = None
            if "get_health_factor"  in reply: obs = tool_hf(row)
            elif "market_context"   in reply: obs = tool_market(row)
            elif "similar_accounts" in reply: obs = tool_similar(hf, vol, lc)
            elif "price_shock"      in reply: obs = tool_shock(hf)

            if obs:
                messages.append({"role":"user","content":f"Observation: {obs}"})
                trace.append(f"  Obs: {obs}\n")
                time.sleep(0.05)

            if "FINAL:" in reply:
                try:
                    raw_final = reply.split("FINAL:")[-1].strip()
                    # Strip markdown code fences
                    raw_final = re.sub(r"```(?:json)?\s*", "", raw_final).strip().rstrip("`").strip()
                    parsed = json.loads(raw_final)
                    prob   = float(parsed["liquidation_prob"])
                except Exception:
                    # Regex fallback: extract the numeric value directly
                    m = re.search(r'"liquidation_prob"\s*:\s*([\d.]+)', reply)
                    if m:
                        prob = float(m.group(1))
                break
    except Exception as e:
        trace.append(f"[Error: {e}]")

    return {"prob": prob, "trace": "\n".join(trace)}

agent_results = {}
print("\n[MODEL 6] LLM Agent (ReAct)...")

for horizon in PREDICTION_HORIZONS:
    lc = f"label_{horizon}d"
    if lc not in df.columns:
        continue
    print(f"\n  ── Horizon: {horizon}d ──")
    if provider is None:
        print("  SKIPPED — set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable.")
        agent_results[horizon] = None
        continue

    sample = test_df.sample(min(AGENT_SAMPLE_SIZE, len(test_df)),
                            random_state=RANDOM_STATE).reset_index(drop=True)
    probs, traces, labels = [], [], []
    for i, (_, row) in enumerate(sample.iterrows()):
        res = run_agent(row, horizon, provider)
        if res["prob"] is not None:
            probs.append(res["prob"])
            traces.append(res["trace"])
            labels.append(int(row[lc]))
        print(f"    {i+1}/{len(sample)} accounts processed...")
        time.sleep(0.1)

    trace_path = os.path.join(RESULTS_DIR, f"agent_traces_{horizon}d.txt")
    with open(trace_path, "w") as f:
        f.write(f"\n{'='*60}\n".join(traces[:5]))
    print(f"  Agent traces saved → {trace_path}")

    if len(probs) >= 5 and sum(labels) >= 1:
        roc = roc_auc_score(labels, probs)
        pr  = average_precision_score(labels, probs)
        bs  = brier_score_loss(labels, probs)
        agent_results[horizon] = {
            "prob_test": np.array(probs), "y_test": np.array(labels),
            "roc_auc": roc, "pr_auc": pr, "brier": bs, "n": len(probs),
        }
        print(f"    ROC-AUC: {roc:.4f}  |  PR-AUC: {pr:.4f}  (n={len(probs)})")
    else:
        agent_results[horizon] = None


# ============================================================
# PART 7 — Grand Comparison Chart
# ============================================================
print("\n[PART 7] Building grand comparison...")

prior_path = os.path.join(RESULTS_DIR, "metrics_summary.csv")
prior = {}
if os.path.exists(prior_path):
    pm = pd.read_csv(prior_path)
    for _, row in pm.iterrows():
        h = int(row["Horizon"].replace("d",""))
        prior.setdefault(h, {})[row["Model"]] = row

MODEL_ORDER = [
    "Logistic Regression",
    "XGBoost",
    "MLP (sklearn)",
    "LSTM (PyTorch)",
    "LLM Zero-shot",
    "LLM Agent (ReAct)",
]
COLORS = {
    "Logistic Regression": "#4e79a7",
    "XGBoost":             "#f28e2b",
    "MLP (sklearn)":       "#e15759",
    "LSTM (PyTorch)":      "#76b7b2",
    "LLM Zero-shot":       "#59a14f",
    "LLM Agent (ReAct)":   "#b07aa1",
}

all_rows = []
for horizon in PREDICTION_HORIZONS:
    # From model.py
    for mname in ["Logistic Regression", "XGBoost"]:
        if horizon in prior and mname in prior[horizon]:
            r = prior[horizon][mname]
            all_rows.append({"Model": mname, "Horizon": f"{horizon}d",
                             "ROC_AUC": r["ROC_AUC"], "PR_AUC": r["PR_AUC"],
                             "F1_optimal": r.get("F1_optimal", np.nan),
                             "Brier": r["Brier"], "Note": "model.py"})
    # MLP
    if horizon in mlp_results:
        r = mlp_results[horizon]
        all_rows.append({"Model":"MLP (sklearn)","Horizon":f"{horizon}d",
                         "ROC_AUC":round(r["roc_auc"],4),"PR_AUC":round(r["pr_auc"],4),
                         "F1_optimal":round(f1_at_optimal(r["y_test"],r["prob_test"]),4),
                         "Brier":round(r["brier"],4),"Note":""})
    # LSTM
    if horizon in lstm_results:
        r = lstm_results[horizon]
        all_rows.append({"Model":"LSTM (PyTorch)","Horizon":f"{horizon}d",
                         "ROC_AUC":round(r["roc_auc"],4),"PR_AUC":round(r["pr_auc"],4),
                         "F1_optimal":round(f1_at_optimal(r["y_test"],r["prob_test"]),4),
                         "Brier":round(r["brier"],4),"Note":""})
    # LLM Zero-shot
    if horizon in llm_zeroshot_results and llm_zeroshot_results[horizon]:
        r = llm_zeroshot_results[horizon]
        all_rows.append({"Model":"LLM Zero-shot","Horizon":f"{horizon}d",
                         "ROC_AUC":round(r["roc_auc"],4),"PR_AUC":round(r["pr_auc"],4),
                         "F1_optimal":round(f1_at_optimal(r["y_test"],r["prob_test"]),4),
                         "Brier":round(r["brier"],4),"Note":f"n={len(r['prob_test'])}"})
    # LLM Agent
    if horizon in agent_results and agent_results[horizon]:
        r = agent_results[horizon]
        all_rows.append({"Model":"LLM Agent (ReAct)","Horizon":f"{horizon}d",
                         "ROC_AUC":round(r["roc_auc"],4),"PR_AUC":round(r["pr_auc"],4),
                         "F1_optimal":round(f1_at_optimal(r["y_test"],r["prob_test"]),4),
                         "Brier":round(r["brier"],4),"Note":f"n={r['n']}"})

comp_df = pd.DataFrame(all_rows)
comp_path = os.path.join(RESULTS_DIR, "grand_comparison.csv")
comp_df.to_csv(comp_path, index=False)
print(f"\n  Grand comparison table saved → {comp_path}")
print(comp_df.to_string(index=False))

# ── Bar chart ───────────────────────────────────────────────
fig, axes = plt.subplots(1, len(PREDICTION_HORIZONS),
                         figsize=(7*len(PREDICTION_HORIZONS), 6))
if len(PREDICTION_HORIZONS) == 1:
    axes = [axes]

for ax, horizon in zip(axes, PREDICTION_HORIZONS):
    sub     = comp_df[comp_df["Horizon"] == f"{horizon}d"]
    models  = [m for m in MODEL_ORDER if m in sub["Model"].values]
    rocs    = [sub[sub["Model"]==m]["ROC_AUC"].values[0] for m in models]
    prs     = [sub[sub["Model"]==m]["PR_AUC"].values[0]  for m in models]
    x       = np.arange(len(models))
    w       = 0.35
    bars1   = ax.bar(x-w/2, rocs, w, label="ROC-AUC",
                     color=[COLORS.get(m,"gray") for m in models], alpha=0.9)
    ax.bar(x+w/2, prs, w, label="PR-AUC",
           color=[COLORS.get(m,"gray") for m in models], alpha=0.45,
           edgecolor="black", linewidth=0.5)
    for bar, v in zip(bars1, rocs):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=22, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{horizon}-day Liquidation Horizon", fontsize=11)
    ax.set_ylabel("Score")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

plt.suptitle(
    "Grand Model Comparison — Aave Liquidation Risk\n"
    "Classical ML  →  Deep Learning  →  LLM Reasoning  →  Agentic AI",
    fontsize=11, y=1.01,
)
plt.tight_layout()
plot_path = os.path.join(RESULTS_DIR, "grand_comparison.png")
plt.savefig(plot_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Grand comparison chart saved → {plot_path}")

print("\n" + "="*65)
print("  models_advanced.py complete!")
models_run = ["MLP (sklearn)"]
if TORCH_AVAILABLE and lstm_results:  models_run.append("LSTM (PyTorch)")
if any(v for v in llm_zeroshot_results.values()): models_run.append("LLM Zero-shot")
if any(v for v in agent_results.values()):         models_run.append("LLM Agent")
print(f"  Models completed: {', '.join(models_run)}")
print(f"  Results → ./{RESULTS_DIR}/grand_comparison.csv")
print(f"  Chart   → ./{RESULTS_DIR}/grand_comparison.png")
print("="*65)
