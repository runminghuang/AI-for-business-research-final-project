"""
fetch_data.py
=============
Build a real Aave V2 account-date dataset from Dune's curated lending tables.

Why this version exists
-----------------------
The previous script depended on public Dune query IDs. Those IDs can disappear or
change ownership, which makes the pipeline brittle. This version talks directly to
the official Dune SQL API and queries curated tables instead:

  - lending.borrow
  - lending.supply

If no valid DUNE_API_KEY is available, the script still falls back to a realistic
synthetic dataset so the downstream pipeline remains runnable.

Usage
-----
  export DUNE_API_KEY="..."
  source .venv/bin/activate
  python fetch_data.py
"""

import os
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import requests


class tqdm:
    def __init__(self, iterable=None, total=None, desc="", unit=""):
        self.iterable = iterable
        self.total = total or (len(iterable) if iterable is not None else 0)
        self.desc = desc
        self.n = 0

    def __iter__(self):
        for item in self.iterable:
            yield item
            self.n += 1
            if self.total and self.n % max(1, self.total // 10) == 0:
                print(f"    {self.desc}: {self.n}/{self.total}", flush=True)

    def update(self, n=1):
        self.n += n

    def close(self):
        print(f"    {self.desc}: done ({self.n} records)", flush=True)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


DUNE_API_KEY = os.getenv("DUNE_API_KEY", "YOUR_DUNE_API_KEY_HERE")
START_DATE = os.getenv("DUNE_START_DATE", "2022-01-01")
END_DATE = os.getenv("DUNE_END_DATE", "2024-01-01")
OUTPUT_DIR = "data"
PROJECT = "aave"
VERSION = "2"
BLOCKCHAIN = "ethereum"
STABLECOIN_SYMBOLS = {"USDC", "USDT", "DAI", "FRAX", "LUSD", "TUSD", "USDP", "GUSD", "sUSD", "RAI"}

os.makedirs(OUTPUT_DIR, exist_ok=True)

DUNE_BASE = "https://api.dune.com/api/v1"
HEADERS = {
    "X-Dune-API-Key": DUNE_API_KEY,
    "Content-Type": "application/json",
}


def execute_sql(sql: str, timeout_s: int = 240) -> list[dict]:
    """Run SQL against Dune and return rows."""
    r = requests.post(
        f"{DUNE_BASE}/sql/execute",
        headers=HEADERS,
        json={"sql": sql, "performance": "free"},
        timeout=60,
    )
    r.raise_for_status()
    execution_id = r.json()["execution_id"]

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        s = requests.get(
            f"{DUNE_BASE}/execution/{execution_id}/status",
            headers=HEADERS,
            timeout=30,
        )
        s.raise_for_status()
        state = s.json()["state"]
        if state == "QUERY_STATE_COMPLETED":
            break
        if state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
            raise RuntimeError(f"Dune SQL failed: {state}")
        time.sleep(2)
    else:
        raise TimeoutError(f"Dune SQL did not finish in {timeout_s}s")

    res = requests.get(
        f"{DUNE_BASE}/execution/{execution_id}/results",
        headers=HEADERS,
        timeout=60,
    )
    res.raise_for_status()
    return res.json().get("result", {}).get("rows", [])


def fetch_real_dune_events() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch Aave V2 event tables from Dune curated lending schemas."""
    print("\n  Querying Dune curated lending tables...")
    # Use quarterly chunks to halve the number of API calls vs monthly.
    # Each chunk is at most ~3 months; still safe for Dune's result-size limits.
    month_starts = pd.date_range(START_DATE, END_DATE, freq="QS")
    if month_starts[-1] != pd.Timestamp(END_DATE):
        month_starts = pd.DatetimeIndex(list(month_starts) + [pd.Timestamp(END_DATE)])

    borrow_parts = []
    supply_parts = []
    borrow_path = os.path.join(OUTPUT_DIR, "raw_borrow_events.csv")
    supply_path = os.path.join(OUTPUT_DIR, "raw_supply_events.csv")

    for start, end in tqdm(list(zip(month_starts[:-1], month_starts[1:])), desc="  Dune monthly chunks"):
        start_s = start.strftime("%Y-%m-%d")
        end_s = end.strftime("%Y-%m-%d")

        sql_borrow = f"""
        SELECT
          CAST(block_time AS TIMESTAMP) AS block_time,
          borrower AS account,
          symbol,
          CAST(amount_usd AS DOUBLE) AS amount_usd,
          transaction_type,
          tx_hash
        FROM lending.borrow
        WHERE blockchain = '{BLOCKCHAIN}'
          AND project = '{PROJECT}'
          AND version = '{VERSION}'
          AND block_time >= TIMESTAMP '{start_s}'
          AND block_time <  TIMESTAMP '{end_s}'
          AND transaction_type IN ('borrow', 'repay', 'borrow_liquidation')
          AND borrower IS NOT NULL
        ORDER BY block_time
        """

        sql_supply = f"""
        SELECT
          CAST(block_time AS TIMESTAMP) AS block_time,
          depositor AS account,
          symbol,
          CAST(amount_usd AS DOUBLE) AS amount_usd,
          transaction_type,
          tx_hash
        FROM lending.supply
        WHERE blockchain = '{BLOCKCHAIN}'
          AND project = '{PROJECT}'
          AND version = '{VERSION}'
          AND block_time >= TIMESTAMP '{start_s}'
          AND block_time <  TIMESTAMP '{end_s}'
          AND transaction_type IN ('deposit', 'withdraw')
          AND depositor IS NOT NULL
        ORDER BY block_time
        """

        borrow_rows = execute_sql(sql_borrow, timeout_s=300)
        supply_rows = execute_sql(sql_supply, timeout_s=300)
        df_borrow_part = pd.DataFrame(borrow_rows)
        df_supply_part = pd.DataFrame(supply_rows)
        borrow_parts.append(df_borrow_part)
        supply_parts.append(df_supply_part)

        # Persist progress after each monthly chunk so partial work is not lost.
        if borrow_parts:
            pd.concat(borrow_parts, ignore_index=True).to_csv(borrow_path, index=False)
        if supply_parts:
            pd.concat(supply_parts, ignore_index=True).to_csv(supply_path, index=False)

    df_borrow = pd.concat(borrow_parts, ignore_index=True) if borrow_parts else pd.DataFrame()
    df_supply = pd.concat(supply_parts, ignore_index=True) if supply_parts else pd.DataFrame()
    print(f"    borrow table rows: {len(df_borrow):,}")
    print(f"    supply table rows: {len(df_supply):,}")

    for df_raw in (df_borrow, df_supply):
        df_raw["block_time"] = pd.to_datetime(df_raw["block_time"], utc=True).dt.tz_localize(None)
        df_raw["date"] = df_raw["block_time"].dt.floor("D")
        df_raw["amount_usd"] = pd.to_numeric(df_raw["amount_usd"], errors="coerce").fillna(0.0)
        df_raw["symbol"] = df_raw["symbol"].fillna("UNKNOWN")
        df_raw["account"] = df_raw["account"].str.lower()

    return df_borrow, df_supply


def fetch_eth_prices() -> pd.DataFrame:
    """Fetch daily ETH prices from Dune prices table; fall back to CoinGecko/synthetic."""
    print("\n  Fetching ETH daily prices...")
    try:
        sql = f"""
        SELECT
          CAST(timestamp AS DATE) AS date,
          AVG(price) AS eth_price_usd
        FROM prices.day
        WHERE blockchain = '{BLOCKCHAIN}'
          AND contract_address IS NULL
          AND symbol = 'ETH'
          AND timestamp >= TIMESTAMP '{START_DATE}'
          AND timestamp <  TIMESTAMP '{END_DATE}'
        GROUP BY 1
        ORDER BY 1
        """
        rows = execute_sql(sql, timeout_s=180)
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df["eth_price_usd"] = pd.to_numeric(df["eth_price_usd"], errors="coerce")
        print(f"    Dune prices rows: {len(df):,}")
    except Exception as e:
        print(f"    Dune prices unavailable ({e.__class__.__name__}) — trying CoinGecko...")
        try:
            url = (
                "https://api.coingecko.com/api/v3/coins/ethereum/market_chart"
                "?vs_currency=usd&days=730&interval=daily"
            )
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            df = pd.DataFrame(data["prices"], columns=["ts_ms", "eth_price_usd"])
            df["date"] = pd.to_datetime(df["ts_ms"], unit="ms").dt.floor("D")
            df = df.drop(columns=["ts_ms"])
        except Exception as inner:
            print(f"    CoinGecko unavailable ({inner.__class__.__name__}) — generating synthetic ETH prices.")
            rng = np.random.default_rng(42)
            n = (pd.Timestamp(END_DATE) - pd.Timestamp(START_DATE)).days
            dates = [pd.Timestamp(START_DATE) + timedelta(days=i) for i in range(n)]
            shocks = rng.normal(0.0003, 0.035, n)
            price = 2500 * np.exp(np.cumsum(shocks))
            df = pd.DataFrame({"date": dates, "eth_price_usd": price})

    df = df.sort_values("date").drop_duplicates("date").reset_index(drop=True)
    df["log_return"] = np.log(df["eth_price_usd"] / df["eth_price_usd"].shift(1))
    df["realized_vol_7d"] = df["log_return"].rolling(7).std() * np.sqrt(365)
    df["eth_return_7d"] = df["eth_price_usd"].pct_change(7)
    df = df.dropna().reset_index(drop=True)

    path = os.path.join(OUTPUT_DIR, "raw_eth_prices.csv")
    df.to_csv(path, index=False)
    print(f"    Saved {len(df):,} rows → {path}")
    return df


def make_synthetic_dataset(n: int = 10_000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    base = pd.Timestamp(START_DATE)
    days = (pd.Timestamp(END_DATE) - base).days

    obs_dates = [base + timedelta(days=int(d)) for d in rng.integers(0, days, n)]
    accounts = [f"0x{rng.integers(0, 5000):04x}" for _ in range(n)]

    collateral = rng.lognormal(mean=10.0, sigma=1.5, size=n)
    debt = collateral / rng.uniform(1.05, 5.0, size=n)
    hf = collateral / (debt + 1e-9)

    stbl_share = rng.beta(1, 4, size=n)
    n_col = rng.integers(1, 5, size=n)
    n_dbt = rng.integers(1, 4, size=n)
    n_dep = rng.integers(0, 20, size=n)
    n_bor = rng.integers(0, 15, size=n)
    n_rep = rng.integers(0, 15, size=n)
    n_wd = rng.integers(0, 10, size=n)
    acct_age = rng.integers(7, 700, size=n)

    eth_price = rng.lognormal(7.5, 0.4, size=n)
    eth_ret7 = rng.normal(0, 0.10, size=n)
    vol7 = np.abs(rng.normal(0.6, 0.3, size=n))

    log_odds = (
        -4.0
        + (-3.5 * np.log(np.clip(hf, 0.5, 10)))
        + (2.0 * vol7)
        + (0.8 * np.abs(eth_ret7))
        + (0.5 * (debt / (collateral + 1)))
        + rng.normal(0, 0.4, size=n)
    )
    prob = 1 / (1 + np.exp(-log_odds))
    label_7 = (rng.uniform(size=n) < prob).astype(int)
    label_14 = np.where(label_7 == 1, 1, (rng.uniform(size=n) < prob * 1.5).astype(int))

    return pd.DataFrame({
        "account": accounts,
        "obs_date": obs_dates,
        "collateral_value_usd": collateral,
        "total_debt_usd": debt,
        "collateralization_ratio": hf,
        "approx_health_factor": hf,
        "n_collateral_assets": n_col,
        "n_debt_assets": n_dbt,
        "stablecoin_debt_share": stbl_share,
        "n_deposits_30d": n_dep,
        "n_borrows_30d": n_bor,
        "n_repays_30d": n_rep,
        "n_withdraws_30d": n_wd,
        "total_tx_30d": n_dep + n_bor + n_rep + n_wd,
        "account_age_days": acct_age,
        "eth_price_usd": eth_price,
        "eth_return_7d": eth_ret7,
        "realized_vol_7d": vol7,
        "label_7d": label_7,
        "label_14d": label_14,
    })


def build_real_dataset(df_borrow: pd.DataFrame, df_supply: pd.DataFrame, df_eth: pd.DataFrame) -> pd.DataFrame:
    """Convert raw Dune events into an account-date dataset."""
    print("\n  Building account-date dataset from real Dune events...")

    df_borrow = df_borrow.copy()
    df_supply = df_supply.copy()

    df_liq = df_borrow[df_borrow["transaction_type"] == "borrow_liquidation"].copy()
    df_b = df_borrow[df_borrow["transaction_type"].isin(["borrow", "repay"])].copy()
    df_s = df_supply[df_supply["transaction_type"].isin(["deposit", "withdraw"])].copy()

    tx_frames = [
        df_s[["account", "date", "block_time", "symbol", "amount_usd", "transaction_type", "tx_hash"]].copy(),
        df_b[["account", "date", "block_time", "symbol", "amount_usd", "transaction_type", "tx_hash"]].copy(),
    ]
    all_tx = pd.concat(tx_frames, ignore_index=True)
    all_tx = all_tx.sort_values(["account", "block_time", "tx_hash"]).reset_index(drop=True)
    all_tx.to_csv(os.path.join(OUTPUT_DIR, "raw_transactions.csv"), index=False)

    liq_dates = (
        df_liq.groupby("account")["date"]
        .apply(lambda s: sorted(pd.to_datetime(s).dt.date.tolist()))
        .to_dict()
    )

    daily = (
        all_tx.groupby(["account", "date", "transaction_type"], observed=True)
        .agg(
            tx_count=("tx_hash", "nunique"),
            amount_usd=("amount_usd", "sum"),
        )
        .reset_index()
    )
    wide = daily.pivot_table(
        index=["account", "date"],
        columns="transaction_type",
        values=["tx_count", "amount_usd"],
        aggfunc="sum",
        fill_value=0.0,
    )
    wide.columns = [f"{a}_{b}" for a, b in wide.columns]
    wide = wide.reset_index().sort_values(["account", "date"]).reset_index(drop=True)

    for col in [
        "tx_count_deposit", "tx_count_withdraw", "tx_count_borrow", "tx_count_repay",
        "amount_usd_deposit", "amount_usd_withdraw", "amount_usd_borrow", "amount_usd_repay",
    ]:
        if col not in wide.columns:
            wide[col] = 0.0

    wide["net_collateral_change_usd"] = wide["amount_usd_deposit"] + wide["amount_usd_withdraw"]
    wide["net_debt_change_usd"] = wide["amount_usd_borrow"] + wide["amount_usd_repay"]

    frames = []
    symbol_daily_supply = (
        df_s.groupby(["account", "date", "symbol"], observed=True)["amount_usd"]
        .sum()
        .reset_index()
        .sort_values(["account", "symbol", "date"])
    )
    symbol_daily_borrow = (
        df_b.groupby(["account", "date", "symbol"], observed=True)["amount_usd"]
        .sum()
        .reset_index()
        .sort_values(["account", "symbol", "date"])
    )

    stable_borrow_daily = (
        df_b[df_b["symbol"].isin(STABLECOIN_SYMBOLS)]
        .groupby(["account", "date"], observed=True)["amount_usd"]
        .sum()
        .reset_index()
        .rename(columns={"amount_usd": "stablecoin_debt_change_usd"})
    )

    pos_supply = symbol_daily_supply.copy()
    pos_supply["cum_amt"] = pos_supply.groupby(["account", "symbol"], observed=True)["amount_usd"].cumsum()
    pos_supply = pos_supply[pos_supply["cum_amt"] > 0]
    pos_supply = (
        pos_supply.groupby(["account", "date"], observed=True)["symbol"]
        .nunique()
        .reset_index()
        .rename(columns={"symbol": "n_collateral_assets"})
    )

    pos_borrow = symbol_daily_borrow.copy()
    pos_borrow["cum_amt"] = pos_borrow.groupby(["account", "symbol"], observed=True)["amount_usd"].cumsum()
    pos_borrow = pos_borrow[pos_borrow["cum_amt"] > 0]
    pos_borrow = (
        pos_borrow.groupby(["account", "date"], observed=True)["symbol"]
        .nunique()
        .reset_index()
        .rename(columns={"symbol": "n_debt_assets"})
    )

    first_seen = all_tx.groupby("account", observed=True)["date"].min().to_dict()

    for account, grp in tqdm(wide.groupby("account", sort=False), total=wide["account"].nunique(), desc="  Accounts"):
        grp = grp.sort_values("date").copy()
        grp["collateral_value_usd"] = grp["net_collateral_change_usd"].cumsum().clip(lower=0)
        grp["total_debt_usd"] = grp["net_debt_change_usd"].cumsum().clip(lower=0)
        grp["stablecoin_debt_change_usd"] = 0.0

        stable_part = stable_borrow_daily[stable_borrow_daily["account"] == account][["date", "stablecoin_debt_change_usd"]]
        if not stable_part.empty:
            grp = grp.merge(stable_part, on="date", how="left", suffixes=("", "_new"))
            if "stablecoin_debt_change_usd_new" in grp.columns:
                grp["stablecoin_debt_change_usd"] = grp["stablecoin_debt_change_usd_new"].fillna(0.0)
                grp = grp.drop(columns=["stablecoin_debt_change_usd_new"])
        grp["stablecoin_debt_usd"] = grp["stablecoin_debt_change_usd"].cumsum().clip(lower=0)

        grp["n_deposits_30d"] = grp["tx_count_deposit"].rolling(30, min_periods=1).sum()
        grp["n_withdraws_30d"] = grp["tx_count_withdraw"].rolling(30, min_periods=1).sum()
        grp["n_borrows_30d"] = grp["tx_count_borrow"].rolling(30, min_periods=1).sum()
        grp["n_repays_30d"] = grp["tx_count_repay"].rolling(30, min_periods=1).sum()
        grp["total_tx_30d"] = (
            grp["n_deposits_30d"] + grp["n_withdraws_30d"] + grp["n_borrows_30d"] + grp["n_repays_30d"]
        )

        grp["stablecoin_debt_share"] = np.where(
            grp["total_debt_usd"] > 0,
            np.clip(grp["stablecoin_debt_usd"] / grp["total_debt_usd"], 0, 1),
            0.0,
        )

        # Note: true Aave HF = Σ(collateral_i × liquidation_threshold_i) / debt
        # We use simple col/debt as an approximation (thresholds vary by asset ~0.80–0.87).
        grp["collateralization_ratio"] = np.where(
            grp["total_debt_usd"] > 0,
            grp["collateral_value_usd"] / grp["total_debt_usd"],
            5.0,
        )
        grp["collateralization_ratio"] = np.clip(grp["collateralization_ratio"], 0, 10)
        grp["approx_health_factor"] = grp["collateralization_ratio"].clip(upper=10)

        account_start = first_seen.get(account, grp["date"].min())
        grp["account_age_days"] = (grp["date"] - account_start).dt.days.clip(lower=0) + 1

        frames.append(grp)

    df = pd.concat(frames, ignore_index=True)
    df = df.merge(pos_supply, on=["account", "date"], how="left")
    df = df.merge(pos_borrow, on=["account", "date"], how="left")
    df["n_collateral_assets"] = df["n_collateral_assets"].fillna(1).astype(int)
    df["n_debt_assets"] = df["n_debt_assets"].fillna(1).astype(int)

    df = df.merge(
        df_eth[["date", "eth_price_usd", "eth_return_7d", "realized_vol_7d"]],
        on="date",
        how="left",
    )

    # ── Vectorized label assignment (replaces slow row-by-row apply) ──────────
    # Build a flat table of all liquidation events, then use a date-range merge.
    print("  Assigning future liquidation labels (vectorized)...")
    liq_records = [
        {"account": acct, "liq_date": pd.Timestamp(d)}
        for acct, dates in liq_dates.items()
        for d in dates
    ]
    if liq_records:
        liq_df = pd.DataFrame(liq_records)
        # Cross-join on account, then filter by day offset
        merged = df[["account", "date"]].merge(liq_df, on="account", how="left")
        merged["diff_days"] = (merged["liq_date"] - merged["date"]).dt.days
        for h in [7, 14]:
            flagged = (
                merged[(merged["diff_days"] > 0) & (merged["diff_days"] <= h)][["account", "date"]]
                .drop_duplicates()
                .assign(**{f"label_{h}d": 1})
            )
            df = df.merge(flagged, on=["account", "date"], how="left")
            df[f"label_{h}d"] = df[f"label_{h}d"].fillna(0).astype(int)
    else:
        df["label_7d"] = 0
        df["label_14d"] = 0

    df = df.rename(columns={"date": "obs_date"})
    keep_cols = [
        "account",
        "obs_date",
        "collateral_value_usd",
        "total_debt_usd",
        "collateralization_ratio",
        "approx_health_factor",
        "n_collateral_assets",
        "n_debt_assets",
        "stablecoin_debt_share",
        "n_deposits_30d",
        "n_borrows_30d",
        "n_repays_30d",
        "n_withdraws_30d",
        "total_tx_30d",
        "account_age_days",
        "eth_price_usd",
        "eth_return_7d",
        "realized_vol_7d",
        "label_7d",
        "label_14d",
    ]
    df = df[keep_cols].sort_values(["obs_date", "account"]).reset_index(drop=True)
    df = df.dropna(subset=["eth_price_usd", "eth_return_7d", "realized_vol_7d"])

    # Drop rows where both collateral and debt are zero (no active position).
    df = df[(df["collateral_value_usd"] > 0) | (df["total_debt_usd"] > 0)].reset_index(drop=True)

    # Drop rows where collateral=0 but debt>0: these are artifacts from accounts
    # that held positions before the study window.  Their cumulative collateral
    # starts at 0 within the window, making approx_health_factor = 0 — misleading
    # for the model.  Filtering them removes ~23 % of rows but improves label quality.
    n_before = len(df)
    df = df[~((df["collateral_value_usd"] == 0) & (df["total_debt_usd"] > 0))].reset_index(drop=True)
    print(f"  Removed {n_before - len(df):,} pre-window artifact rows (collateral=0/debt>0) "
          f"→ {len(df):,} rows remain")

    return df


if __name__ == "__main__":
    print("=" * 60)
    print("  Aave Liquidation Risk — Data Fetching (Dune SQL)")
    print("=" * 60)

    using_real_data = DUNE_API_KEY not in ("", "YOUR_DUNE_API_KEY_HERE")

    if using_real_data:
        print(f"\n  API key found — fetching real Aave V{VERSION} data from Dune...")
        print(f"  Study window: {START_DATE} → {END_DATE}")
        try:
            df_borrow, df_supply = fetch_real_dune_events()
            df_eth = fetch_eth_prices()
            df = build_real_dataset(df_borrow, df_supply, df_eth)
        except Exception as e:
            print(f"\n  Dune fetch failed: {e}")
            print("  Falling back to synthetic data for pipeline testing.")
            using_real_data = False

    if not using_real_data:
        print("\n  No valid DUNE_API_KEY detected.")
        print("  Generating high-quality synthetic dataset instead.\n")
        df = make_synthetic_dataset(n=10_000)
        df_eth = fetch_eth_prices()

    path = os.path.join(OUTPUT_DIR, "dataset.csv")
    df.to_csv(path, index=False)

    print(f"\n{'=' * 60}")
    print(f"  Done! dataset.csv → {len(df):,} rows × {len(df.columns)} cols")
    for h in [7, 14]:
        col = f"label_{h}d"
        if col in df.columns:
            pos = int(df[col].sum())
            print(f"  label_{h}d: {pos:,} positives / {len(df):,} ({pos / len(df) * 100:.2f}%)")
    print(f"  Output folder: ./{OUTPUT_DIR}/")
    print(f"{'=' * 60}")
    print("\n  Next: run model.py then models_advanced.py")
