"""
fetch_gaps.py
=============
Fill exactly the 5 missing quarters and rebuild dataset.csv.

Missing quarters:
  2021-01-01 → 2021-04-01  (Q1)
  2021-04-01 → 2021-07-01  (Q2)
  2021-07-01 → 2021-10-01  (Q3)
  2025-07-01 → 2025-10-01  (Q3)
  2025-10-01 → 2026-01-01  (Q4)
"""

import os, sys, time, importlib.util
import pandas as pd
import requests

DUNE_API_KEY = os.getenv("DUNE_API_KEY", "BLmzfm0d1QiW3WPYWK4IHlQVE1z290xO")
OUTPUT_DIR   = "data"
BLOCKCHAIN, PROJECT, VERSION = "ethereum", "aave", "2"

HEADERS = {
    "X-Dune-API-Key": DUNE_API_KEY,
    "Content-Type": "application/json",
}
BASE = "https://api.dune.com/api/v1"

MISSING_QUARTERS = [
    ("2021-01-01", "2021-04-01"),
    ("2021-04-01", "2021-07-01"),
    ("2021-07-01", "2021-10-01"),
    ("2025-07-01", "2025-10-01"),
    ("2025-10-01", "2026-01-01"),
]


def execute_sql(sql: str, timeout_s: int = 300) -> list[dict]:
    r = requests.post(f"{BASE}/sql/execute", headers=HEADERS,
                      json={"sql": sql, "performance": "free"}, timeout=60)
    r.raise_for_status()
    eid = r.json()["execution_id"]
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        s = requests.get(f"{BASE}/execution/{eid}/status", headers=HEADERS, timeout=30)
        s.raise_for_status()
        state = s.json()["state"]
        if state == "QUERY_STATE_COMPLETED":
            break
        if state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
            raise RuntimeError(f"Dune failed: {s.json().get('error',{}).get('message','')}")
        time.sleep(3)
    else:
        raise TimeoutError(f"Dune timeout after {timeout_s}s")
    res = requests.get(f"{BASE}/execution/{eid}/results", headers=HEADERS, timeout=60)
    res.raise_for_status()
    return res.json().get("result", {}).get("rows", [])


def fetch_quarter(qs: str, qe: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sql_b = f"""
    SELECT CAST(block_time AS TIMESTAMP) AS block_time, borrower AS account,
           symbol, CAST(amount_usd AS DOUBLE) AS amount_usd, transaction_type, tx_hash
    FROM lending.borrow
    WHERE blockchain='{BLOCKCHAIN}' AND project='{PROJECT}' AND version='{VERSION}'
      AND block_time >= TIMESTAMP '{qs}' AND block_time < TIMESTAMP '{qe}'
      AND transaction_type IN ('borrow','repay','borrow_liquidation')
      AND borrower IS NOT NULL
    ORDER BY block_time
    """
    sql_s = f"""
    SELECT CAST(block_time AS TIMESTAMP) AS block_time, depositor AS account,
           symbol, CAST(amount_usd AS DOUBLE) AS amount_usd, transaction_type, tx_hash
    FROM lending.supply
    WHERE blockchain='{BLOCKCHAIN}' AND project='{PROJECT}' AND version='{VERSION}'
      AND block_time >= TIMESTAMP '{qs}' AND block_time < TIMESTAMP '{qe}'
      AND transaction_type IN ('deposit','withdraw')
      AND depositor IS NOT NULL
    ORDER BY block_time
    """
    print(f"  borrow {qs}→{qe}...", flush=True)
    b = execute_sql(sql_b)
    print(f"    → {len(b):,} rows", flush=True)
    print(f"  supply {qs}→{qe}...", flush=True)
    s = execute_sql(sql_s)
    print(f"    → {len(s):,} rows", flush=True)
    return pd.DataFrame(b), pd.DataFrame(s)


def normalise(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["block_time"] = pd.to_datetime(df["block_time"], utc=True, errors="coerce").dt.tz_localize(None)
    df["date"]       = df["block_time"].dt.floor("D")
    df["amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce").fillna(0.0)
    df["symbol"]     = df["symbol"].fillna("UNKNOWN")
    df["account"]    = df["account"].str.lower()
    return df


print("=" * 65)
print("  fetch_gaps.py — filling 5 missing quarters")
print("=" * 65)

borrow_path = os.path.join(OUTPUT_DIR, "raw_borrow_events.csv")
supply_path = os.path.join(OUTPUT_DIR, "raw_supply_events.csv")

existing_b = pd.read_csv(borrow_path) if os.path.exists(borrow_path) else pd.DataFrame()
existing_s = pd.read_csv(supply_path) if os.path.exists(supply_path) else pd.DataFrame()
print(f"\n  Existing borrow: {len(existing_b):,} rows")
print(f"  Existing supply: {len(existing_s):,} rows")

new_b, new_s = [], []

for i, (qs, qe) in enumerate(MISSING_QUARTERS, 1):
    print(f"\n[{i}/{len(MISSING_QUARTERS)}] {qs} → {qe}")
    try:
        df_b, df_s = fetch_quarter(qs, qe)
        if not df_b.empty: new_b.append(normalise(df_b))
        if not df_s.empty: new_s.append(normalise(df_s))

        # Save progress after each quarter
        if new_b:
            cb = pd.concat([existing_b] + new_b, ignore_index=True)
            if "tx_hash" in cb.columns: cb = cb.drop_duplicates(subset=["tx_hash"])
            cb.to_csv(borrow_path, index=False)
        if new_s:
            cs = pd.concat([existing_s] + new_s, ignore_index=True)
            if "tx_hash" in cs.columns: cs = cs.drop_duplicates(subset=["tx_hash"])
            cs.to_csv(supply_path, index=False)

    except Exception as e:
        print(f"  WARNING: {qs}→{qe} failed ({e}) — skipping")

# Final combined
final_b = pd.concat([existing_b] + new_b, ignore_index=True) if new_b else existing_b
final_s = pd.concat([existing_s] + new_s, ignore_index=True) if new_s else existing_s
if "tx_hash" in final_b.columns: final_b = final_b.drop_duplicates(subset=["tx_hash"])
if "tx_hash" in final_s.columns: final_s = final_s.drop_duplicates(subset=["tx_hash"])
final_b.to_csv(borrow_path, index=False)
final_s.to_csv(supply_path, index=False)

print(f"\n  Final borrow: {len(final_b):,} rows")
print(f"  Final supply: {len(final_s):,} rows")

# Rebuild dataset.csv
print("\n  Rebuilding dataset.csv...")

os.environ["DUNE_START_DATE"] = "2021-01-01"
os.environ["DUNE_END_DATE"]   = "2026-01-01"
spec = importlib.util.spec_from_file_location("fetch_data", "fetch_data.py")
fd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fd)

df_eth = pd.read_csv(os.path.join(OUTPUT_DIR, "raw_eth_prices.csv"), parse_dates=["date"])

def norm2(df):
    df = df.copy()
    df["block_time"] = pd.to_datetime(df["block_time"], errors="coerce").dt.tz_localize(None)
    df["date"]       = df["block_time"].dt.floor("D")
    df["amount_usd"] = pd.to_numeric(df["amount_usd"], errors="coerce").fillna(0.0)
    df["symbol"]     = df["symbol"].fillna("UNKNOWN")
    df["account"]    = df["account"].str.lower()
    return df

dataset = fd.build_real_dataset(norm2(final_b), norm2(final_s), df_eth)
dataset.to_csv(os.path.join(OUTPUT_DIR, "dataset.csv"), index=False)

print(f"  dataset.csv: {len(dataset):,} rows")
print(f"  Date range : {dataset['obs_date'].min()} → {dataset['obs_date'].max()}")
print(f"  Accounts   : {dataset['account'].nunique():,}")
print(f"  label_7d   : {dataset['label_7d'].sum():,} positives ({dataset['label_7d'].mean()*100:.2f}%)")
print(f"  label_14d  : {dataset['label_14d'].sum():,} positives ({dataset['label_14d'].mean()*100:.2f}%)")

print("\n" + "=" * 65)
print("  fetch_gaps.py complete.")
print("=" * 65)
