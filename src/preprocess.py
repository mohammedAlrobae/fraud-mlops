"""
src/preprocess.py

Single script to preprocess the PaySim dataset and perform temporal data splitting.
Contains the reusable `build_features()` function used by both batch preprocessing and live serving.

Output files created in data/processed/:
  - train.csv       (~65% of rows, steps up to 65th pct)
  - validation.csv  (~15% of rows, steps 65th-80th pct)
  - test.csv        (~20% of rows, steps above 80th pct, includes isFraud)
"""

import gc
import os
import resource
import pandas as pd

RAW_DATA_PATH = "data/raw/paysim.csv"
PROCESSED_DIR = "data/processed"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Takes a raw-shaped dataframe with columns: step, type, amount,
    oldbalanceOrg, nameDest, oldbalanceDest.
    Returns the engineered, one-hot-encoded feature dataframe -
    exactly the same transformation used for both batch training
    data and live serving requests, to avoid train-serving skew.
    """
    res = df.copy()

    # Engineer new features
    res["isMerchantDest"] = res["nameDest"].astype(str).str.startswith("M").astype(int)
    res["hourOfDay"] = res["step"] % 24

    # Drop non-feature, leaky, and identifier columns if present
    cols_to_drop = [
        "step",
        "nameDest",
        "nameOrig",
        "isFlaggedFraud",
        "newbalanceOrig",
        "newbalanceDest",
        "isFraud",
        "transaction_id",
    ]
    existing_drops = [c for c in cols_to_drop if c in res.columns]
    res = res.drop(columns=existing_drops)

    # One-hot encode 'type' column using fixed categories to ensure single-row inference produces identical dummy columns
    type_categories = pd.CategoricalDtype(categories=["CASH_IN", "CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"])
    res["type"] = res["type"].astype(type_categories)
    res = pd.get_dummies(res, columns=["type"], drop_first=True, dtype=int)

    return res


def main():
    # 1. Load raw data with explicit dtypes to reduce memory usage, add transaction_id BEFORE dropping/splitting
    print(f"1. Loading raw dataset from '{RAW_DATA_PATH}'...")
    raw_dtypes = {
        "step": "int32",
        "amount": "float32",
        "oldbalanceOrg": "float32",
        "newbalanceOrig": "float32",
        "oldbalanceDest": "float32",
        "newbalanceDest": "float32",
        "isFraud": "int8",
        "isFlaggedFraud": "int8",
    }
    df_raw = pd.read_csv(RAW_DATA_PATH, dtype=raw_dtypes)
    total = len(df_raw)
    print(f"   Loaded {total:,} rows.")

    print("2. Adding 'transaction_id' column from row index...")
    df_raw["transaction_id"] = df_raw.index

    # 2. Split into train/val/test chronologically using ORIGINAL step column
    print("3. Performing temporal split by original 'step' column (row-weighted quantiles)...")
    cutoff_val  = df_raw["step"].quantile(0.65)
    cutoff_test = df_raw["step"].quantile(0.80)
    print(f"   Train/Validation cutoff (65th pct): step <= {cutoff_val}")
    print(f"   Validation/Test cutoff  (80th pct): step <= {cutoff_test}")

    raw_train = df_raw[df_raw["step"] <= cutoff_val].copy()
    raw_val   = df_raw[(df_raw["step"] > cutoff_val) & (df_raw["step"] <= cutoff_test)].copy()
    raw_test  = df_raw[df_raw["step"] > cutoff_test].copy()

    # Free df_raw immediately after splitting
    del df_raw
    gc.collect()

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    train_path = os.path.join(PROCESSED_DIR, "train.csv")
    val_path   = os.path.join(PROCESSED_DIR, "validation.csv")
    test_path  = os.path.join(PROCESSED_DIR, "test.csv")

    def assemble_split(raw_sub_df: pd.DataFrame) -> pd.DataFrame:
        feats = build_features(raw_sub_df)
        out = feats.copy()
        out.insert(0, "step", raw_sub_df["step"].values)
        out.insert(0, "transaction_id", raw_sub_df["transaction_id"].values)
        if "isFraud" in raw_sub_df.columns:
            out["isFraud"] = raw_sub_df["isFraud"].values
        return out

    # 4. Process and save each split one at a time
    print("4. Building features, saving CSVs, and releasing split memory sequentially...")

    print(f"   -> Processing and saving {train_path}...")
    train_df = assemble_split(raw_train)
    del raw_train
    gc.collect()
    train_stats = (len(train_df), train_df['isFraud'].mean(), train_df['isFraud'].sum())
    train_df.to_csv(train_path, index=False)
    del train_df
    gc.collect()

    print(f"   -> Processing and saving {val_path}...")
    val_df = assemble_split(raw_val)
    del raw_val
    gc.collect()
    val_stats = (len(val_df), val_df['isFraud'].mean(), val_df['isFraud'].sum())
    val_df.to_csv(val_path, index=False)
    del val_df
    gc.collect()

    print(f"   -> Processing and saving {test_path}...")
    test_df = assemble_split(raw_test)
    del raw_test
    gc.collect()
    test_stats = (len(test_df), test_df['isFraud'].mean(), test_df['isFraud'].sum())
    test_df.to_csv(test_path, index=False)
    del test_df
    gc.collect()

    # 5. Print summary
    print("\n" + "=" * 58)
    print("              PREPROCESSING SUMMARY")
    print("=" * 58)
    print(f"  Cutoff Train/Val  (65th pct row-weighted): step <= {cutoff_val}")
    print(f"  Cutoff Val/Test   (80th pct row-weighted): step <= {cutoff_test}")
    print("-" * 58)
    print(f"  {'Set':<12} {'Rows':>12}  {'% of Total':>10}  {'Fraud Rate':>10}  {'Fraud Rows':>10}")
    print("-" * 58)
    print(f"  {'Train':<12} {train_stats[0]:>12,}  {train_stats[0]/total:>10.2%}  {train_stats[1]:>10.4%}  {train_stats[2]:>10,}")
    print(f"  {'Validation':<12} {val_stats[0]:>12,}  {val_stats[0]/total:>10.2%}  {val_stats[1]:>10.4%}  {val_stats[2]:>10,}")
    print(f"  {'Test':<12} {test_stats[0]:>12,}  {test_stats[0]/total:>10.2%}  {test_stats[1]:>10.4%}  {test_stats[2]:>10,}")
    print("-" * 58)
    print(f"  {'TOTAL':<12} {total:>12,}  {'100.00%':>10}")
    print("=" * 58)
    print("✅ Preprocessing and temporal splitting completed successfully!")

    # 6. Print peak memory usage
    print(f"Peak memory: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")


if __name__ == "__main__":
    main()
