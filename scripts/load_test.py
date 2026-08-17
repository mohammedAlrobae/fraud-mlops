"""
scripts/load_test.py

Load test script to generate alternating steady and burst traffic to the BentoML fraud detection service.
Handles temporary service downtime gracefully so alerts can be triggered and tested.
"""

import time
import random
import requests
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

RAW_DATA_PATH = "data/raw/paysim.csv"
PREDICT_URL = "http://localhost:3000/predict"
SAMPLE_SIZE = 3000
STEADY_BATCH = 30
BURST_BATCH = 20


def send_request(req_data: dict) -> bool:
    try:
        # Send payload with nested request key (standard BentoML schema)
        resp = requests.post(PREDICT_URL, json={"request": req_data}, timeout=5)
        if resp.status_code == 200:
            return True
        elif resp.status_code == 400:
            # Fallback to flat json
            resp2 = requests.post(PREDICT_URL, json=req_data, timeout=5)
            return resp2.status_code == 200
        else:
            return False
    except requests.exceptions.RequestException:
        # Service is down (simulated outage for alert testing)
        return False


def main():
    print(f"1. Loading test transactions (step > 355) from '{RAW_DATA_PATH}'...")
    
    usecols = ["step", "type", "amount", "oldbalanceOrg", "nameDest", "oldbalanceDest"]
    chunks = []
    for chunk in pd.read_csv(RAW_DATA_PATH, usecols=usecols, chunksize=200000):
        filtered = chunk[chunk["step"] > 355]
        if not filtered.empty:
            chunks.append(filtered)
            if sum(len(c) for c in chunks) >= 10000:
                break

    df_test = pd.concat(chunks, ignore_index=True)
    sample_n = min(SAMPLE_SIZE, len(df_test))
    sampled_df = df_test.sample(n=sample_n, random_state=42).reset_index(drop=True)
    print(f"   Sampled {sample_n} transactions for load testing.")

    requests_list = []
    for _, row in sampled_df.iterrows():
        requests_list.append({
            "step": int(row["step"]),
            "type": str(row["type"]),
            "amount": float(row["amount"]),
            "oldbalanceOrg": float(row["oldbalanceOrg"]),
            "nameDest": str(row["nameDest"]),
            "oldbalanceDest": float(row["oldbalanceDest"]),
        })

    total_sent = 0
    success_count = 0
    idx = 0
    start_time = time.time()

    print(f"\n2. Starting traffic generation against {PREDICT_URL}...\n")

    while idx < len(requests_list):
        # --- STEADY PHASE ---
        steady_items = requests_list[idx : idx + STEADY_BATCH]
        idx += len(steady_items)
        if steady_items:
            print(f"\n-- STEADY phase, {len(steady_items)} requests --", flush=True)
            for item in steady_items:
                ok = send_request(item)
                total_sent += 1
                if ok:
                    success_count += 1
                    print(".", end="", flush=True)
                else:
                    print("x", end="", flush=True)
                time.sleep(random.uniform(0.3, 1.0))
            print("", flush=True)

        if idx >= len(requests_list):
            break

        # --- BURST PHASE ---
        burst_items = requests_list[idx : idx + BURST_BATCH]
        idx += len(burst_items)
        if burst_items:
            print(f"\n-- BURST phase, {len(burst_items)} concurrent requests --", flush=True)
            with ThreadPoolExecutor(max_workers=BURST_BATCH) as executor:
                results = list(executor.map(send_request, burst_items))
            total_sent += len(burst_items)
            success_count += sum(results)
            burst_chars = "".join("*" if r else "x" for r in results)
            print(burst_chars, flush=True)
            time.sleep(random.uniform(1.0, 2.0))

    elapsed = time.time() - start_time
    print("\n" + "=" * 60)
    print("LOAD TEST COMPLETED")
    print(f"• Total requests sent : {total_sent}")
    print(f"• Successful (200 OK) : {success_count}")
    print(f"• Outage/Failed       : {total_sent - success_count}")
    print(f"• Total elapsed time  : {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
    print(f"• Average throughput  : {total_sent/elapsed:.2f} req/s")
    print("=" * 60)


if __name__ == "__main__":
    main()
