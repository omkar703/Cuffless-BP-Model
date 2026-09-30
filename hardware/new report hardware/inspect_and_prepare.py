import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INPUT_FILE = "final_dataset_4.csv"
OUTPUT_FILE = "final_dataset_ready.csv"

def inspect_and_prepare():
    print("=" * 70)
    print("MAX30102 PPG HARDWARE DATA INSPECTION (ADAPTED COMPATIBILITY)")
    print("=" * 70)

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(f"'{INPUT_FILE}' not found in the current working directory.")

    df = pd.read_csv(INPUT_FILE)
    print(f"\nRaw samples loaded: {len(df)}")

    # Backward compatibility: generate expected_timestamp_ms from index
    if "expected_timestamp_ms" not in df.columns:
        print("[INFO] Generating 'expected_timestamp_ms' (sample_index * 10 ms timeline)")
        df["expected_timestamp_ms"] = df["sample_index"] * 10

    # Map existing timestamp_ms to host_timestamp_ms
    if "host_timestamp_ms" not in df.columns and "timestamp_ms" in df.columns:
        print("[INFO] Mapping 'timestamp_ms' -> 'host_timestamp_ms'")
        df["host_timestamp_ms"] = df["timestamp_ms"]

    required = [
        "sample_index",
        "expected_timestamp_ms",
        "host_timestamp_ms",
        "ir",
        "red",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # 1. SAMPLE INDEX INTEGRITY
    index_diff = df["sample_index"].diff().dropna()
    missing_index_count = int((index_diff != 1).sum())
    print("\n--- SAMPLE INDEX CHECK ---")
    print(f"Index discontinuities: {missing_index_count}")
    if missing_index_count == 0:
        print("[PASS] No sample-index gaps.")
    else:
        print("[WARN] Sample-index gaps detected.")

    # 2. EXPECTED 100-HZ TIMING
    expected_diff = df["expected_timestamp_ms"].diff().dropna()
    print("\n--- EXPECTED SENSOR TIMING ---")
    print(f"Mean interval: {expected_diff.mean():.3f} ms")
    if np.all(expected_diff == 10):
        print("[PASS] Expected timeline is exactly 100 Hz.")
    else:
        print("[WARN] Expected timestamp sequence is not exactly 10 ms.")

    # 3. HOST TIMING
    host_diff = df["host_timestamp_ms"].diff().dropna()
    print("\n--- HOST READ TIMING ---")
    print(f"Mean:   {host_diff.mean():.3f} ms")
    print(f"Median: {host_diff.median():.3f} ms")
    print(f"Min:    {host_diff.min():.3f} ms")
    print(f"Max:    {host_diff.max():.3f} ms")

    # 4. EFFECTIVE SAMPLE RATE FROM INDEX
    if len(df) > 1:
        duration_seconds = (df["sample_index"].iloc[-1] - df["sample_index"].iloc[0] + 1) / 100.0
        effective_hz = len(df) / duration_seconds
        print("\n--- EFFECTIVE SAMPLE RATE ---")
        print(f"Estimated rate: {effective_hz:.3f} Hz")
        if abs(effective_hz - 100.0) < 0.5:
            print("[PASS] Acquisition is consistent with 100 Hz.")
        else:
            print("[WARN] Acquisition deviates from 100 Hz.")

    # 5. IR QUALITY DESCRIPTIVE ONLY (No row deletion)
    ir = df["ir"].astype(float)
    print("\n--- IR SIGNAL ---")
    print(f"Mean:   {ir.mean():.2f}")
    print(f"Median: {ir.median():.2f}")
    print(f"Min:    {ir.min():.2f}")
    print(f"Max:    {ir.max():.2f}")
    print(f"Std:    {ir.std():.2f}")

    active_fraction = (ir > 40000).mean() * 100
    print(f"Samples above 40,000 counts: {active_fraction:.2f}%")

    # 6. RED CHANNEL
    red = df["red"].astype(float)
    print("\n--- RED CHANNEL ---")
    print(f"Mean: {red.mean():.2f}")
    print(f"Min:  {red.min():.2f}")
    print(f"Max:  {red.max():.2f}")
    if red.mean() < 1000:
        print("[PASS] Red channel appears inactive/ambient-noise dominated.")
    else:
        print("[WARN] Red channel has substantial amplitude.")

    # 7. EXPORT RAW FIVE-COLUMN DATASET WITHOUT INTERNAL ROW DELETION
    ready = df[required].copy()
    ready.to_csv(OUTPUT_FILE, index=False)
    print("\n--- EXPORT ---")
    print(f"Saved: {OUTPUT_FILE}")
    print(f"Rows retained: {len(ready)}")
    print("[PASS] Raw timing structure preserved.")

    # 8. PLOT RAW IR (FIRST 5 SECONDS)
    plot_df = ready.iloc[:500]
    plt.figure(figsize=(12, 5))
    plt.plot(
        plot_df["expected_timestamp_ms"] / 1000.0,
        plot_df["ir"],
        linewidth=1.0,
        color="#0055ff"
    )
    plt.xlabel("Time (seconds)")
    plt.ylabel("IR ADC Counts")
    plt.title("MAX30102 Raw IR PPG Waveform 100 Hz Acquisition")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("raw_ir_5sec.png", dpi=300)
    plt.close()
    print("Saved: raw_ir_5sec.png")
    print("\n[DONE]")

if __name__ == "__main__":
    inspect_and_prepare()