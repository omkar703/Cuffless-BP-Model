import os
import pandas as pd
import matplotlib.pyplot as plt

def run_pipeline(input_filename="final_dataset_4.csv", output_filename="final_dataset_ready.csv"):
    print("=" * 60)
    print("   MAX30102 PPG PIPELINE: INSPECTION & SANITIZATION")
    print("=" * 60)

    # 1. LOAD FILE
    if not os.path.exists(input_filename):
        print(f"[ERROR] '{input_filename}' not found.")
        print("Make sure the script is running in the same directory as your CSV file.")
        return

    df = pd.read_csv(input_filename)
    total_raw_samples = len(df)
    print(f"\n[FILE LOADED] Raw sample count: {total_raw_samples}")

    # 2. PRACTICAL HARDWARE INSPECTION
    print("\n--- STAGE 1: HARDWARE & TIMING QUALITY CHECK ---")
    
    # Timing calculation
    df['delta_t'] = df['timestamp_ms'].diff()
    mean_interval = df['delta_t'].mean()
    effective_hz = 1000.0 / mean_interval if mean_interval > 0 else 0
    jitter_min = df['delta_t'].min()
    jitter_max = df['delta_t'].max()

    print(f"1. Sampling Interval : {mean_interval:.2f} ms (Effective: {effective_hz:.2f} Hz)")
    print(f"   Jitter Range      : {jitter_min:.1f} ms to {jitter_max:.1f} ms")
    if 9.0 <= mean_interval <= 11.0:
        print("   -> STATUS: [PASS] Sampling rate matches 100 Hz target.")
    else:
        print("   -> STATUS: [WARN] Sampling rate deviates from 100 Hz.")

    # Red LED check
    red_mean = df['red'].mean()
    print(f"2. Red LED Amplitude : {red_mean:.1f} counts (Mean)")
    if red_mean < 1000:
        print("   -> STATUS: [PASS] Red channel inactive (ambient noise only).")
    else:
        print("   -> STATUS: [WARN] Red channel is actively firing.")

    # Active IR signal check
    active_samples = df[df['ir'] >= 40000]
    coverage = (len(active_samples) / total_raw_samples) * 100
    print(f"3. Active Signal     : {coverage:.1f}% ({len(active_samples)} / {total_raw_samples} samples)")
    if coverage >= 90.0:
        print("   -> STATUS: [PASS] Stable finger contact maintained.")
    else:
        print("   -> STATUS: [WARN] Significant finger disconnects detected.")

    # 3. DATA CLEANING & EXPORT
    print("\n--- STAGE 2: SANITIZATION & EXPORT ---")
    
    # Drop samples captured before stable finger contact (under 50k counts)
    clean_df = df[df['ir'] >= 50000].copy().reset_index(drop=True)
    
    # Re-index samples and re-zero millisecond timestamps from first valid sample
    clean_df['timestamp_ms'] = clean_df['timestamp_ms'] - clean_df['timestamp_ms'].iloc[0]
    clean_df['sample_index'] = clean_df.index
    
    # Re-calculate timing on clean segment
    clean_interval = clean_df['timestamp_ms'].diff().mean()
    clean_hz = 1000.0 / clean_interval if clean_interval > 0 else 0

    # Save to CSV (keeping required columns)
    export_columns = ['sample_index', 'timestamp_ms', 'ir', 'red']
    clean_df[export_columns].to_csv(output_filename, index=False)
    
    print(f"Sanitized rows exported : {len(clean_df)} (dropped {total_raw_samples - len(clean_df)} noise samples)")
    print(f"Final Sampling Rate     : {clean_hz:.2f} Hz ({clean_interval:.2f} ms avg interval)")
    print(f"Saved file destination  : {output_filename}")

    # 4. MORPHOLOGY PLOTTING
    print("\n--- STAGE 3: GENERATING WAVEFORM PLOT ---")
    plt.figure(figsize=(11, 6))

    # Zoom into a clean 5-second slice (500 samples at 100 Hz)
    window_start = 100
    window_end = min(window_start + 500, len(clean_df))
    plot_segment = clean_df.iloc[window_start:window_end]

    plt.subplot(2, 1, 1)
    plt.plot(plot_segment['timestamp_ms'], plot_segment['ir'], color='#0055ff', linewidth=1.2)
    plt.title('Sanitized DC-Coupled IR Pulse Waveform (5-Second Representative Window)')
    plt.ylabel('Amplitude (ADC Counts)')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(2, 1, 2)
    plt.plot(plot_segment['timestamp_ms'], plot_segment['red'], color='#777777', linewidth=1.0)
    plt.title('Red Channel Baseline (Ambient Noise Verification)')
    plt.xlabel('Timestamp (ms)')
    plt.ylabel('Amplitude (ADC Counts)')
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_filename = "inspection_and_clean_waveform.png"
    plt.savefig(plot_filename, dpi=300)
    print(f"Plot saved successfully as : {plot_filename}")
    print("\n[COMPLETE] Dataset is validated, sanitized, and ready for model training.")

if __name__ == "__main__":
    run_pipeline("final_dataset_4.csv", "final_dataset_ready.csv")