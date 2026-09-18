#!/usr/bin/env python3
"""
Phase 4A Runner: Neural PPG Baseline (1D CNN).

Usage:
    conda run -n ppg_bp python code/scripts/run_phase4a.py --smoke-test
    conda run -n ppg_bp python code/scripts/run_phase4a.py --full-run
    conda run -n ppg_bp python code/scripts/run_phase4a.py --full-run --model-a-only
    conda run -n ppg_bp python code/scripts/run_phase4a.py --full-run --model-b-only
"""

import argparse
import sys
import time
from pathlib import Path

# Ensure project code root is on sys.path
SCRIPT_DIR = Path(__file__).resolve().parent
CODE_DIR = SCRIPT_DIR.parent
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import torch
from phase4a.train import (
    run_smoke_test,
    train_one_model,
    write_reports,
    LOG_DIR,
    PHASE3A_SBP_MAE,
    PHASE3A_DBP_MAE,
)
from phase4a.utils import set_seed, setup_logger, gpu_memory_summary


def main():
    parser = argparse.ArgumentParser(description="Phase 4A: Neural PPG Baseline")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run smoke test only (small subset, verify pipeline)")
    parser.add_argument("--full-run", action="store_true",
                        help="Run full training on all data")
    parser.add_argument("--model-a-only", action="store_true",
                        help="Train only Model A (PPG only)")
    parser.add_argument("--model-b-only", action="store_true",
                        help="Train only Model B (PPG+VPG+APG)")
    args = parser.parse_args()

    if not args.smoke_test and not args.full_run:
        parser.print_help()
        sys.exit(1)

    logger = setup_logger(
        "phase4a",
        log_dir=LOG_DIR,
        log_filename="phase4a_training.log",
    )

    set_seed(42)

    # Device detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"Initial {gpu_memory_summary()}")

    # Print run summary
    logger.info("=" * 70)
    logger.info("PHASE 4A — Neural PPG Baseline")
    logger.info("=" * 70)
    logger.info(f"Mode: {'SMOKE TEST' if args.smoke_test else 'FULL RUN'}")
    logger.info(f"Phase 3A baselines: SBP={PHASE3A_SBP_MAE} mmHg, DBP={PHASE3A_DBP_MAE} mmHg")

    # =====================================================================
    # Smoke test (mandatory before full run)
    # =====================================================================
    if args.smoke_test or args.full_run:
        passed = run_smoke_test(device=device, logger=logger)
        if not passed:
            logger.error("SMOKE TEST FAILED. Aborting.")
            sys.exit(1)

    if args.smoke_test and not args.full_run:
        logger.info("Smoke test complete. Exiting (use --full-run to train).")
        return

    # =====================================================================
    # Full training
    # =====================================================================
    run_model_a = not args.model_b_only
    run_model_b = not args.model_a_only

    t0 = time.time()

    model_a = None
    metrics_val_a = None
    metrics_test_a = None
    train_losses_a, val_losses_a = [], []

    model_b = None
    metrics_val_b = None
    metrics_test_b = None
    train_losses_b, val_losses_b = [], []

    if run_model_a:
        metrics_val_a, metrics_test_a, train_losses_a, val_losses_a, model_a = train_one_model(
            model_name="model_a_ppg_only",
            n_channels=1,
            device=device,
            logger=logger,
        )

    if run_model_b:
        metrics_val_b, metrics_test_b, train_losses_b, val_losses_b, model_b = train_one_model(
            model_name="model_b_ppg_vpg_apg",
            n_channels=3,
            device=device,
            logger=logger,
        )

    # Write reports and figures (both models required)
    if run_model_a and run_model_b:
        write_reports(
            metrics_val_a, metrics_test_a,
            metrics_val_b, metrics_test_b,
            model_a, model_b,
            device, logger,
        )

    total_elapsed = time.time() - t0

    # =====================================================================
    # Final Summary
    # =====================================================================
    logger.info("\n" + "=" * 70)
    logger.info("=== PHASE 4A COMPLETE ===")
    logger.info("=" * 70)

    if run_model_a and metrics_test_a:
        comb_a = (metrics_test_a["sbp"]["mae"] + metrics_test_a["dbp"]["mae"]) / 2
        logger.info("PPG-only (Model A):")
        logger.info(f"  SBP MAE = {metrics_test_a['sbp']['mae']:.2f} mmHg")
        logger.info(f"  DBP MAE = {metrics_test_a['dbp']['mae']:.2f} mmHg")
        logger.info(f"  Combined MAE = {comb_a:.2f} mmHg")

    if run_model_b and metrics_test_b:
        comb_b = (metrics_test_b["sbp"]["mae"] + metrics_test_b["dbp"]["mae"]) / 2
        logger.info("PPG+VPG+APG (Model B):")
        logger.info(f"  SBP MAE = {metrics_test_b['sbp']['mae']:.2f} mmHg")
        logger.info(f"  DBP MAE = {metrics_test_b['dbp']['mae']:.2f} mmHg")
        logger.info(f"  Combined MAE = {comb_b:.2f} mmHg")

    logger.info("Phase 3A:")
    logger.info(f"  SBP MAE = {PHASE3A_SBP_MAE}")
    logger.info(f"  DBP MAE = {PHASE3A_DBP_MAE}")

    if run_model_a and metrics_test_a:
        logger.info(f"Best validation model (A): epoch {model_a._best_epoch}")
    if run_model_b and metrics_test_b:
        logger.info(f"Best validation model (B): epoch {model_b._best_epoch}")

    logger.info("Test-set evaluation performed exactly once: YES")
    logger.info(f"Total elapsed: {total_elapsed / 60:.1f} min")
    logger.info("=" * 70)
    logger.info("Phase 4A outputs: code/outputs/phase4a/")
    logger.info("DO NOT start GRU/LSTM/Transformer until reviewing Phase 4A report.")


if __name__ == "__main__":
    main()
