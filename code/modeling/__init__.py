"""
Phase 3A: PPG-Only Classical Baseline, Evaluation & Error Analysis.

Modules:
- data_loader: Window streamer and slice extractor from raw MAT files.
- features: Handcrafted feature extraction across Branches A, B, C and Groups A-F.
- preprocessing_pipeline: Leakage-safe scaling, imputation, and canonical HDF5 caching.
- classical_models: Classical baseline model definitions and training wrappers.
- evaluation: Window-weighted and record-weighted regression metrics.
- error_analysis: BP range, quality strata, error correlations, and feature ablations.
"""

__version__ = "1.0.0"
