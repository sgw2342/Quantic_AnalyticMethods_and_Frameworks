#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
novabank_rule_baseline.py

Purpose
-------
Simple, transparent baseline policy:
  Contact (1) only if the prior campaign outcome ('poutcome') was 'success'.
Otherwise do not contact (0).

Why this matters
----------------
Provides an easy-to-understand yardstick for model ROI (precision will be high,
coverage will be low).
"""
import numpy as np
import pandas as pd


def rule_predict_prior_success(X: pd.DataFrame) -> np.ndarray:
    """Return a 0/1 array: 1 if poutcome == 'success', else 0.
    If the column is missing, default to all zeros (no contacts)."""
    if "poutcome" not in X.columns:
        return np.zeros(len(X), dtype=int)
    return (X["poutcome"].astype(str).str.strip().str.lower() == "success").astype(int).values
