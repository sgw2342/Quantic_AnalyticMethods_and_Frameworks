#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
novabank_data_setup.py

Purpose
-------
Load and lightly clean the UCI bank marketing dataset (semicolon-delimited),
split into TRAIN/TEST with stratification, and build a preprocessing pipeline:
- Numeric features → StandardScaler (with_mean=False for sparse-friendliness)
- Categorical features → OneHotEncoder(handle_unknown="ignore", dense output)

Why it matters
--------------
Every downstream model (logistic, tree/ensemble) uses the SAME preprocessing,
which prevents data leakage and keeps comparisons fair.

Returns
-------
load_and_prepare(data_path, test_size=0.25, random_state=42) ->
    df                : original cleaned DataFrame (with 'target' column added)
    X_train, X_test   : feature matrices (raw/untransformed)
    y_train, y_test   : target Series (0/1)
    feature_cols      : list of feature column names used for modeling
    num_cols, cat_cols: numeric/categorical feature list
    preprocessor      : fitted ColumnTransformer to plug into sklearn Pipelines
"""
from typing import Tuple, List
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def load_and_prepare(
    data_path: str,
    test_size: float = 0.25,
    random_state: int = 42
):
    """Read dataset, normalize text fields, apply light validity checks,
    derive the binary target, and return split data and a fitted preprocessor.
    """
    # Read the UCI file (semicolon-separated CSV)
    df = pd.read_csv(data_path, sep=";")

    # Normalize object columns to stripped strings (avoid spurious levels)
    for c in df.select_dtypes(include="object").columns:
        df[c] = df[c].astype(str).str.strip()

    # Remove exact duplicate rows (protect against accidental duplication)
    df = df.drop_duplicates()

    # Light validity guard on age if present
    if "age" in df.columns:
        df = df[df["age"].between(18, 95)]

    # Binary target: marketing “yes” → 1; “no” → 0
    df["target"] = (df["y"] == "yes").astype(int)

    # Remove columns not available at score time (duration leaks target),
    # plus the original 'y' label and the derived 'target' itself.
    drop_cols = [c for c in ["y", "target", "duration"] if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # Split into X (features) and y (target)
    X = df[feature_cols].copy()
    y = df["target"].copy()

    # Stratified split preserves the positive rate in both folds
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Identify numeric and categorical columns for preprocessing
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

    # ColumnTransformer: scale numeric; one-hot encode categoricals
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ],
        # Force dense output to make downstream use simpler/consistent
        sparse_threshold=0.0
    )

    # Fit on TRAIN ONLY to avoid leakage
    preprocessor.fit(X_train)

    return df, X_train, X_test, y_train, y_test, feature_cols, num_cols, cat_cols, preprocessor
