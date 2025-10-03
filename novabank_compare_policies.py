#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
novabank_compare_policies.py

Compare:
- Logistic Regression (profit-weighted or not)
- Gradient Boosting (profit-weighted or not)
- Gradient Boosting (profit-weighted + calibrated via prefit split)
- XGB (profit-weighted + calibrated via prefit split)
- Pruned Decision Tree (profit-weighted or not)
- Rule baseline and Do-Nothing

Why profit-weighted?
--------------------
We weight TRAIN samples by your economics:
  weight(y=1)  ~ value_if_positive   (FN costs ≈ value missed)
  weight(y=0)  ~ offer_cost          (FP costs ≈ wasted offer)
This nudges the learner to reduce €-expensive mistakes.

Calibration
-----------
We calibrate a profit-weighted GB using CalibratedClassifierCV(cv="prefit"):
- Split TRAIN into inner TRAIN/CAL sets.
- Fit GB on inner TRAIN with sample_weight.
- Calibrate on CAL (no weights needed).
This keeps the profit-weighted training intact and improves probability reliability.

What’s included
---------------
- Profit-max thresholding (sweeps thresholds and picks the best net €)
- OR Capacity-aware targeting (top-K by score across the TEST set)
- ROC/PR curves + AUC/AP table (discrimination)
- Calibration plots (reliability + score histograms; confidence)
- Optional SHAP reason-code exports for tree models (GB/DT/XGB) with plain language option

Why it matters
--------------
Gives executives the clearest picture of which policy yields the most net euros
under current economics — with diagnostics that show both accuracy and trust.
"""

from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBClassifier

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import (
    confusion_matrix,
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
)

# Optional SHAP (tree models). Script works without SHAP.
try:
    import shap
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

from novabank_data_setup import load_and_prepare
from novabank_rule_baseline import rule_predict_prior_success

# Helper to deal with BaseScore Float bug

def _normalize_xgb_base_score(xgb_pipeline):
    """
    Some XGBoost versions store base_score as a bracketed string like "[1.125E-1]".
    This converts it to a plain float in the booster config so downstream tools don't crash.
    """
    try:
        import json
        booster = xgb_pipeline.named_steps["xgb"].get_booster()
        cfg = json.loads(booster.save_config())
        bs = cfg.get("learner", {}).get("learner_model_param", {}).get("base_score", None)
        if isinstance(bs, str) and bs.startswith("[") and bs.endswith("]"):
            # Strip brackets and cast to float
            cfg["learner"]["learner_model_param"]["base_score"] = float(bs.strip("[]"))
            booster.load_config(json.dumps(cfg))
    except Exception as e:
        print(f"NOTE: could not normalize XGB base_score (safe to ignore if all runs succeed): {e}")

# ---------- Helpers: profit math ----------
def total_net_with_fn_penalty(y_true, y_pred, offer_cost, value_if_positive) -> Tuple[float, Tuple[int,int,int,int]]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    net = tp*(value_if_positive - offer_cost) - fp*offer_cost - fn*value_if_positive
    return float(net), (int(tp), int(fp), int(fn), int(tn))

def added_net_vs_do_nothing(y_true, y_pred, offer_cost, value_if_positive) -> float:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return float(tp*value_if_positive - (tp+fp)*offer_cost)

def sweep_thresholds(y_true, scores, offer_cost, value_if_positive, steps=199) -> pd.DataFrame:
    thresholds = np.round(np.linspace(0.005, 0.995, steps), 3)
    rows = []
    for thr in thresholds:
        y_pred = (scores >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        targeted = int(y_pred.sum())
        precision = float(tp/targeted) if targeted>0 else 0.0
        recall = float(tp/(tp+fn)) if (tp+fn)>0 else 0.0
        net = tp*(value_if_positive - offer_cost) - fp*offer_cost - fn*value_if_positive
        added = tp*value_if_positive - (tp+fp)*offer_cost
        rows.append({
            "threshold": float(thr),
            "targeted": targeted,
            "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "precision": precision, "recall": recall,
            "net_profit_eur": float(net),
            "added_vs_do_nothing_eur": float(added),
        })
    return pd.DataFrame(rows).sort_values("net_profit_eur", ascending=False).reset_index(drop=True)


# ---------- Sample weights ----------
def build_profit_weights(y: np.ndarray, offer_cost: float, value_if_positive: float, normalize: str = "avg") -> np.ndarray:
    """Return per-sample weights aligned to € economics."""
    y = np.asarray(y).reshape(-1)
    w_pos = float(value_if_positive)
    w_neg = float(offer_cost)
    w = np.where(y==1, w_pos, w_neg).astype(float)
    if normalize == "avg":
        w = w / np.mean(w)
    elif normalize == "sum":
        w = w * (len(w) / np.sum(w))
    # normalize=="none": leave as-is
    return w


# ---------- Diagnostics ----------
def build_auc_ap_table(y_true, model_scores: dict) -> pd.DataFrame:
    rows = []
    for name, s in model_scores.items():
        rows.append({
            "model": name,
            "roc_auc": float(roc_auc_score(y_true, s)),
            "avg_precision": float(average_precision_score(y_true, s))
        })
    return pd.DataFrame(rows).sort_values("roc_auc", ascending=False).reset_index(drop=True)

def plot_roc_curves(y_true, model_scores: dict, path: Path):
    plt.figure()
    for name, s in model_scores.items():
        fpr, tpr, _ = roc_curve(y_true, s); auc = roc_auc_score(y_true, s)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0,1],[0,1],"--",label="Random chance")
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC — TEST"); plt.legend(loc="lower right"); plt.tight_layout()
    plt.savefig(path, bbox_inches="tight"); plt.close()

def plot_pr_curves(y_true, model_scores: dict, path: Path):
    prevalence = float(np.mean(y_true))
    plt.figure()
    for name, s in model_scores.items():
        prec, rec, _ = precision_recall_curve(y_true, s); ap = average_precision_score(y_true, s)
        plt.plot(rec, prec, label=f"{name} (AP={ap:.3f})")
    plt.hlines(prevalence, 0, 1, linestyles="--", label="Prevalence")
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall — TEST")
    plt.legend(loc="upper right"); plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()

def plot_calibration(y_true, scores, model_name, out_path: Path, n_bins=10):
    frac_pos, mean_pred = calibration_curve(y_true, scores, n_bins=n_bins, strategy="uniform")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot([0,1],[0,1],"--",label="Perfect calibration")
    ax[0].plot(mean_pred, frac_pos, marker="o", label=model_name)
    ax[0].set_xlabel("Predicted probability"); ax[0].set_ylabel("Observed frequency")
    ax[0].set_title("Reliability"); ax[0].legend(loc="best")
    ax[1].hist(scores, bins=20, edgecolor="white")
    ax[1].set_xlabel("Predicted probability"); ax[1].set_ylabel("Count")
    ax[1].set_title("Score distribution")
    fig.suptitle(f"Calibration — {model_name}")
    plt.tight_layout(); plt.savefig(out_path, bbox_inches="tight"); plt.close()

def _get_feature_names_from_preprocessor(preprocessor) -> List[str]:
    try:
        return [str(n) for n in preprocessor.get_feature_names_out()]
    except Exception:
        return [f"feature_{i}" for i in range(getattr(preprocessor, "n_features_in_", 0))]



def _explain_tree_model(model_name: str,
                        pipe: Pipeline,
                        preprocessor,
                        X_raw: pd.DataFrame,
                        scores: np.ndarray,
                        explain_top_n: int,
                        out_path: Path,
                        top_features_per_customer: int = 5,
                        plain_language: bool = False,
                        plain_out_path: Optional[Path] = None) -> Optional[Path]:
    """Export top-N customer reason codes for a tree model (GB/DT) using SHAP.
       - Writes a detailed CSV of top features with signed SHAP contributions (log-odds).
       - If plain_language=True, also writes a '..._plain.csv' with human-friendly narratives.
    """
    if not _HAS_SHAP:
        return None

    # Transform exactly as the model sees the data
    X_trans = preprocessor.transform(X_raw)

    # ColumnTransformer feature names → list[str]
    feature_names = _get_feature_names_from_preprocessor(preprocessor)
    feature_names = [str(f) for f in feature_names]

    # Find fitted estimator explicitly
    est = pipe.named_steps.get("gb", None)
    if est is None:
        est = pipe.named_steps.get("dt", None)
    if est is None:
        est = pipe.named_steps.get("xgb", None)
    if est is None:
        return None

    # SHAP for tree models: use raw/log-odds output (stable) and convert to prob for display
    explainer = shap.TreeExplainer(est, feature_names=feature_names, model_output="raw")
    shap_vals = explainer.shap_values(X_trans)

    # Handle binary-class API variants; force shape (n_samples, n_features)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[-1]
    shap_vals = np.asarray(shap_vals)
    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(-1, 1)

    # expected_value can be scalar or vector; pick positive-class/base scalar
    base = np.asarray(explainer.expected_value).reshape(-1)
    base_val = float(base[-1])

    # Ensure scores is 1D float array
    scores = np.asarray(scores, dtype=float).reshape(-1)

    # Guard top-N selection
    N = int(min(max(0, explain_top_n), scores.shape[0]))
    if N == 0:
        return None
    top_idx = np.argsort(scores)[-N:][::-1]

    # Simple sigmoid (no SciPy dependency)
    def _sigmoid(z):
        z = float(z)
        return 1.0 / (1.0 + np.exp(-z))

    # Helper: convert engineered feature name → human-friendly label
    # Examples: "num__age" → ("age", None); "cat__job_blue-collar" → ("job", "blue-collar")
    def _parse_feat_name(name: str):
        if "__" in name:
            _, rest = name.split("__", 1)
        else:
            rest = name
        if "_" in rest and any(rest.startswith(prefix + "_") for prefix in X_raw.columns if X_raw[ prefix if prefix in rest else prefix].dtype == "O"):
            # looks like one-hot "col_value"
            col, val = rest.split("_", 1)
            return col, val
        # numeric or unknown pattern
        return rest, None

    # Helper: arrow direction by contribution sign + rough strength by magnitude
    def _arrow_and_strength(v: float):
        av = abs(v)
        if v > 0:
            if av >= 1.0:  return "↑↑ (strong)"
            if av >= 0.4:  return "↑ (medium)"
            return "↑ (light)"
        elif v < 0:
            if av >= 1.0:  return "↓↓ (strong)"
            if av >= 0.4:  return "↓ (medium)"
            return "↓ (light)"
        else:
            return "↔"

    # Build detailed rows and (optional) plain-language narratives
    detailed_rows = []
    plain_rows = []

    for i in top_idx:
        contribs = np.asarray(shap_vals[i], dtype=float).reshape(-1)  # (n_features,)
        pred_logodds = base_val + float(np.sum(contribs))
        prob_from_shap = _sigmoid(pred_logodds)
        prob_model = float(scores[i])

        # Rank features by absolute impact
        order = np.argsort(np.abs(contribs))[::-1]
        k = int(min(top_features_per_customer, order.size))

        # ---- Detailed row (machine-friendly) ----
        det = {
            "row_id": int(X_raw.index[i]),
            "score_model_prob": prob_model,
            "pred_logodds_from_shap": float(pred_logodds),
            "pred_prob_from_shap": float(prob_from_shap),
            "base_value_logodds": float(base_val),
        }
        for r in range(k):
            f_idx = int(order[r])
            f_name = feature_names[f_idx] if f_idx < len(feature_names) else f"feature_{f_idx}"
            det[f"top_feature_{r+1}"] = f_name
            det[f"contribution_logodds_{r+1}"] = float(contribs[f_idx])
        detailed_rows.append(det)

        # ---- Plain-language row (human-friendly) ----
        if plain_language:
            reasons = []
            for r in range(k):
                f_idx = int(order[r])
                f_name = feature_names[f_idx] if f_idx < len(feature_names) else f"feature_{f_idx}"
                col, val = _parse_feat_name(f_name)
                arrow = _arrow_and_strength(float(contribs[f_idx]))

                # Compose a short phrase for each top reason
                if val is None:
                    # Numeric or non-one-hot: try to include the current value if present in raw data
                    current = None
                    if col in X_raw.columns:
                        try:
                            current = X_raw.iloc[i][col]
                        except Exception:
                            current = None
                    if current is not None:
                        reasons.append(f"{col} = {current} {arrow}")
                    else:
                        reasons.append(f"{col} {arrow}")
                else:
                    # One-hot categorical
                    reasons.append(f"{col} is {val} {arrow}")

            narrative = f"Predicted acceptance {prob_model:.1%}. Key drivers: " + "; ".join(reasons)
            plain_rows.append({
                "row_id": int(X_raw.index[i]),
                "narrative": narrative
            })

    # Write detailed CSV
    pd.DataFrame(detailed_rows).to_csv(out_path, index=False)

    # Write plain CSV if requested
    if plain_language and plain_out_path is not None:
        pd.DataFrame(plain_rows).to_csv(plain_out_path, index=False)

    return out_path



# ---------- CLI ----------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Compare profit-weighted vs standard training across models.")
    parser.add_argument("data", help="Path to bank-additional-full.csv (semicolon-delimited).")
    parser.add_argument("--offer-cost", type=float, default=20.0, help="€ cost per contacted customer.")
    parser.add_argument("--value-if-positive", type=float, default=250.0, help="€ value if a contacted positive accepts.")
    parser.add_argument("--outdir", default="./nb_compare_models", help="Directory to write outputs.")
    parser.add_argument("--steps", type=int, default=199, help="Threshold steps (profit mode).")
    parser.add_argument("--test-size", type=float, default=0.25, help="Test fraction for split.")
    parser.add_argument("--random-state", type=int, default=42, help="Seed.")

    # Targeting mode
    parser.add_argument("--policy-mode", choices=["profit", "capacity"], default="profit",
                        help="Profit-max thresholds or capacity-aware top-K.")
    parser.add_argument("--capacity", type=int, default=0, help="Absolute capacity (customers) in capacity mode.")
    parser.add_argument("--capacity-rate", type=float, default=0.0,
                        help="Target top-rate fraction of TEST (e.g., 0.12). Ignored if --capacity > 0.")

    # Profit-weighted training
    parser.add_argument("--use-profit-weights", action="store_true",
                        help="Train models with sample_weight aligned to € (pos≈value, neg≈cost).")
    parser.add_argument("--weight-normalize", choices=["avg","sum","none"], default="avg",
                        help="Normalize weights to keep training numerically stable (default: avg=mean 1).")

    # Calibrated GB (prefit with inner split)
    parser.add_argument("--calibrate-gb", action="store_true", help="Also build a *calibrated* GB from a profit-weighted base.")
    parser.add_argument("--calibration-method", choices=["isotonic","sigmoid"], default="isotonic",
                        help="Calibration mapping.")
    parser.add_argument("--calibration-split", type=float, default=0.2,
                        help="Fraction of TRAIN used as calibration set for prefit GB (default 0.2).")

    # Plots / SHAP
    parser.add_argument("--calib-bins", type=int, default=10, help="Bins for calibration curve.")
    parser.add_argument("--explain-top-n", type=int, default=0,
                        help="Export SHAP reason codes for top-N (uncalibrated GB + DT + XGB).")
    parser.add_argument("--top-features-per-customer", type=int, default=5,
                        help="How many top SHAP features per customer.")

    parser.add_argument("--use-xgboost", action="store_true",
                        help="Include an XGBoost model in the comparison.")
    parser.add_argument("--calibrate-xgb", action="store_true",
                        help="Also build a calibrated XGBoost from a profit-weighted prefit base.")

    parser.add_argument("--plain-language", action="store_true",
        help="Also export plain-language narratives for GB/DT/XGB reason codes.")

    args = parser.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    # Load data & preprocessor
    df, X_train, X_test, y_train, y_test, feature_cols, num_cols, cat_cols, preprocessor = load_and_prepare(
        args.data, test_size=args.test_size, random_state=args.random_state
    )

    # Profit weights for TRAIN (optional)
    sample_weight = None
    sample_weight_tr = None
    if args.use_profit_weights:
        sample_weight = build_profit_weights(y_train.values, args.offer_cost, args.value_if_positive, args.weight_normalize)
        sample_weight_tr = sample_weight  # alias

    # ---- Train models ----
    # Logistic (supports sample_weight via fit param "logit__sample_weight")
    pipe_logit = Pipeline([
        ("prep", preprocessor),
        ("logit", LogisticRegression(solver="lbfgs", max_iter=1000, class_weight=None)),
    ])
    fit_params = {"logit__sample_weight": sample_weight_tr} if sample_weight_tr is not None else {}
    pipe_logit.fit(X_train, y_train, **fit_params)

    # GB (uncalibrated)
    pipe_gb_uncal = Pipeline([
        ("prep", preprocessor),
        ("gb", GradientBoostingClassifier(n_estimators=120, learning_rate=0.06, max_depth=3, subsample=0.8, random_state=42)),
    ])
    fit_params = {"gb__sample_weight": sample_weight_tr} if sample_weight_tr is not None else {}
    pipe_gb_uncal.fit(X_train, y_train, **fit_params)

    # DT (pruned)
    pipe_dt = Pipeline([
        ("prep", preprocessor),
        ("dt", DecisionTreeClassifier(max_depth=6, min_samples_leaf=120, class_weight=None, random_state=42)),
    ])
    fit_params = {"dt__sample_weight": sample_weight_tr} if sample_weight_tr is not None else {}
    pipe_dt.fit(X_train, y_train, **fit_params)

    # XGBoost (optional)
    xgb_pipe = None
    xgb_calibrated = None
    if args.use_xgboost:
        xgb_pipe = Pipeline([
            ("prep", preprocessor),
            ("xgb", XGBClassifier(
                n_estimators=600, learning_rate=0.05, max_depth=3,
                subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                reg_lambda=2.0, objective="binary:logistic", eval_metric="logloss",
                tree_method="hist", n_jobs=-1, random_state=42
            )),
        ])
        fit_params = {"xgb__sample_weight": sample_weight_tr} if sample_weight_tr is not None else {}
        xgb_pipe.fit(X_train, y_train, **fit_params)
        #_normalize_xgb_base_score(xgb_pipe)

        # (Optional) Calibrated XGB via prefit split (mirrors GB calibration pattern)
        if args.calibrate_xgb:
            from sklearn.model_selection import train_test_split
            X_tr, X_cal, y_tr, y_cal = train_test_split(
                X_train, y_train, test_size=args.calibration_split,
                random_state=args.random_state, stratify=y_train
            )
            w_tr = None
            if args.use_profit_weights:
                mask = y_train.index.isin(y_tr.index)
                w_tr = sample_weight_tr[mask]

            xgb_prefit = Pipeline([
                ("prep", preprocessor),
                ("xgb", XGBClassifier(
                    n_estimators=600, learning_rate=0.05, max_depth=3,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=20,
                    reg_lambda=2.0, objective="binary:logistic", eval_metric="logloss",
                    tree_method="hist", n_jobs=-1, random_state=42
                )),
            ])
            fit_params_prefit = {"xgb__sample_weight": w_tr} if w_tr is not None else {}
            xgb_prefit.fit(X_tr, y_tr, **fit_params_prefit)
            #_normalize_xgb_base_score(xgb_prefit)

            try:
                xgb_calibrated = CalibratedClassifierCV(estimator=xgb_prefit, method=args.calibration_method,
                                                        cv="prefit")
            except TypeError:
                xgb_calibrated = CalibratedClassifierCV(base_estimator=xgb_prefit, method=args.calibration_method,
                                                        cv="prefit")
            xgb_calibrated.fit(X_cal, y_cal)

    # Calibrated GB (prefit on inner split, ONLY if requested)
    pipe_gb_cal = None
    if args.calibrate_gb:
        # Inner split from TRAIN to create a calibration set
        X_tr, X_cal, y_tr, y_cal = train_test_split(
            X_train, y_train, test_size=args.calibration_split, random_state=args.random_state, stratify=y_train
        )
        w_tr = None
        if args.use_profit_weights:
            # Weights only for the *training* part of the prefit estimator
            mask = y_train.index.isin(y_tr.index)
            w_tr = sample_weight_tr[mask]

        pipe_gb_prefit = Pipeline([
            ("prep", preprocessor),
            ("gb", GradientBoostingClassifier(n_estimators=120, learning_rate=0.06, max_depth=3, subsample=0.8, random_state=42)),
        ])
        fit_params_prefit = {"gb__sample_weight": w_tr} if w_tr is not None else {}
        pipe_gb_prefit.fit(X_tr, y_tr, **fit_params_prefit)

        # Calibrate the already trained estimator on the CAL set
        try:
            pipe_gb_cal = CalibratedClassifierCV(estimator=pipe_gb_prefit, method=args.calibration_method, cv="prefit")
        except TypeError:
            # Older sklearn
            pipe_gb_cal = CalibratedClassifierCV(base_estimator=pipe_gb_prefit, method=args.calibration_method, cv="prefit")
        pipe_gb_cal.fit(X_cal, y_cal)  # calibration itself typically unweighted

    # ---- Score on TEST ----
    scores = {
        "Logistic Regression": pipe_logit.predict_proba(X_test)[:, 1],
        "Gradient Boosting":   pipe_gb_uncal.predict_proba(X_test)[:, 1],
        "Pruned Decision Tree": pipe_dt.predict_proba(X_test)[:, 1],
        "Rule: prior success": rule_predict_prior_success(X_test).astype(float),
    }
    if pipe_gb_cal is not None:
        scores["Gradient Boosting (Calibrated)"] = pipe_gb_cal.predict_proba(X_test)[:, 1]

    if args.use_xgboost and xgb_pipe is not None:
        scores["XGBoost"] = xgb_pipe.predict_proba(X_test)[:, 1]
    if args.use_xgboost and xgb_calibrated is not None:
        scores["XGBoost (Calibrated)"] = xgb_calibrated.predict_proba(X_test)[:, 1]

    # ---- Build policies ----
    rows = []
    sweeps = {}
    if args.policy_mode == "profit":
        for name, s in scores.items():
            sw = sweep_thresholds(y_test.values, s, args.offer_cost, args.value_if_positive, steps=args.steps)
            sweeps[name] = sw
            thr = float(sw.iloc[0]["threshold"])
            y_pred = (s >= thr).astype(int)
            net, _ = total_net_with_fn_penalty(y_test.values, y_pred, args.offer_cost, args.value_if_positive)
            rows.append({
                "policy": name,
                **{
                    k: v for k, v in {
                        "targeted": int(y_pred.sum()),
                        "TP": int(confusion_matrix(y_test.values, y_pred).ravel()[3]),
                        "FP": int(confusion_matrix(y_test.values, y_pred).ravel()[1]),
                        "FN": int(confusion_matrix(y_test.values, y_pred).ravel()[2]),
                        "TN": int(confusion_matrix(y_test.values, y_pred).ravel()[0]),
                        "precision": float(np.sum((y_pred==1)&(y_test.values==1))/max(1, np.sum(y_pred==1))),
                        "recall": float(np.sum((y_pred==1)&(y_test.values==1))/max(1, np.sum(y_test.values==1))),
                        "net_profit_eur": float(net),
                        "added_vs_do_nothing_eur": added_net_vs_do_nothing(y_test.values, y_pred, args.offer_cost, args.value_if_positive),
                        "chosen_threshold": thr,
                    }.items()
                }
            })
        thr_note = "Profit-max thresholds (per model). See *_threshold_sweep.csv for details."
    else:
        # Capacity-aware top-K
        n_test = len(X_test)
        if args.capacity and args.capacity > 0:
            k = int(min(args.capacity, n_test))
        else:
            rate = float(np.clip(args.capacity_rate, 0.0, 1.0))
            k = int(round(rate * n_test))
        for name, s in scores.items():
            idx = np.argpartition(s, -k)[-k:]
            y_pred = np.zeros_like(y_test.values, dtype=int); y_pred[idx] = 1
            rows.append({
                "policy": name,
                **{
                    k2: v2 for k2, v2 in {
                        "targeted": int(y_pred.sum()),
                        "TP": int(confusion_matrix(y_test.values, y_pred).ravel()[3]),
                        "FP": int(confusion_matrix(y_test.values, y_pred).ravel()[1]),
                        "FN": int(confusion_matrix(y_test.values, y_pred).ravel()[2]),
                        "TN": int(confusion_matrix(y_test.values, y_pred).ravel()[0]),
                        "precision": float(np.sum((y_pred==1)&(y_test.values==1))/max(1, np.sum(y_pred==1))),
                        "recall": float(np.sum((y_pred==1)&(y_test.values==1))/max(1, np.sum(y_test.values==1))),
                        "net_profit_eur": float(total_net_with_fn_penalty(y_test.values, y_pred, args.offer_cost, args.value_if_positive)[0]),
                        "added_vs_do_nothing_eur": added_net_vs_do_nothing(y_test.values, y_pred, args.offer_cost, args.value_if_positive),
                        "chosen_threshold": np.nan,
                    }.items()
                }
            })
        thr_note = f"Capacity mode — targeted {k} customers per model (top by score)."

    # Do-Nothing baseline
    y_none = np.zeros_like(y_test.values, dtype=int)
    net_none, _ = total_net_with_fn_penalty(y_test.values, y_none, args.offer_cost, args.value_if_positive)
    rows.append({
        "policy": "Do-Nothing",
        "targeted": 0, "TP": 0, "FP": 0, "FN": int(np.sum(y_test.values==1)), "TN": int(np.sum(y_test.values==0)),
        "precision": 0.0, "recall": 0.0, "net_profit_eur": float(net_none), "added_vs_do_nothing_eur": 0.0,
        "chosen_threshold": np.nan
    })

    comp = pd.DataFrame(rows).sort_values("net_profit_eur", ascending=False).reset_index(drop=True)
    comp.to_csv(outdir / "policies_overview.csv", index=False)

    # Save threshold sweeps (profit mode)
    if args.policy_mode == "profit":
        for name, sw in sweeps.items():
            fn = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_") + "_threshold_sweep.csv"
            sw.to_csv(outdir / fn, index=False)

    # Bar chart
    plt.figure()
    plt.bar(comp["policy"], comp["net_profit_eur"])
    plt.ylabel("Net profit (€)"); plt.title(f"Policies — Net € on TEST ({args.policy_mode})")
    plt.xticks(rotation=18); plt.tight_layout()
    plt.savefig(outdir / "policies_overview_net.png", bbox_inches="tight"); plt.close()

    # ROC / PR tables & plots
    auc_ap_df = build_auc_ap_table(y_test.values, scores); auc_ap_df.to_csv(outdir / "roc_auc_table.csv", index=False)
    plot_roc_curves(y_test.values, scores, outdir / "roc_curves.png")
    plot_pr_curves(y_test.values, scores, outdir / "pr_curves.png")

    # Calibration plots
    for name, s in scores.items():
        fn = name.lower().replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
        plot_calibration(y_test.values, s, name, outdir / f"calibration_{fn}.png")

    # Optional SHAP (uncal GB + DT)
    if args.explain_top_n > 0:
        _ = _explain_tree_model("Gradient Boosting", pipe_gb_uncal, preprocessor, X_test, scores["Gradient Boosting"],
                                args.explain_top_n, outdir / "reason_codes_GB.csv",
                                top_features_per_customer=args.top_features_per_customer,
                                plain_language=args.plain_language,
                                plain_out_path=(outdir / "reason_codes_GB_plain.csv") if args.plain_language else None)
        _ = _explain_tree_model("Pruned Decision Tree", pipe_dt, preprocessor, X_test, scores["Pruned Decision Tree"],
                                args.explain_top_n, outdir / "reason_codes_DT.csv",
                                top_features_per_customer=args.top_features_per_customer,
                                plain_language=args.plain_language,
                                plain_out_path=(outdir / "reason_codes_DT_plain.csv") if args.plain_language else None)
        _ = _explain_tree_model(
            "XGBoost",
            xgb_pipe,  # the fitted XGB pipeline
            preprocessor,  # same preprocessor used for training
            X_test,  # raw test frame (not transformed)
            scores["XGBoost"],  # model probabilities on TEST
            args.explain_top_n,  # how many customers to export
            outdir / "reason_codes_XGB.csv",
            top_features_per_customer=args.top_features_per_customer,
            plain_language=args.plain_language,
            plain_out_path=(outdir / "reason_codes_XGB_plain.csv") if args.plain_language else None
        )
        if not _HAS_SHAP:
            print("NOTE: SHAP not installed — skipping reason-code exports.")

    print(thr_note)
    print("Saved:", (outdir / "policies_overview.csv").resolve())

if __name__ == "__main__":
    main()
