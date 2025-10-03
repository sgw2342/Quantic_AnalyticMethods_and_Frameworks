NovaBank Retention — RUN BOOK 
=================================================

This run book explains how to use the script: `novabank_compare_policies.py`
Recommendation: **Use XGBoost + Plain Language** as your standard setting.

-------------------------------------------------
0) Environment & install
-------------------------------------------------
- Python 3.9–3.12
- Install dependencies:
    pip install -r requirements.txt
  (Includes: scikit-learn, xgboost, shap, numpy, pandas, matplotlib)

Data:
- Source CSV: bank-additional-full.csv (semicolon-delimited from UCI)
- Target variable used: target = 1 if y == "yes" else 0
- We drop `duration` to avoid leakage (it is only known after a call)

-------------------------------------------------
1) What the script does
-------------------------------------------------
`novabank_compare_policies.py`:
- Loads & cleans data, builds a train/test split.
- Trains: Logistic Regression, Gradient Boosting (GB), Pruned Decision Tree (DT).
- (Optional) Trains **XGBoost** when `--use-xgboost` is passed.
- (Optional) Uses **profit-weighted training** so models value positives vs negatives in €.
- (Optional) Calibrates GB and/or XGB for better probability reliability.
- Scores the TEST set, and for each model finds either:
  - **Profit-max threshold** ("profit" mode), or
  - **Top-K customers** ("capacity" mode).
- Writes a comparison table and diagnostic plots.
- (Optional) Exports **reason codes** (SHAP for GB/DT/XGB) + **plain-language** narratives if flagged.

-------------------------------------------------
2) Standard command (recommended defaults)
-------------------------------------------------
Use XGBoost + plain-language reason codes, with profit-max thresholding:

python3 novabank_compare_policies.py ./bank-additional-full.csv   --use-xgboost   --plain-language   --explain-top-n 500   --outdir ./nb_compare_final

Notes:
- Default economics: --offer-cost 20 --value-if-positive 250
- Default split: --test-size 0.25 --random-state 42
- You can add calibration for GB/XGB if desired (see §3b).

-------------------------------------------------
3) Useful variants
-------------------------------------------------
a) Capacity-aware (limited outreach slots)
   Target the top-K customers (e.g., 20,000) instead of using a fixed threshold:
   python3 novabank_compare_policies.py ./bank-additional-full.csv      --use-xgboost  --plain-language --explain-top-n 500      --policy-mode capacity --capacity 20000      --outdir ./nb_compare_capacity

b) Add calibration (improves probability reliability near the cutoff)
   For GB:
     --calibrate-gb --calibration-method isotonic
   For XGB:
     --calibrate-xgb --calibration-method isotonic
   Example:
   python3 novabank_compare_policies.py ./bank-additional-full.csv      --use-xgboost  --plain-language --explain-top-n 500      --calibrate-xgb --calibration-method isotonic      --outdir ./nb_compare_xgb_cal

c) Sensitivity to money assumptions
   Change offer cost or value per save to re-find the best cutoff:
   --offer-cost 25 --value-if-positive 200

d) Lighter explain exports
   Reduce SHAP rows or features per customer:
   --explain-top-n 200 --top-features-per-customer 5

-------------------------------------------------
4) Key command-line flags 
-------------------------------------------------
Required:
  data                          Path to bank-additional-full.csv

Economics & split:
  --offer-cost FLOAT            € per contacted customer (default: 20.0)
  --value-if-positive FLOAT     € value when a contacted positive accepts (default: 250.0)
  --test-size FLOAT             Test fraction (default: 0.25)
  --random-state INT            Seed (default: 42)
  --steps INT                   Threshold sweep steps in profit mode (default: 199)
  --outdir PATH                 Output directory (default: ./nb_compare)

Policy mode:
  --policy-mode {profit,capacity}   Default: profit
  --capacity INT                    Absolute top-K when in capacity mode (default: 0)
  --capacity-rate FLOAT             Fraction of TEST when in capacity mode (default: 0.0)

Training options:
  --use-profit-weights          Train with €-aligned sample weights (pos≈value, neg≈cost)
  --weight-normalize {avg,sum,none} Normalize weights (default: avg)

Model toggles:
  --calibrate-gb                Add calibrated GB (prefit)
  --calibrate-xgb               Add calibrated XGB (prefit)
  --calibration-method {isotonic,sigmoid}  Mapping used for calibration (default: isotonic)
  --calibration-split FLOAT     Fraction of TRAIN held out for calibration (default: 0.2)
  --use-xgboost                 Include XGBoost model

Explainability:
  --explain-top-n INT           Export reason codes for top-N scored customers (default: 0 = off)
  --top-features-per-customer INT  Number of top features per customer (default: 5)
  --plain-language              Also export plain-language reason codes per customer

Plotting / calibration:
  --calib-bins INT              Bins for calibration curve (default: 10)

-------------------------------------------------
5) Outputs you’ll get in --outdir
-------------------------------------------------
Core comparison:
  policies_overview.csv         Table of policies with net_profit_eur, precision, recall, threshold/top-K
  policies_overview_net.png     Bar chart of net € per policy
  roc_auc_table.csv             AUC / Average Precision table
  roc_curves.png                ROC plot (all models)
  pr_curves.png                 Precision–Recall plot (all models)

Calibration:
  calibration_*.png             Reliability + score distribution per model

Threshold sweeps (profit mode only):
  <model>_threshold_sweep.csv   Per-threshold TP/FP/FN/TN, precision/recall, net €, added vs do-nothing

Reason codes (if --explain-top-n > 0):
  reason_codes_GB.csv           SHAP top-N for Gradient Boosting
  reason_codes_DT.csv           SHAP top-N for Pruned Decision Tree
  reason_codes_XGB.csv          SHAP top-N for XGBoost (when --use-xgboost)
  reason_codes_*_plain.csv      Plain-language narratives (when --plain-language)

-------------------------------------------------
6) How to read the results (quick guide)
-------------------------------------------------
Pick the winner by **net_profit_eur** in policies_overview.csv. That metric already prices:
- Wasted offers (false positives) via the contact cost
- Missed saves (false negatives) via the lost value

Check:
- **chosen_threshold** (profit mode) — the cutoff to use in production now.
- **precision** (“hit rate”) — of those contacted, what % said yes (efficiency).
- **recall** (“coverage”) — of those who would say yes, what % did we contact (saves captured).
- **calibration_*.png** — curves closer to the diagonal mean more trustworthy probabilities near your cutoff.

If capacity-limited:
- Re-run in **capacity** mode and use the **top-K** customers per period. That’s equivalent to using a dynamic cutoff at the top of the list.

-------------------------------------------------
7) Standard operating recommendation
-------------------------------------------------
- Use **XGBoost + Plain Language** as your default:
    --use-xgboost --plain-language
- Start in **profit** mode; use the **chosen_threshold** from policies_overview.csv.
- If capacity-limited, switch to **capacity** mode and set **--capacity** to your weekly slots.
- Re-check monthly:
    - If offer cost or value changes, the optimal cutoff moves (re-run to refit).
    - Watch calibration plots; if curves drift off the diagonal, enable model calibration or re-train.
- Keep the **Pruned Decision Tree** exports for transparency and coaching; share **plain-language reason codes** with agents.

-------------------------------------------------
8) Example one-liners to copy/paste
-------------------------------------------------
Profit mode, recommended default:
  python3 novabank_compare_policies.py ./bank-additional-full.csv \
    --use-xgboost  --plain-language --explain-top-n 500 \
    --outdir ./nb_compare_final

Capacity mode (20k max outreach per cycle):
  python3 novabank_compare_policies.py ./bank-additional-full.csv \
    --use-xgboost  --plain-language --explain-top-n 500 \
    --policy-mode capacity --capacity 20000 \
    --outdir ./nb_compare_capacity

Add calibration for XGB:
  python3 novabank_compare_policies.py ./bank-additional-full.csv \
    --use-xgboost  --plain-language --explain-top-n 500 \
    --calibrate-xgb --calibration-method isotonic \
    --outdir ./nb_compare_xgb_cal

-------------------------------------------------
9) Troubleshooting
-------------------------------------------------
- Different numbers across scripts:
  Ensure identical --test-size, --random-state, economics (cost/value), and policy mode.
  The comparison script uses profit-max threshold by default; 
- Calibration keyword error:
  scikit-learn versions differ; the code handles `estimator` vs `base_estimator` internally.
- SHAP too slow/large:
  Lower --explain-top-n, reduce --top-features-per-customer, or skip --plain-language.
- Memory issues:
  Close other apps; reduce OneHot dimensionality if you’ve added many new categories.

-------------------------------------------------
10) Governance & safety
-------------------------------------------------
- Ensure inputs exclude post-contact info (no leakage, e.g., `duration`).
- Run periodic fairness checks (segment performance, adverse impact).
- Apply contact caps and suppression rules to protect customer experience.
