# scripts/predict.py
import argparse
from pathlib import Path
import os
import sys

import numpy as np
import pandas as pd
import joblib

def load_feature_schema(models_dir: Path):
    """Pick the right feature schema (prefer the one with decades)."""
    with_decade = models_dir / "feature_columns_with_decade.pkl"
    plain = models_dir / "feature_columns.pkl"
    if with_decade.exists():
        cols = joblib.load(with_decade)
        print(f"[info] Loaded feature schema: {with_decade.name} ({len(cols)} features)")
        return cols, True
    elif plain.exists():
        cols = joblib.load(plain)
        print(f"[info] Loaded feature schema: {plain.name} ({len(cols)} features)")
        return cols, False
    else:
        raise FileNotFoundError("No feature schema found (feature_columns*.pkl). Train models first.")

def choose_model(models_dir: Path):
    """Try best->fallback model order."""
    order = [
        "rf_tuned_with_decade.pkl",
        "rf_tuned.pkl",
        "random_forest.pkl",
        "logistic_pipeline.pkl",
    ]
    for name in order:
        p = models_dir / name
        if p.exists():
            model = joblib.load(p)
            print(f"[info] Loaded model: {name}")
            # Show class order if available
            try:
                print("[info] model classes_:", getattr(model, "classes_", None))
            except Exception:
                pass
            return model, name
    raise FileNotFoundError("No model .pkl found in models/. Run training first.")

def load_inputs(path: Path, feature_cols, decade_expected: bool):
    """Load CSV, align to training feature schema, fill missing columns, default a decade if needed."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    print(f"[info] Loaded input: {path} shape={df.shape}")

    # Fill any missing expected columns with zeros
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0

    # If extra columns exist, drop them (strict schema)
    df = df[feature_cols]

    # If decade dummies are part of the schema, ensure at least one is set per row
    if decade_expected:
        decade_cols = [c for c in feature_cols if c.startswith("decade_")]
        if decade_cols:
            none_set = (df[decade_cols].sum(axis=1) == 0)
            # Default to 2010s if present, else first decade col
            default_col = "decade_2010s" if "decade_2010s" in decade_cols else decade_cols[0]
            if none_set.any():
                df.loc[none_set, default_col] = 1
    return df

def predict(model, X: pd.DataFrame):
    """Return probability of positive class and hard predictions if available."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]  # positive class prob (class index 1)
        preds = (proba >= 0.5).astype(int)
        return proba, preds
    else:
        preds = model.predict(X)
        proba = preds.astype(float)  # no probability: cast labels to float to keep CSV uniform
        return proba, preds

def compute_shap(model, X: pd.DataFrame, out_path: Path, max_rows: int = None):
    """Compute SHAP for the positive class, robust to 2-D/3-D outputs."""
    import shap  # import here to avoid dependency if not used
    Xexp = X if max_rows is None else X.head(max_rows).copy()

    # Heuristic: tree vs linear vs other
    model_name = model.__class__.__name__.lower()
    is_tree = any(k in model_name for k in ["randomforest", "xgb", "lightgbm", "catboost"])
    is_linear = "logisticregression" in model_name

    if is_tree:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(Xexp)
        # Normalize to 2-D for the positive class
        if isinstance(shap_vals, list):
            sv = shap_vals[1] if len(shap_vals) > 1 else shap_vals[0]
        else:
            sv = shap_vals
            if getattr(sv, "ndim", 2) == 3:
                sv = sv[:, :, 1] if sv.shape[2] > 1 else sv[:, :, 0]

    elif is_linear:
        try:
            explainer = shap.LinearExplainer(model, X)
            sv = explainer.shap_values(Xexp)
            if isinstance(sv, list):
                sv = sv[0]
        except Exception:
            # Fallback to KernelExplainer if pipeline/solver issues
            bg = X.sample(min(200, len(X)), random_state=42)
            f = lambda data: model.predict_proba(pd.DataFrame(data, columns=X.columns))[:, 1]
            explainer = shap.KernelExplainer(f, bg)
            sv = explainer.shap_values(Xexp, nsamples=100)
    else:
        # Generic fallback
        bg = X.sample(min(200, len(X)), random_state=42)
        if hasattr(model, "predict_proba"):
            f = lambda data: model.predict_proba(pd.DataFrame(data, columns=X.columns))[:, 1]
        else:
            f = lambda data: model.predict(pd.DataFrame(data, columns=X.columns)).astype(float)
        explainer = shap.KernelExplainer(f, bg)
        sv = explainer.shap_values(Xexp, nsamples=100)

    sv = np.asarray(sv)
    if sv.ndim != 2:
        raise ValueError(f"Expected 2-D SHAP array, got shape={sv.shape}")

    pd.DataFrame(sv, columns=Xexp.columns).to_csv(out_path, index=False)
    print(f"[info] Wrote SHAP values: {out_path} (shape={sv.shape})")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to inference CSV")
    parser.add_argument("--out", required=True, help="Where to write predictions CSV")
    parser.add_argument("--shap", default=None, help="Optional path to write SHAP CSV")
    parser.add_argument("--shap_n", type=int, default=200, help="Max rows for SHAP (to keep it fast)")
    args = parser.parse_args()

    ROOT = Path(__file__).resolve().parents[1]
    models_dir = ROOT / "models"

    # Load model + schema
    model, model_name = choose_model(models_dir)
    feature_cols, decade_expected = load_feature_schema(models_dir)

    # Load inputs and align
    X = load_inputs(ROOT / args.input, feature_cols, decade_expected)

    # Predict
    proba, preds = predict(model, X)
    out_df = pd.DataFrame({"hit_proba": proba, "hit_pred": preds})
    out_path = ROOT / args.out
    out_df.to_csv(out_path, index=False)
    print(f"[info] Wrote predictions: {args.out} (shape={out_df.shape})")

    # Optional SHAP
    if args.shap:
        try:
            compute_shap(model, X, ROOT / args.shap, max_rows=args.shap_n)
        except Exception as e:
            print(f"[warn] SHAP failed: {e}")

if __name__ == "__main__":
    main()