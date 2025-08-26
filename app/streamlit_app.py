# app/streamlit_app.py
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import streamlit as st
import matplotlib.pyplot as plt

# Optional SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except Exception:
    SHAP_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"

st.set_page_config(page_title="Music Popularity Prediction", layout="wide")

# -----------------------------
# Utilities
# -----------------------------
@st.cache_resource
def load_model_and_schema():
    """Load the best-available model and matching feature schema, with graceful fallback."""
    candidates = [
        ("rf_tuned_with_decade.pkl", "feature_columns_with_decade.pkl"),
        ("rf_tuned.pkl", "feature_columns.pkl"),
        ("random_forest.pkl", "feature_columns.pkl"),
        ("logistic_pipeline.pkl", "feature_columns.pkl"),
    ]
    for model_name, schema_name in candidates:
        model_path = MODELS_DIR / model_name
        schema_path = MODELS_DIR / schema_name
        if model_path.exists() and schema_path.exists():
            model = joblib.load(model_path)
            feature_cols = joblib.load(schema_path)
            return model, feature_cols, model_name
    raise FileNotFoundError("No model/schema pair found in models/. Please train models first.")

def ensure_columns(df, feature_cols):
    """Add any missing expected columns with 0, and handle decade one-hot safety; then align order."""
    df = df.copy()
    for c in feature_cols:
        if c not in df.columns:
            df[c] = 0
    decade_cols = [c for c in feature_cols if c.startswith("decade_")]
    if decade_cols:
        none_set = (df[decade_cols].sum(axis=1) == 0)
        if none_set.any():
            default_decade = "decade_2010s" if "decade_2010s" in df.columns else decade_cols[-1]
            df.loc[none_set, default_decade] = 1
    return df[feature_cols]

def predict_proba_and_label(model, X, threshold=0.5):
    """Return hit probability and binary class based on threshold."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[:, 1]
    elif hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        proba = 1 / (1 + np.exp(-scores))
    else:
        preds = model.predict(X)
        proba = preds.astype(float)
    preds = (proba >= float(threshold)).astype(int)
    return proba, preds

def shap_for_single(model, X_row, background=None, feature_names=None):
    """
    Robust SHAP explanation for a single row:
    1) TreeExplainer (interventional, probability) for tree models
    2) Fallback to modern Explainer
    """
    if not SHAP_AVAILABLE:
        return None, "SHAP not installed."

    bg = background if background is not None else X_row

    # Preferred path: interventional + probability (works for RF/XGB)
    try:
        explainer = shap.TreeExplainer(
            model, data=bg, feature_perturbation="interventional", model_output="probability"
        )
        ex = explainer(X_row)
        # Collapse to positive class if multi-class-shaped output
        if getattr(ex.values, "ndim", 2) == 3:
            pos_idx = 1
            if hasattr(model, "classes_"):
                classes = list(model.classes_)
                if 1 in classes:
                    pos_idx = classes.index(1)
            values = ex.values[:, :, pos_idx]
            base_vals = ex.base_values
            if np.ndim(base_vals) == 2:
                base_vals = base_vals[:, pos_idx]
            ex = shap.Explanation(
                values=values,
                base_values=base_vals,
                data=ex.data,
                feature_names=feature_names or X_row.columns.tolist(),
            )
        return ex, None
    except Exception as e_tree:
        # Fallback: modern API
        try:
            ex = shap.Explainer(model, bg)(X_row)
            if hasattr(model, "classes_") and getattr(ex.values, "ndim", 2) == 3:
                classes = list(model.classes_)
                pos_idx = classes.index(1) if 1 in classes else (1 if len(classes) > 1 else 0)
                values = ex.values[:, :, pos_idx]
                base_vals = ex.base_values
                if np.ndim(base_vals) == 2:
                    base_vals = base_vals[:, pos_idx]
                ex = shap.Explanation(
                    values=values,
                    base_values=base_vals,
                    data=ex.data,
                    feature_names=feature_names or X_row.columns.tolist(),
                )
            return ex, None
        except Exception as e_new:
            return None, f"SHAP failed: {e_tree} | fallback: {e_new}"

def load_feature_importances():
    """Try to load an importances CSV to visualize."""
    paths = [
        MODELS_DIR / "rf_tuned_feature_importances_with_decade.csv",
        MODELS_DIR / "rf_tuned_with_decade_feature_importances.csv",
        MODELS_DIR / "rf_tuned_feature_importances.csv",
    ]
    for p in paths:
        if p.exists():
            df = pd.read_csv(p)
            if set(df.columns) >= {"feature", "importance"}:
                return df[["feature", "importance"]]
            if "Unnamed: 0" in df.columns and df.shape[1] == 2:
                df.columns = ["feature", "importance"]
                return df
    return None

# -----------------------------
# Sidebar / Model
# -----------------------------
st.sidebar.title("Music Popularity Prediction")
model, feature_cols, model_name = load_model_and_schema()
st.sidebar.success(f"Loaded model: **{model_name}**")
st.sidebar.caption(f"Features: {len(feature_cols)}")

tab1, tab2, tab3, tab4 = st.tabs(["Single Prediction", "Batch Scoring", "Insights", "Validation"])

# -----------------------------
# Tab 1: Single Prediction
# -----------------------------
with tab1:
    st.subheader("Single Prediction (Manual Input)")
    st.caption("Adjust audio features and decade, then choose the decision threshold and predict.")

    # Defaults
    defaults = {
        "acousticness": 0.3, "danceability": 0.55, "energy": 0.6, "instrumentalness": 0.05,
        "liveness": 0.15, "speechiness": 0.08, "valence": 0.5, "tempo": 120.0, "loudness": -10.0,
        "dance_energy": 0.55 * 0.6, "valence_acoustic": 0.5 * 0.3, "loudness_norm": 0.7,
    }

    # Numeric sliders (except decade dummies)
    num_inputs = {}
    cols = st.columns(3)
    i = 0
    for f in feature_cols:
        if f.startswith("decade_"):
            continue
        if f == "tempo":
            val = cols[i % 3].slider("tempo (BPM)", 40.0, 240.0, float(defaults.get("tempo", 120.0)))
        elif f == "loudness":
            val = cols[i % 3].slider("loudness (dBFS)", -60.0, 5.0, float(defaults.get("loudness", -10.0)))
        else:
            val = cols[i % 3].slider(f, 0.0, 1.0, float(defaults.get(f, 0.5)))
        num_inputs[f] = val
        i += 1

    # Decade select → one-hot
    decade_options = ["1960s", "1970s", "1980s", "1990s", "2000s", "2010s"]
    selected_decade = st.selectbox("Decade", options=decade_options, index=5)

    # Decision threshold (moved here under the sliders)
    threshold = st.slider(
        "Decision threshold (≥ = hit)", 0.00, 1.00, 0.50, 0.01,
        help="Scores ≥ threshold are labelled as 'Hit'. Lower it to be more inclusive (higher recall); raise it to be stricter (higher precision)."
    )

    # Build row & engineered features
    row = pd.DataFrame([num_inputs])
    if "dance_energy" in feature_cols:
        row["dance_energy"] = row["danceability"] * row["energy"]
    if "valence_acoustic" in feature_cols:
        row["valence_acoustic"] = row["valence"] * row["acousticness"]
    if "loudness_norm" in feature_cols and "loudness_norm" not in row.columns:
        row["loudness_norm"] = (row["loudness"] - (-60.0)) / (5.0 - (-60.0))
        row["loudness_norm"] = row["loudness_norm"].clip(0, 1)

    for d in decade_options:
        col_name = f"decade_{d}"
        if col_name in feature_cols:
            row[col_name] = 1 if d == selected_decade else 0

    X_single = ensure_columns(row, feature_cols)

    if st.button("Predict"):
        proba, pred = predict_proba_and_label(model, X_single, threshold=threshold)
        st.success(f"Hit probability: **{proba[0]:.3f}**  →  Class @ {threshold:.2f}: **{int(pred[0])}**")

        if SHAP_AVAILABLE:
            with st.expander("Explain this prediction (SHAP)"):
                # Try to use a small background sample (improves stability)
                try:
                    bg_path = DATA_DIR / "cleaned_music_data_with_decade.csv"
                    if bg_path.exists():
                        bg_df = pd.read_csv(bg_path)
                        bg_df = bg_df.drop(columns=[c for c in ["popularity"] if c in bg_df.columns])
                        bg_df = ensure_columns(bg_df, feature_cols)
                        background = bg_df.sample(min(200, len(bg_df)), random_state=42)
                    else:
                        background = X_single
                except Exception:
                    background = X_single

                ex, err = shap_for_single(model, X_single, background=background, feature_names=feature_cols)
                if err:
                    st.warning(err)
                elif ex is not None:
                    # BAR
                    try:
                        shap.plots.bar(ex, max_display=12, show=False)
                        fig_bar = plt.gcf()
                        st.pyplot(fig_bar, clear_figure=True)
                        plt.clf()
                    except Exception as e:
                        st.info(f"Bar plot unavailable: {e}")
                    # WATERFALL
                    try:
                        shap.plots.waterfall(ex[0], max_display=12, show=False)
                        fig_wf = plt.gcf()
                        st.pyplot(fig_wf, clear_figure=True)
                        plt.clf()
                    except Exception as e:
                        st.info(f"Waterfall unavailable: {e}")
                else:
                    st.info("SHAP explanation not available for this model.")
        else:
            st.caption("Install SHAP to view local explanations: `pip install shap`")

# -----------------------------
# Tab 2: Batch Scoring
# -----------------------------
with tab2:
    st.subheader("Batch Scoring (CSV)")
    st.caption("Upload a CSV with your features. We will align to the model schema and predict.")

    threshold_b = st.slider(
        "Decision threshold for batch (≥ = hit)", 0.00, 1.00, 0.50, 0.01,
        help="Scores ≥ threshold are labelled as 'Hit' for the batch predictions."
    )
    up = st.file_uploader("Upload CSV", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        if "popularity" in df.columns:
            df = df.drop(columns=["popularity"])
        X_batch = ensure_columns(df, feature_cols)
        proba, pred = predict_proba_and_label(model, X_batch, threshold=threshold_b)

        out = df.copy()
        out["hit_proba"] = proba
        out["hit_pred"] = pred
        st.write("Preview:", out.head())

        st.download_button(
            "Download predictions CSV",
            data=out.to_csv(index=False).encode("utf-8"),
            file_name="predictions_batch.csv",
            mime="text/csv"
        )

# -----------------------------
# Tab 3: Insights (Feature Importances)
# -----------------------------
with tab3:
    st.subheader("Model Insights")
    st.caption("Top feature importances from the tuned random forest (if available).")

    imp = load_feature_importances()
    if imp is None:
        st.info("No importances CSV found in models/. Train the tuned RF to produce it.")
    else:
        imp = imp.sort_values("importance", ascending=False).head(20)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.barh(imp["feature"][::-1], imp["importance"][::-1])
        ax.set_xlabel("Importance")
        ax.set_title("Top Feature Importances")
        fig.tight_layout()
        st.pyplot(fig)

# -----------------------------
# Tab 4: Validation (with ground truth)
# -----------------------------
with tab4:
    st.subheader("Validation with Ground Truth")
    st.caption("Upload a CSV that includes the model features AND a binary ground-truth label column (default: 'popularity'). We'll align features to the trained schema, score with the model, and compute metrics at your chosen threshold.")

    up_val = st.file_uploader("Upload validation CSV", type=["csv"], key="val_csv")
    if up_val is None:
        st.info("Please upload a CSV to begin.")
    else:
        df_val_raw = pd.read_csv(up_val)

        # Choose/confirm label column
        options = list(df_val_raw.columns)
        default_label = "popularity" if "popularity" in df_val_raw.columns else options[0]
        label_col = st.selectbox("Ground-truth label column (0/1):", options=options, index=options.index(default_label))

        if label_col not in df_val_raw.columns:
            st.warning("Please choose a valid label column.")
        else:
            y_true = df_val_raw[label_col].astype(int).values
            X_val = df_val_raw.drop(columns=[label_col], errors="ignore")
            X_val = ensure_columns(X_val, feature_cols)

            thr = st.slider("Decision threshold (validation)", 0.0, 1.0, 0.50, 0.01, key="val_thr")
            proba, y_pred = predict_proba_and_label(model, X_val, threshold=thr)

            from sklearn.metrics import (
                accuracy_score, precision_score, recall_score, f1_score,
                roc_auc_score, average_precision_score, confusion_matrix,
                RocCurveDisplay, PrecisionRecallDisplay
            )

            acc  = accuracy_score(y_true, y_pred)
            prec = precision_score(y_true, y_pred, zero_division=0)
            rec  = recall_score(y_true, y_pred, zero_division=0)
            f1   = f1_score(y_true, y_pred, zero_division=0)
            try:
                roc = roc_auc_score(y_true, proba)
            except Exception:
                roc = float("nan")
            try:
                pr_auc = average_precision_score(y_true, proba)
            except Exception:
                pr_auc = float("nan")

            c1, c2, c3, c4, c5, c6 = st.columns(6)
            c1.metric("Accuracy", f"{acc:.3f}")
            c2.metric("Precision", f"{prec:.3f}")
            c3.metric("Recall", f"{rec:.3f}")
            c4.metric("F1", f"{f1:.3f}")
            c5.metric("ROC-AUC", f"{roc:.3f}" if not np.isnan(roc) else "N/A")
            c6.metric("PR-AUC", f"{pr_auc:.3f}" if not np.isnan(pr_auc) else "N/A")

            st.markdown("#### Confusion Matrix")
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            cm_df = pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
            st.dataframe(cm_df)

            st.markdown("#### ROC & Precision-Recall Curves")
            fig1, ax1 = plt.subplots()
            try:
                RocCurveDisplay.from_predictions(y_true, proba, ax=ax1)
                ax1.set_title("ROC Curve")
                st.pyplot(fig1)
            except Exception:
                st.info("ROC curve not available (no predicted probabilities).")

            fig2, ax2 = plt.subplots()
            try:
                PrecisionRecallDisplay.from_predictions(y_true, proba, ax=ax2)
                ax2.set_title("Precision-Recall Curve")
                st.pyplot(fig2)
            except Exception:
                st.info("PR curve not available (no predicted probabilities).")

            # Threshold suggestions
            st.markdown("#### Suggested Thresholds")
            cand = np.unique(np.concatenate([[0.0, 1.0], proba]))
            cand = np.clip(cand, 0.0, 1.0)
            best_f1, best_f1_thr = -1.0, 0.5
            best_j, best_j_thr = -1.0, 0.5
            for t in cand:
                yp = (proba >= t).astype(int)
                f1_t = f1_score(y_true, yp, zero_division=0)
                if f1_t > best_f1:
                    best_f1, best_f1_thr = f1_t, t
                tn, fp, fn, tp = confusion_matrix(y_true, yp, labels=[0, 1]).ravel()
                tpr = tp / (tp + fn) if (tp + fn) else 0.0
                fpr = fp / (fp + tn) if (fp + tn) else 0.0
                j = tpr - fpr
                if j > best_j:
                    best_j, best_j_thr = j, t
            st.write(f"**Best F1 threshold:** {best_f1_thr:.3f}  (F1={best_f1:.3f})")
            st.write(f"**Best Youden’s J threshold:** {best_j_thr:.3f}  (J={best_j:.3f})")

            # Per-decade performance (if decade dummies exist)
            decade_cols = [c for c in feature_cols if c.startswith("decade_")]
            if decade_cols:
                st.markdown("#### Per-Decade Breakdown (current threshold)")
                dec_ix = X_val[decade_cols].values.argmax(axis=1)
                dec_names = np.array(decade_cols)[dec_ix]
                df_eval = pd.DataFrame({"decade": dec_names, "y_true": y_true, "y_pred": y_pred})
                rows = []
                for d in decade_cols:
                    sub = df_eval[df_eval["decade"] == d]
                    if sub.empty:
                        continue
                    acc_d  = accuracy_score(sub["y_true"], sub["y_pred"])
                    prec_d = precision_score(sub["y_true"], sub["y_pred"], zero_division=0)
                    rec_d  = recall_score(sub["y_true"], sub["y_pred"], zero_division=0)
                    f1_d   = f1_score(sub["y_true"], sub["y_pred"], zero_division=0)
                    rows.append([d, len(sub), acc_d, prec_d, rec_d, f1_d])
                if rows:
                    per_decade = pd.DataFrame(rows, columns=["decade", "N", "Accuracy", "Precision", "Recall", "F1"]).sort_values("decade")
                    st.dataframe(per_decade, use_container_width=True)
                else:
                    st.caption("No decade dummies found in this file or none are set.")

st.sidebar.markdown("---")
st.sidebar.caption("Tip: This demo uses Streamlit. A React/Next.js UI is optional for later.")