# scripts/train_model.py
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def load_cleaned_dataframe(root: Path) -> tuple[pd.DataFrame, str]:
    """
    Try to load the 'with decade' file first; if missing, fall back to the baseline file.
    Returns (df, tag) where tag is either 'with_decade' or 'no_decade' for naming artifacts.
    """
    with_decade = root / "data" / "processed" / "cleaned_music_data_with_decade.csv"
    baseline    = root / "data" / "processed" / "cleaned_music_data.csv"

    if with_decade.exists():
        df = pd.read_csv(with_decade)
        return df, "with_decade"
    elif baseline.exists():
        df = pd.read_csv(baseline)
        return df, "no_decade"
    else:
        raise FileNotFoundError(
            f"Neither {with_decade} nor {baseline} exists. "
            "Run your cleaning script first."
        )

def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], bool]:
    """
    Builds X, y. If 'decade' exists, one-hot encodes it and returns use_decade=True.
    Returns (X, y, num_feats, feature_cols, use_decade)
    """
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower()

    # target
    if "popularity" not in df.columns:
        if "target" in df.columns:
            df.rename(columns={"target": "popularity"}, inplace=True)
        else:
            raise KeyError("No 'popularity' or 'target' column found.")
    df["popularity"] = df["popularity"].astype(int)
    y = df["popularity"]

    # numeric acoustic + engineered features (keep only those present)
    numeric_candidates = [
        "acousticness","danceability","energy","instrumentalness","liveness",
        "speechiness","valence","tempo","loudness",
        "dance_energy","valence_acoustic","loudness_norm"
    ]
    num_feats = [c for c in numeric_candidates if c in df.columns]

    # optional decade one-hot
    use_decade = "decade" in df.columns
    if use_decade:
        dummies = pd.get_dummies(df["decade"], prefix="decade", drop_first=False)
        X = pd.concat([df[num_feats], dummies], axis=1)
    else:
        X = df[num_feats]

    feature_cols = X.columns.tolist()
    return X, y, num_feats, feature_cols, use_decade

def train_log_reg_with_optional_decade(X_train, X_test, y_train, y_test, num_feats, use_decade, models_dir: Path, tag: str):
    """
    Scale numeric features only; concatenate decade dummies (already in X) back after scaling.
    Saves logistic model bundle with scaler & metadata.
    """
    # figure out which columns are decade dummies
    other_cols = [c for c in X_train.columns if c not in num_feats]

    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[num_feats])
    X_test_num  = scaler.transform(X_test[num_feats])

    if use_decade and other_cols:
        X_train_lr = np.hstack([X_train_num, X_train[other_cols].values])
        X_test_lr  = np.hstack([X_test_num,  X_test[other_cols].values])
    else:
        X_train_lr, X_test_lr = X_train_num, X_test_num

    log_reg = LogisticRegression(max_iter=2000, random_state=42)
    log_reg.fit(X_train_lr, y_train)
    y_pred = log_reg.predict(X_test_lr)

    print(f"\n=== Logistic Regression ({tag}) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    try:
        proba = log_reg.predict_proba(X_test_lr)[:, 1]
        print("ROC AUC:", roc_auc_score(y_test, proba))
    except Exception:
        pass
    print(classification_report(y_test, y_pred))

    joblib.dump(
        {"scaler": scaler, "model": log_reg, "num_feats": num_feats, "use_decade": use_decade,
         "feature_cols_full": X_train.columns.tolist()},
        models_dir / f"logistic_{tag}.pkl"
    )
    print(f"Saved logistic_{tag}.pkl")

def train_random_forest(X_train, X_test, y_train, y_test, feature_cols, models_dir: Path, tag: str):
    rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    print(f"\n=== Random Forest ({tag}) ===")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    try:
        proba = rf.predict_proba(X_test)[:, 1]
        print("ROC AUC:", roc_auc_score(y_test, proba))
    except Exception:
        pass
    print(classification_report(y_test, y_pred))

    joblib.dump(rf, models_dir / f"random_forest_{tag}.pkl")
    print(f"Saved random_forest_{tag}.pkl")
    return rf

def train_xgboost_optional(X_train, X_test, y_train, y_test, models_dir: Path, tag: str):
    try:
        import xgboost as xgb
        xgb_model = xgb.XGBClassifier(
            eval_metric="logloss", random_state=42, n_jobs=-1, tree_method="hist"
        )
        xgb_model.fit(X_train, y_train)
        y_pred = xgb_model.predict(X_test)

        print(f"\n=== XGBoost ({tag}) ===")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        try:
            proba = xgb_model.predict_proba(X_test)[:, 1]
            print("ROC AUC:", roc_auc_score(y_test, proba))
        except Exception:
            pass
        print(classification_report(y_test, y_pred))

        joblib.dump(xgb_model, models_dir / f"xgboost_{tag}.pkl")
        print(f"Saved xgboost_{tag}.pkl")
    except Exception as e:
        print(f"\nXGBoost not available or failed to train: {e}")

def tune_rf_and_save(X_train, X_test, y_train, y_test, feature_cols, models_dir: Path, tag: str):
    from sklearn.model_selection import RandomizedSearchCV
    param_dist = {
        "n_estimators": [200, 300, 400],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "bootstrap": [True, False],
    }
    rs = RandomizedSearchCV(
        RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=12, cv=3, scoring="accuracy",
        n_jobs=-1, random_state=42, verbose=1
    )
    rs.fit(X_train, y_train)
    best_rf = rs.best_estimator_
    y_pred = best_rf.predict(X_test)

    print(f"\n=== Tuned Random Forest ({tag}) ===")
    print("Best params:", rs.best_params_)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    try:
        proba = best_rf.predict_proba(X_test)[:, 1]
        print("ROC AUC:", roc_auc_score(y_test, proba))
    except Exception:
        pass
    print(classification_report(y_test, y_pred))

    joblib.dump(best_rf, models_dir / f"rf_tuned_{tag}.pkl")
    importances = pd.Series(best_rf.feature_importances_, index=feature_cols).sort_values(ascending=False)
    importances.to_csv(models_dir / f"rf_tuned_{tag}_feature_importances.csv", index=True)
    print(f"Saved rf_tuned_{tag}.pkl and rf_tuned_{tag}_feature_importances.csv")

    return best_rf

def per_decade_report(best_rf, X_test, y_test, df_full, tag: str):
    if "decade" not in df_full.columns:
        return
    print(f"\n=== Per-Decade Metrics ({tag}, tuned RF) ===")
    test_idx = X_test.index
    decade_test = df_full.loc[test_idx, "decade"]
    y_pred = best_rf.predict(X_test)
    out = (pd.DataFrame({"decade": decade_test, "y_true": y_test, "y_pred": y_pred})
             .groupby("decade").apply(lambda g: accuracy_score(g["y_true"], g["y_pred"])))
    print(out.sort_index())

if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[1]
    models_dir = ROOT / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Load
    df, tag = load_cleaned_dataframe(ROOT)

    # Build features
    X, y, num_feats, feature_cols, use_decade = build_feature_matrix(df)

    # Persist exact feature set for inference alignment
    joblib.dump(feature_cols, models_dir / f"feature_columns_{tag}.pkl")
    print(f"Saved feature_columns_{tag}.pkl ({len(feature_cols)} columns)")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # Train
    train_log_reg_with_optional_decade(X_train, X_test, y_train, y_test, num_feats, use_decade, models_dir, tag)
    rf = train_random_forest(X_train, X_test, y_train, y_test, feature_cols, models_dir, tag)
    train_xgboost_optional(X_train, X_test, y_train, y_test, models_dir, tag)
    best_rf = tune_rf_and_save(X_train, X_test, y_train, y_test, feature_cols, models_dir, tag)

    # Per-decade bias check (only meaningful if we have decade)
    per_decade_report(best_rf, X_test, y_test, df, tag)

    print("\nAll artifacts saved in:", models_dir)