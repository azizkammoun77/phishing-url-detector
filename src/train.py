"""Train and compare hostname-only phishing models, tracked with MLflow.

Hostnames are only ever handled as text; nothing is visited or resolved.
Run from the project root:  python -m src.train
"""
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd
from mlflow.models import infer_signature
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from src.features import FEATURE_NAMES, extract_features_df

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
STRESS_PATH = ROOT / "data" / "stress" / "legit_subdomains.csv"

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "phishing-hostname-v1"
DATA_VERSION = "v2_umbrella"  # v1 = PhiUSIIL only
RANDOM_STATE = 42
THRESHOLD = 0.5

# Features that mostly encode "has a subdomain / is on free hosting".
EXCLUDED_IN_B = {"num_subdomain_levels", "num_labels", "is_free_hosting"}
FEATURE_SETS = {
    "A_all": list(FEATURE_NAMES),
    "B_no_subdomain_hosting": [f for f in FEATURE_NAMES if f not in EXCLUDED_IN_B],
}

LOGREG_PARAMS = {"class_weight": "balanced", "max_iter": 1000}
XGB_PARAMS = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
              "eval_metric": "logloss", "random_state": RANDOM_STATE}


def make_model(model_type: str):
    if model_type == "logreg":
        return Pipeline([("scaler", StandardScaler()),
                         ("clf", LogisticRegression(**LOGREG_PARAMS))])
    return XGBClassifier(**XGB_PARAMS)


def run_one(feature_set, model_type, cols, data, folds):
    X_tr, y_tr = data["X_train"][cols], data["y_train"]
    X_te, y_te = data["X_test"][cols], data["y_test"]
    X_st = data["X_stress"][cols]

    with mlflow.start_run(run_name=f"{feature_set}__{model_type}"):
        hyper = LOGREG_PARAMS if model_type == "logreg" else XGB_PARAMS
        mlflow.log_params({"data_version": DATA_VERSION, "feature_set": feature_set, "model_type": model_type,
                           "n_features": len(cols), "features": ",".join(cols),
                           **{f"hp_{k}": v for k, v in hyper.items()}})

        # Grouped CV on train (folds built once, shared by all runs).
        pr, roc = [], []
        for tr_idx, va_idx in folds:
            m = make_model(model_type).fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
            p = m.predict_proba(X_tr.iloc[va_idx])[:, 1]
            pr.append(average_precision_score(y_tr.iloc[va_idx], p))
            roc.append(roc_auc_score(y_tr.iloc[va_idx], p))
        metrics = {"cv_pr_auc_mean": np.mean(pr), "cv_pr_auc_std": np.std(pr),
                   "cv_roc_auc_mean": np.mean(roc)}

        # Final fit; test set is evaluated exactly once.
        model = make_model(model_type).fit(X_tr, y_tr)
        p_te = model.predict_proba(X_te)[:, 1]
        pred = (p_te >= THRESHOLD).astype(int)
        metrics.update({
            "test_pr_auc": average_precision_score(y_te, p_te),
            "test_roc_auc": roc_auc_score(y_te, p_te),
            "precision": precision_score(y_te, pred, zero_division=0),
            "recall": recall_score(y_te, pred),
            "f1": f1_score(y_te, pred),
        })

        # Source shortcut check: mean P(phishing) on legitimate test rows per source.
        legit = (data["y_test"] == 0).to_numpy()
        for src in ("phiusiil", "umbrella"):
            m = legit & (data["test_source"] == src)
            metrics[f"test_mean_prob_legit_{src}"] = p_te[m].mean()

        # Stress test: every hostname is legitimate, so any flag is a false positive.
        p_st = model.predict_proba(X_st)[:, 1]
        flagged = pd.DataFrame({"hostname": data["stress_hosts"], "phishing_prob": p_st})
        flagged = flagged[flagged.phishing_prob >= THRESHOLD].sort_values(
            "phishing_prob", ascending=False)
        metrics["stress_fp_rate"] = len(flagged) / len(p_st)
        mlflow.log_table(flagged, "stress_flagged_hostnames.json")
        mlflow.log_text(flagged.to_csv(index=False), "stress_flagged_hostnames.csv")

        importance = None
        if model_type == "xgboost":
            importance = (pd.DataFrame({"feature": cols,
                                        "importance": model.feature_importances_})
                          .sort_values("importance", ascending=False))
            mlflow.log_table(importance, "feature_importance.json")
            mlflow.log_text(importance.to_csv(index=False), "feature_importance.csv")

        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})

        example = X_tr.head(5)
        sig = infer_signature(example, model.predict_proba(example)[:, 1])
        log_model = mlflow.sklearn.log_model if model_type == "logreg" else mlflow.xgboost.log_model
        log_model(model, name="model", input_example=example, signature=sig)

    return {"feature_set": feature_set, "model_type": model_type, **metrics,
            "flagged": flagged, "importance": importance}


def main() -> None:
    warnings.filterwarnings("ignore")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    train, test = pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)
    stress = pd.read_csv(STRESS_PATH)
    # float64 everywhere keeps the MLflow model signature free of int/float issues.
    data = {
        "X_train": extract_features_df(train["hostname"]).astype("float64"),
        "y_train": train["label"].reset_index(drop=True),
        "X_test": extract_features_df(test["hostname"]).astype("float64"),
        "y_test": test["label"].reset_index(drop=True),
        "test_source": test["source"].to_numpy(),
        "X_stress": extract_features_df(stress["hostname"]).astype("float64"),
        "stress_hosts": stress["hostname"].tolist(),
    }

    # Build the (slow) grouped folds once and reuse them for every run.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(data["X_train"], data["y_train"], groups=train["registered_domain"]))

    results = [run_one(fs, mt, cols, data, folds)
               for fs, cols in FEATURE_SETS.items() for mt in ("logreg", "xgboost")]

    cols = ["cv_pr_auc_mean", "cv_pr_auc_std", "test_pr_auc", "test_roc_auc",
            "precision", "recall", "stress_fp_rate"]
    table = pd.DataFrame(results).set_index(["feature_set", "model_type"])[cols]
    pd.set_option("display.width", 200)
    print("\n=== Comparison ===")
    print(table.round(4).to_string())

    print("\n=== Source shortcut check (mean P(phishing), legitimate test rows) ===")
    for r in results:
        print(f"{r['feature_set']:<24}{r['model_type']:<9}"
              f"phiusiil={r['test_mean_prob_legit_phiusiil']:.4f}  "
              f"umbrella={r['test_mean_prob_legit_umbrella']:.4f}")

    print("\n=== Top 10 XGBoost feature importances ===")
    for r in results:
        if r["importance"] is not None:
            print(f"\n[{r['feature_set']}]")
            print(r["importance"].head(10).round(4).to_string(index=False))

    print("\n=== Stress hostnames flagged by the best model per feature set "
          "(best = highest cv_pr_auc_mean) ===")
    for fs in FEATURE_SETS:
        best = max((r for r in results if r["feature_set"] == fs),
                   key=lambda r: r["cv_pr_auc_mean"])
        print(f"\n[{fs}] best={best['model_type']} "
              f"({len(best['flagged'])}/{len(data['stress_hosts'])} flagged)")
        print(best["flagged"].round(3).to_string(index=False) or "none")


if __name__ == "__main__":
    main()
