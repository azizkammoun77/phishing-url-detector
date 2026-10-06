"""Train and compare hostname-only phishing models, tracked with MLflow.

Models: XGBoost on hand-made features, a char n-gram logistic regression, and a
stacked XGBoost that also sees the n-gram model's out-of-fold probability.
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
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.features import FEATURE_NAMES, FEATURE_NAMES_V1, extract_features_df

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
STRESS_PATH = ROOT / "data" / "stress" / "legit_subdomains.csv"

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "phishing-hostname-v1"
DATA_VERSION = "v2_umbrella"  # v1 = PhiUSIIL only
FEATURE_VERSION = "v1.1"
RANDOM_STATE = 42
THRESHOLD = 0.5
TARGET_FPR = 0.01

XGB_PARAMS = {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1,
              "eval_metric": "logloss", "random_state": RANDOM_STATE}
NGRAM_PARAMS = {"analyzer": "char_wb", "ngram_range": (3, 5), "min_df": 5,
                "max_features": 200000, "sublinear_tf": True}
LR_PARAMS = {"class_weight": "balanced", "max_iter": 2000}


def make_xgb():
    return XGBClassifier(**XGB_PARAMS)


def make_ngram():
    # Vectorizer lives inside the pipeline, so it is refit on each training fold.
    return Pipeline([("tfidf", ColumnTransformer(
                         [("hostname", TfidfVectorizer(**NGRAM_PARAMS), "hostname")])),
                     ("clf", LogisticRegression(**LR_PARAMS))])


def fit_eval(make, X_tr, y_tr, X_te, X_st, folds):
    """Grouped CV (out-of-fold probs) + full-train fit and predictions."""
    oof = np.zeros(len(X_tr))
    pr, roc = [], []
    for tr_idx, va_idx in folds:
        m = make().fit(X_tr.iloc[tr_idx], y_tr.iloc[tr_idx])
        oof[va_idx] = p = m.predict_proba(X_tr.iloc[va_idx])[:, 1]
        pr.append(average_precision_score(y_tr.iloc[va_idx], p))
        roc.append(roc_auc_score(y_tr.iloc[va_idx], p))
    model = make().fit(X_tr, y_tr)
    return {"oof": oof, "pr": pr, "roc": roc, "model": model,
            "p_te": model.predict_proba(X_te)[:, 1],
            "p_st": model.predict_proba(X_st)[:, 1]}


def pick_threshold(oof, y, target_fpr=TARGET_FPR):
    """Smallest threshold whose OOF false-positive rate on legit rows is <= target."""
    legit = oof[np.asarray(y) == 0]
    q = np.quantile(legit, 1 - target_fpr, method="higher")
    t = float(np.nextafter(q, np.inf))  # predict phishing when p >= t (i.e. p > q)
    assert (legit >= t).mean() <= target_fpr
    return t


def clf_metrics(y, p, thr, prefix):
    y = np.asarray(y)
    pred = p >= thr
    return {f"{prefix}precision": precision_score(y, pred, zero_division=0),
            f"{prefix}recall": recall_score(y, pred),
            f"{prefix}f1": f1_score(y, pred),
            f"{prefix}fpr": pred[y == 0].mean()}


def log_run(name, model_type, feature_set, cols, res, data, extra_params,
            log_model_fn, importance=None, feature_version=FEATURE_VERSION):
    y_te = data["y_test"].to_numpy()
    p_te, p_st = res["p_te"], res["p_st"]
    thr = pick_threshold(res["oof"], data["y_train"])

    metrics = {"cv_pr_auc_mean": np.mean(res["pr"]), "cv_pr_auc_std": np.std(res["pr"]),
               "cv_roc_auc_mean": np.mean(res["roc"]),
               "test_pr_auc": average_precision_score(y_te, p_te),
               "test_roc_auc": roc_auc_score(y_te, p_te),
               "threshold_fpr1": thr,
               "oof_fpr_at_threshold_fpr1": (res["oof"][data["y_train"].to_numpy() == 0] >= thr).mean()}
    # Test metrics: at the OOF-chosen FPR<=1% threshold ("fpr1_") and at 0.5 ("t05_").
    metrics.update(clf_metrics(y_te, p_te, thr, "test_fpr1_"))
    metrics.update(clf_metrics(y_te, p_te, THRESHOLD, "test_t05_"))
    metrics["stress_fp_rate_fpr1"] = (p_st >= thr).mean()
    metrics["stress_fp_rate_t05"] = (p_st >= THRESHOLD).mean()

    # Source shortcut check on legitimate test rows.
    legit = y_te == 0
    for src in ("phiusiil", "umbrella"):
        m = legit & (data["test_source"] == src)
        metrics[f"test_mean_prob_legit_{src}"] = p_te[m].mean()
        metrics[f"test_fpr1_fpr_legit_{src}"] = (p_te[m] >= thr).mean()

    stress = pd.DataFrame({"hostname": data["stress_hosts"], "phishing_prob": p_st})
    flagged = stress[stress.phishing_prob >= thr].sort_values("phishing_prob", ascending=False)
    flagged_05 = stress[stress.phishing_prob >= THRESHOLD].sort_values(
        "phishing_prob", ascending=False)

    with mlflow.start_run(run_name=name):
        mlflow.log_params({"data_version": DATA_VERSION, "feature_version": feature_version,
                           "feature_set": feature_set, "model_type": model_type,
                           "n_features": len(cols), "features": ",".join(cols),
                           **extra_params})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.log_table(flagged, "stress_flagged_hostnames_fpr1.json")
        mlflow.log_text(flagged.to_csv(index=False), "stress_flagged_hostnames_fpr1.csv")
        mlflow.log_text(flagged_05.to_csv(index=False), "stress_flagged_hostnames_t05.csv")
        if importance is not None:
            mlflow.log_table(importance, "feature_importance.json")
            mlflow.log_text(importance.to_csv(index=False), "feature_importance.csv")
            mlflow.log_table(importance.head(15), "feature_importance_top15.json")
        log_model_fn()

    return {"name": name, **metrics, "flagged": flagged, "flagged_05": flagged_05,
            "importance": importance}


def xgb_importance(model, cols):
    return (pd.DataFrame({"feature": cols, "importance": model.feature_importances_})
            .sort_values("importance", ascending=False).reset_index(drop=True))


def main() -> None:
    warnings.filterwarnings("ignore")
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)

    train, test = pd.read_csv(TRAIN_PATH), pd.read_csv(TEST_PATH)
    stress = pd.read_csv(STRESS_PATH)
    y_tr = train["label"].reset_index(drop=True)
    data = {
        "y_train": y_tr,
        "y_test": test["label"].reset_index(drop=True),
        "test_source": test["source"].to_numpy(),
        "stress_hosts": stress["hostname"].tolist(),
    }
    # float64 everywhere keeps the MLflow model signature free of int/float issues.
    X_tr = extract_features_df(train["hostname"]).astype("float64").reset_index(drop=True)
    X_te = extract_features_df(test["hostname"]).astype("float64").reset_index(drop=True)
    X_st = extract_features_df(stress["hostname"]).astype("float64").reset_index(drop=True)
    H_tr = train[["hostname"]].reset_index(drop=True)
    H_te = test[["hostname"]].reset_index(drop=True)
    H_st = stress[["hostname"]].reset_index(drop=True)

    # Build the grouped folds once and reuse them for every model.
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    folds = list(cv.split(X_tr, y_tr, groups=train["registered_domain"]))

    v1_cols, all_cols = list(FEATURE_NAMES_V1), list(FEATURE_NAMES)
    xgb_hp = {f"hp_{k}": v for k, v in XGB_PARAMS.items()}
    results = []

    def log_xgb(model, X):
        ex = X.head(5)
        mlflow.xgboost.log_model(model, name="model", input_example=ex,
                                 signature=infer_signature(ex, model.predict_proba(ex)[:, 1]))

    # 0) Previous best re-run with the same folds, for a like-for-like comparison.
    print("fitting xgb_baseline (v1 features)...", flush=True)
    r = fit_eval(make_xgb, X_tr[v1_cols], y_tr, X_te[v1_cols], X_st[v1_cols], folds)
    results.append(log_run(
        "A_all__xgboost_baseline", "xgboost", "A_all_v1_features", v1_cols, r, data, xgb_hp,
        lambda: log_xgb(r["model"], X_tr[v1_cols]), xgb_importance(r["model"], v1_cols),
        feature_version="v1"))

    # a) XGBoost on all hand-made features.
    print("fitting xgb_features...", flush=True)
    r = fit_eval(make_xgb, X_tr[all_cols], y_tr, X_te[all_cols], X_st[all_cols], folds)
    results.append(log_run(
        "xgb_features", "xgboost", "all_handmade", all_cols, r, data, xgb_hp,
        lambda: log_xgb(r["model"], X_tr[all_cols]), xgb_importance(r["model"], all_cols)))

    # b) Char n-gram logistic regression (vectorizer refit inside every fold).
    print("fitting ngram_lr...", flush=True)
    ng = fit_eval(make_ngram, H_tr, y_tr, H_te, H_st, folds)

    def log_ngram():
        ex = H_tr.head(5)
        mlflow.sklearn.log_model(ng["model"], name="model", input_example=ex,
                                 signature=infer_signature(ex, ng["model"].predict_proba(ex)[:, 1]))

    ng_params = {**{f"ngram_{k}": v for k, v in NGRAM_PARAMS.items()},
                 **{f"hp_{k}": v for k, v in LR_PARAMS.items()}}
    results.append(log_run("ngram_lr", "ngram_logreg", "hostname_char_ngrams", ["hostname"],
                           ng, data, ng_params, log_ngram))

    # c) Stacked: hand-made features + n-gram probability. Train column is out-of-fold;
    #    test/stress use the n-gram model fitted on the full train set.
    print("fitting stacked...", flush=True)
    st_cols = all_cols + ["ngram_prob"]
    S_tr = X_tr[all_cols].assign(ngram_prob=ng["oof"])
    S_te = X_te[all_cols].assign(ngram_prob=ng["p_te"])
    S_st = X_st[all_cols].assign(ngram_prob=ng["p_st"])
    r = fit_eval(make_xgb, S_tr, y_tr, S_te, S_st, folds)

    def log_stacked():
        log_xgb(r["model"], S_tr)
        ex = H_tr.head(5)
        mlflow.sklearn.log_model(ng["model"], name="ngram_model", input_example=ex,
                                 signature=infer_signature(ex, ng["model"].predict_proba(ex)[:, 1]))

    results.append(log_run("stacked", "xgboost_stacked", "all_handmade+ngram_prob", st_cols,
                           r, data, {**xgb_hp, **{f"ngram_{k}": v for k, v in NGRAM_PARAMS.items()}},
                           log_stacked, xgb_importance(r["model"], st_cols)))

    report(results, len(data["stress_hosts"]))


def report(results, n_stress):
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    cols = ["cv_pr_auc_mean", "cv_pr_auc_std", "test_pr_auc", "threshold_fpr1",
            "test_fpr1_precision", "test_fpr1_recall", "test_fpr1_fpr",
            "stress_fp_rate_fpr1", "stress_fp_rate_t05"]
    table = pd.DataFrame(results).set_index("name")[cols]
    print("\n=== Comparison (fpr1 = threshold with OOF FPR<=1%) ===")
    print(table.round(4).to_string())

    cols05 = ["test_t05_precision", "test_t05_recall", "test_t05_f1", "test_t05_fpr",
              "test_fpr1_f1"]
    print("\n=== Test metrics at 0.5 (and F1 at fpr1) ===")
    print(pd.DataFrame(results).set_index("name")[cols05].round(4).to_string())

    print("\n=== Source shortcut check (legitimate TEST rows) ===")
    for r in results:
        print(f"{r['name']:<26}mean P: phiusiil={r['test_mean_prob_legit_phiusiil']:.4f} "
              f"umbrella={r['test_mean_prob_legit_umbrella']:.4f} | FPR@fpr1: "
              f"phiusiil={r['test_fpr1_fpr_legit_phiusiil']:.4f} "
              f"umbrella={r['test_fpr1_fpr_legit_umbrella']:.4f}")

    for r in results:
        if r["importance"] is not None and r["name"] in ("xgb_features", "stacked"):
            print(f"\n=== Top 15 importances: {r['name']} ===")
            print(r["importance"].head(15).round(4).to_string(index=False))

    print("\n=== Stress hostnames flagged at threshold_fpr1 ===")
    for r in results:
        print(f"\n[{r['name']}] {len(r['flagged'])}/{n_stress} flagged")
        if len(r["flagged"]):
            print(r["flagged"].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
