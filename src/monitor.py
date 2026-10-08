"""Production monitoring of the simulated traffic (text only, no network access).

Reference period : TEST rows that the simulation never used (the 20% random holdout from
                   src/monitoring_data.py), scored by the same champion model. Using
                   unused rows keeps the reference independent of the analysis period; a
                   random holdout keeps its mix natural (about 21% phishing). The full
                   test set would overlap with the simulated rows.
Analysis period  : the 30 simulated days in logs/simulated_predictions.jsonl, joined with
                   logs/simulated_labels.csv only for the "realized" checks.

Per day it computes:
 a. Data drift per hand-made feature. Score = KS statistic (continuous features) or the
    absolute difference in proportions (binary features); both are on a 0-1 scale and,
    unlike p-values, do not explode with sample size. A feature "drifts" when its score
    exceeds ITS OWN threshold = max(DRIFT_FLOOR, 1.1 x the 99th percentile of the score
    seen when a random 2,000-row slice of the reference is compared with the rest of the
    reference, i.e. pure sampling noise). DRIFT_FLOOR (0.05) keeps rare features from
    triggering on a handful of rows.
 b. Prediction drift: share of safe / phishing / review decisions, mean probability.
 c. Estimated performance WITHOUT labels (CBPE idea from NannyML), see cbpe().
 d. Realized precision and recall WITH labels.
 e. Alerts (day, check, value, limit). Limits for the shares, mean probability and the
    four precision/recall series are "reference mean +/- 3 standard deviations", where the
    standard deviation comes from many random day-sized (2,000-row) slices of the
    reference. The drifting-feature count alerts when MORE THAN 2 features drift.

"Flagged" means probability >= threshold, i.e. decision "phishing" or "review". Precision
and recall (estimated and realized) are about flagged rows vs. true phishing labels.

NannyML was not used (see the notebook for why); CBPE is implemented directly with numpy
because it is only a few lines.

Run:  python -m src.monitor
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.features import FEATURE_NAMES, extract_features_df
from src.monitoring_data import (load_champion, load_test, predict_proba,
                                 split_reference_pool)
from src.simulate_traffic import LABELS_PATH, LOG_PATH, SCENARIOS, SIM_START

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
BINARY_FEATURES = ("is_ip", "is_punycode", "brand_impersonation", "is_free_hosting",
                   "tld_risky", "name_brand_lookalike", "name_homoglyph_brand")
DRIFT_FLOOR = 0.05
DRIFT_NULL_MARGIN = 1.1
MAX_DRIFTING_FEATURES = 2      # alert when more than this many features drift
SIGMA = 3.0
DAY_SIZE = 2000
N_NULL = 200
SEED = 42
SCENARIO_COLORS = ["#e8eef7", "#fbeed5", "#f3dfe9"]


# ----------------------------------------------------------------------- metrics
def cbpe(p: np.ndarray, threshold: float) -> tuple[float, float]:
    """Confidence-based performance estimation (the CBPE idea from NannyML).

    If the probabilities are calibrated, row i is phishing with probability p_i, so
    without any label we can take expectations:
        flagged rows (p >= threshold):  expected TP = sum p_i,  expected FP = sum (1 - p_i)
        unflagged rows (p <  threshold): expected FN = sum p_i
        precision = TP / (TP + FP)       recall = TP / (TP + FN)
    """
    flagged = p >= threshold
    tp = p[flagged].sum()
    fp = (1 - p[flagged]).sum()
    fn = p[~flagged].sum()
    return (tp / (tp + fp) if tp + fp else np.nan), (tp / (tp + fn) if tp + fn else np.nan)


def cbpe_interval(p: np.ndarray, threshold: float, n_draws: int = 200,
                  seed: int = SEED) -> tuple[tuple[float, float], tuple[float, float]]:
    """95% sampling interval of the estimates: draw labels ~ Bernoulli(p) many times."""
    rng = np.random.default_rng(seed)
    flagged = p >= threshold
    y = rng.random((n_draws, len(p))) < p
    tp = (y & flagged).sum(1)
    prec = tp / max(int(flagged.sum()), 1)
    rec = tp / np.maximum(y.sum(1), 1)
    return (tuple(np.percentile(prec, [2.5, 97.5])), tuple(np.percentile(rec, [2.5, 97.5])))


def realized(flagged: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    tp = int((flagged & (y == 1)).sum())
    return (tp / flagged.sum() if flagged.sum() else np.nan,
            tp / (y == 1).sum() if (y == 1).any() else np.nan)


def drift_score(name: str, ref: pd.Series, cur: pd.Series) -> float:
    if name in BINARY_FEATURES:
        return abs(float(cur.mean()) - float(ref.mean()))
    return float(ks_2samp(ref, cur).statistic)


def decisions_of(p: np.ndarray, free: np.ndarray, threshold: float) -> np.ndarray:
    """Same rule as the API: below threshold safe, free hosting above it -> review."""
    return np.where(p < threshold, "safe", np.where(free == 1, "review", "phishing"))


def day_metrics(p, free, y, threshold) -> dict:
    """Prediction-drift, estimated and realized metrics for one batch of rows."""
    dec = decisions_of(p, free, threshold)
    est_p, est_r = cbpe(p, threshold)
    real_p, real_r = realized(p >= threshold, y)
    return {"share_safe": float((dec == "safe").mean()),
            "share_phishing": float((dec == "phishing").mean()),
            "share_review": float((dec == "review").mean()),
            "mean_probability": float(p.mean()),
            "est_precision": est_p, "est_recall": est_r,
            "real_precision": real_p, "real_recall": real_r}


# --------------------------------------------------------------------- reference
def build_reference(model):
    """Reference data: features, probabilities and labels of the unused test rows."""
    reference, _ = split_reference_pool(load_test())
    feats = extract_features_df(reference["hostname"]).reset_index(drop=True)
    p = predict_proba(model, reference["hostname"])
    return feats, p, reference["label"].to_numpy()


def null_distribution(feats, p, y, threshold, seed=SEED):
    """What a normal day looks like: many random 2,000-row slices of the reference."""
    rng = np.random.default_rng(seed)
    free = feats["is_free_hosting"].to_numpy()
    n = len(p)
    scores, metrics = [], []
    for _ in range(N_NULL):
        idx = rng.choice(n, DAY_SIZE, replace=False)
        rest = np.setdiff1d(np.arange(n), idx)
        scores.append({f: drift_score(f, feats[f].iloc[rest], feats[f].iloc[idx])
                       for f in FEATURE_NAMES})
        metrics.append(day_metrics(p[idx], free[idx], y[idx], threshold))
    scores, metrics = pd.DataFrame(scores), pd.DataFrame(metrics)
    thresholds = scores.quantile(0.99).mul(DRIFT_NULL_MARGIN).clip(lower=DRIFT_FLOOR)
    counts = (scores > thresholds).sum(axis=1)
    return thresholds, metrics, counts


def limits_from(metrics: pd.DataFrame) -> pd.DataFrame:
    mu, sd = metrics.mean(), metrics.std()
    return pd.DataFrame({"low": mu - SIGMA * sd, "high": mu + SIGMA * sd, "reference": mu})


# ---------------------------------------------------------------------- analysis
def load_analysis() -> pd.DataFrame:
    log = pd.read_json(LOG_PATH, lines=True, dtype={"request_id": str})
    labels = pd.read_csv(LABELS_PATH, dtype={"request_id": str})
    log["day"] = (pd.to_datetime(log["timestamp"], utc=True).dt.normalize()
                  - pd.Timestamp(SIM_START)).dt.days + 1
    df = log.merge(labels[["request_id", "label"]], on="request_id", how="left")
    if df["label"].isna().any():
        raise SystemExit("some logged requests have no label")
    return df


def run_monitoring():
    """Return (daily, drift_scores, thresholds, limits, alerts, info)."""
    model, threshold = load_champion()
    ref_feats, ref_p, ref_y = build_reference(model)
    thresholds, null_metrics, null_counts = null_distribution(ref_feats, ref_p, ref_y,
                                                              threshold)
    limits = limits_from(null_metrics)

    df = load_analysis()
    feats = extract_features_df(df["hostname"]).reset_index(drop=True)
    free = feats["is_free_hosting"].to_numpy()
    daily, scores = [], []
    for day, g in df.groupby("day"):
        idx = g.index.to_numpy()
        s = {f: drift_score(f, ref_feats[f], feats[f].iloc[idx]) for f in FEATURE_NAMES}
        drifting = [f for f in FEATURE_NAMES if s[f] > thresholds[f]]
        p = g["probability"].to_numpy()
        m = day_metrics(p, free[idx], g["label"].to_numpy(), threshold)
        m["n_requests"] = len(g)
        m["true_phishing_share"] = float(g["label"].mean())
        m["n_drifting"] = len(drifting)
        m["drifting_features"] = ", ".join(drifting)
        (pl, ph), (rl, rh) = cbpe_interval(p, threshold)
        m.update(est_precision_low=pl, est_precision_high=ph,
                 est_recall_low=rl, est_recall_high=rh)
        daily.append({"day": day, **m})
        scores.append({"day": day, **s})
    daily = pd.DataFrame(daily).set_index("day")
    scores = pd.DataFrame(scores).set_index("day")

    alerts = []
    for day, row in daily.iterrows():
        if row["n_drifting"] > MAX_DRIFTING_FEATURES:
            alerts.append((day, "drifting_features", row["n_drifting"], MAX_DRIFTING_FEATURES))
        for check in limits.index:
            v, lo, hi = row[check], limits.loc[check, "low"], limits.loc[check, "high"]
            if v < lo:
                alerts.append((day, check, v, lo))
            elif v > hi:
                alerts.append((day, check, v, hi))
    alerts = pd.DataFrame(alerts, columns=["day", "check", "value", "limit"])
    alerts["direction"] = np.where(alerts["value"] > alerts["limit"], "above", "below")
    info = {"threshold": threshold, "ref_rows": len(ref_p), "null_counts": null_counts,
            "ref_phishing_share": float(ref_y.mean())}
    return daily, scores, thresholds, limits, alerts, info


# ------------------------------------------------------------------------ charts
def _shade(ax):
    for (lo, hi, _), c in zip(SCENARIOS, SCENARIO_COLORS):
        ax.axvspan(lo - 0.5, hi + 0.5, color=c, zorder=0)
    ax.set_xlim(0.5, 30.5)
    ax.set_xlabel("simulated day")


def _label_scenarios(ax):
    for lo, hi, name in SCENARIOS:
        ax.text((lo + hi) / 2, 1.02, name, transform=ax.get_xaxis_transform(),
                ha="center", fontsize=8, color="#444")


def _save(fig, path):
    REPORTS.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    return fig


def chart_drifting_features(daily, path=REPORTS / "drifting_features.png"):
    fig, ax = plt.subplots(figsize=(10, 3.6))
    _shade(ax)
    ax.bar(daily.index, daily["n_drifting"], color="#2a6fdb", zorder=2)
    ax.axhline(MAX_DRIFTING_FEATURES, color="#c0392b", ls="--", zorder=3)
    ax.text(30.4, MAX_DRIFTING_FEATURES, "alert limit ", color="#c0392b", va="bottom",
            ha="right", fontsize=8)
    ax.set_ylabel("features drifting (of 22)")
    ax.set_title("Data drift: how many features look different from the reference",
                 pad=18, loc="left")
    _label_scenarios(ax)
    return _save(fig, path)


def chart_drift_heatmap(scores, thresholds, path=REPORTS / "drift_heatmap.png"):
    ratio = scores.div(thresholds, axis=1).T      # above 1 means "drifting"
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(ratio.clip(upper=4), aspect="auto", cmap="viridis", vmin=0, vmax=4,
                   extent=(0.5, 30.5, len(ratio) - 0.5, -0.5))
    ax.set_yticks(range(len(ratio)), ratio.index, fontsize=8)
    ax.set_xlabel("simulated day")
    ax.set_title("Drift score / feature threshold (above 1 = drifting)", loc="left")
    fig.colorbar(im, ax=ax, label="score / threshold (capped at 4)")
    return _save(fig, path)


def chart_decision_shares(daily, limits, path=REPORTS / "decision_shares.png"):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 6.5), sharex=True)
    for ax in (a1, a2):
        _shade(ax)
    for col, color, lab in (("share_safe", "#2a6fdb", "safe"),
                            ("share_phishing", "#d97706", "phishing"),
                            ("share_review", "#7a3fb0", "review")):
        a1.plot(daily.index, daily[col], "o-", ms=3, color=color, label=lab, zorder=3)
        a1.axhspan(limits.loc[col, "low"], limits.loc[col, "high"], color=color, alpha=0.12,
                   zorder=1)
    a1.set_ylabel("share of requests")
    a1.set_title("Prediction drift: decision shares (bands = normal range from the reference)",
                 pad=18, loc="left")
    a1.legend(ncol=3, loc="center right")
    _label_scenarios(a1)
    a2.plot(daily.index, daily["mean_probability"], "o-", ms=3, color="#2a6fdb", zorder=3)
    a2.axhspan(limits.loc["mean_probability", "low"], limits.loc["mean_probability", "high"],
               color="#2a6fdb", alpha=0.12, zorder=1)
    a2.set_ylabel("mean predicted probability")
    return _save(fig, path)


def chart_estimated_vs_realized(daily, limits, path=REPORTS / "estimated_vs_realized.png"):
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for ax, metric, title in zip(axes, ("precision", "recall"), ("Precision", "Recall")):
        _shade(ax)
        ax.fill_between(daily.index, daily[f"est_{metric}_low"], daily[f"est_{metric}_high"],
                        color="#2a6fdb", alpha=0.18, zorder=2)
        ax.plot(daily.index, daily[f"est_{metric}"], "o-", ms=3, color="#2a6fdb",
                label="estimated without labels (CBPE, band = 95% range)", zorder=3)
        ax.plot(daily.index, daily[f"real_{metric}"], "s--", ms=3, color="#d97706",
                label="realized with labels", zorder=3)
        ax.axhspan(limits.loc[f"est_{metric}", "low"], limits.loc[f"est_{metric}", "high"],
                   color="#888", alpha=0.15, zorder=1)
        ax.set_ylabel(title)
    axes[0].set_title("Estimated vs. realized performance (grey band = normal range)",
                      pad=18, loc="left")
    axes[0].legend(loc="lower left")
    _label_scenarios(axes[0])
    return _save(fig, path)


def main() -> None:
    daily, scores, thresholds, limits, alerts, info = run_monitoring()
    REPORTS.mkdir(exist_ok=True)
    daily.to_csv(REPORTS / "monitoring_daily.csv")
    chart_drifting_features(daily)
    chart_drift_heatmap(scores, thresholds)
    chart_decision_shares(daily, limits)
    chart_estimated_vs_realized(daily, limits)

    print(f"reference rows {info['ref_rows']}, threshold {info['threshold']:.4f}, "
          f"null max drifting features {int(info['null_counts'].max())}")
    print("\nlimits (reference mean +/- 3 sd):")
    print(limits.round(4).to_string())
    print("\nALERTS (day, check, value, limit):")
    print(alerts.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    cols = ["true_phishing_share", "n_drifting", "share_safe", "share_phishing",
            "share_review", "mean_probability", "est_precision", "real_precision",
            "est_recall", "real_recall"]
    print("\nDAILY:")
    print(daily[cols].round(3).to_string())


if __name__ == "__main__":
    main()
