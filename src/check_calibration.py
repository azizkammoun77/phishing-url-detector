"""Calibration check of the champion model on the TEST set.

Prints the Brier score, a 10-bin reliability table, the expected calibration
error (ECE) and a plain-language verdict; saves reports/calibration.png.
Run from the project root:  python -m src.check_calibration
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.monitoring_data import load_champion, load_test, predict_proba

REPORTS = Path(__file__).resolve().parents[1] / "reports"
N_BINS = 10
ECE_OK = 0.02       # "well calibrated" limit on ECE
BIN_GAP_OK = 0.10   # no well-populated bin may be off by more than this
MIN_BIN_ROWS = 200  # bins smaller than this are too noisy to judge


def reliability_table(y: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> pd.DataFrame:
    """Equal-width bins on the predicted probability."""
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = bins == b
        n = int(m.sum())
        rows.append({"bin": f"{b / n_bins:.1f}-{(b + 1) / n_bins:.1f}", "n": n,
                     "mean_predicted": p[m].mean() if n else np.nan,
                     "actual_phishing_rate": y[m].mean() if n else np.nan})
    t = pd.DataFrame(rows)
    t["gap"] = (t["mean_predicted"] - t["actual_phishing_rate"]).abs()
    return t


def expected_calibration_error(t: pd.DataFrame) -> float:
    """ECE = sum over bins of (bin share of rows) * |mean predicted - actual rate|."""
    t = t[t["n"] > 0]
    return float((t["n"] / t["n"].sum() * t["gap"]).sum())


def main() -> None:
    test = load_test()
    model, threshold = load_champion()
    y = test["label"].to_numpy()
    p = predict_proba(model, test["hostname"])

    brier = float(np.mean((p - y) ** 2))
    base_brier = float(np.mean((y.mean() - y) ** 2))  # always predicting the base rate
    t = reliability_table(y, p)
    ece = expected_calibration_error(t)

    print(f"Test rows: {len(y)}, phishing rate {y.mean():.3f}, threshold {threshold:.4f}")
    print(f"Brier score: {brier:.4f}  (baseline that always says the base rate: {base_brier:.4f})")
    print(f"Expected calibration error (ECE): {ece:.4f}\n")
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    big = t[t["n"] >= MIN_BIN_ROWS]
    worst = big.loc[big["gap"].idxmax()]
    ok = ece <= ECE_OK and worst["gap"] <= BIN_GAP_OK
    print("\nVERDICT:")
    if ok:
        print(f"  Probabilities look trustworthy. ECE is {ece:.3f} (limit {ECE_OK}) and the "
              f"worst well-populated bin is off by {worst['gap']:.3f} (bin {worst['bin']}). "
              "A predicted 0.30 really does mean roughly 30% phishing, so averaging the "
              "probabilities is a reasonable way to ESTIMATE precision and recall without labels.")
    else:
        print(f"  Probabilities are NOT reliable enough. ECE is {ece:.3f} (limit {ECE_OK}); "
              f"the worst well-populated bin ({worst['bin']}) is off by {worst['gap']:.3f} "
              f"(limit {BIN_GAP_OK}). Label-free performance estimates would be biased.")
    print("  Caveat: this holds for traffic that looks like the test set. If the kind of "
          "phishing changes (concept drift), the probabilities can stay confident and be wrong, "
          "and any label-free estimate will miss it.")

    REPORTS.mkdir(exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    a1.plot([0, 1], [0, 1], "--", color="#888", label="perfect calibration")
    a1.plot(t["mean_predicted"], t["actual_phishing_rate"], "o-", color="#2a6fdb", label="champion")
    a1.set(xlabel="mean predicted probability", ylabel="actual phishing rate",
           title=f"Reliability (10 bins) - ECE {ece:.3f}, Brier {brier:.3f}",
           xlim=(0, 1), ylim=(0, 1))
    a1.legend(loc="upper left")
    a2.bar(range(N_BINS), t["n"], color="#2a6fdb")
    a2.set(xticks=range(N_BINS), xticklabels=t["bin"], yscale="log",
           xlabel="predicted probability bin", ylabel="rows (log scale)",
           title="How many test rows fall in each bin")
    a2.tick_params(axis="x", rotation=60)
    fig.tight_layout()
    fig.savefig(REPORTS / "calibration.png", dpi=130)
    print(f"\nSaved {REPORTS / 'calibration.png'}")


if __name__ == "__main__":
    main()
