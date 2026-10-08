"""SIMULATED production traffic: 30 days of ~2,000 requests sent through the real API.

THIS IS A SIMULATION. No real users, no real traffic, and no URL or hostname is ever
visited, fetched or resolved: hostnames are only text rows of data/processed/test.csv
(never train.csv). Each request goes through the real FastAPI app via TestClient, which
scores the text with the champion model and appends a line to
logs/simulated_predictions.jsonl (LOG_PATH). The true labels are written SEPARATELY to
logs/simulated_labels.csv (request_id, day, label) to mimic ground truth that only
arrives later (e.g. after analysts review reports).

Sampling pool: the test set is first split (src/monitoring_data.py) into a 20% random
REFERENCE holdout, which is never sampled here, and an 80% POOL. Every day is drawn from
the pool with random_state=42-derived seeds, without replacement inside a day. Hostnames
may recur on different days (as in real traffic); the clean-looking phishing pool is
smaller than 10 days x 500 rows, so repeats are unavoidable there.

Scenarios (2,000 requests per day):
  Days 1-10  "normal": a plain random sample of the pool, i.e. the test set's natural
             mix (about 21% phishing).
  Days 11-20 "free-hosting campaign": 35% phishing (700 rows). 414 of them are ordinary
             phishing drawn at random (the natural ~20.7% share); the 286 EXTRA ones are
             85% free-hosting phishing (is_free_hosting == 1) and 15% ordinary phishing.
             Benign rows (1,300) are a random sample of benign hostnames.
  Days 21-30 "clean-looking phishing": 25% phishing (500 rows), sampled ONLY from phishing
             hostnames that look ordinary: no subdomain, no digits, no hyphens, TLD not in
             RISKY_TLDS, not on free hosting (and not an IP). Benign rows (1,500) are a
             random sample of benign hostnames. The model is largely blind to this kind of phishing,
             so recall drops. Note the inputs ALSO shift: removing every phishing hostname with
             a subdomain, digits, hyphens or free hosting changes the overall feature mix.

Each day gets a simulated clock starting at SIM_START (one day per simulated day, request
times spread randomly over the day), injected by replacing src.api.utc_now.

Run from the project root:  python -m src.simulate_traffic
"""
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LOG_PATH = ROOT / "logs" / "simulated_predictions.jsonl"
LABELS_PATH = ROOT / "logs" / "simulated_labels.csv"
SIM_START = datetime(2026, 9, 1, tzinfo=timezone.utc)
SIM_KEY = "simulation-only-key"
SEED = 42
N_PER_DAY = 2000
N_DAYS = 30
SCENARIOS = [(1, 10, "normal"), (11, 20, "free-hosting campaign"),
             (21, 30, "clean-looking phishing")]


def scenario_of(day: int) -> str:
    return next(name for lo, hi, name in SCENARIOS if lo <= day <= hi)


def clean_looking_mask(feats: pd.DataFrame) -> pd.Series:
    """Phishing that looks ordinary: no subdomain/digit/hyphen, safe TLD, not free hosting."""
    return ((feats["num_subdomain_levels"] == 0) & (feats["num_digits"] == 0)
            & (feats["num_hyphens"] == 0) & (feats["tld_risky"] == 0)
            & (feats["is_free_hosting"] == 0) & (feats["is_ip"] == 0))


def build_day(day: int, pool: pd.DataFrame, feats: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return the 2,000 sampled pool rows for one day (a shuffled DataFrame)."""
    rng = np.random.default_rng(seed)
    phish, benign = pool[pool.label == 1], pool[pool.label == 0]
    kind = scenario_of(day)

    if kind == "normal":
        rows = pool.sample(N_PER_DAY, random_state=int(rng.integers(2**31)))
    else:
        n_phish = round((0.35 if kind == "free-hosting campaign" else 0.25) * N_PER_DAY)
        if kind == "free-hosting campaign":
            n_extra = n_phish - round(0.207 * N_PER_DAY)       # phishing above the natural share
            n_free = round(0.85 * n_extra)
            free = phish[feats.loc[phish.index, "is_free_hosting"] == 1]
            p_free = free.sample(n_free, random_state=int(rng.integers(2**31)))
            rest = phish.drop(p_free.index)
            p_rest = rest.sample(n_phish - n_free, random_state=int(rng.integers(2**31)))
            p_rows = pd.concat([p_free, p_rest])
        else:
            clean = phish[clean_looking_mask(feats.loc[phish.index])]
            p_rows = clean.sample(n_phish, random_state=int(rng.integers(2**31)))
        b_rows = benign.sample(N_PER_DAY - n_phish, random_state=int(rng.integers(2**31)))
        rows = pd.concat([p_rows, b_rows])
    return rows.sample(frac=1, random_state=int(rng.integers(2**31)))


def main() -> None:
    # Environment must be set BEFORE importing the API.
    os.environ["API_KEY"] = SIM_KEY
    os.environ["LOG_PATH"] = str(LOG_PATH)
    sys.path.insert(0, str(ROOT))
    from fastapi.testclient import TestClient

    import src.api as api
    from src.features import extract_features_df
    from src.monitoring_data import load_test, split_reference_pool

    test = load_test()
    _, pool = split_reference_pool(test)
    feats = extract_features_df(pool["hostname"])
    feats.index = pool.index
    print(f"pool rows: {len(pool)} (phishing {int(pool.label.sum())})")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.unlink(missing_ok=True)   # outputs of an earlier simulation run only
    labels, failed = [], 0
    clock = {"now": SIM_START}
    api.utc_now = lambda: clock["now"]  # simulated timestamps

    with TestClient(api.app) as client:
        for day in range(1, N_DAYS + 1):
            rows = build_day(day, pool, feats, seed=SEED * 1000 + day)
            day_start = SIM_START + timedelta(days=day - 1)
            offsets = np.sort(np.random.default_rng(SEED + day).uniform(0, 86400, len(rows)))
            for (_, row), off in zip(rows.iterrows(), offsets):
                clock["now"] = day_start + timedelta(seconds=float(off))
                r = client.post("/check-url", json={"url": f"https://{row.hostname}/"},
                                headers={"X-API-Key": SIM_KEY})
                if r.status_code != 200:
                    failed += 1
                    continue
                labels.append((r.headers["X-Request-ID"], day, int(row.label)))
            share = rows.label.mean()
            print(f"day {day:2d} [{scenario_of(day)}] sent {len(rows)}, phishing {share:.1%}",
                  flush=True)

    pd.DataFrame(labels, columns=["request_id", "day", "label"]).to_csv(LABELS_PATH, index=False)
    print(f"wrote {len(labels)} labels to {LABELS_PATH} ({failed} requests failed)")


if __name__ == "__main__":
    main()
