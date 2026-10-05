"""Clean, normalize and split the raw PhiUSIIL URLs into train/test hostname sets.

URLs are treated strictly as text; nothing is ever visited or resolved.
"""
import ipaddress
import re
from pathlib import Path

import pandas as pd
import tldextract
from sklearn.model_selection import StratifiedGroupKFold

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "phiusiil_urls.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"

SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)
RANDOM_STATE = 42

# Bundled public suffix snapshot only: no network access.
_extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


def normalize_hostname(url: str) -> str:
    """Return the lowercased host of a URL string (pure string handling)."""
    s = str(url).strip()
    s = SCHEME_RE.sub("", s)
    host = re.split(r"[/?#]", s, maxsplit=1)[0]
    host = host.rsplit("@", 1)[-1]  # drop user:password@
    if host.startswith("["):  # bracketed IPv6, drop port after "]"
        host = host.split("]", 1)[0].lstrip("[")
    else:
        host = host.split(":", 1)[0]  # drop port
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host.rstrip(".")


def registered_domain(hostname: str) -> str:
    """domain + suffix via tldextract; the IP itself for raw IPs."""
    try:
        ipaddress.ip_address(hostname)
        return hostname
    except ValueError:
        pass
    rd = _extract(hostname).top_domain_under_public_suffix
    return rd or hostname  # no known suffix (e.g. "localhost"): use the host


def main() -> None:
    steps = []

    def log(name: str, before: int, after: int) -> None:
        steps.append((name, before - after, after))

    df = pd.read_csv(RAW_PATH)
    n_raw = len(df)

    n = len(df)
    df = df.drop_duplicates(subset="url")
    log("drop exact duplicate URLs", n, len(df))

    df["hostname"] = df["url"].map(normalize_hostname)
    n = len(df)
    df = df[df["hostname"] != ""]
    log("drop empty hostnames", n, len(df))

    df = df.drop(columns="url")
    n = len(df)
    df = df.drop_duplicates(subset=["hostname", "label"])
    log("dedupe on (hostname, label)", n, len(df))

    labels_per_host = df.groupby("hostname")["label"].nunique()
    conflicting = labels_per_host[labels_per_host > 1].index
    n = len(df)
    df = df[~df["hostname"].isin(conflicting)]
    log(f"drop hostnames with BOTH labels ({len(conflicting)} hostnames)", n, len(df))

    df = df.reset_index(drop=True)
    df["registered_domain"] = df["hostname"].map(registered_domain)

    # Domain-grouped, class-stratified 80/20 split (1 of 5 folds is the test set).
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(
        sgkf.split(df, df["label"], groups=df["registered_domain"])
    )
    cols = ["hostname", "registered_domain", "label"]
    train, test = df.iloc[train_idx][cols], df.iloc[test_idx][cols]

    overlap = set(train["registered_domain"]) & set(test["registered_domain"])
    assert not overlap, f"{len(overlap)} registered domains in both train and test"

    TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_PATH, index=False)
    test.to_csv(TEST_PATH, index=False)

    print("=" * 64)
    print("PREPARE DATA REPORT")
    print("=" * 64)
    print(f"{'raw rows':<58}{n_raw:>10,}")
    print("-" * 64)
    print(f"{'step':<48}{'removed':>8}{'remaining':>10}")
    for name, removed, remaining in steps:
        print(f"{name:<48}{removed:>8,}{remaining:>10,}")
    print("-" * 64)
    print(f"Registered-domain overlap train/test: {len(overlap)} (verified zero)")
    print()
    print(f"{'':<10}{'rows':>9}{'phishing':>10}{'legit':>9}{'% phish':>9}{'domains':>9}")
    for name, part in [("train", train), ("test", test)]:
        phish = int(part["label"].sum())
        print(
            f"{name:<10}{len(part):>9,}{phish:>10,}{len(part) - phish:>9,}"
            f"{phish / len(part) * 100:>8.2f}%{part['registered_domain'].nunique():>9,}"
        )
    print(f"\nSaved {TRAIN_PATH.name} and {TEST_PATH.name} to {TRAIN_PATH.parent}")


if __name__ == "__main__":
    main()
