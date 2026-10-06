"""Clean, normalize and split the raw PhiUSIIL URLs into train/test hostname sets.

URLs are treated strictly as text; nothing is ever visited or resolved.
"""
import ipaddress
import zipfile
from pathlib import Path

import pandas as pd
import tldextract
from sklearn.model_selection import StratifiedGroupKFold

from src.features import FREE_HOSTING_DOMAINS
from src.normalize import normalize_url_to_hostname

ROOT = Path(__file__).resolve().parents[1]
RAW_PATH = ROOT / "data" / "raw" / "phiusiil_urls.csv"
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
TEST_PATH = ROOT / "data" / "processed" / "test.csv"
UMBRELLA_PATH = ROOT / "data" / "raw" / "umbrella-top-1m.csv.zip"
TRANCO_PATH = ROOT / "data" / "raw" / "tranco-top-1m.csv"
STRESS_PATH = ROOT / "data" / "stress" / "legit_subdomains.csv"

TRANCO_TOP_N = 100_000
MAX_PER_DOMAIN = 20

RANDOM_STATE = 42

# Bundled public suffix snapshot only: no network access.
_extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)


def registered_domain(hostname: str) -> str:
    """domain + suffix via tldextract; the IP itself for raw IPs."""
    try:
        ipaddress.ip_address(hostname)
        return hostname
    except ValueError:
        pass
    rd = _extract(hostname).top_domain_under_public_suffix
    return rd or hostname  # no known suffix (e.g. "localhost"): use the host


def has_subdomain(hostname: str) -> bool:
    """True when the hostname has at least one label left of its registered domain."""
    return registered_domain(hostname) != hostname


def build_umbrella_legit(phishing_hosts: set, stress_domains: set):
    """Legitimate hostnames with subdomains from Umbrella; returns (df, step list)."""
    with zipfile.ZipFile(UMBRELLA_PATH) as z:
        with z.open(z.namelist()[0]) as f:
            um = pd.read_csv(f, header=None, names=["rank", "raw"], dtype={"raw": str})
    tr = pd.read_csv(TRANCO_PATH, header=None, names=["rank", "domain"], nrows=TRANCO_TOP_N)
    tranco_top = set(tr["domain"].str.lower())

    steps = []

    def log(name, before, after):
        steps.append((name, before - after, after))

    um["hostname"] = um["raw"].map(normalize_url_to_hostname)
    um = um[um["hostname"] != ""]
    um = um.sort_values("rank").drop_duplicates("hostname")
    steps.append(("umbrella rows after normalization + dedupe", 0, len(um)))

    n = len(um)
    um["registered_domain"] = um["hostname"].map(registered_domain)
    # Valid public suffix and at least one subdomain level (bundled list only).
    ext = um["hostname"].map(_extract)
    valid = ext.map(lambda e: bool(e.suffix))
    um = um[valid & um["hostname"].map(has_subdomain)]
    log("drop no valid suffix / no subdomain level", n, len(um))

    n = len(um)
    um = um[um["registered_domain"].isin(tranco_top)]
    log(f"drop registered domain not in Tranco top {TRANCO_TOP_N:,}", n, len(um))

    n = len(um)
    um = um[~um["hostname"].isin(phishing_hosts)]
    log("drop hostnames present in PhiUSIIL phishing data", n, len(um))

    n = len(um)
    um = um[~um["registered_domain"].isin(stress_domains)]
    log("drop registered domains in the stress set", n, len(um))

    n = len(um)
    um = um.sort_values("rank").groupby("registered_domain").head(MAX_PER_DOMAIN)
    log(f"cap at {MAX_PER_DOMAIN} hostnames per registered domain", n, len(um))

    um = um[["hostname", "registered_domain"]].assign(label=0, source="umbrella")
    return um.reset_index(drop=True), steps


def main() -> None:
    steps = []

    def log(name: str, before: int, after: int) -> None:
        steps.append((name, before - after, after))

    df = pd.read_csv(RAW_PATH)
    n_raw = len(df)

    n = len(df)
    df = df.drop_duplicates(subset="url")
    log("drop exact duplicate URLs", n, len(df))

    df["hostname"] = df["url"].map(normalize_url_to_hostname)
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
    df["source"] = "phiusiil"
    n_phiusiil = len(df)

    # Legitimate hostnames with subdomains from Umbrella (filtered by Tranco).
    stress = pd.read_csv(STRESS_PATH)
    stress_domains = set(stress["hostname"].map(registered_domain))
    um, um_steps = build_umbrella_legit(
        set(df.loc[df["label"] == 1, "hostname"]), stress_domains
    )

    df = pd.concat([df, um], ignore_index=True)
    steps.append(("add Umbrella legitimate rows", -len(um), len(df)))
    n = len(df)
    df = df.drop_duplicates(subset=["hostname", "label"])  # keeps the PhiUSIIL copy
    log("dedupe on (hostname, label) after merge", n, len(df))
    n = len(df)
    labels_per_host = df.groupby("hostname")["label"].nunique()
    conflicting = labels_per_host[labels_per_host > 1].index
    df = df[~df["hostname"].isin(conflicting)]
    log(f"drop hostnames with BOTH labels after merge ({len(conflicting)})", n, len(df))
    df = df.reset_index(drop=True)
    n_umbrella_final = int((df["source"] == "umbrella").sum())

    # Domain-grouped, class-stratified 80/20 split (1 of 5 folds is the test set).
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(
        sgkf.split(df, df["label"], groups=df["registered_domain"])
    )
    cols = ["hostname", "registered_domain", "label", "source"]
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
    print("UMBRELLA FILTERING")
    print(f"{'step':<48}{'removed':>8}{'remaining':>10}")
    for name, removed, remaining in um_steps:
        print(f"{name:<48}{removed:>8,}{remaining:>10,}")
    print(f"Umbrella rows added: {len(um):,}; surviving the final merge: "
          f"{n_umbrella_final:,}; stress-set registered domains: {len(stress_domains)}")
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
    print("\nTrain composition by class (hostnames):")
    print(f"{'':<12}{'% with subdomain':>18}{'% free hosting':>16}")
    free = train["registered_domain"].isin(FREE_HOSTING_DOMAINS)
    sub = train["hostname"].map(has_subdomain)
    for lab, name in [(0, "legitimate"), (1, "phishing")]:
        m = train["label"] == lab
        print(f"{name:<12}{sub[m].mean() * 100:>17.2f}%{free[m].mean() * 100:>15.2f}%")
    print("\nTrain rows by source and class:")
    print(train.groupby(["source", "label"]).size().to_string())
    print(f"\nSaved {TRAIN_PATH.name} and {TEST_PATH.name} to {TRAIN_PATH.parent}")


if __name__ == "__main__":
    main()
