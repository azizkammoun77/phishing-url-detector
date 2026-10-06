"""Hostname-only feature extraction (version 1).

Pure Python, no network calls: hostnames are only ever handled as text.
tldextract is configured with its bundled suffix list, so it never goes online.
"""
import ipaddress
import math
import re
from collections import Counter

import pandas as pd
import tldextract

# NOTE: the lists below are a starting point, not an exhaustive or authoritative
# source. Extend and tune them as the project evolves.

# brand keyword -> registered domains that legitimately belong to that brand
BRAND_DOMAINS = {
    "paypal": {"paypal.com", "paypal.me"},
    "apple": {"apple.com", "icloud.com"},
    "microsoft": {"microsoft.com", "live.com", "office.com", "microsoftonline.com"},
    "amazon": {"amazon.com", "amazon.co.uk", "amazon.de", "amazon.fr", "amazon.in",
               "amazonaws.com"},
    "netflix": {"netflix.com"},
    "google": {
        "google.com", "google.co.uk", "google.fr", "google.de", "googleapis.com",
        "gmail.com", "youtube.com",
    },
    "facebook": {"facebook.com", "fb.com", "facebookmail.com", "fbcdn.net"},
    "instagram": {"instagram.com"},
    "whatsapp": {"whatsapp.com", "whatsapp.net"},
    "dhl": {"dhl.com", "dhl.de"},
    "office365": {"office365.com", "office.com", "microsoft.com"},
    "outlook": {"outlook.com", "live.com", "office.com"},
    "chase": {"chase.com"},
    "wellsfargo": {"wellsfargo.com"},
    "bankofamerica": {"bankofamerica.com"},
    "citibank": {"citibank.com", "citi.com"},
    "hsbc": {"hsbc.com", "hsbc.co.uk"},
    "barclays": {"barclays.com", "barclays.co.uk"},
}

# Short or ambiguous brand names: only a match when the brand is a whole token
# of the hostname split on "." and "-" (so "pineapple" or "purchase" don't hit).
# All other brands are distinctive enough for plain substring matching.
WHOLE_TOKEN_BRANDS = frozenset({"apple", "chase", "dhl", "citibank", "hsbc", "outlook"})

_TOKEN_SPLIT = re.compile(r"[.\-]")

SUSPICIOUS_KEYWORDS = (
    "login", "secure", "verify", "account", "update",
    "signin", "banking", "confirm", "wallet", "support",
)

FREE_HOSTING_DOMAINS = frozenset({
    "web.app", "firebaseapp.com", "repl.co", "weeblysite.com", "ipfs.io",
    "github.io", "netlify.app", "vercel.app", "pages.dev", "wixsite.com",
    "blogspot.com", "000webhostapp.com", "glitch.me",
})

RISKY_TLDS = frozenset({
    "xyz", "top", "club", "online", "site", "icu", "buzz", "live", "shop", "info",
})

VOWELS = frozenset("aeiou")

# Bundled public suffix snapshot only: no network access.
_extract = tldextract.TLDExtract(suffix_list_urls=(), cache_dir=None)

FEATURE_NAMES = (
    "hostname_length", "num_labels", "num_subdomain_levels", "num_digits",
    "digit_ratio", "num_hyphens", "longest_label_length", "domain_entropy",
    "vowel_ratio", "is_ip", "is_punycode", "brand_impersonation",
    "suspicious_keyword_count", "is_free_hosting", "tld_risky",
)


def _is_ip(hostname: str) -> bool:
    # Cheap pre-check so ordinary hostnames never hit the exception path.
    if ":" not in hostname and not hostname.replace(".", "").isdigit():
        return False
    try:
        ipaddress.ip_address(hostname)
        return True
    except ValueError:
        return False


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    n = len(text)
    return -sum(c / n * math.log2(c / n) for c in Counter(text).values())


def extract_features(hostname: str) -> dict:
    """Return the version-1 feature dict for a normalized hostname."""
    hostname = str(hostname).strip().lower()
    labels = hostname.split(".") if hostname else []
    is_ip = _is_ip(hostname)

    if is_ip:
        # No domain structure for a raw IP.
        registered, domain_name, suffix, subdomain = hostname, hostname, "", ""
    else:
        ext = _extract(hostname)
        registered = ext.top_domain_under_public_suffix or hostname
        domain_name, suffix, subdomain = ext.domain, ext.suffix, ext.subdomain

    num_digits = sum(ch.isdigit() for ch in hostname)
    hostname_length = len(hostname)
    tokens = set(_TOKEN_SPLIT.split(hostname))
    brand_hit = any(
        registered not in real
        and ((brand in tokens) if brand in WHOLE_TOKEN_BRANDS else (brand in hostname))
        for brand, real in BRAND_DOMAINS.items()
    )

    return {
        "hostname_length": hostname_length,
        "num_labels": len(labels),
        "num_subdomain_levels": len(subdomain.split(".")) if subdomain else 0,
        "num_digits": num_digits,
        "digit_ratio": num_digits / hostname_length if hostname_length else 0.0,
        "num_hyphens": hostname.count("-"),
        "longest_label_length": max((len(label) for label in labels), default=0),
        "domain_entropy": _entropy(domain_name),
        "vowel_ratio": (
            sum(ch in VOWELS for ch in domain_name) / len(domain_name)
            if domain_name else 0.0
        ),
        "is_ip": int(is_ip),
        "is_punycode": int("xn--" in hostname),
        "brand_impersonation": int(brand_hit),
        "suspicious_keyword_count": sum(kw in hostname for kw in SUSPICIOUS_KEYWORDS),
        "is_free_hosting": int(registered in FREE_HOSTING_DOMAINS),
        "tld_risky": int(suffix.rsplit(".", 1)[-1] in RISKY_TLDS),
    }


def extract_features_df(hostnames) -> pd.DataFrame:
    """Apply extract_features to a list/Series of hostnames; keeps a Series' index."""
    index = hostnames.index if isinstance(hostnames, pd.Series) else None
    rows = [extract_features(h) for h in hostnames]
    return pd.DataFrame(rows, index=index, columns=list(FEATURE_NAMES))
