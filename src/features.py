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
from rapidfuzz.distance import Levenshtein

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

FEATURE_NAMES_V1 = (
    "hostname_length", "num_labels", "num_subdomain_levels", "num_digits",
    "digit_ratio", "num_hyphens", "longest_label_length", "domain_entropy",
    "vowel_ratio", "is_ip", "is_punycode", "brand_impersonation",
    "suspicious_keyword_count", "is_free_hosting", "tld_risky",
)
# Features computed on the "effective name" plus lookalike detection (feature_version v1.1).
FEATURE_NAMES_V1_1 = (
    "name_length", "name_entropy", "name_digit_ratio", "name_num_hyphens",
    "name_keyword_count", "name_brand_lookalike", "name_homoglyph_brand",
)
FEATURE_NAMES = FEATURE_NAMES_V1 + FEATURE_NAMES_V1_1

# Only brands this long are used for edit-distance lookalike matching.
LOOKALIKE_MIN_BRAND_LEN = 5
LOOKALIKE_MIN_TOKEN_LEN = 5
LOOKALIKE_MAX_DISTANCE = 2
# Applied in this order: multi-character sequences first.
_HOMOGLYPH_SEQUENCES = (("rn", "m"), ("vv", "w"))
_HOMOGLYPH_CHARS = str.maketrans({"0": "o", "1": "l", "3": "e", "5": "s"})
_LOOKALIKE_BRANDS = tuple(b for b in BRAND_DOMAINS if len(b) >= LOOKALIKE_MIN_BRAND_LEN)


def registered_domain_of(hostname: str) -> str:
    """Registered domain of a normalized hostname (the host itself for raw IPs)."""
    hostname = str(hostname).strip().lower()
    if _is_ip(hostname):
        return hostname
    return _extract(hostname).top_domain_under_public_suffix or hostname


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


def _effective_name(registered: str, subdomain: str, domain_name: str) -> str:
    """Name a site owner actually chose: the label left of a free-hosting domain
    (paypa1-verify in paypa1-verify.github.io), else the domain without its suffix."""
    if registered in FREE_HOSTING_DOMAINS and subdomain:
        return subdomain.rsplit(".", 1)[-1]
    return domain_name


def _unnormalize_homoglyphs(token: str) -> str:
    for seq, repl in _HOMOGLYPH_SEQUENCES:
        token = token.replace(seq, repl)
    return token.translate(_HOMOGLYPH_CHARS)


def _lookalike_flags(name: str, registered: str) -> tuple[int, int]:
    """(name_brand_lookalike, name_homoglyph_brand) for an effective name."""
    lookalike = homoglyph = 0
    for token in name.split("-"):
        if not token:
            continue
        if len(token) >= LOOKALIKE_MIN_TOKEN_LEN:
            for brand in _LOOKALIKE_BRANDS:
                if (registered not in BRAND_DOMAINS[brand] and token != brand
                        and Levenshtein.distance(
                            token, brand, score_cutoff=LOOKALIKE_MAX_DISTANCE
                        ) <= LOOKALIKE_MAX_DISTANCE):
                    lookalike = 1
                    break
        fixed = _unnormalize_homoglyphs(token)
        if fixed != token and fixed in BRAND_DOMAINS                 and registered not in BRAND_DOMAINS[fixed]:
            homoglyph = 1
    return lookalike, homoglyph


def extract_features(hostname: str) -> dict:
    """Return the feature dict (v1 + v1.1) for a normalized hostname."""
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

    name = "" if is_ip else _effective_name(registered, subdomain, domain_name)
    lookalike, homoglyph = _lookalike_flags(name, registered)

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
        "name_length": len(name),
        "name_entropy": _entropy(name),
        "name_digit_ratio": sum(ch.isdigit() for ch in name) / len(name) if name else 0.0,
        "name_num_hyphens": name.count("-"),
        "name_keyword_count": sum(kw in name for kw in SUSPICIOUS_KEYWORDS),
        "name_brand_lookalike": lookalike,
        "name_homoglyph_brand": homoglyph,
    }


def extract_features_df(hostnames) -> pd.DataFrame:
    """Apply extract_features to a list/Series of hostnames; keeps a Series' index."""
    index = hostnames.index if isinstance(hostnames, pd.Series) else None
    rows = [extract_features(h) for h in hostnames]
    return pd.DataFrame(rows, index=index, columns=list(FEATURE_NAMES))
