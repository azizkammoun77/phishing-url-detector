import math

import pandas as pd
import pytest

from src.features import (FEATURE_NAMES, _effective_name, extract_features,
                          extract_features_df)



def test_normal_domain():
    f = extract_features("example.com")
    assert f["hostname_length"] == 11
    assert f["num_labels"] == 2
    assert f["num_subdomain_levels"] == 0
    assert f["num_digits"] == 0 and f["digit_ratio"] == 0
    assert f["num_hyphens"] == 0
    assert f["longest_label_length"] == 7
    assert f["is_ip"] == 0 and f["is_punycode"] == 0
    assert f["brand_impersonation"] == 0
    assert f["suspicious_keyword_count"] == 0
    assert f["is_free_hosting"] == 0 and f["tld_risky"] == 0
    # "example": e x a m p l e -> 'e' appears twice among 7 chars
    expected = -(2 / 7 * math.log2(2 / 7) + 5 * (1 / 7) * math.log2(1 / 7))
    assert f["domain_entropy"] == pytest.approx(expected)
    assert f["vowel_ratio"] == pytest.approx(3 / 7)  # e, a, e


def test_subdomain():
    f = extract_features("login.mail.example.com")
    assert f["num_labels"] == 4
    assert f["num_subdomain_levels"] == 2
    assert f["suspicious_keyword_count"] == 1  # login


def test_raw_ipv4():
    f = extract_features("192.168.0.1")
    assert f["is_ip"] == 1
    assert f["num_labels"] == 4
    assert f["num_subdomain_levels"] == 0
    assert f["num_digits"] == 8
    assert f["is_punycode"] == 0 and f["tld_risky"] == 0


def test_raw_ipv6():
    assert extract_features("2001:db8::1")["is_ip"] == 1


def test_punycode():
    f = extract_features("xn--bcher-kva.example.com")
    assert f["is_punycode"] == 1
    assert extract_features("example.com")["is_punycode"] == 0


def test_brand_impersonation_on_free_hosting():
    f = extract_features("paypal-secure.web.app")
    assert f["brand_impersonation"] == 1
    assert f["is_free_hosting"] == 1
    assert f["suspicious_keyword_count"] == 1  # secure
    assert f["num_hyphens"] == 1


def test_brand_on_lookalike_domain():
    assert extract_features("paypal.com.evil-site.xyz")["brand_impersonation"] == 1


def test_real_brand_domain_is_not_flagged():
    assert extract_features("paypal.com")["brand_impersonation"] == 0
    assert extract_features("login.microsoftonline.com")["brand_impersonation"] == 0


def test_co_uk_domain():
    f = extract_features("mail.example.co.uk")
    assert f["num_labels"] == 4
    assert f["num_subdomain_levels"] == 1  # "mail"; "co.uk" is the suffix
    assert f["longest_label_length"] == 7
    assert f["tld_risky"] == 0


def test_risky_tld():
    assert extract_features("example.xyz")["tld_risky"] == 1
    assert extract_features("example.com")["tld_risky"] == 0


def test_empty_hostname_does_not_crash():
    f = extract_features("")
    assert f["hostname_length"] == 0 and f["num_labels"] == 0


def test_extract_features_df():
    s = pd.Series(["example.com", "paypal-secure.web.app"], index=[10, 20])
    df = extract_features_df(s)
    assert list(df.columns) == list(FEATURE_NAMES)
    assert list(df.index) == [10, 20]
    assert df.loc[20, "brand_impersonation"] == 1


@pytest.mark.parametrize("hostname, expected", [
    ("purchase-guide.com", 0),       # "chase" only as a substring
    ("pineapple-recipes.com", 0),    # "apple" only as a substring
    ("chase-login.xyz", 1),          # whole token "chase"
    ("paypalsecure-verify.com", 1),  # distinctive brand, substring match
    ("facebookmail.com", 0),         # official facebook domain
])
def test_brand_impersonation_matching(hostname, expected):
    assert extract_features(hostname)["brand_impersonation"] == expected


@pytest.mark.parametrize("hostname, lookalike, homoglyph", [
    ("pytorch.github.io", 0, 0),
    ("paypa1-verify.github.io", None, 1),  # also a distance-1 lookalike
    ("paypai-login.com", 1, 0),
    ("arnazon-support.com", None, 1),     # also a distance-2 lookalike
    ("paypal.com", 0, 0),
    ("microsoft.com", 0, 0),
])
def test_lookalike_and_homoglyph(hostname, lookalike, homoglyph):
    f = extract_features(hostname)
    if lookalike is not None:
        assert f["name_brand_lookalike"] == lookalike
    assert f["name_homoglyph_brand"] == homoglyph


@pytest.mark.parametrize("hostname, expected", [
    ("paypa1-verify.github.io", "paypa1-verify"),
    ("mail.google.com", "google"),
    ("a.b.netlify.app", "b"),
    ("github.io", "github"),
])
def test_effective_name(hostname, expected):
    from src.features import _extract
    ext = _extract(hostname)
    registered = ext.top_domain_under_public_suffix
    assert _effective_name(registered, ext.subdomain, ext.domain) == expected


def test_name_features():
    f = extract_features("paypa1-verify.github.io")
    assert f["name_length"] == 13
    assert f["name_num_hyphens"] == 1
    assert f["name_keyword_count"] == 1  # verify
    assert f["name_digit_ratio"] == pytest.approx(1 / 13)
    assert extract_features("mail.google.com")["name_length"] == 6
