import pytest

from src.normalize import normalize_url_to_hostname


@pytest.mark.parametrize("url, expected", [
    ("https://www.Example.com/login?x=1#top", "example.com"),
    ("http://user:pass@sub.example.com:8080/a/b", "sub.example.com"),
    ("example.com/login", "example.com"),
    ("www.example.com", "example.com"),
    ("EXAMPLE.COM.", "example.com"),
    ("https://example.com?q=1", "example.com"),
    ("https://example.com#frag", "example.com"),
    ("ftp://files.example.org/x", "files.example.org"),
    ("http://192.168.0.1:8000/admin", "192.168.0.1"),
    ("http://[2001:db8::1]:8080/x", "2001:db8::1"),
    ("  https://paypa1-verify.github.io/  ", "paypa1-verify.github.io"),
    ("https://www.", ""),
    ("", ""),
    ("   ", ""),
    ("https:///path-only", ""),
])
def test_normalize(url, expected):
    assert normalize_url_to_hostname(url) == expected


def test_path_and_query_do_not_change_hostname():
    bare = normalize_url_to_hostname("evil.example.xyz")
    full = normalize_url_to_hostname("https://evil.example.xyz/a/b/c?token=abc&x=1#f")
    assert bare == full == "evil.example.xyz"
