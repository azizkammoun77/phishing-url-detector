"""URL -> hostname normalization shared by data preparation and the API.

Pure string handling: URLs are never visited or resolved.
"""
import re

SCHEME_RE = re.compile(r"^[a-z][a-z0-9+.\-]*://", re.IGNORECASE)


def normalize_url_to_hostname(url: str) -> str:
    """Return the lowercased host of a URL string.

    Strips scheme, credentials, port, path, query and fragment, a leading "www."
    and a trailing dot. Works for URLs without a scheme ("example.com/login").
    Returns "" when no host can be extracted.
    """
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
