"""Download the Cisco Umbrella and Tranco top-1M ranking lists into data/raw/.

Only the two ranking-list files (and Tranco's list metadata) are fetched, from
their official sources. Hostnames inside the lists are never visited or resolved.
"""
import json
import urllib.request
from datetime import date
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw"
UMBRELLA_URL = "http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip"
TRANCO_LATEST_API = "https://tranco-list.eu/api/lists/date/latest"
TRANCO_FALLBACK_URL = "https://tranco-list.eu/top-1m.csv.zip"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "phishing-url-project"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return resp.read()


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)

    umbrella = _get(UMBRELLA_URL)
    (RAW / "umbrella-top-1m.csv.zip").write_bytes(umbrella)
    print(f"Umbrella: {len(umbrella):,} bytes")

    info = json.loads(_get(TRANCO_LATEST_API))
    list_id = info["list_id"]
    tranco_url = info.get("download") or TRANCO_FALLBACK_URL
    tranco = _get(tranco_url)
    (RAW / "tranco-top-1m.csv").write_bytes(tranco)
    print(f"Tranco list {list_id}: {len(tranco):,} bytes")

    meta = {
        "download_date": date.today().isoformat(),
        "umbrella": {"url": UMBRELLA_URL, "file": "umbrella-top-1m.csv.zip"},
        "tranco": {
            "list_id": list_id,
            "url": tranco_url,
            "api_url": TRANCO_LATEST_API,
            "list_date": info.get("created_on"),
            "file": "tranco-top-1m.csv",
        },
    }
    (RAW / "benign_lists_metadata.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
