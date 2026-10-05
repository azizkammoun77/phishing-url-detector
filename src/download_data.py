"""Download the PhiUSIIL Phishing URL dataset and keep only url + label.

URLs are treated strictly as text; nothing in the dataset is ever visited.
"""
from pathlib import Path

from ucimlrepo import fetch_ucirepo

OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "raw" / "phiusiil_urls.csv"


def main() -> None:
    dataset = fetch_ucirepo(id=967)
    # Original labels: 1 = legitimate, 0 = phishing.
    # Flip so that 1 = phishing, 0 = legitimate.
    df = dataset.data.original[["URL", "label"]].copy()
    df.columns = ["url", "label"]
    df["label"] = 1 - df["label"].astype(int)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUTPUT_PATH}")
    print(df["label"].value_counts().rename({1: "phishing", 0: "legitimate"}))


if __name__ == "__main__":
    main()
