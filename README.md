# Phishing URL Detector

A production-style ML service that detects phishing URLs in real time, with explanations and drift monitoring.

## Project structure

- `data/raw/` - original, untouched datasets (not tracked by Git).
- `data/processed/` - cleaned data and extracted features ready for training (not tracked by Git).
- `notebooks/` - Jupyter notebooks for exploration and experiments.
- `src/` - source code: feature extraction, training, and the API.
- `tests/` - automated tests.

## Setup

**Windows (PowerShell)**

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**Linux / macOS**

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data notes

The data comes from the PhiUSIIL Phishing URL dataset (UCI, id 967), keeping only the URL and the label (1 = phishing, 0 = legitimate). Exploration showed a strong shortcut problem: every legitimate URL is a bare `https://www.` homepage with no path, query or trailing slash, while phishing URLs are full URLs in many shapes, so a model on raw URLs could score well without learning anything about phishing.

Decisions taken (implemented in `src/prepare_data.py`):

- **Hostname-only v1:** each URL is normalized to its hostname (scheme, credentials, port, path, query and fragment removed; lowercased; leading `www.` and trailing dot stripped), which removes the shortcut features.
- **Cleaning:** exact duplicate URLs are dropped, then rows are deduplicated on (hostname, label), and hostnames that appear with both labels are dropped as conflicting.
- **Domain-grouped split:** an 80/20 train/test split grouped by registered domain (via `tldextract`, offline), stratified by class, so no registered domain appears in both sets.
- URLs are only ever handled as text and are never visited or resolved.

To rebuild the data: `python src/download_data.py` then `python src/prepare_data.py`.

## Run the API

The API scores the **hostname** of a URL with the registered `phishing-hostname-detector@champion` model (the stacked model). URLs are only ever handled as text: the service never visits, fetches or resolves them.

1. Package and register the model (once; needs the trained "stacked" MLflow run): `python -m src.package_model`
2. Create your `.env` from the template and set a real key:

   ```bash
   cp .env.example .env     # Windows PowerShell: Copy-Item .env.example .env
   ```

   Edit `.env` and replace `change-me` with a long random value (the server refuses to start with the placeholder).
3. Start the server: `uvicorn src.api:app --reload`
4. Open the interactive docs at http://127.0.0.1:8000/docs

Example request:

```bash
curl -X POST http://127.0.0.1:8000/check-url \
  -H "Content-Type: application/json" \
  -H "X-API-Key: <your key>" \
  -d '{"url": "https://paypa1-verify.github.io/login?next=/account"}'
```

Example response:

```json
{
  "hostname": "paypa1-verify.github.io",
  "decision": "review",
  "probability": 0.9995,
  "threshold": 0.626,
  "signals": ["free hosting", "brand lookalike", "brand homoglyph", "suspicious keywords"],
  "model_version": "1"
}
```

- `decision` is `safe` below the threshold, `phishing` at or above it, and `review` at or above it when the site is on a free-hosting domain (e.g. `github.io`), where the hostname alone cannot tell a real page from a phishing page.
- `GET /health` needs no key. A wrong or missing `X-API-Key` returns 401.
- Each prediction appends one line (timestamp, hostname, probability, decision, model version) to `logs/predictions.jsonl`. Only the hostname is logged, never the full URL.
