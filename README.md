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
