"""FastAPI service: POST /check-url scores the HOSTNAME of a URL.

URLs are treated strictly as text. The service never visits, fetches or resolves
a URL or hostname it receives (no DNS, no WHOIS).
"""
import json
import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import mlflow
import mlflow.pyfunc
import pandas as pd
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from mlflow import MlflowClient
from pydantic import BaseModel, Field

from src.features import FREE_HOSTING_DOMAINS, extract_features, registered_domain_of
from src.normalize import normalize_url_to_hostname

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

MODEL_NAME = "phishing-hostname-detector"
MODEL_URI = f"models:/{MODEL_NAME}@champion"
DEFAULT_TRACKING_URI = f"sqlite:///{(ROOT / 'mlflow.db').as_posix()}"
DEFAULT_LOG_PATH = ROOT / "logs" / "predictions.jsonl"
MAX_HOSTNAME_LEN = 253

logger = logging.getLogger("phishing_api")
_log_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    api_key = os.environ.get("API_KEY", "")
    if not api_key or api_key == "change-me":
        raise RuntimeError("Set API_KEY in .env (copy .env.example and pick a real key).")
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    model = mlflow.pyfunc.load_model(MODEL_URI)  # loaded once, reused per request
    version = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion").version
    meta = model.metadata.metadata or {}
    app.state.model = model
    app.state.model_name = MODEL_NAME
    app.state.model_version = str(version)
    app.state.threshold = float(meta["threshold"])
    app.state.data_version = meta.get("data_version")
    app.state.feature_version = meta.get("feature_version")
    yield


app = FastAPI(title="Phishing hostname detector", lifespan=lifespan)


class CheckUrlRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


class CheckUrlResponse(BaseModel):
    hostname: str
    decision: str
    probability: float
    threshold: float
    signals: list[str]
    model_version: str


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("API_KEY", "")
    if not expected or x_api_key is None or not secrets.compare_digest(
            x_api_key.encode("utf-8"), expected.encode("utf-8")):
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


def get_signals(hostname: str) -> list[str]:
    f = extract_features(hostname)
    signals = []
    if f["is_free_hosting"]:
        signals.append("free hosting")
    if f["tld_risky"]:
        signals.append("risky TLD")
    if f["brand_impersonation"]:
        signals.append("brand impersonation")
    if f["name_brand_lookalike"]:
        signals.append("brand lookalike")
    if f["name_homoglyph_brand"]:
        signals.append("brand homoglyph")
    if f["suspicious_keyword_count"]:
        signals.append("suspicious keywords")
    if f["is_ip"]:
        signals.append("IP address")
    if f["is_punycode"]:
        signals.append("punycode")
    return signals


def write_prediction_log(hostname: str, probability: float, decision: str,
                         model_version: str) -> None:
    """Append one JSON line. Hostname only: never the full URL."""
    path = Path(os.environ.get("PREDICTION_LOG", DEFAULT_LOG_PATH))
    record = {"timestamp": datetime.now(timezone.utc).isoformat(), "hostname": hostname,
              "probability": probability, "decision": decision,
              "model_version": model_version}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with _log_lock, path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError:
        logger.exception("could not write prediction log")


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "model_name": request.app.state.model_name,
            "model_version": request.app.state.model_version}


@app.post("/check-url", response_model=CheckUrlResponse,
          dependencies=[Depends(require_api_key)])
def check_url(body: CheckUrlRequest, request: Request):
    hostname = normalize_url_to_hostname(body.url)
    if (not hostname or len(hostname) > MAX_HOSTNAME_LEN
            or any(ch.isspace() or ord(ch) < 32 for ch in hostname)):
        raise HTTPException(
            status_code=422, detail="Could not extract a valid hostname from the url")

    state = request.app.state
    probability = float(state.model.predict(pd.DataFrame({"hostname": [hostname]}))[0])
    if probability < state.threshold:
        decision = "safe"
    elif registered_domain_of(hostname) in FREE_HOSTING_DOMAINS:
        decision = "review"
    else:
        decision = "phishing"

    write_prediction_log(hostname, probability, decision, state.model_version)
    return CheckUrlResponse(
        hostname=hostname, decision=decision, probability=round(probability, 4),
        threshold=round(state.threshold, 4), signals=get_signals(hostname),
        model_version=state.model_version)
