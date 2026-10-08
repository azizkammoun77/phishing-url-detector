"""Integration tests: need the registered model (python -m src.package_model)."""
import json
import os

import pytest
from fastapi.testclient import TestClient

TEST_KEY = "test-key-for-pytest"
DECISIONS = {"safe", "phishing", "review"}
RESPONSE_FIELDS = {"hostname", "decision", "probability", "threshold", "signals",
                   "model_version"}


@pytest.fixture(scope="module")
def log_path(tmp_path_factory):
    return tmp_path_factory.mktemp("logs") / "predictions.jsonl"


@pytest.fixture(scope="module")
def client(log_path):
    old = {k: os.environ.get(k) for k in ("API_KEY", "LOG_PATH")}
    os.environ["API_KEY"] = TEST_KEY
    os.environ["LOG_PATH"] = str(log_path)
    from src.api import app
    try:
        with TestClient(app) as c:  # runs lifespan: loads the model once
            yield c
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


AUTH = {"X-API-Key": TEST_KEY}


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_name"] == "phishing-hostname-detector"
    assert body["model_version"]


def test_missing_key_is_401(client):
    assert client.post("/check-url", json={"url": "https://example.com"}).status_code == 401


def test_wrong_key_is_401(client):
    r = client.post("/check-url", json={"url": "https://example.com"},
                    headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_empty_url_is_422(client):
    assert client.post("/check-url", json={"url": ""}, headers=AUTH).status_code == 422


def test_url_without_hostname_is_422(client):
    r = client.post("/check-url", json={"url": "https://"}, headers=AUTH)
    assert r.status_code == 422
    assert "hostname" in r.json()["detail"]


def test_too_long_url_is_422(client):
    r = client.post("/check-url", json={"url": "a" * 2049}, headers=AUTH)
    assert r.status_code == 422


def test_valid_request_has_all_fields(client):
    r = client.post("/check-url", json={"url": "https://www.example.com/"}, headers=AUTH)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == RESPONSE_FIELDS
    assert body["decision"] in DECISIONS
    assert 0.0 <= body["probability"] <= 1.0
    assert isinstance(body["signals"], list)
    assert body["hostname"] == "example.com"


def test_path_and_query_do_not_change_hostname(client):
    bare = client.post("/check-url", json={"url": "mail.google.com"}, headers=AUTH).json()
    full = client.post(
        "/check-url",
        json={"url": "https://mail.google.com/a/b/c?token=secret123&x=1#frag"},
        headers=AUTH).json()
    assert bare["hostname"] == full["hostname"] == "mail.google.com"
    assert bare["probability"] == full["probability"]


def test_free_hosting_never_returns_phishing(client):
    body = client.post("/check-url", json={"url": "https://paypa1-verify.github.io/x"},
                       headers=AUTH).json()
    assert "free hosting" in body["signals"]
    assert body["decision"] in {"safe", "review"}


def test_log_line_has_hostname_only(client, log_path):
    secret = "very-secret-path-and-query"
    client.post("/check-url", json={"url": f"https://logtest.example.org/{secret}?k={secret}"},
                headers=AUTH)
    text = log_path.read_text(encoding="utf-8")
    assert secret not in text and "https://" not in text and "?k=" not in text
    record = json.loads(text.strip().splitlines()[-1])
    assert set(record) == {"request_id", "timestamp", "hostname", "probability",
                           "decision", "model_version"}
    assert record["hostname"] == "logtest.example.org"


def test_request_id_is_unique_and_matches_log(client, log_path):
    ids = []
    for _ in range(2):
        r = client.post("/check-url", json={"url": "https://idtest.example.org/"},
                        headers=AUTH)
        ids.append(r.headers["X-Request-ID"])
    assert ids[0] != ids[1]
    logged = [json.loads(line)["request_id"]
              for line in log_path.read_text(encoding="utf-8").splitlines()]
    assert logged[-2:] == ids
