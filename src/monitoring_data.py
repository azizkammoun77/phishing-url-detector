"""Shared helpers for calibration, simulation and monitoring (text only, no network).

Loads the champion model and splits the TEST set into two disjoint parts:
  * a REFERENCE holdout: never used by the traffic simulation
  * a simulation POOL: the only rows the simulated traffic is sampled from
"""
import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

from src.train import TEST_PATH, TRACKING_URI

MODEL_URI = "models:/phishing-hostname-detector@champion"
REFERENCE_FRACTION = 0.20
SPLIT_SEED = 42


def load_champion():
    """Return (model, threshold) for the registered champion."""
    mlflow.set_tracking_uri(TRACKING_URI)
    model = mlflow.pyfunc.load_model(MODEL_URI)
    return model, float((model.metadata.metadata or {})["threshold"])


def predict_proba(model, hostnames, batch: int = 20000) -> np.ndarray:
    hostnames = list(hostnames)
    parts = [np.asarray(model.predict(pd.DataFrame({"hostname": hostnames[i:i + batch]})),
                        dtype=float)
             for i in range(0, len(hostnames), batch)]
    return np.concatenate(parts)


def load_test() -> pd.DataFrame:
    return pd.read_csv(TEST_PATH)


def split_reference_pool(test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Random reference holdout (stratified by label) and the remaining simulation pool."""
    reference = (test.groupby("label", group_keys=False)
                 .sample(frac=REFERENCE_FRACTION, random_state=SPLIT_SEED))
    pool = test.drop(reference.index)
    return reference, pool
