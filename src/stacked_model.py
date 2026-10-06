"""MLflow pyfunc wrapper around the full stacked hostname pipeline.

Pipeline: hostname -> char n-gram LR probability, hostname -> hand-made features,
then XGBoost on features + ngram_prob. Pure text processing: hostnames are never
visited or resolved.
"""
import mlflow.pyfunc
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd

from src.features import FEATURE_NAMES, extract_features_df

NGRAM_COLUMN = "ngram_prob"


class StackedPhishingModel(mlflow.pyfunc.PythonModel):
    """Input: DataFrame with a "hostname" column (or a list/Series of hostnames).
    Output: 1-D float array of phishing probabilities."""

    def load_context(self, context):
        self.ngram = mlflow.sklearn.load_model(context.artifacts["ngram_model"])
        self.xgb = mlflow.xgboost.load_model(context.artifacts["xgb_model"])

    @staticmethod
    def _hostnames(model_input) -> pd.Series:
        if isinstance(model_input, pd.DataFrame):
            col = "hostname" if "hostname" in model_input.columns else model_input.columns[0]
            return model_input[col].astype(str).reset_index(drop=True)
        return pd.Series(list(model_input), dtype=str)

    def predict(self, context, model_input, params=None):
        hostnames = self._hostnames(model_input)
        feats = extract_features_df(hostnames).astype("float64").reset_index(drop=True)
        ngram_prob = self.ngram.predict_proba(
            pd.DataFrame({"hostname": hostnames}))[:, 1]
        stacked = feats[list(FEATURE_NAMES)].assign(**{NGRAM_COLUMN: ngram_prob})
        return self.xgb.predict_proba(stacked)[:, 1].astype("float64")
