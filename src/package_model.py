"""Package the trained "stacked" model as one MLflow pyfunc model and register it.

Does NOT retrain: it loads the two trained components from the stacked run's
artifacts. Run from the project root:  python -m src.package_model
"""
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from mlflow.models import infer_signature

from src.features import FEATURE_NAMES, extract_features_df
from src.stacked_model import NGRAM_COLUMN, StackedPhishingModel
from src.train import EXPERIMENT, TEST_PATH, TRACKING_URI

REGISTERED_NAME = "phishing-hostname-detector"
ALIAS = "champion"
N_VERIFY = 1000


def find_stacked_run(client: MlflowClient):
    exp = client.get_experiment_by_name(EXPERIMENT)
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string="tags.mlflow.runName = 'stacked' and params.feature_version = 'v1.1' "
                      "and params.data_version = 'v2_umbrella' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        raise SystemExit("no finished stacked run found")
    return runs[0]


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    src_run = find_stacked_run(client)
    run_id = src_run.info.run_id
    threshold = src_run.data.metrics["threshold_fpr1"]
    meta = {"threshold": threshold, "data_version": src_run.data.params["data_version"],
            "feature_version": src_run.data.params["feature_version"],
            "source_run_id": run_id}
    print("source run:", run_id, meta)

    xgb_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/model")
    ngram_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/ngram_model")

    test = pd.read_csv(TEST_PATH).head(N_VERIFY)
    example = pd.DataFrame({"hostname": test["hostname"].head(5)})

    # Reference probabilities straight from the original run's components.
    ngram = mlflow.sklearn.load_model(ngram_path)
    xgb = mlflow.xgboost.load_model(xgb_path)
    feats = extract_features_df(test["hostname"]).astype("float64").reset_index(drop=True)
    ref = xgb.predict_proba(
        feats.assign(**{NGRAM_COLUMN: ngram.predict_proba(
            test[["hostname"]].reset_index(drop=True))[:, 1]})
    )[:, 1]

    wrapper = StackedPhishingModel()
    signature = infer_signature(example, ref[:5])
    with mlflow.start_run(run_name="package_stacked"):
        mlflow.log_params({"source_run_id": run_id, **{k: v for k, v in meta.items()
                                                       if k != "source_run_id"}})
        info = mlflow.pyfunc.log_model(
            name="model", python_model=wrapper,
            artifacts={"xgb_model": xgb_path, "ngram_model": ngram_path},
            code_paths=["src"], signature=signature, input_example=example,
            metadata=meta, registered_model_name=REGISTERED_NAME,
            pip_requirements=["mlflow", "scikit-learn", "xgboost", "pandas", "numpy",
                              "tldextract", "rapidfuzz"])
        packaged_run = mlflow.active_run().info.run_id

    version = info.registered_model_version
    client.set_registered_model_alias(REGISTERED_NAME, ALIAS, str(version))
    for k, v in meta.items():
        client.set_model_version_tag(REGISTERED_NAME, str(version), k, str(v))

    # Verify the packaged model reproduces the original probabilities.
    loaded = mlflow.pyfunc.load_model(f"models:/{REGISTERED_NAME}@{ALIAS}")
    got = np.asarray(loaded.predict(test[["hostname"]]))
    max_diff = float(np.max(np.abs(got - ref)))
    print(f"\nPackaged-vs-original max abs difference on {len(test)} test rows: {max_diff:.3e}")
    print(f"metadata in model: {loaded.metadata.metadata}")
    print(f"Registered: name={REGISTERED_NAME} version={version} alias={ALIAS} "
          f"(packaging run {packaged_run})")
    assert max_diff < 1e-9, "packaged model does not match the original"


if __name__ == "__main__":
    main()
