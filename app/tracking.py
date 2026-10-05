"""MLflow helpers. Tracking failures are non-fatal: analysis must never die because MLflow is down."""
import logging

import mlflow

from app.config import settings

log = logging.getLogger("ads.mlflow")


def setup():
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(settings.mlflow_experiment)


def log_to_run(run_id: str | None, metrics: dict | None = None, artifacts: list[str] | None = None, tags: dict | None = None):
    if not run_id:
        return
    try:
        setup()
        c = mlflow.tracking.MlflowClient()
        for k, v in (metrics or {}).items():
            if isinstance(v, (int, float)):
                c.log_metric(run_id, k, float(v))
        for k, v in (tags or {}).items():
            c.set_tag(run_id, k, str(v))
        for a in artifacts or []:
            c.log_artifact(run_id, a)
    except Exception:  # noqa: BLE001
        log.warning("MLflow logging failed", exc_info=True)
