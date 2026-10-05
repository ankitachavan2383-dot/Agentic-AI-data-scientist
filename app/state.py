import operator
from typing import Annotated, Any, TypedDict


class AgentState(TypedDict, total=False):
    # identifiers / inputs
    run_id: int
    dataset_id: int
    question: str
    raw_path: str
    run_dir: str
    table_name: str
    # agent outputs (JSON-safe dicts; DataFrames live on disk in run_dir)
    plan: dict[str, Any]
    profile: dict[str, Any]
    cleaning: dict[str, Any]
    eda: dict[str, Any]
    features: dict[str, Any]
    model: dict[str, Any]
    evaluation: dict[str, Any]
    sql: dict[str, Any]
    figures: list[str]
    report: str
    report_quality: dict[str, Any]
    # control flow
    retries: int
    needs_retrain: bool
    mlflow_run_id: str | None
    # append-only channels
    evidence: Annotated[list, operator.add]
    log: Annotated[list, operator.add]
    errors: Annotated[list, operator.add]
