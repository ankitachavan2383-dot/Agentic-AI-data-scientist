import datetime as dt
from pathlib import Path
from typing import Callable

from app.config import settings
from app.db import SessionLocal
from app.graph import get_graph
from app.models import Dataset, Run
from app.utils import jsonable


def run_graph(raw_path: str, question: str, dataset_id: int, run_id: int, on_progress: Callable[[str], None] | None = None) -> dict:
    run_dir = Path(settings.data_dir) / "runs" / str(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    state = {"run_id": run_id, "dataset_id": dataset_id, "question": question, "raw_path": raw_path, "run_dir": str(run_dir),
             "table_name": f"run_{run_id}_data", "evidence": [], "log": [], "errors": [], "retries": 0}
    final, seen = state, 0
    for final in get_graph().stream(state, stream_mode="values", config={"recursion_limit": 40}):
        if on_progress and len(final.get("log", [])) > seen:
            for msg in final["log"][seen:]:
                on_progress(msg)
            seen = len(final["log"])
    return final


def execute_run(run_id: int) -> None:
    with SessionLocal() as s:
        run = s.get(Run, run_id)
        ds = s.get(Dataset, run.dataset_id)
        run.status, run.progress = "running", []
        s.commit()

        def progress(msg: str):
            run.progress = [*run.progress, msg]
            s.commit()

        try:
            final = run_graph(ds.path, run.question, ds.id, run.id, progress)
            run.result = jsonable({k: final.get(k) for k in ("plan", "profile", "cleaning", "eda", "features", "model", "evaluation", "sql", "figures", "evidence", "errors", "report_quality")}
                                  | {"report_path": str(Path(final["run_dir"]) / "report.md"), "mlflow_run_id": final.get("mlflow_run_id")})
            run.status = "completed"
        except Exception as e:  # noqa: BLE001
            run.status, run.error = "failed", f"{type(e).__name__}: {e}"
        run.finished_at = dt.datetime.utcnow()
        s.commit()
