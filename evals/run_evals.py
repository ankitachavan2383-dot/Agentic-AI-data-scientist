"""Run: python -m evals.run_evals  (works offline with LLM_PROVIDER=none; set a key to evaluate the LLM paths)."""
import json
import sys
import tempfile
import time
from pathlib import Path

import yaml

from app.config import settings
from app.db import init_db
from app.service import run_graph
from evals.datasets import GENERATORS
from evals.metrics import score_case


def main(threshold: float = 0.9) -> int:
    init_db()
    cases = yaml.safe_load((Path(__file__).parent / "cases.yaml").read_text())
    tmp, results = Path(tempfile.mkdtemp()), []
    for i, case in enumerate(cases):
        raw = tmp / f"{case['dataset']}.csv"
        GENERATORS[case["dataset"]]().to_csv(raw, index=False)
        t0 = time.time()
        state = run_graph(str(raw), case["question"], dataset_id=9000 + i, run_id=9000 + i)
        checks = score_case(case, state)
        results.append({"case": case["name"], "passed": all(checks.values()), "checks": checks, "seconds": round(time.time() - t0, 1)})
        print(f"{'PASS' if results[-1]['passed'] else 'FAIL'}  {case['name']}  ({results[-1]['seconds']}s)")
        for k, v in checks.items():
            if not v:
                print(f"      x {k}")
    rate = sum(r["passed"] for r in results) / len(results)
    check_rate = sum(v for r in results for v in r["checks"].values()) / sum(len(r["checks"]) for r in results)
    summary = {"llm_provider": settings.llm_provider, "case_pass_rate": rate, "check_pass_rate": round(check_rate, 3), "results": results}
    Path("eval_results.json").write_text(json.dumps(summary, indent=2))
    print(f"\ncase pass rate {rate:.0%} | check pass rate {check_rate:.0%}  -> eval_results.json")
    try:
        import mlflow
        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment("agent-evals")
        with mlflow.start_run(run_name=f"eval-{settings.llm_provider}"):
            mlflow.log_metrics({"case_pass_rate": rate, "check_pass_rate": check_rate})
            mlflow.log_artifact("eval_results.json")
    except Exception as e:  # noqa: BLE001
        print("MLflow logging skipped:", e)
    return 0 if rate >= threshold else 1


if __name__ == "__main__":
    sys.exit(main())
