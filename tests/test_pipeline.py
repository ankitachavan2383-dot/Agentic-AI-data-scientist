import tempfile
from pathlib import Path

from app.db import init_db
from app.service import run_graph
from evals.datasets import churn, sales
from evals.metrics import narrative, citation_validity


def test_predictive_pipeline_end_to_end():
    init_db()
    raw = Path(tempfile.mkdtemp()) / "c.csv"
    churn().to_csv(raw, index=False)
    s = run_graph(str(raw), "What factors drive customer churn?", 1, 1)
    assert not s["errors"], s["errors"]
    assert s["plan"]["target"] == "churn" and s["model"]["task"] == "classification"
    assert s["evaluation"]["metrics"]["lift"] > 0.05
    ids = {e["id"] for e in s["evidence"]}
    assert len(ids) == len(s["evidence"])
    assert citation_validity(narrative(s["report"]), ids) == 1.0
    assert "feature_importance.png" in s["figures"]


def test_descriptive_question_skips_ml():
    init_db()
    raw = Path(tempfile.mkdtemp()) / "s.csv"
    sales().to_csv(raw, index=False)
    s = run_graph(str(raw), "Which region generates the highest total revenue?", 2, 2)
    assert s["plan"]["intent"] == "descriptive" and "model" not in s
    assert any(e["kind"] == "sql" for e in s["evidence"])


def test_api_flow():
    from fastapi.testclient import TestClient
    from app.main import app

    with TestClient(app) as c:
        raw = Path(tempfile.mkdtemp()) / "c.csv"
        churn().to_csv(raw, index=False)
        ds = c.post("/datasets", files={"file": ("c.csv", raw.read_bytes())}).json()
        run = c.post("/runs", json={"dataset_id": ds["id"], "question": "What drives churn?"}).json()
        r = c.get(f"/runs/{run['id']}").json()  # TestClient executes background tasks before returning
        assert r["status"] == "completed", r
        assert "## Answer" in c.get(f"/runs/{run['id']}/report").text
        assert c.post(f"/runs/{run['id']}/sql", json={"query": "SELECT COUNT(*) FROM run_%d_data" % run["id"]}).status_code == 200
        assert c.post(f"/runs/{run['id']}/sql", json={"query": "DROP TABLE datasets"}).status_code == 400
