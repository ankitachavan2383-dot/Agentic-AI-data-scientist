# 🤖 Agentic AI Data Scientist

> Upload a CSV or Excel file, ask a question in plain English, and a team of specialised AI agents profiles, cleans, explores, models, queries, visualises and reports, with **every claim in the final report traceable to evidence**.

![Python](https://img.shields.io/badge/python-3.11+-blue) ![LangGraph](https://img.shields.io/badge/orchestration-LangGraph-orange) ![FastAPI](https://img.shields.io/badge/API-FastAPI-009688) ![PostgreSQL](https://img.shields.io/badge/DB-PostgreSQL-336791) ![MLflow](https://img.shields.io/badge/tracking-MLflow-0194E2) ![Docker](https://img.shields.io/badge/deploy-Docker-2496ED)

---

## ✨ Highlights

- **Multi-agent workflow (LangGraph)**: 10 specialised agents with conditional routing and a self-correcting retrain loop.
- **Evidence-based insights**: each agent writes numbered evidence items (`E1`, `E2`, ...). The report must cite them, and invalid citations are stripped automatically.
- **Explainable ML**: leakage-safe sklearn pipelines, baseline comparison, permutation importance, and direction-of-effect notes.
- **LLM with tool calling**: a SQL analyst agent calls a guarded, read-only SQL tool.
- **RAG**: curated statistics/ML guidance is retrieved and cited in reports.
- **Works without an API key**: deterministic fallbacks keep the whole pipeline running (`LLM_PROVIDER=none`).
- **Production-style stack**: FastAPI, PostgreSQL, MLflow, Docker Compose.
- **Built-in evaluation system**: synthetic datasets with known ground truth, scored on correctness, citation validity and numeric grounding.

---

## 🧠 How it works

```
                      ┌────────────┐  question
 CSV/XLSX ──► FastAPI ─►  planner   │  intent + target + step plan
                      └─────┬──────┘
   profiler ► cleaner ► eda ─┤ predictive question?
                             ├─yes► feature_eng ► trainer ► evaluator ──(lift too low: retry once)──┐
                             │                       ▲   (MLflow)  │                                 │
                             │                       └─────────────┴─────────────────────────────────┘
                             └─no───────────────────────────────┐   │
                                          sql_analyst  ◄─────────┴───┘   LLM tool-calling over guarded SQL
                                               ▼
                                visualizer ► reporter (LLM + RAG, citation-verified) ► report.md
```

| Agent | Responsibility | Uses LLM? |
|---|---|---|
| **planner** | Detects intent (predictive vs descriptive), picks the target column, builds the step plan | Optional |
| **profiler** | Types, missingness, duplicates, ID/constant/high-cardinality flags | No |
| **cleaner** | De-duplication, NA tokens, numeric/date parsing, junk-column removal, loads data into PostgreSQL | No |
| **eda** | Collinearity, outliers, target distribution, mutual-information ranking | No |
| **feature_eng** | Date parts, missing indicators, rare-level grouping | No |
| **trainer** | Cross-validates 3 candidate models in a leakage-safe pipeline, logs to MLflow | No |
| **evaluator** | Held-out metrics vs baseline, permutation importance, leakage/stability warnings, triggers retrain | No |
| **sql_analyst** | Tool-calling loop (`get_table_schema`, `run_sql`) with a SELECT-only guard | Optional |
| **visualizer** | Missingness, target, correlation, importance, confusion matrix / predicted-vs-actual | No |
| **reporter** | Writes the report from the evidence ledger + RAG guidance and verifies citations | Optional |

---

## 🧰 Tech stack

Python 3.11 · LangGraph · LangChain (Anthropic / OpenAI) · FastAPI · SQLAlchemy · PostgreSQL · scikit-learn · pandas · MLflow · matplotlib · Docker Compose · pytest

---

## 🚀 Quick start (Docker)

```bash
git clone <your-repo-url>
cd agentic-data-scientist
cp .env.example .env          # Windows: copy .env.example .env
# edit .env: add ANTHROPIC_API_KEY, or set LLM_PROVIDER=none
docker compose up --build
```

| Service | URL |
|---|---|
| API + interactive docs | http://localhost:8000/docs |
| MLflow UI | http://localhost:5000 |
| PostgreSQL | localhost:5432 (user/pass/db: `ads`) |

### Use it

1. **Upload**: `POST /datasets` (choose a CSV/Excel file) → returns `dataset_id`
2. **Ask**: `POST /runs` with `{"dataset_id": 1, "question": "What factors drive customer churn?"}`
3. **Track**: `GET /runs/{id}` → status and live agent progress
4. **Read**: `GET /runs/{id}/report` (Markdown) · `GET /runs/{id}/result` (full evidence + metrics JSON)
5. **Query**: `POST /runs/{id}/sql` → guarded read-only SQL on the cleaned data (table `run_{id}_data`)

```bash
# macOS / Linux
curl -F "file=@sample_data/churn.csv" localhost:8000/datasets
curl -X POST localhost:8000/runs -H 'content-type: application/json' \
     -d '{"dataset_id":1,"question":"What factors drive customer churn?"}'
```
```powershell
# Windows PowerShell
curl.exe -F "file=@sample_data/churn.csv" http://localhost:8000/datasets
Invoke-RestMethod -Method Post http://localhost:8000/runs -ContentType "application/json" -Body '{"dataset_id":1,"question":"What factors drive customer churn?"}'
```

### Run locally without Docker

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
LLM_PROVIDER=none uvicorn app.main:app --reload       # uses SQLite + local MLflow DB by default
```

---

## 🔐 Safety & trust

- **Evidence ledger**: every number in the report links back to an agent-produced evidence item.
- **Leakage protection**: imputation, scaling and encoding live *inside* the CV pipeline; the evaluator flags dominant features and near-perfect scores.
- **SQL guard**: single statement, `SELECT`/`WITH` only, table allow-list, forbidden keywords blocked, row limit, and a read-only transaction with a 5 s timeout on PostgreSQL.
- **Fault tolerance**: an agent failure is recorded and surfaced in the report's limitations instead of crashing the run.
- **Careful wording**: reports use associational language and never claim causation.

---

## 📊 Evaluation

```bash
make test     # unit + end-to-end tests (offline)
make eval     # scores the agents on synthetic datasets with known answers
```

Checks include: correct intent / target / task type, expected drivers in the top 3, lift over baseline, SQL answers the question, citation validity and coverage, and numeric grounding. Results are saved to `eval_results.json` and MLflow (`agent-evals`). The runner exits non-zero below the pass threshold, so it works as a CI gate. Add scenarios in `evals/cases.yaml`.

---

## 📁 Project structure

```
agentic-data-scientist/
├── app/
│   ├── main.py            FastAPI endpoints
│   ├── graph.py           LangGraph workflow and routing
│   ├── service.py         run execution and progress tracking
│   ├── state.py           shared agent state
│   ├── llm.py             LLM factory with graceful fallback
│   ├── tracking.py        MLflow helpers
│   ├── agents/            planner, profiler, cleaner, eda, feature_eng,
│   │                      trainer, evaluator, sql_agent, visualizer, reporter
│   ├── tools/             loaders.py, sql_tools.py (guarded SQL)
│   └── rag/               TF-IDF retrieval over knowledge/
├── knowledge/             RAG corpus (metrics, leakage, causal language)
├── evals/                 datasets, cases.yaml, metrics, runner
├── tests/                 SQL guard, pipeline, API tests
├── docker/ · Dockerfile · docker-compose.yml
└── INTERVIEW_QUESTIONS.md
```

---

## ⚙️ Configuration (`.env`)

| Variable | Description | Default |
|---|---|---|
| `LLM_PROVIDER` | `anthropic`, `openai`, or `none` | `anthropic` |
| `LLM_MODEL` | Model name | `claude-sonnet-5-5` |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | Provider key | empty |
| `DATABASE_URL` | SQLAlchemy URL | SQLite (Docker: PostgreSQL) |
| `MLFLOW_TRACKING_URI` | MLflow server/DB | local SQLite (Docker: `http://mlflow:5000`) |
| `MAX_UPLOAD_MB` | Upload size limit | `50` |

---

## 🛣️ Roadmap / known limitations

- Single-table datasets only (no multi-file joins yet)
- Random train/test split; time-series splitting not implemented
- Background tasks run in-process → move to Celery/Arq for scale
- No authentication or per-user isolation yet
- Planned: SHAP explanations, hyper-parameter search, `/predict` endpoint, pgvector RAG, local-LLM support (Ollama), Streamlit/React front-end

---

## 🤝 Contributing

1. Fork the repo and create a feature branch
2. Add or update tests (`make test`) and eval cases (`make eval`)
3. Open a pull request describing the change

## 📄 License

Add a license of your choice (e.g. MIT) as `LICENSE` before publishing.