import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse
from sqlalchemy.orm import Session

from app.config import settings
from app.db import SessionLocal, init_db
from app.models import Dataset, Run
from app.schemas import DatasetOut, RunCreate, RunOut, SqlQuery
from app.service import execute_run
from app.tools.loaders import load_raw
from app.tools.sql_tools import SqlGuardError, run_select


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    Path(settings.data_dir, "uploads").mkdir(parents=True, exist_ok=True)
    yield


app = FastAPI(title="Agentic Data Scientist", version="1.0.0", lifespan=lifespan)


def db():
    with SessionLocal() as s:
        yield s


@app.get("/health")
def health():
    return {"status": "ok", "llm_provider": settings.llm_provider}


@app.post("/datasets", response_model=DatasetOut)
async def upload_dataset(file: UploadFile = File(...), s: Session = Depends(db)):
    ext = Path(file.filename or "").suffix.lower()
    if ext not in (".csv", ".xlsx", ".xls", ".xlsm"):
        raise HTTPException(400, "Upload a .csv or Excel file.")
    data = await file.read()
    if len(data) > settings.max_upload_mb * 1_048_576:
        raise HTTPException(413, f"File exceeds {settings.max_upload_mb} MB.")
    path = Path(settings.data_dir, "uploads", f"{uuid.uuid4().hex}{ext}")
    path.write_bytes(data)
    try:
        df = load_raw(str(path))
    except Exception as e:  # noqa: BLE001
        path.unlink(missing_ok=True)
        raise HTTPException(400, f"Could not parse file: {e}")
    ds = Dataset(name=file.filename, path=str(path), n_rows=len(df), n_cols=df.shape[1])
    s.add(ds)
    s.commit()
    return ds


@app.get("/datasets", response_model=list[DatasetOut])
def list_datasets(s: Session = Depends(db)):
    return s.query(Dataset).order_by(Dataset.id.desc()).all()


@app.post("/runs", response_model=RunOut, status_code=202)
def create_run(body: RunCreate, bg: BackgroundTasks, s: Session = Depends(db)):
    if not s.get(Dataset, body.dataset_id):
        raise HTTPException(404, "Dataset not found")
    run = Run(dataset_id=body.dataset_id, question=body.question, progress=[])
    s.add(run)
    s.commit()
    bg.add_task(execute_run, run.id)
    return run


def _run(s, run_id) -> Run:
    run = s.get(Run, run_id)
    if not run:
        raise HTTPException(404, "Run not found")
    return run


@app.get("/runs/{run_id}", response_model=RunOut)
def get_run(run_id: int, s: Session = Depends(db)):
    return _run(s, run_id)


@app.get("/runs/{run_id}/result")
def get_result(run_id: int, s: Session = Depends(db)):
    run = _run(s, run_id)
    if run.status != "completed":
        raise HTTPException(409, f"Run is {run.status}")
    return run.result


@app.get("/runs/{run_id}/report", response_class=PlainTextResponse)
def get_report(run_id: int):
    p = Path(settings.data_dir, "runs", str(run_id), "report.md")
    if not p.exists():
        raise HTTPException(404, "Report not ready")
    return p.read_text()


@app.get("/runs/{run_id}/figures/{name}")
def get_figure(run_id: int, name: str):
    p = Path(settings.data_dir, "runs", str(run_id), "figures", Path(name).name)
    if not p.exists():
        raise HTTPException(404, "Figure not found")
    return FileResponse(p)


@app.post("/runs/{run_id}/sql")
def ad_hoc_sql(run_id: int, body: SqlQuery):
    """Guarded read-only SQL against the cleaned dataset of a run."""
    try:
        return run_select(body.query, {f"run_{run_id}_data"})
    except SqlGuardError as e:
        raise HTTPException(400, str(e))
