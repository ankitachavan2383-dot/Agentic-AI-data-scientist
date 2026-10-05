import os
import tempfile

_tmp = tempfile.mkdtemp()
os.environ.update(LLM_PROVIDER="none", DATABASE_URL=f"sqlite:///{_tmp}/t.db", DATA_DIR=f"{_tmp}/data", MLFLOW_TRACKING_URI=f"sqlite:///{_tmp}/mlflow.db")
