import re
from pathlib import Path

import pandas as pd


def standardize(name) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip()).strip("_").lower() or "col"


def load_raw(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".xlsx", ".xlsm", ".xls"):
        df = pd.read_excel(p)
    else:
        try:
            df = pd.read_csv(p)
        except UnicodeDecodeError:
            df = pd.read_csv(p, encoding="latin-1")
        except pd.errors.ParserError:
            df = pd.read_csv(p, sep=None, engine="python")
    seen: dict[str, int] = {}
    cols = []
    for c in df.columns:
        s = standardize(c)
        seen[s] = seen.get(s, 0) + 1
        cols.append(s if seen[s] == 1 else f"{s}_{seen[s]}")
    df.columns = cols
    return df


def load_clean(state: dict) -> pd.DataFrame:
    return pd.read_parquet(Path(state["run_dir"]) / "clean.parquet")
