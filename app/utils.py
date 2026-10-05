import datetime as dt
import functools
import logging
import math
import time

import numpy as np
import pandas as pd

log = logging.getLogger("ads")


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, set)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else round(f, 6)
    if isinstance(o, (pd.Timestamp, dt.datetime, dt.date)):
        return o.isoformat()
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    return o


class Evidence:
    """Builds citable evidence items. IDs (E1, E2, ...) are globally unique within a run."""

    def __init__(self, state: dict, agent: str):
        self.agent, self.base, self.items = agent, len(state.get("evidence", [])), []

    def add(self, kind: str, claim: str, data: dict | None = None) -> str:
        eid = f"E{self.base + len(self.items) + 1}"
        self.items.append({"id": eid, "agent": self.agent, "kind": kind, "claim": claim, "data": jsonable(data or {})})
        return eid


def infer_task(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s) or not pd.api.types.is_numeric_dtype(s):
        return "classification"
    return "classification" if s.nunique() <= 10 else "regression"


def safe_node(name: str):
    """Agents never crash the graph: errors are recorded and downstream agents degrade gracefully."""

    def deco(fn):
        @functools.wraps(fn)
        def wrapper(state):
            t0 = time.time()
            try:
                out = fn(state) or {}
            except Exception as e:  # noqa: BLE001
                log.exception("agent %s failed", name)
                out = {"errors": [f"{name}: {type(e).__name__}: {e}"]}
            out["log"] = out.get("log", []) + [f"{name} finished in {time.time() - t0:.1f}s"]
            return out

        return wrapper

    return deco


def text_of(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def is_text(s: pd.Series) -> bool:
    """True for string/object/category columns (works for pandas 2 'object' and pandas 3 'str' dtypes)."""
    return not (pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s) or pd.api.types.is_datetime64_any_dtype(s))


def text_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if is_text(df[c])]
