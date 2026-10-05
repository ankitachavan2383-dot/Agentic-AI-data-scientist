import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from app.tools.loaders import load_raw
from app.tools.sql_tools import load_table
from app.utils import Evidence, safe_node, text_cols, is_text

NA_TOKENS = {"", "na", "n/a", "nan", "null", "none", "-", "?", "missing"}


@safe_node("cleaner")
def cleaner_node(state):
    df = load_raw(state["raw_path"])
    target = (state.get("plan") or {}).get("target")
    actions: list[str] = []
    n0 = len(df)

    df = df.drop_duplicates()
    if len(df) < n0:
        actions.append(f"removed {n0 - len(df)} duplicate rows")

    for c in text_cols(df):
        s = df[c].astype("string").str.strip()
        df[c] = s.mask(s.str.lower().isin(NA_TOKENS)).astype(object).where(s.notna() & ~s.str.lower().isin(NA_TOKENS), np.nan)

    for c in text_cols(df):
        nn = df[c].notna().sum()
        if nn == 0:
            continue
        num = pd.to_numeric(df[c].astype(str).str.replace(r"[,$%\s]", "", regex=True).where(df[c].notna()), errors="coerce")
        if num.notna().sum() >= 0.95 * nn:
            df[c] = num
            actions.append(f"converted '{c}' from text to numeric")
            continue
        sample = df[c].dropna().astype(str).head(200)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(sample, errors="coerce")
        if len(sample) and parsed.notna().mean() >= 0.9 and sample.str.contains(r"[-/:]").mean() > 0.9:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                df[c] = pd.to_datetime(df[c], errors="coerce")
            actions.append(f"parsed '{c}' as datetime")

    n = len(df)
    drop = []
    for c in df.columns:
        if c == target:
            continue
        nun = df[c].nunique(dropna=True)
        miss = df[c].isna().mean()
        id_name = c.endswith("_id") or c == "id" or c.endswith("_index")
        if nun <= 1:
            drop.append((c, "constant"))
        elif miss > 0.6:
            drop.append((c, f"{miss:.0%} missing"))
        elif n > 20 and ((id_name and nun == n) or (is_text(df[c]) and nun / n > 0.95)):
            drop.append((c, "identifier / free text"))
    if drop:
        df = df.drop(columns=[c for c, _ in drop])
        actions.append("dropped columns: " + ", ".join(f"{c} ({why})" for c, why in drop))

    if target and target in df.columns:
        before = len(df)
        df = df.dropna(subset=[target])
        if len(df) < before:
            actions.append(f"dropped {before - len(df)} rows with missing target '{target}'")

    df = df.reset_index(drop=True)
    df.to_parquet(Path(state["run_dir"]) / "clean.parquet", index=False)
    load_table(df, state["table_name"])
    remaining_missing = {c: round(float(v), 4) for c, v in df.isna().mean().items() if v > 0}

    ev = Evidence(state, "cleaner")
    ev.add("cleaning", f"Cleaning produced {len(df)} rows x {df.shape[1]} columns. Actions: " + ("; ".join(actions) if actions else "none needed") + ". Remaining missing values are imputed inside the model pipeline using training data only (prevents leakage).",
           {"actions": actions, "rows": len(df), "cols": df.shape[1], "remaining_missing": remaining_missing})
    return {"cleaning": {"actions": actions, "rows": len(df), "cols": df.shape[1], "dropped": [c for c, _ in drop]}, "evidence": ev.items}
