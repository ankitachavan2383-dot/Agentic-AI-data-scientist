import pandas as pd

from app.tools.loaders import load_raw
from app.utils import Evidence, safe_node


@safe_node("profiler")
def profiler_node(state):
    df = load_raw(state["raw_path"])
    n = len(df)
    cols, flags = [], {"id_like": [], "constant": [], "high_cardinality": [], "mostly_missing": []}
    for c in df.columns:
        s = df[c]
        nun = int(s.nunique(dropna=True))
        info = {"name": c, "dtype": str(s.dtype), "missing_pct": round(float(s.isna().mean() * 100), 2), "n_unique": nun}
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            d = s.describe()
            info.update({k: round(float(d[k]), 4) for k in ("mean", "std", "min", "50%", "max") if k in d})
        else:
            info["top_values"] = {str(k): int(v) for k, v in s.value_counts().head(5).items()}
        if nun <= 1:
            flags["constant"].append(c)
        elif nun == n and n > 20:
            flags["id_like"].append(c)
        elif not pd.api.types.is_numeric_dtype(s) and nun / max(n, 1) > 0.5:
            flags["high_cardinality"].append(c)
        if info["missing_pct"] > 60:
            flags["mostly_missing"].append(c)
        cols.append(info)
    dups = int(df.duplicated().sum())
    profile = {"n_rows": n, "n_cols": df.shape[1], "duplicates": dups, "columns": cols, "flags": flags}
    ev = Evidence(state, "profiler")
    ev.add("profile", f"Dataset has {n} rows and {df.shape[1]} columns; {dups} exact duplicate rows ({dups / max(n, 1):.1%}).", {"n_rows": n, "n_cols": df.shape[1], "duplicates": dups})
    miss = sorted(((c["name"], c["missing_pct"]) for c in cols if c["missing_pct"] > 0), key=lambda x: -x[1])
    if miss:
        ev.add("profile", "Columns with missing values: " + ", ".join(f"{k} ({v}%)" for k, v in miss[:8]) + ".", {"missing": dict(miss)})
    flagged = {k: v for k, v in flags.items() if v}
    if flagged:
        ev.add("profile", "Data-quality flags: " + "; ".join(f"{k.replace('_', '-')}: {', '.join(v)}" for k, v in flagged.items()) + ".", flagged)
    return {"profile": profile, "evidence": ev.items}
