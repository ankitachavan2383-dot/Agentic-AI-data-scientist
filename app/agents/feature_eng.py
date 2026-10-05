from pathlib import Path

import pandas as pd

from app.tools.loaders import load_clean
from app.utils import Evidence, safe_node, text_cols


@safe_node("feature_eng")
def feature_eng_node(state):
    df = load_clean(state)
    target = state["plan"]["target"]
    derived, dropped = [], []

    for c in [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c])]:
        for part, val in (("year", df[c].dt.year), ("month", df[c].dt.month), ("dayofweek", df[c].dt.dayofweek)):
            df[f"{c}_{part}"] = val
            derived.append(f"{c}_{part}")
        df = df.drop(columns=[c])

    for c in [c for c in df.columns if c != target]:
        m = df[c].isna().mean()
        if 0.05 <= m <= 0.6:
            df[f"{c}_was_missing"] = df[c].isna().astype(int)
            derived.append(f"{c}_was_missing")

    for c in [c for c in text_cols(df) if c != target]:
        nun = df[c].nunique()
        if nun > 100:
            df = df.drop(columns=[c])
            dropped.append(f"{c} (>{100} levels)")
        elif nun > 20:
            vc = df[c].value_counts(normalize=True)
            rare = vc[vc < 0.01].index
            df[c] = df[c].where(~df[c].isin(rare), "__other__")
            derived.append(f"{c} (rare levels grouped)")

    feats = [c for c in df.columns if c != target]
    numeric = [c for c in feats if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])]
    categorical = [c for c in feats if c not in numeric]
    df.to_parquet(Path(state["run_dir"]) / "features.parquet", index=False)

    ev = Evidence(state, "feature_eng")
    ev.add("features", f"Feature set: {len(numeric)} numeric, {len(categorical)} categorical. Derived: {', '.join(derived) or 'none'}. Dropped: {', '.join(dropped) or 'none'}.",
           {"numeric": numeric, "categorical": categorical, "derived": derived, "dropped": dropped})
    return {"features": {"numeric": numeric, "categorical": categorical, "derived": derived, "dropped": dropped}, "evidence": ev.items}
