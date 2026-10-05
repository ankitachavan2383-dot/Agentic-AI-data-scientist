import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from app.tools.loaders import load_clean
from app.utils import Evidence, infer_task, is_text, safe_node


def _encode(X: pd.DataFrame):
    X = X.copy()
    discrete = []
    for c in X.columns:
        if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c]):
            X[c] = X[c].fillna(X[c].median())
            discrete.append(False)
        else:
            X[c] = pd.factorize(X[c].astype(str))[0]
            discrete.append(True)
    return X, np.array(discrete)


@safe_node("eda")
def eda_node(state):
    df = load_clean(state)
    target = (state.get("plan") or {}).get("target")
    ev = Evidence(state, "eda")
    out: dict = {}
    num = [c for c in df.select_dtypes(include="number").columns if c != target]

    if len(num) >= 2:
        corr = df[num].corr().abs()
        pairs = sorted(((corr.loc[a, b], a, b) for i, a in enumerate(num) for b in num[i + 1:]), reverse=True)
        strong = [(a, b, round(float(r), 3)) for r, a, b in pairs if r >= 0.8][:5]
        out["collinear_pairs"] = strong
        if strong:
            ev.add("eda", "Highly correlated feature pairs (|r|>=0.8): " + ", ".join(f"{a}~{b} ({r})" for a, b, r in strong) + ". Importance may be shared between them.", {"pairs": strong})

    outl = {}
    for c in num:
        s = df[c].dropna()
        if s.nunique() > 10:
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            pct = float(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).mean())
            if iqr > 0 and pct > 0.01:
                outl[c] = round(pct, 4)
    out["outliers"] = outl
    if outl:
        ev.add("eda", "Columns with >1% IQR outliers: " + ", ".join(f"{c} ({p:.1%})" for c, p in outl.items()) + ". Outliers were flagged, not removed.", {"outliers": outl})

    if target and target in df.columns:
        task = infer_task(df[target])
        y = df[target]
        if task == "classification":
            dist = y.astype(str).value_counts(normalize=True).round(4).to_dict()
            ev.add("eda", f"Target '{target}' class distribution: " + ", ".join(f"{k}={v:.1%}" for k, v in dist.items()) + ".", {"distribution": dist})
            out["target_distribution"] = dist
        else:
            d = y.describe()
            ev.add("eda", f"Target '{target}' summary: mean={d['mean']:.3f}, std={d['std']:.3f}, min={d['min']:.3f}, max={d['max']:.3f}; skew={y.skew():.2f}.", {k: float(d[k]) for k in ("mean", "std", "min", "max")})
        feats = [c for c in df.columns if c != target and (pd.api.types.is_numeric_dtype(df[c]) or is_text(df[c]) or pd.api.types.is_bool_dtype(df[c]))]
        if feats:
            sub = df.dropna(subset=[target]).sample(min(len(df), 20000), random_state=0)
            X, disc = _encode(sub[feats])
            yy = sub[target].astype(str) if task == "classification" else sub[target]
            fn = mutual_info_classif if task == "classification" else mutual_info_regression
            mi = pd.Series(fn(X, yy, discrete_features=disc, random_state=0), index=feats).sort_values(ascending=False)
            top = {k: round(float(v), 4) for k, v in mi.head(6).items()}
            out["mutual_info"] = top
            ev.add("eda", f"Strongest univariate associations with '{target}' (mutual information): " + ", ".join(f"{k} ({v})" for k, v in top.items()) + ". Association is not causation.", {"mutual_info": top})
    return {"eda": out, "evidence": ev.items}
