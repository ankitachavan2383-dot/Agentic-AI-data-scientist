from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score

from app import tracking
from app.utils import Evidence, safe_node


def _score(task, y, pred):
    return f1_score(y, pred, average="macro") if task == "classification" else r2_score(y, pred)


def _direction_notes(pipe, Xte, task, top_features):
    if task == "classification":
        proba = pipe.predict_proba(Xte)
        cls = list(pipe.classes_)
        pos = cls[1] if len(cls) == 2 else cls[int(np.argmin(proba.mean(0)))]
        out = pd.Series(proba[:, cls.index(pos)], index=Xte.index)
        label = f"probability of '{pos}'"
    else:
        out, label = pd.Series(pipe.predict(Xte), index=Xte.index), "predicted value"
    notes = []
    for f in top_features:
        s = Xte[f]
        if pd.api.types.is_numeric_dtype(s) and s.nunique() > 2:
            r = s.corr(out, method="spearman")
            if pd.notna(r):
                notes.append(f"higher {f} -> {'higher' if r > 0 else 'lower'} {label} (Spearman {r:.2f})")
        else:
            g = out.groupby(s.astype(str)).mean().sort_values()
            if len(g) >= 2:
                notes.append(f"{f}='{g.index[-1]}' has the highest mean {label} ({g.iloc[-1]:.2f}) vs '{g.index[0]}' ({g.iloc[0]:.2f})")
    return notes


@safe_node("evaluator")
def evaluator_node(state):
    if not state.get("model"):
        return {}
    run_dir = Path(state["run_dir"])
    pipe, sp = joblib.load(run_dir / "model.joblib"), joblib.load(run_dir / "split.joblib")
    Xte, yte, base = sp["Xte"], sp["yte"], sp["base"]
    m = state["model"]
    task, ev = m["task"], Evidence(state, "evaluator")
    pred, bpred = pipe.predict(Xte), base.predict(Xte)
    score, bscore = _score(task, yte, pred), _score(task, yte, bpred)
    lift = score - bscore
    metrics = {"primary_metric": m["scoring"], "test_score": score, "baseline_score": bscore, "lift": lift}

    if task == "classification":
        metrics["accuracy"] = accuracy_score(yte, pred)
        try:
            proba, cls = pipe.predict_proba(Xte), list(pipe.classes_)
            metrics["roc_auc"] = roc_auc_score(yte == cls[1], proba[:, 1]) if len(cls) == 2 else roc_auc_score(yte, proba, multi_class="ovr", labels=cls)
        except Exception:  # noqa: BLE001
            pass
        txt = f"accuracy {metrics['accuracy']:.3f}" + (f", ROC-AUC {metrics['roc_auc']:.3f}" if "roc_auc" in metrics else "")
    else:
        metrics.update(rmse=float(np.sqrt(mean_squared_error(yte, pred))), mae=float(mean_absolute_error(yte, pred)))
        txt = f"RMSE {metrics['rmse']:.3f}, MAE {metrics['mae']:.3f} (target std {yte.std():.3f})"
    ev.add("performance", f"On {len(yte)} held-out rows, '{m['best']}' scored {m['scoring']}={score:.3f} versus {bscore:.3f} for a naive baseline (lift {lift:+.3f}); {txt}.", metrics)

    sub = Xte.sample(min(len(Xte), 1500), random_state=0)
    pi = permutation_importance(pipe, sub, yte.loc[sub.index], n_repeats=5, random_state=0, scoring=m["scoring"])
    imp = sorted(({"feature": f, "importance": float(a), "std": float(s)} for f, a, s in zip(Xte.columns, pi.importances_mean, pi.importances_std)), key=lambda d: -d["importance"])
    top = [d for d in imp if d["importance"] > 2 * d["std"] and d["importance"] > 0][:6]
    if top:
        ev.add("importance", f"Top predictive features by permutation importance (drop in {m['scoring']} when shuffled): " + ", ".join(f"{d['feature']} ({d['importance']:.3f})" for d in top) + ". This shows predictive relevance, not causation.", {"importance": top})
        notes = _direction_notes(pipe, Xte, task, [d["feature"] for d in top[:3]])
        if notes:
            ev.add("direction", "Direction of effects among top features: " + "; ".join(notes) + ".", {"notes": notes})
    else:
        ev.add("importance", "No feature shows a statistically meaningful permutation importance; the features carry little predictive signal.", {"importance": imp[:6]})

    warnings = []
    pos_total = sum(d["importance"] for d in imp if d["importance"] > 0)
    if pos_total > 0 and top and top[0]["importance"] / pos_total > 0.8:
        warnings.append(f"'{top[0]['feature']}' accounts for >80% of importance; check for target leakage")
    if score > 0.98:
        warnings.append("near-perfect test score suggests possible leakage or a trivially separable target")
    if m["candidates"][m["best"]]["cv_std"] > 0.1:
        warnings.append("cross-validation scores vary widely across folds; estimates are unstable")
    if len(yte) < 100:
        warnings.append(f"small test set (n={len(yte)}); metrics are noisy")
    if warnings:
        ev.add("warning", "Validity warnings: " + "; ".join(warnings) + ".", {"warnings": warnings})

    out = {"evaluation": {"metrics": metrics, "importance": imp[:15], "warnings": warnings}, "evidence": ev.items}
    retries = state.get("retries", 0)
    if lift < 0.05 and retries < 1:
        ev.add("decision", f"Lift over baseline is only {lift:+.3f}; evaluator requested one retraining round with a larger, tuned model family.", {"lift": lift})
        out.update(needs_retrain=True, retries=retries + 1)
    tracking.log_to_run(state.get("mlflow_run_id"), metrics={f"test_{k}": v for k, v in metrics.items()})
    return out
