from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app import tracking
from app.config import settings
from app.utils import Evidence, infer_task, safe_node

SEED = 42


def candidates(task: str, attempt: int) -> dict:
    if task == "classification":
        if attempt == 0:
            return {"logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
                    "random_forest": RandomForestClassifier(300, class_weight="balanced", n_jobs=-1, random_state=SEED),
                    "hist_gradient_boosting": HistGradientBoostingClassifier(random_state=SEED)}
        return {"hist_gb_tuned": HistGradientBoostingClassifier(learning_rate=0.05, max_iter=400, l2_regularization=1.0, random_state=SEED),
                "random_forest_deep": RandomForestClassifier(600, min_samples_leaf=2, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED),
                "extra_trees": ExtraTreesClassifier(500, class_weight="balanced", n_jobs=-1, random_state=SEED)}
    if attempt == 0:
        return {"ridge": Ridge(alpha=1.0),
                "random_forest": RandomForestRegressor(300, n_jobs=-1, random_state=SEED),
                "hist_gradient_boosting": HistGradientBoostingRegressor(random_state=SEED)}
    return {"hist_gb_tuned": HistGradientBoostingRegressor(learning_rate=0.05, max_iter=400, l2_regularization=1.0, random_state=SEED),
            "random_forest_deep": RandomForestRegressor(600, min_samples_leaf=2, n_jobs=-1, random_state=SEED),
            "extra_trees": ExtraTreesRegressor(500, n_jobs=-1, random_state=SEED)}


def make_pipeline(num, cat, est) -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("sc", StandardScaler())]), num),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01))]), cat),
    ])
    return Pipeline([("pre", pre), ("model", est)])


@safe_node("trainer")
def trainer_node(state):
    if "features" not in state:
        raise RuntimeError("no feature set available")
    target, attempt = state["plan"]["target"], state.get("retries", 0)
    run_dir = Path(state["run_dir"])
    df = pd.read_parquet(run_dir / "features.parquet")
    f = state["features"]
    num, cat = f["numeric"], f["categorical"]
    if not (num or cat):
        raise RuntimeError("no usable features")
    if len(df) > settings.max_train_rows:
        df = df.sample(settings.max_train_rows, random_state=SEED)
    X, y = df[num + cat].copy(), df[target]
    for c in cat:
        X[c] = X[c].astype(object).where(X[c].notna(), np.nan)
    task = infer_task(y)
    strat = y if task == "classification" and y.value_counts().min() >= 2 else None
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=strat)

    if task == "classification":
        k = min(5, int(ytr.value_counts().min()))
        if k < 2:
            raise RuntimeError("a target class has fewer than 2 training samples")
        cv, scoring = StratifiedKFold(k, shuffle=True, random_state=SEED), "f1_macro"
        baseline = DummyClassifier(strategy="most_frequent")
    else:
        cv, scoring = KFold(5, shuffle=True, random_state=SEED), "r2"
        baseline = DummyRegressor()

    results = {}
    for name, est in candidates(task, attempt).items():
        sc = cross_val_score(make_pipeline(num, cat, est), Xtr, ytr, cv=cv, scoring=scoring)
        results[name] = {"cv_mean": float(sc.mean()), "cv_std": float(sc.std())}
    best = max(results, key=lambda k: results[k]["cv_mean"])
    pipe = make_pipeline(num, cat, candidates(task, attempt)[best]).fit(Xtr, ytr)
    base = make_pipeline(num, cat, baseline).fit(Xtr, ytr)
    joblib.dump(pipe, run_dir / "model.joblib")
    joblib.dump({"Xtr": Xtr, "Xte": Xte, "ytr": ytr, "yte": yte, "base": base}, run_dir / "split.joblib")

    mlflow_run_id = state.get("mlflow_run_id")
    try:
        tracking.setup()
        with mlflow.start_run(run_name=f"run-{state['run_id']}-attempt-{attempt}") as r:
            mlflow_run_id = r.info.run_id
            mlflow.log_params({"target": target, "task": task, "best_model": best, "attempt": attempt, "n_train": len(Xtr), "n_features": len(num) + len(cat), "question": state["question"][:250]})
            for n, v in results.items():
                mlflow.log_metric(f"cv_{scoring}_{n}", v["cv_mean"])
            try:
                mlflow.sklearn.log_model(pipe, name="model", serialization_format="cloudpickle")  # mlflow>=3
            except TypeError:
                mlflow.sklearn.log_model(pipe, artifact_path="model", serialization_format="cloudpickle")  # mlflow 2.x
    except Exception as e:  # noqa: BLE001
        tracking.log.warning("MLflow unavailable: %s", e)

    model = {"task": task, "scoring": scoring, "best": best, "candidates": results, "attempt": attempt, "n_train": len(Xtr), "n_test": len(Xte)}
    ev = Evidence(state, "trainer")
    ev.add("model", f"Trained {len(results)} candidate models with {cv.get_n_splits()}-fold CV on {len(Xtr)} rows ({task}, metric {scoring}): "
           + ", ".join(f"{n}={v['cv_mean']:.3f}±{v['cv_std']:.3f}" for n, v in results.items()) + f". Selected '{best}'.", model)
    return {"model": model, "mlflow_run_id": mlflow_run_id, "needs_retrain": False, "evidence": ev.items}
