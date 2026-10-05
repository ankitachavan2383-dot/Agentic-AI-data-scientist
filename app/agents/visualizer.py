from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from app.tools.loaders import load_clean  # noqa: E402
from app.utils import is_text, safe_node  # noqa: E402


def _save(fig, d: Path, name: str, made: list):
    fig.tight_layout()
    fig.savefig(d / name, dpi=110)
    plt.close(fig)
    made.append(name)


@safe_node("visualizer")
def viz_node(state):
    run_dir = Path(state["run_dir"])
    d = run_dir / "figures"
    d.mkdir(exist_ok=True)
    df, made = load_clean(state), []
    target = (state.get("plan") or {}).get("target")

    miss = df.isna().mean().sort_values(ascending=False)
    miss = miss[miss > 0].head(15)
    if len(miss):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        miss[::-1].mul(100).plot.barh(ax=ax, color="#c0504d")
        ax.set(title="Missing values after cleaning (%)", xlabel="%")
        _save(fig, d, "missingness.png", made)

    if target and target in df.columns:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        if is_text(df[target]) or df[target].nunique() <= 10:
            df[target].astype(str).value_counts().plot.bar(ax=ax, color="#4f81bd")
        else:
            ax.hist(df[target].dropna(), bins=30, color="#4f81bd")
        ax.set(title=f"Target distribution: {target}")
        _save(fig, d, "target_distribution.png", made)

    num = [c for c in df.select_dtypes("number").columns][:12]
    if len(num) >= 3:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        im = ax.imshow(df[num].corr(), cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(num)), num, rotation=90)
        ax.set_yticks(range(len(num)), num)
        fig.colorbar(im)
        ax.set_title("Correlation matrix")
        _save(fig, d, "correlation.png", made)

    imp = (state.get("evaluation") or {}).get("importance", [])[:10]
    if imp:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh([i["feature"] for i in imp][::-1], [i["importance"] for i in imp][::-1], xerr=[i["std"] for i in imp][::-1], color="#9bbb59")
        ax.set(title="Permutation importance (test set)")
        _save(fig, d, "feature_importance.png", made)
        try:
            pipe, sp = joblib.load(run_dir / "model.joblib"), joblib.load(run_dir / "split.joblib")
            pred = pipe.predict(sp["Xte"])
            fig, ax = plt.subplots(figsize=(5, 5))
            if state["model"]["task"] == "regression":
                ax.scatter(sp["yte"], pred, s=8, alpha=0.5)
                lim = [min(sp["yte"].min(), pred.min()), max(sp["yte"].max(), pred.max())]
                ax.plot(lim, lim, "k--")
                ax.set(title="Predicted vs actual", xlabel="actual", ylabel="predicted")
                _save(fig, d, "predicted_vs_actual.png", made)
            else:
                from sklearn.metrics import ConfusionMatrixDisplay
                ConfusionMatrixDisplay.from_predictions(sp["yte"], pred, ax=ax, colorbar=False)
                ax.set_title("Confusion matrix (test set)")
                _save(fig, d, "confusion_matrix.png", made)
        except Exception:  # noqa: BLE001
            plt.close("all")
    return {"figures": made}
