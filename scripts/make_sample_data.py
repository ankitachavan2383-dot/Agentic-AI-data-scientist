from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evals.datasets import churn  # noqa: E402

out = Path("sample_data"); out.mkdir(exist_ok=True)
churn().to_csv(out / "churn.csv", index=False)
print("wrote", out / "churn.csv")
