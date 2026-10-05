"""Planner: interprets the question, picks a target (if predictive) and decides which agents run."""
import re

from app.llm import llm_json
from app.tools.loaders import load_raw
from app.utils import Evidence, safe_node

PRED_KW = re.compile(r"\b(predict|forecast|classif|driv|influenc|affect|explain|important|factor|determin|caus|why|model)", re.I)


def heuristic_plan(question: str, columns: list[str]) -> dict:
    q = question.lower().replace("_", " ")
    mentions = sorted((q.find(c.replace("_", " ")), c) for c in columns if c.replace("_", " ") in q)
    kw = PRED_KW.search(q)
    target = None
    if kw and mentions:
        after = [c for pos, c in mentions if pos >= kw.start()]
        target = after[0] if after else mentions[-1][1]
    return {"intent": "predictive" if target else "descriptive", "target": target, "source": "heuristic",
            "rationale": "Keyword/column matching on the question."}


@safe_node("planner")
def planner_node(state):
    df = load_raw(state["raw_path"])
    cols = list(df.columns)
    plan = None
    res = llm_json(
        "You plan data-science analyses. Given a question and dataset columns decide: "
        'intent ("predictive" if the user wants drivers/prediction of one column, else "descriptive"), '
        'target (exact column name or null), rationale (one sentence). Keys: intent, target, rationale.',
        f"Question: {state['question']}\nColumns (name:dtype): " + ", ".join(f"{c}:{df[c].dtype}" for c in cols),
    )
    if res and res.get("intent") in ("predictive", "descriptive") and (res.get("target") in cols or res.get("target") is None):
        plan = {**res, "source": "llm"}
        if plan["intent"] == "predictive" and not plan.get("target"):
            plan = None
    plan = plan or heuristic_plan(state["question"], cols)
    steps = ["profile", "clean", "eda"] + (["feature_eng", "train", "evaluate"] if plan["intent"] == "predictive" else []) + ["sql", "visualize", "report"]
    plan["steps"] = steps
    ev = Evidence(state, "planner")
    ev.add("plan", f"Question interpreted as {plan['intent']}" + (f" with target '{plan['target']}'" if plan.get("target") else "") + f" (planner: {plan['source']}).", plan)
    return {"plan": plan, "evidence": ev.items, "retries": 0}
