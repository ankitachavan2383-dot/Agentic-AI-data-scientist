import re


def citation_validity(report_body: str, evidence_ids: set[str]) -> float:
    cites = re.findall(r"\[(E\d+)\]", report_body)
    return 1.0 if not cites else sum(c in evidence_ids for c in cites) / len(cites)


def numeric_grounding(report_body: str, evidence_text: str) -> float:
    """Share of numbers in the narrative that literally appear in the evidence ledger."""
    nums = [n for n in re.findall(r"\d+\.\d+|\d{2,}", re.sub(r"\[[EK]\d+\]", "", report_body))]
    return 1.0 if not nums else sum(n in evidence_text for n in nums) / len(nums)


def narrative(report: str) -> str:
    return report.split("## Figures")[0]


def score_case(case: dict, state: dict) -> dict:
    exp, plan, ev = case["expect"], state.get("plan", {}), state.get("evidence", [])
    ev_text = " ".join(e["claim"] + str(e["data"]) for e in ev)
    body = narrative(state.get("report", ""))
    checks = {
        "no_agent_errors": not state.get("errors"),
        "intent": plan.get("intent") == exp["intent"],
        "citations_valid": citation_validity(body, {e["id"] for e in ev}) == 1.0,
        "citation_coverage>=0.8": state.get("report_quality", {}).get("citation_coverage", 0) >= 0.8,
        "numeric_grounding>=0.8": numeric_grounding(body, ev_text) >= 0.8,
        "report_exists": len(body) > 200,
    }
    if "target" in exp:
        checks["target"] = plan.get("target") == exp["target"]
    if "task_type" in exp:
        checks["task_type"] = (state.get("model") or {}).get("task") == exp["task_type"]
    if "min_lift" in exp:
        checks["lift"] = ((state.get("evaluation") or {}).get("metrics") or {}).get("lift", -1) >= exp["min_lift"]
    if "top_features_any" in exp:
        top3 = [d["feature"] for d in (state.get("evaluation") or {}).get("importance", [])[:3]]
        checks["top_features"] = any(f in top3 for f in exp["top_features_any"])
    sqls = [e for e in ev if e["kind"] == "sql"]
    if "sql_contains" in exp:
        checks["sql_used"] = any(exp["sql_contains"] in e["data"].get("sql", "").lower() for e in sqls)
    if "sql_top_row_contains" in exp:
        checks["sql_answer"] = any(e["data"].get("rows") and exp["sql_top_row_contains"] in str(e["data"]["rows"][0]).lower() and exp["sql_contains"] in " ".join(e["data"].get("columns", [])).lower() for e in sqls)
    return checks
