"""Reporter: writes a report grounded ONLY in the evidence ledger, then verifies every citation."""
import json
import re
from pathlib import Path

from app import tracking
from app.llm import llm_text
from app.rag.knowledge import retrieve
from app.utils import safe_node

SYSTEM = """You are a senior data scientist writing an executive report.
Rules:
1. Use ONLY facts from the EVIDENCE list. Never invent numbers.
2. Every sentence that states a finding must end with citations like [E3] or [E3][E5]. Use [K1] only for guidance snippets.
3. Use associational language (not causal) unless evidence says otherwise.
4. Structure: '## Answer' (2-4 sentences directly answering the question), '## Key findings', '## Data quality & method', '## Limitations & caveats', '## Recommended next steps'.
5. If evidence is insufficient to answer, say so plainly."""


def _fallback(state, evidence) -> str:
    by = lambda *kinds: [e for e in evidence if e["kind"] in kinds]  # noqa: E731
    cite = lambda es: "".join(f"[{e['id']}]" for e in es)  # noqa: E731
    lines = ["## Answer"]
    findings = by("importance", "performance", "sql", "eda")
    lines.append(f"> Question: {state['question']}")
    if by("importance"):
        lines.append(by("importance")[0]["claim"] + f" [{by('importance')[0]['id']}]")
    if by("performance"):
        lines.append(by("performance")[0]["claim"] + f" [{by('performance')[0]['id']}]")
    if not (by("importance") or by("performance")) and by("sql"):
        last = by("sql")[-1]  # the most specific query is run last
        lines.append("Query result (" + last["data"]["sql"][:90] + "...): " + last["claim"].split(" -> ")[-1] + f" [{last['id']}]")
    lines += ["", "## Key findings"] + [f"- {e['claim']} [{e['id']}]" for e in findings]
    lines += ["", "## Data quality & method"] + [f"- {e['claim']} [{e['id']}]" for e in by("plan", "profile", "cleaning", "features", "model")]
    lines += ["", "## Limitations & caveats"] + ([f"- {e['claim']} [{e['id']}]" for e in by("warning", "decision")] or ["- No validity warnings were raised by the evaluator."])
    lines += [f"> Agent error: {e}" for e in state.get("errors", [])]
    lines += ["> Note: associations reported here are not causal effects."]
    return "\n".join(lines)


@safe_node("reporter")
def reporter_node(state):
    evidence = state.get("evidence", [])
    ids = {e["id"] for e in evidence}
    kb = retrieve(state["question"] + " " + " ".join(e["claim"] for e in evidence if e["kind"] in ("warning", "performance", "eda")), k=3)
    ledger = "\n".join(f"[{e['id']}] ({e['agent']}/{e['kind']}) {e['claim']}" for e in evidence)
    guidance = "\n".join(f"[{k['id']}] {k['title']}: {k['text']}" for k in kb)
    body = llm_text(SYSTEM, f"QUESTION: {state['question']}\n\nEVIDENCE:\n{ledger}\n\nGUIDANCE:\n{guidance}\n\nAgent errors: {state.get('errors', [])}")
    used_llm = bool(body)
    body = body or _fallback(state, evidence)

    cited = re.findall(r"\[E(\d+)\]", body)
    invalid = sorted({c for c in cited if f"E{c}" not in ids})
    for c in invalid:
        body = body.replace(f"[E{c}]", "")
    lines = [l for l in body.splitlines() if l.strip() and not l.startswith(("#", ">"))]
    cov = sum(bool(re.search(r"\[[EK]\d+\]", l)) for l in lines) / max(len(lines), 1)
    quality = {"llm_written": used_llm, "n_citations": len(cited) - sum(cited.count(c) for c in invalid), "invalid_citations_removed": invalid, "citation_coverage": round(cov, 3), "kb_used": [k["id"] for k in kb]}

    run_dir = Path(state["run_dir"])
    figs = "\n".join(f"![{f}](figures/{f})" for f in state.get("figures", []))
    appendix = "\n\n## Figures\n" + (figs or "_none_") + "\n\n## Evidence ledger\n" + "\n".join(f"- **{e['id']}** `{e['agent']}/{e['kind']}`: {e['claim']}" for e in evidence)
    if kb:
        appendix += "\n\n## Guidance consulted (RAG)\n" + "\n".join(f"- **{k['id']}** {k['title']} ({k['source']})" for k in kb)
    report = f"# Analysis report\n\n{body.strip()}{appendix}\n"
    (run_dir / "report.md").write_text(report)
    (run_dir / "evidence.json").write_text(json.dumps(evidence, indent=2))
    tracking.log_to_run(state.get("mlflow_run_id"), metrics={"citation_coverage": cov}, artifacts=[str(run_dir / "report.md"), str(run_dir / "evidence.json")], tags={"question": state["question"][:200]})
    return {"report": report, "report_quality": quality}
