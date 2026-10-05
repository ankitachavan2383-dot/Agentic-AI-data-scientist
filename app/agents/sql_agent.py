"""SQL analyst: LLM tool-calling loop over a guarded read-only SQL tool, with a deterministic fallback."""
import logging

import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from app.llm import get_llm
from app.tools.loaders import load_clean
from app.tools.sql_tools import SqlGuardError, get_schema, run_select
from app.utils import Evidence, is_text, safe_node, text_of

log = logging.getLogger("ads.sql")
q = lambda c: '"' + c.replace('"', '""') + '"'  # noqa: E731


def _fallback_queries(state, df: pd.DataFrame) -> list[str]:
    t, plan = state["table_name"], state.get("plan") or {}
    target, qs = plan.get("target"), [f"SELECT COUNT(*) AS n_rows FROM {state['table_name']}"]
    question = state["question"].lower().replace("_", " ")
    cats = [c for c in df.columns if is_text(df[c]) and df[c].nunique() <= 50]
    nums = [c for c in df.select_dtypes("number").columns]
    mc = [c for c in cats if c.replace("_", " ") in question]
    mn = [c for c in nums if c.replace("_", " ") in question]
    if mc and mn:
        qs.append(f"SELECT {q(mc[0])}, SUM({q(mn[0])}) AS total, AVG({q(mn[0])}) AS average, COUNT(*) AS n FROM {t} GROUP BY {q(mc[0])} ORDER BY total DESC")
    top = [d["feature"] for d in (state.get("evaluation") or {}).get("importance", [])[:2]]
    if target and target in df.columns:
        tnum = pd.api.types.is_numeric_dtype(df[target]) and df[target].nunique() > 10
        pos = None if tnum else str(df[target].astype(str).value_counts().index[-1]).replace("'", "''")
        for f in top:
            if f not in df.columns:
                continue
            if f in cats:
                agg = f"AVG({q(target)})" if tnum else f"AVG(CASE WHEN CAST({q(target)} AS TEXT) = '{pos}' THEN 1.0 ELSE 0.0 END)"
                qs.append(f"SELECT {q(f)}, COUNT(*) AS n, {agg} AS target_rate_or_avg FROM {t} GROUP BY {q(f)} ORDER BY n DESC")
            elif pd.api.types.is_numeric_dtype(df[f]) and not tnum:
                qs.append(f"SELECT CAST({q(target)} AS TEXT) AS {q(target)}, COUNT(*) AS n, AVG({q(f)}) AS avg_{f} FROM {t} GROUP BY 1")
    if not mc and not target and cats:
        qs.append(f"SELECT {q(cats[0])}, COUNT(*) AS n FROM {t} GROUP BY {q(cats[0])} ORDER BY n DESC")
    return qs


def _fmt(res: dict, n=6) -> str:
    head = " | ".join(res["columns"])
    return head + " :: " + " ; ".join(", ".join(str(v if not isinstance(v, float) else round(v, 3)) for v in r) for r in res["rows"][:n])


@safe_node("sql_analyst")
def sql_node(state):
    table, df = state["table_name"], load_clean(state)
    allowed, ev, executed = {table}, Evidence(state, "sql_analyst"), []

    def execute(sql: str) -> str:
        try:
            res = run_select(sql, allowed)
        except SqlGuardError as e:
            return f"REJECTED: {e}"
        except Exception as e:  # noqa: BLE001
            return f"SQL ERROR: {str(e)[:300]}"
        executed.append({"sql": sql, "result": res})
        ev.add("sql", f"SQL `{sql}` returned {res['n_rows']} rows -> {_fmt(res)}", {"sql": sql, "columns": res["columns"], "rows": res["rows"][:20]})
        return _fmt(res, 20)

    llm, answer = get_llm(), None
    if llm is not None:
        @tool
        def get_table_schema() -> str:
            """Return the schema of the analysis table."""
            return get_schema(table)

        @tool
        def run_sql(query: str) -> str:
            """Run ONE read-only SELECT query on the analysis table and return up to 20 rows."""
            return execute(query)

        tools = {t.name: t for t in (get_table_schema, run_sql)}
        msgs = [SystemMessage(content=f"You are a careful SQL analyst. Answer the question using at most 4 SELECT queries against table {table}. Call get_table_schema first. Report only numbers returned by tools."),
                HumanMessage(content=state["question"])]
        try:
            bound = llm.bind_tools(list(tools.values()))
            for _ in range(6):
                ai = bound.invoke(msgs)
                msgs.append(ai)
                if not ai.tool_calls:
                    answer = text_of(ai.content)
                    break
                for tc in ai.tool_calls:
                    msgs.append(ToolMessage(content=tools[tc["name"]].invoke(tc["args"]), tool_call_id=tc["id"]))
        except Exception:  # noqa: BLE001
            log.exception("LLM SQL loop failed; falling back")
    if not executed:
        for sql in _fallback_queries(state, df):
            execute(sql)
    return {"sql": {"queries": [e["sql"] for e in executed], "llm_answer": answer}, "evidence": ev.items}
