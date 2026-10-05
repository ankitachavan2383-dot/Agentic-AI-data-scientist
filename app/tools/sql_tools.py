"""Guarded, read-only SQL execution against per-run tables in PostgreSQL."""
import re

import pandas as pd
from sqlalchemy import inspect

from app.db import engine
from app.utils import jsonable

FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy|attach|pragma|into|call|execute|vacuum)\b"
    r"|\bpg_\w+|information_schema|sqlite_master",
    re.I,
)


class SqlGuardError(ValueError):
    pass


def validate_select(query: str, allowed_tables: set[str]) -> str:
    q = re.sub(r"--[^\n]*|/\*.*?\*/", "", query, flags=re.S).strip().rstrip(";").strip()
    if ";" in q:
        raise SqlGuardError("Only a single statement is allowed.")
    if not re.match(r"^(select|with)\b", q, re.I):
        raise SqlGuardError("Only SELECT queries are allowed.")
    chk = re.sub(r"'[^']*'", "''", q)
    if FORBIDDEN.search(chk):
        raise SqlGuardError("Query contains a forbidden keyword or object.")
    chk = re.sub(r"\b(extract|substring|trim|overlay)\s*\([^()]*\)", "", chk, flags=re.I)
    ctes = {m.lower() for m in re.findall(r"\b(\w+)\s+as\s*\(", chk, re.I)}
    allowed = {t.lower() for t in allowed_tables}
    for m in re.finditer(r"\b(?:from|join)\s+([\"\w\.]+)", chk, re.I):
        name = m.group(1).strip('"').lower()
        if name not in allowed and name not in ctes:
            raise SqlGuardError(f"Table '{name}' is not allowed. Use: {sorted(allowed)}")
    return q


def run_select(query: str, allowed_tables: set[str], limit: int = 200) -> dict:
    q = validate_select(query, allowed_tables)
    wrapped = f"SELECT * FROM ({q}) AS _q LIMIT {int(limit)}"
    pg = engine.dialect.name == "postgresql"
    with engine.connect() as conn:
        if pg:
            conn.exec_driver_sql("SET TRANSACTION READ ONLY")
            conn.exec_driver_sql("SET LOCAL statement_timeout = 5000")
        res = conn.exec_driver_sql(wrapped.replace("%", "%%") if pg else wrapped)
        cols = list(res.keys())
        rows = [list(r) for r in res.fetchall()]
    return {"columns": cols, "rows": jsonable(rows), "n_rows": len(rows)}


def get_schema(table: str) -> str:
    cols = inspect(engine).get_columns(table)
    return f"table {table}(" + ", ".join(f"{c['name']} {c['type']}" for c in cols) + ")"


def load_table(df: pd.DataFrame, table: str) -> None:
    df.to_sql(table, engine, if_exists="replace", index=False, chunksize=2000)
