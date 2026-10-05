import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from app.config import settings
from app.utils import text_of

log = logging.getLogger("ads.llm")


def get_llm():
    """Returns a LangChain chat model, or None -> agents use deterministic fallbacks."""
    try:
        if settings.llm_provider == "anthropic" and settings.anthropic_api_key:
            from langchain_anthropic import ChatAnthropic
            return ChatAnthropic(model=settings.llm_model, temperature=0, max_tokens=2500, api_key=settings.anthropic_api_key)
        if settings.llm_provider == "openai" and settings.openai_api_key:
            from langchain_openai import ChatOpenAI
            return ChatOpenAI(model=settings.llm_model, temperature=0, api_key=settings.openai_api_key)
    except Exception:  # noqa: BLE001
        log.exception("LLM init failed; using fallbacks")
    return None


def llm_text(system: str, user: str) -> str | None:
    llm = get_llm()
    if llm is None:
        return None
    try:
        return text_of(llm.invoke([SystemMessage(content=system), HumanMessage(content=user)]).content)
    except Exception:  # noqa: BLE001
        log.exception("LLM call failed")
        return None


def llm_json(system: str, user: str) -> dict | None:
    txt = llm_text(system + "\nRespond with a single JSON object and nothing else.", user)
    if not txt:
        return None
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0)) if m else None
    except json.JSONDecodeError:
        return None
