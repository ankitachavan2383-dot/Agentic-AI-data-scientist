"""LangGraph orchestration.

planner -> profiler -> cleaner -> eda -> (predictive?) -> feature_eng -> trainer -> evaluator --retrain--> trainer
                                      \\-> sql_analyst  <---------------------------------------------/
sql_analyst -> visualizer -> reporter -> END
"""
from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from app.agents.cleaner import cleaner_node
from app.agents.eda import eda_node
from app.agents.evaluator import evaluator_node
from app.agents.feature_eng import feature_eng_node
from app.agents.planner import planner_node
from app.agents.profiler import profiler_node
from app.agents.reporter import reporter_node
from app.agents.sql_agent import sql_node
from app.agents.trainer import trainer_node
from app.agents.visualizer import viz_node
from app.state import AgentState

# NOTE: LangGraph forbids node names equal to state keys, hence the *_agent suffix.
NODES = {"planner_agent": planner_node, "profiler_agent": profiler_node, "cleaner_agent": cleaner_node, "eda_agent": eda_node,
         "feature_agent": feature_eng_node, "trainer_agent": trainer_node, "evaluator_agent": evaluator_node,
         "sql_agent": sql_node, "viz_agent": viz_node, "reporter_agent": reporter_node}


def route_after_eda(state) -> str:
    wants_ml = "train" in (state.get("plan") or {}).get("steps", [])
    return "feature_agent" if wants_ml and (state.get("cleaning")) else "sql_agent"


def route_after_eval(state) -> str:
    return "trainer_agent" if state.get("needs_retrain") else "sql_agent"


@lru_cache(maxsize=1)
def get_graph():
    g = StateGraph(AgentState)
    for name, fn in NODES.items():
        g.add_node(name, fn)
    g.add_edge(START, "planner_agent")
    g.add_edge("planner_agent", "profiler_agent")
    g.add_edge("profiler_agent", "cleaner_agent")
    g.add_edge("cleaner_agent", "eda_agent")
    g.add_conditional_edges("eda_agent", route_after_eda, {"feature_agent": "feature_agent", "sql_agent": "sql_agent"})
    g.add_edge("feature_agent", "trainer_agent")
    g.add_edge("trainer_agent", "evaluator_agent")
    g.add_conditional_edges("evaluator_agent", route_after_eval, {"trainer_agent": "trainer_agent", "sql_agent": "sql_agent"})
    g.add_edge("sql_agent", "viz_agent")
    g.add_edge("viz_agent", "reporter_agent")
    g.add_edge("reporter_agent", END)
    return g.compile()
