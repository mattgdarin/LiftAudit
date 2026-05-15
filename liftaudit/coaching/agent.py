from dataclasses import dataclass
from datetime import date
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from scipy.stats import linregress

from liftaudit.dbstore.connection import DEFAULT_DB_PATH, connect, connect_readonly


DEFAULT_COACHING_MODEL = "gpt-4o-mini"

SYSTEM_PROMPT = """You are a strength coaching assistant. You have access to the user's workout history.
Use your tools to look up their data before giving advice. Be concise and specific — cite actual numbers.
Do not invent data. If you don't have enough data to answer, say so."""


# --- Data types ---

@dataclass
class TrendResult:
    exercise: str
    slope_per_week: float
    r_squared: float
    n: int
    first_date: date
    last_date: date


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    preferences: str
    strengths: list[str]
    weaknesses: list[str]
    active_lifts: list[str]
    plan: str | None


# --- Pure helpers ---

def one_rep_max(weight: float, reps: float) -> float:
    return weight * (1 + reps / 30)


def compute_trend(exercise: str, points: list[tuple[str, float]]) -> TrendResult:
    n = len(points)
    if n < 2:
        return TrendResult(
            exercise=exercise, slope_per_week=0.0, r_squared=0.0, n=n,
            first_date=points[0][0] if n else None,
            last_date=points[-1][0] if n else None,
        )
    weeks = [(date.fromisoformat(p[0]) - date.fromisoformat(points[0][0])).days / 7 for p in points]
    y = [p[1] for p in points]
    result = linregress(weeks, y)
    return TrendResult(
        exercise=exercise,
        slope_per_week=round(result.slope, 2),
        r_squared=round(result.rvalue ** 2, 3),
        n=n,
        first_date=points[0][0],
        last_date=points[-1][0],
    )


def query_lift(lift: str, start_date: date = date.min, most_recent_k: int | None = None) -> list[dict]:
    with connect(DEFAULT_DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT performed_on, sets, reps, weight, unit, rir
            FROM workout_sets
            WHERE (canonical_exercise = ? OR raw_exercise_name = ?)
              AND performed_on >= ?
            ORDER BY performed_on ASC
            """,
            (lift, lift, start_date.isoformat()),
        ).fetchall()
    if most_recent_k is not None:
        rows = rows[-most_recent_k:]
    return [
        {"date": r[0], "sets": r[1], "reps": r[2], "weight": r[3], "unit": r[4], "rir": r[5]}
        for r in rows
    ]


_SCHEMA_SUMMARY = """
Tables:
  workout_sets(id, batch_id, performed_on TEXT, raw_exercise_name, canonical_exercise, sets, reps, weight, unit, rir, notes, created_at)
  ingestion_batches(id, source_type, source_path, rows_processed, created_at)
  ingestion_warnings(id, batch_id, row_number, field, message, suggested_correction_json, created_at)
  unresolved_exercises(id, batch_id, input_name, status, canonical_name, matched_alias, score, created_at)
"""


# --- Tools ---

@tool("list_lifts")
def list_lifts() -> list[str]:
    """Return all distinct exercises recorded in the database."""
    with connect(DEFAULT_DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT DISTINCT COALESCE(canonical_exercise, raw_exercise_name)
            FROM workout_sets
            ORDER BY 1
            """
        ).fetchall()
    return [r[0] for r in rows if r[0]]


@tool("curr_max")
def curr_max(lift: str, n: int = 5) -> Optional[float]:
    """Returns the estimated current 1RM for a lift, averaged over the n most recent sets."""
    data = query_lift(lift, most_recent_k=n)
    maxes = [
        one_rep_max(r["weight"], r["reps"] + (r["rir"] or 0))
        for r in data
        if r["weight"] is not None and r["reps"] is not None
    ]
    return round(sum(maxes) / len(maxes), 1) if maxes else None


@tool("max_trend")
def max_trend(lift: str, start_date: date) -> TrendResult:
    """Compute estimated 1RM trend for a lift since start_date."""
    rows = query_lift(lift, start_date)
    points = [
        (r["date"], one_rep_max(r["weight"], r["reps"]))
        for r in rows
        if r["weight"] is not None and r["reps"] is not None
    ]
    return compute_trend(lift, points)


@tool("query_by_muscle")
def query_by_muscle(muscle: str) -> dict:
    """Return all sets where a muscle is trained, split into primary and secondary.

    Args:
        muscle: muscle name to search for (e.g. 'Chest', 'Quads', 'Glutes')

    Returns a dict with keys 'primary' and 'secondary', each a list of sets.
    """
    pattern = f"%{muscle}%"
    with connect_readonly(DEFAULT_DB_PATH) as conn:
        primary = conn.execute(
            """
            SELECT performed_on, canonical_exercise, sets, reps, weight, unit, rir, target_muscles
            FROM workout_sets
            WHERE target_muscles LIKE ?
            ORDER BY performed_on ASC
            """,
            (pattern,),
        ).fetchall()
        secondary = conn.execute(
            """
            SELECT performed_on, canonical_exercise, sets, reps, weight, unit, rir, secondary_muscles
            FROM workout_sets
            WHERE secondary_muscles LIKE ?
              AND (target_muscles NOT LIKE ? OR target_muscles IS NULL)
            ORDER BY performed_on ASC
            """,
            (pattern, pattern),
        ).fetchall()

    def row_to_dict(r, muscle_key):
        return {
            "date": r[0], "exercise": r[1], "sets": r[2],
            "reps": r[3], "weight": r[4], "unit": r[5],
            "rir": r[6], muscle_key: r[7],
        }

    return {
        "primary": [row_to_dict(r, "target_muscles") for r in primary],
        "secondary": [row_to_dict(r, "secondary_muscles") for r in secondary],
    }


@tool("query_sql")
def query_sql(sql: str) -> list[dict]:
    """Run a read-only SQL SELECT query against the workout database and return rows as dicts. Only use if other rules do not suffice

    Schema:
      workout_sets(id, batch_id, performed_on, raw_exercise_name, canonical_exercise, sets, reps, weight, unit, rir, notes, created_at)
      ingestion_batches(id, source_type, source_path, rows_processed, created_at)

    Only SELECT statements are permitted. The connection is read-only at the driver level."""
    with connect_readonly(DEFAULT_DB_PATH) as conn:
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]



# --- Agent ---

_tools = [list_lifts, curr_max, max_trend, query_by_muscle, query_sql]
_tools_str = ', '.join(t.name for t in _tools)


PLAN_PROMPT = f"""You are a planning assistant for a strength coaching agent.
Given the user's message, write a brief step-by-step plan for how to answer it.
Identify which tools are needed and in what order. Be concise — 2 to 4 steps max.
Available tools: {_tools_str}"""


def get_coaching_agent():
    load_dotenv()
    planner = ChatOpenAI(model=DEFAULT_COACHING_MODEL, temperature=0)
    llm = ChatOpenAI(model=DEFAULT_COACHING_MODEL, temperature=0).bind_tools(_tools)

    def plan_node(state: State) -> State:
        last_user = next(
            (m for m in reversed(state["messages"]) if m.type == "human"), None
        )
        if last_user is None:
            return {"plan": None}
        response = planner.invoke([
            {"role": "system", "content": PLAN_PROMPT},
            {"role": "user", "content": last_user.content},
        ])

        print(response.content)
        return {"plan": response.content}

    def call_model(state: State) -> State:
        system = SYSTEM_PROMPT
        if state.get("plan"):
            system += f"\n\nPlan for this response:\n{state['plan']}"
        messages = [{"role": "system", "content": system}] + state["messages"]
        return {"messages": [llm.invoke(messages)]}

    def route(state: State) -> str:
        last = state["messages"][-1]
        if hasattr(last, "tool_calls") and last.tool_calls:
            return "call_tools"
        return END

    graph = StateGraph(State)
    graph.add_node("plan_node", plan_node)
    graph.add_node("call_model", call_model)
    graph.add_node("call_tools", ToolNode(_tools))
    graph.add_edge(START, "plan_node")
    graph.add_edge("plan_node", "call_model")
    graph.add_conditional_edges("call_model", route, {"call_tools": "call_tools", END: END})
    graph.add_edge("call_tools", "call_model")

    return graph.compile()
