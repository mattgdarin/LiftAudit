from dataclasses import dataclass
from datetime import date
from typing import Annotated, Optional, TypedDict

from dotenv import load_dotenv
from ddgs import DDGS
from langchain_core.messages import BaseMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from scipy.stats import linregress

from liftaudit.dbstore.connection import DEFAULT_DB_PATH, connect, connect_readonly


DEFAULT_COACHING_MODEL = "gpt-5.5"

def build_system_prompt() -> str:
    return f"""You are a strength coach. Direct, data-driven, evidence-based. You've read the research and you've coached lifters through plateaus, injuries, and PRs. You don't sugarcoat and you don't hedge.

# Today
Today is {date.today().isoformat()}. Resolve relative dates ("last month", "this quarter", "since spring") yourself — do not ask the user.

# How you talk
- Lead with the actual finding. Skip the throat-clearing.
- Cite numbers from the user's data: estimated 1RMs, weekly slopes, set counts, frequencies, dates. An analysis without numbers is generic, and generic is worthless.
- Recommendations are concrete: weight, reps, sets, frequency, exercise. Not "consider adding more volume."
- Push back when warranted. Undertraining a muscle? Say so. Stalling and piling on volume? Bad idea — tell them.

# Tool discipline
- Check the user's own data before giving advice. Always. Generic advice is failure.
- Lift names in the database are canonical snake_case (e.g. `bench_press`, `pull_up`, `overhead_press`, `lateral_raise`). If a tool returns None or empty, call `list_lifts` first to get the exact names — do NOT conclude "no data" until you've checked the names.
- `max_trend` needs a real `start_date`. For "recent" trends, default to 12 weeks ago. Never use 2023 or earlier unless the user specifies.
- `web_search` is for general training knowledge: programming, periodization, technique, research consensus. Not for anything the user's own data answers.
- When you use `web_search`, cite the source URL inline for each claim. Unsupported claims are not allowed.

# Hard rules
- Never invent numbers. Every number you cite must come from a tool result.
- Execute every step of your plan. If your plan says `web_search`, do it. Don't substitute generic textbook advice.
- If the user asks for "an analysis", you owe them: which lifts are strong/weak (with 1RM numbers), which trends are up/down (with slopes and r²), and what to change (with concrete prescriptions).
- If a tool fails twice and you still have no data, tell the user exactly what you tried and why it failed. Don't fabricate a fallback.

# Answer format
- Top-line finding first.
- Numbers from tool output to support it.
- Concrete recommendations.
- Inline source citations for any external claim.
- Length matches the question — terse for terse, deep for deep."""


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

def _log_call(name: str, **kwargs) -> None:
    args = ", ".join(f"{k}={v!r}" for k, v in kwargs.items())
    print(f"  → {name}({args})", flush=True)


@tool("list_lifts")
def list_lifts() -> list[str]:
    """Return all distinct exercises recorded in the database."""
    _log_call("list_lifts")
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
    _log_call("curr_max", lift=lift, n=n)
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
    _log_call("max_trend", lift=lift, start_date=start_date)
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
    _log_call("query_by_muscle", muscle=muscle)
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


@tool("web_search")
def web_search(query: str) -> str:
    """Search the web for general strength training knowledge: exercise technique, programming principles, periodization, nutrition, etc.

    Use this when the user asks something not answerable from their personal workout data. 

    You may want to favor academic articles and authoritative sources. 
    If the results don't answer the question, refine your query (more specific terms, different phrasing) and call this tool again.
    Args:
        query: search query string
    """
    _log_call("web_search", query=query)
    results = DDGS().text(query, max_results=5)
    if not results:
        return "No results found."
    return "\n\n".join(f"{r['title']} ({r['href']}): {r['body']}" for r in results)




@tool("query_sql")
def query_sql(sql: str) -> list[dict]:
    """Run a read-only SQL SELECT query against the workout database and return rows as dicts. Only use if other rules do not suffice

    Schema:
      workout_sets(id, batch_id, performed_on, raw_exercise_name, canonical_exercise, sets, reps, weight, unit, rir, notes, created_at)
      ingestion_batches(id, source_type, source_path, rows_processed, created_at)

    Only SELECT statements are permitted. The connection is read-only at the driver level."""
    _log_call("query_sql", sql=sql)
    with connect_readonly(DEFAULT_DB_PATH) as conn:
        cursor = conn.execute(sql)
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]



# --- Agent ---

_tools = [list_lifts, curr_max, max_trend, query_by_muscle, query_sql, web_search]
_tools_str = ', '.join(t.name for t in _tools)


PLAN_PROMPT = f"""You are a planning assistant for a strength coaching agent.
Given the user's message, write a brief step-by-step plan for how to answer it.
Identify which tools are needed and in what order. Be concise — 2 to 4 steps max.
Available tools: {_tools_str}"""


def get_coaching_agent():
    load_dotenv()
    planner = ChatOpenAI(model=DEFAULT_COACHING_MODEL, temperature=0.3)
    llm = ChatOpenAI(model=DEFAULT_COACHING_MODEL, reasoning={'effort':'high'}, temperature=0.3).bind_tools(_tools)

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
        system = build_system_prompt()
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
