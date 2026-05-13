from dotenv import load_dotenv
from liftaudit.dbstore.connection import DEFAULT_DB_PATH, connect
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage
from dataclasses import dataclass
from datetime import date
from typing import Annotated, TypedDict
from scipy.stats import linregress


def get_coaching_agent():
    raise NotImplementedError

def one_rep_max(weight, reps):
    return weight * (1+reps/30)

@dataclass
class TrendResult:
    exercise: str
    slope_per_week: float      # estimated 1RM change per week
    r_squared: float           # 0–1, how linear the trend is
    n: int                     # number of data points
    first_date: date
    last_date: date

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    preferences: str
    strengths: list[str]
    weakneases: list[str]
    active_lifts: list[str]





def compute_trend(exercise: str, points: list[tuple[str, float]]) -> TrendResult:
    n = len(points)
    if n < 2:
        return TrendResult(exercise=exercise, slope_per_week=0.0, r_squared=0.0, n=n,
                           first_date=points[0][0] if n else None, last_date=points[-1][0] if n else None)

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


def query_lift(lift: str, start_date: date) -> list[dict]:
    """Return all sets for a given lift on or after start_date."""
    with connect(DEFAULT_DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT performed_on, sets, reps, weight, unit
            FROM workout_sets
            WHERE (canonical_exercise = ? OR raw_exercise_name = ?)
              AND performed_on >= ?
            ORDER BY performed_on ASC
            """,
            (lift, lift, start_date.isoformat()),
        ).fetchall()
    return [
        {"date": r[0], "sets": r[1], "reps": r[2], "weight": r[3], "unit": r[4]}
        for r in rows
    ]

@tool("Max_Trend")
def max_trend(lift: str, start_date: date) -> TrendResult:
    """Compute estimated 1RM trend for a lift since start_date."""
    rows = query_lift(lift, start_date)
    points = [
        (r["date"], one_rep_max(r["weight"], r["reps"]))
        for r in rows
        if r["weight"] is not None and r["reps"] is not None
    ]
    return compute_trend(lift, points)