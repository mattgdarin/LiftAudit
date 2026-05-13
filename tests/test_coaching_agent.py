import pytest
from datetime import date
from unittest.mock import MagicMock, patch

from liftaudit.coaching.agent import (
    State,
    TrendResult,
    compute_trend,
    curr_max,
    get_coaching_agent,
    list_lifts,
    max_trend,
    one_rep_max,
    query_lift,
)


# --- one_rep_max ---

def test_one_rep_max_single_rep():
    assert one_rep_max(100, 1) == pytest.approx(100 * (1 + 1 / 30))


def test_one_rep_max_five_reps():
    assert one_rep_max(185, 5) == pytest.approx(185 * (1 + 5 / 30))


# --- compute_trend ---

def test_compute_trend_insufficient_data():
    result = compute_trend("bench_press", [])
    assert result.slope_per_week == 0.0
    assert result.r_squared == 0.0
    assert result.n == 0

    result = compute_trend("bench_press", [("2026-01-01", 200.0)])
    assert result.n == 1
    assert result.slope_per_week == 0.0


def test_compute_trend_perfect_linear():
    # 5 lb/week increase over 4 weeks
    points = [
        ("2026-01-01", 200.0),
        ("2026-01-08", 205.0),
        ("2026-01-15", 210.0),
        ("2026-01-22", 215.0),
    ]
    result = compute_trend("bench_press", points)
    assert result.slope_per_week == pytest.approx(5.0, abs=0.1)
    assert result.r_squared == pytest.approx(1.0, abs=0.01)
    assert result.n == 4


def test_compute_trend_declining():
    points = [
        ("2026-01-01", 300.0),
        ("2026-01-08", 295.0),
        ("2026-01-15", 290.0),
    ]
    result = compute_trend("deadlift", points)
    assert result.slope_per_week < 0


# --- query_lift (mocked DB) ---

MOCK_ROWS = [
    ("2026-03-03", 4, 5, 175.0, "lb", 3),
    ("2026-03-10", 4, 5, 180.0, "lb", 2),
    ("2026-03-17", 4, 5, 185.0, "lb", 2),
    ("2026-03-24", 4, 5, 190.0, "lb", 1),
    ("2026-03-31", 4, 5, 195.0, "lb", 1),
]


def _mock_connect(rows):
    conn = MagicMock()
    conn.__enter__ = MagicMock(return_value=conn)
    conn.__exit__ = MagicMock(return_value=False)
    conn.execute.return_value.fetchall.return_value = rows
    return conn


@patch("liftaudit.coaching.agent.connect")
def test_query_lift_returns_all(mock_connect):
    mock_connect.return_value = _mock_connect(MOCK_ROWS)
    rows = query_lift("bench_press")
    assert len(rows) == 5
    assert rows[0] == {"date": "2026-03-03", "sets": 4, "reps": 5, "weight": 175.0, "unit": "lb", "rir": 3}


@patch("liftaudit.coaching.agent.connect")
def test_query_lift_most_recent_k(mock_connect):
    mock_connect.return_value = _mock_connect(MOCK_ROWS)
    rows = query_lift("bench_press", most_recent_k=2)
    assert len(rows) == 2
    assert rows[0]["weight"] == 190.0
    assert rows[1]["weight"] == 195.0


@patch("liftaudit.coaching.agent.connect")
def test_query_lift_start_date_filters(mock_connect):
    filtered = [r for r in MOCK_ROWS if r[0] >= "2026-03-17"]
    mock_connect.return_value = _mock_connect(filtered)
    rows = query_lift("bench_press", start_date=date(2026, 3, 17))
    assert len(rows) == 3


# --- curr_max tool ---

@patch("liftaudit.coaching.agent.connect")
def test_curr_max_averages_recent_sets(mock_connect):
    mock_connect.return_value = _mock_connect(MOCK_ROWS[-3:])
    result = curr_max.invoke({"lift": "bench_press", "n": 3})
    expected = sum(one_rep_max(w, 5 + r) for _, _, _, w, _, r in MOCK_ROWS[-3:]) / 3
    assert result == pytest.approx(expected, abs=0.5)


@patch("liftaudit.coaching.agent.connect")
def test_curr_max_returns_none_on_empty(mock_connect):
    mock_connect.return_value = _mock_connect([])
    result = curr_max.invoke({"lift": "unknown_lift", "n": 5})
    assert result is None


# --- list_lifts tool ---

@patch("liftaudit.coaching.agent.connect")
def test_list_lifts(mock_connect):
    conn = _mock_connect([("bench_press",), ("deadlift",), ("squat",)])
    mock_connect.return_value = conn
    result = list_lifts.invoke({})
    assert result == ["bench_press", "deadlift", "squat"]


# --- max_trend tool ---

@patch("liftaudit.coaching.agent.connect")
def test_max_trend_positive_slope(mock_connect):
    mock_connect.return_value = _mock_connect(MOCK_ROWS)
    result = max_trend.invoke({"lift": "bench_press", "start_date": date(2026, 3, 1)})
    assert isinstance(result, TrendResult)
    assert result.slope_per_week > 0
    assert result.n == 5


# --- get_coaching_agent wires up without error ---

def test_get_coaching_agent_compiles():
    with patch("liftaudit.coaching.agent.ChatOpenAI"):
        agent = get_coaching_agent()
    assert agent is not None
