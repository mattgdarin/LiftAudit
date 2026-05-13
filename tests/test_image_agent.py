from pathlib import Path

import pytest


from liftaudit.ingestion.image_ingestion import agent
from liftaudit.ingestion.image_ingestion.schemas import ImageResponse, ImageWorkoutRow
from liftaudit.ingestion.image_ingestion.state import ImageIngestionState


ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_IMAGE = ROOT / "examples" / "9db610b3-9fcd-4763-b355-636fea9bb3cb.png"


def test_extract_workout_csv_from_image_updates_state(monkeypatch):
    calls = []

    class FakeStructuredLlm:
        def invoke(self, messages):
            calls.append(messages)
            return ImageResponse(
                rows=[
                    ImageWorkoutRow(
                        date="",
                        exercise="bench",
                        sets="3",
                        reps="5",
                        weight="185",
                        unit="lb",
                        rir="2",
                        notes="",
                    )
                ],
                needs_review=False,
                review_notes=[],
                source_description="mock workout screenshot",
            )

    class FakeChatOpenAI:
        def __init__(self, **_kwargs):
            pass

        def with_structured_output(self, schema):
            assert schema is ImageResponse
            return FakeStructuredLlm()

    monkeypatch.setattr(agent, "ChatOpenAI", FakeChatOpenAI)

    state = ImageIngestionState(image_path=EXAMPLE_IMAGE)
    result = agent.extract_workout_csv_from_image(state)

    assert result.extraction is not None
    assert result.extraction.rows[0].exercise == "bench"
    assert result.needs_review is False
    assert calls[0][1]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_extract_workout_csv_from_image_includes_human_review_context(monkeypatch):
    prompts = []
    previous_extraction = ImageResponse(
        rows=[
            ImageWorkoutRow(
                date="",
                exercise="unclear press",
                sets="5",
                reps="3",
                weight="185",
                unit="lb",
                rir="",
                notes="",
            )
        ],
        needs_review=True,
        review_notes=["Could not tell whether 5x3 means sets x reps or reps x sets."],
        source_description="mock workout screenshot",
    )

    class FakeStructuredLlm:
        def invoke(self, messages):
            prompts.append(messages[1]["content"][0]["text"])
            return ImageResponse(
                rows=previous_extraction.rows,
                needs_review=False,
                review_notes=[],
                source_description="mock workout screenshot",
            )

    class FakeChatOpenAI:
        def __init__(self, **_kwargs):
            pass

        def with_structured_output(self, _schema):
            return FakeStructuredLlm()

    monkeypatch.setattr(agent, "ChatOpenAI", FakeChatOpenAI)

    state = ImageIngestionState(
        image_path=EXAMPLE_IMAGE,
        extraction=previous_extraction,
        review_notes=previous_extraction.review_notes,
        human_notes=["5x3 means 5 sets of 3 reps."],
    )
    agent.extract_workout_csv_from_image(state)

    prompt = prompts[0]
    assert "5x3 means 5 sets of 3 reps." in prompt
    assert "Previous issues of concern" in prompt
    assert "Could not tell whether 5x3" in prompt
    assert "Previous extracted rows" in prompt


def test_add_human_feedback_appends_note_and_attempt(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda _prompt: "The first number is sets.")

    state = ImageIngestionState(
        image_path=EXAMPLE_IMAGE,
        review_notes=["Ambiguous set and rep notation."],
    )
    result = agent.add_human_feedback(state)

    assert result.review_attempts == 1
    assert result.human_notes == ["The first number is sets."]
    assert result.errors == []


def test_real_extract_workout_csv_from_example_image():
    state = ImageIngestionState(image_path=EXAMPLE_IMAGE)

    result = agent.extract_workout_csv_from_image(state)

    print(result.extraction)
    assert result.extraction is not None
    assert result.extraction.source_description
    assert result.extraction.rows
    assert any("bench" in row.exercise.lower() for row in result.extraction.rows)
