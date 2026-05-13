from pathlib import Path
from types import SimpleNamespace

from liftaudit.ingestion.pipeline import ingest_csv


ROOT = Path(__file__).resolve().parents[1]


def test_valid_csv_parses():
    result = ingest_csv(ROOT / "examples" / "sample_workout_log.csv")

    assert result.rows_processed == 3
    assert len(result.sets) == 3
    assert result.sets[0].exercise == "bench"
    assert result.sets[0].sets == 3
    assert result.sets[0].weight == 185
    assert result.warnings == []


def test_aliases_resolve():
    result = ingest_csv(ROOT / "examples" / "sample_workout_log.csv")

    bench = result.sets[0]
    squat = result.sets[1]

    assert bench.canonical_exercise == "bench_press"
    assert bench.resolution.status == "alias"
    assert squat.canonical_exercise == "squat"
    assert squat.resolution.status == "alias"


def test_unknown_exercises_are_unresolved():
    result = ingest_csv(ROOT / "examples" / "sample_bad_workout_log.csv")

    unresolved = [item.input_name for item in result.unresolved_exercises]

    assert "mystery press" in unresolved
    mystery_set = next(item for item in result.sets if item.exercise == "mystery press")
    assert mystery_set.canonical_exercise is None
    assert mystery_set.resolution.status == "unresolved"


def test_lateral_raise_200_creates_suggested_correction():
    result = ingest_csv(ROOT / "examples" / "sample_bad_workout_log.csv")

    warning = next(item for item in result.warnings if item.field == "weight")
    lateral_raise = result.sets[0]

    assert warning.suggested_correction == {"weight": 20.0}
    assert lateral_raise.weight == 200


def test_missing_date_is_allowed():
    result = ingest_csv(ROOT / "examples" / "sample_bad_workout_log.csv")

    bench = next(item for item in result.sets if item.exercise == "bench")

    assert bench.date is None
    assert not any(warning.field == "date" for warning in result.warnings)


def test_date_column_is_optional(tmp_path):
    csv_path = tmp_path / "workout_without_date.csv"
    csv_path.write_text(
        "exercise,sets,reps,weight,unit,rir,notes\n"
        "bench,3,5,185,lb,2,Felt strong\n",
        encoding="utf-8",
    )

    result = ingest_csv(csv_path)

    assert len(result.sets) == 1
    assert result.sets[0].date is None
    assert result.warnings == []


def test_csv_columns_with_typos_are_matched(tmp_path):
    csv_path = tmp_path / "workout_with_typo_headers.csv"
    csv_path.write_text(
        "exercize,sets,reps,weigt,unit,rir,notes\n"
        "bench,3,5,185,lb,2,Felt strong\n",
        encoding="utf-8",
    )

    result = ingest_csv(csv_path)

    assert len(result.sets) == 1
    assert result.sets[0].exercise == "bench"
    assert result.sets[0].weight == 185
    assert result.warnings == []


def test_llm_column_mapping_fallback(monkeypatch, tmp_path):
    csv_path = tmp_path / "workout_with_external_headers.csv"
    csv_path.write_text(
        "movement,set_count,rep_count,load,load_unit,effort,comment\n"
        "bench,3,5,185,lb,2,Felt strong\n",
        encoding="utf-8",
    )

    class FakeSuggestion:
        mappings = [
            SimpleNamespace(original_header="movement", canonical_column="exercise"),
            SimpleNamespace(original_header="set_count", canonical_column="sets"),
            SimpleNamespace(original_header="rep_count", canonical_column="reps"),
            SimpleNamespace(original_header="load", canonical_column="weight"),
            SimpleNamespace(original_header="load_unit", canonical_column="unit"),
            SimpleNamespace(original_header="effort", canonical_column="rir"),
            SimpleNamespace(original_header="comment", canonical_column="notes"),
        ]

    monkeypatch.setattr(
        "liftaudit.ingestion.generic_csv.suggest_header_mapping",
        lambda **_kwargs: FakeSuggestion(),
    )

    result = ingest_csv(csv_path, use_llm_fallback=True)

    assert len(result.sets) == 1
    assert result.sets[0].exercise == "bench"
    assert result.sets[0].weight == 185
