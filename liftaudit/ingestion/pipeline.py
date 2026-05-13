from pathlib import Path
from typing import List

from liftaudit.ingestion.exercise_resolver import ExerciseResolver
from liftaudit.ingestion.generic_csv import read_generic_csv
from liftaudit.ingestion.schemas import IngestionResult, IngestionWarning, WorkoutSet
from liftaudit.ingestion.validators import (
    should_skip_row,
    validate_parseable_values,
    validate_required_fields,
    validate_suspicious_entries,
)


def ingest_csv(path: str | Path, use_llm_fallback: bool = True) -> IngestionResult:
    rows = read_generic_csv(path, use_llm_fallback=use_llm_fallback)
    resolver = ExerciseResolver()
    workout_sets: List[WorkoutSet] = []
    warnings: List[IngestionWarning] = []
    unresolved = []

    for index, row in enumerate(rows, start=2):
        row_warnings = validate_required_fields(row, index)
        row_warnings.extend(validate_parseable_values(row, index))

        resolution = resolver.resolve(row.get("exercise", ""))
        if resolution.status == "unresolved":
            unresolved.append(resolution)

        row_warnings.extend(validate_suspicious_entries(row, index, resolution))
        warnings.extend(row_warnings)

        if should_skip_row(row_warnings):
            continue

        workout_sets.append(
            WorkoutSet(
                date=(row.get("date") or "").strip() or None,
                exercise=row["exercise"].strip(),
                canonical_exercise=resolution.canonical_name,
                sets=int(row["sets"]),
                reps=int(row["reps"]),
                weight=float(row["weight"]),
                unit=row["unit"].strip(),
                rir=_optional_float(row.get("rir")),
                notes=(row.get("notes") or "").strip() or None,
                resolution=resolution,
            )
        )

    return IngestionResult(
        sets=workout_sets,
        warnings=warnings,
        unresolved_exercises=unresolved,
        rows_processed=len(rows),
    )


def _optional_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    return float(value)
