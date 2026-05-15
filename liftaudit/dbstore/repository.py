import json
from pathlib import Path
from typing import Optional

from liftaudit.dbstore.connection import DEFAULT_DB_PATH, connect, initialize_database
from liftaudit.ingestion.schemas import IngestionResult


def save_ingestion_result(
    result: IngestionResult,
    db_path: str | Path = DEFAULT_DB_PATH,
    source_type: str = "csv",
    source_path: Optional[str | Path] = None,
) -> int:
    initialize_database(db_path)

    with connect(db_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO ingestion_batches (source_type, source_path, rows_processed)
            VALUES (?, ?, ?)
            """,
            (
                source_type,
                str(source_path) if source_path is not None else None,
                result.rows_processed,
            ),
        )
        batch_id = int(cursor.lastrowid)

        for workout_set in result.sets:
            connection.execute(
                """
                INSERT INTO workout_sets (
                    batch_id,
                    performed_on,
                    raw_exercise_name,
                    canonical_exercise,
                    sets,
                    reps,
                    weight,
                    unit,
                    rir,
                    notes,
                    target_muscles,
                    secondary_muscles
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    workout_set.date.isoformat() if workout_set.date else None,
                    workout_set.exercise,
                    workout_set.canonical_exercise,
                    workout_set.sets,
                    workout_set.reps,
                    workout_set.weight,
                    workout_set.unit,
                    workout_set.rir,
                    workout_set.notes,
                    json.dumps(workout_set.resolution.target_muscles) if workout_set.resolution.target_muscles else None,
                    json.dumps(workout_set.resolution.secondary_muscles) if workout_set.resolution.secondary_muscles else None,
                ),
            )

        for warning in result.warnings:
            connection.execute(
                """
                INSERT INTO ingestion_warnings (
                    batch_id,
                    row_number,
                    field,
                    message,
                    suggested_correction_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    warning.row_number,
                    warning.field,
                    warning.message,
                    json.dumps(warning.suggested_correction)
                    if warning.suggested_correction is not None
                    else None,
                ),
            )

        for exercise in result.unresolved_exercises:
            connection.execute(
                """
                INSERT INTO unresolved_exercises (
                    batch_id,
                    input_name,
                    status,
                    canonical_name,
                    matched_alias,
                    score
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    exercise.input_name,
                    exercise.status,
                    exercise.canonical_name,
                    exercise.matched_alias,
                    exercise.score,
                ),
            )

    return batch_id

