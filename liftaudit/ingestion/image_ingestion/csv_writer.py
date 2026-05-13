import csv
from pathlib import Path

from liftaudit.ingestion.generic_csv import CSV_COLUMNS
from liftaudit.ingestion.image_ingestion.schemas import ImageWorkoutCsv
from liftaudit.ingestion.schemas import IngestionResult


def write_image_csv(extraction: ImageWorkoutCsv, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in extraction.rows:
            writer.writerow(row.model_dump())

    return path


def write_normalized_csv(result: IngestionResult, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for workout_set in result.sets:
            writer.writerow(
                {
                    "date": workout_set.date.isoformat() if workout_set.date else "",
                    "exercise": workout_set.canonical_exercise or workout_set.exercise,
                    "sets": workout_set.sets,
                    "reps": workout_set.reps,
                    "weight": workout_set.weight,
                    "unit": workout_set.unit,
                    "rir": workout_set.rir if workout_set.rir is not None else "",
                    "notes": workout_set.notes or "",
                }
            )

    return path
