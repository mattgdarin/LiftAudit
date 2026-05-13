from pathlib import Path

from liftaudit.ingestion.image_ingestion.agent import extract_workout_csv_from_image
from liftaudit.ingestion.image_ingestion.csv_writer import (
    write_image_csv,
    write_normalized_csv,
)
from liftaudit.ingestion.image_ingestion.state import ImageIngestionState
from liftaudit.ingestion.pipeline import ingest_csv
from liftaudit.ingestion.schemas import IngestionResult


def image_to_csv(
    image_path: str | Path,
    output_csv_path: str | Path,
    model: str | None = None,
) -> Path:
    state = extract_workout_csv_from_image(ImageIngestionState(image_path=Path(image_path)))
    if state.extraction is None:
        raise ValueError("Image extraction did not produce any rows.")
    return write_image_csv(state.extraction, output_csv_path)


def ingest_image(
    image_path: str | Path,
    output_csv_path: str | Path,
    model: str | None = None,
    use_llm_fallback: bool = True,
) -> IngestionResult:
    csv_path = image_to_csv(image_path, output_csv_path, model=model)
    return ingest_csv(csv_path, use_llm_fallback=use_llm_fallback)


def image_to_normalized_csv(
    image_path: str | Path,
    draft_csv_path: str | Path,
    normalized_csv_path: str | Path,
    model: str | None = None,
    use_llm_fallback: bool = True,
) -> ImageIngestionState:
    state = ImageIngestionState(
        image_path=Path(image_path),
        draft_csv_path=Path(draft_csv_path),
    )
    state = extract_workout_csv_from_image(state)
    if state.extraction is None:
        raise ValueError("Image extraction did not produce any rows.")

    write_image_csv(state.extraction, draft_csv_path)
    state.ingestion_result = ingest_csv(draft_csv_path, use_llm_fallback=use_llm_fallback)
    state.approved_csv_path = write_normalized_csv(
        state.ingestion_result,
        normalized_csv_path,
    )
    return state
