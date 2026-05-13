from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field

from liftaudit.ingestion.image_ingestion.schemas import ImageResponse
from liftaudit.ingestion.schemas import IngestionResult


class ImageIngestionState(BaseModel):
    image_path: Path
    draft_csv_path: Optional[Path] = None
    approved_csv_path: Optional[Path] = None
    extraction: Optional[ImageResponse] = None
    ingestion_result: Optional[IngestionResult] = None
    needs_review: bool = True
    review_notes: List[str] = Field(default_factory=list)
    human_notes: List[str] = Field(default_factory=list)
    review_attempts: int = 0
    max_review_attempts: int = 2
    errors: List[str] = Field(default_factory=list)
