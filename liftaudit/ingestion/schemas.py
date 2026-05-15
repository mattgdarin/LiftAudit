from datetime import date as Date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class ExerciseResolution(BaseModel):
    input_name: str
    canonical_name: Optional[str] = None
    status: str
    matched_alias: Optional[str] = None
    score: Optional[float] = None
    target_muscles: List[str] = []
    secondary_muscles: List[str] = []
    body_parts: List[str] = []


class IngestionWarning(BaseModel):
    row_number: int
    field: Optional[str] = None
    message: str
    suggested_correction: Optional[Dict[str, Any]] = None


class WorkoutSet(BaseModel):
    date: Optional[Date] = None
    exercise: str
    canonical_exercise: Optional[str] = None
    sets: int
    reps: int
    weight: float
    unit: str
    rir: Optional[float] = None
    notes: Optional[str] = None
    resolution: ExerciseResolution


class IngestionResult(BaseModel):
    sets: List[WorkoutSet]
    warnings: List[IngestionWarning]
    unresolved_exercises: List[ExerciseResolution]
    rows_processed: int
