from typing import List

from pydantic import BaseModel, Field


class ImageWorkoutRow(BaseModel):
    date: str = Field(description="Workout date, or blank if not visible.")
    exercise: str = Field(description="Exercise name exactly as visible.")
    sets: str = Field(description="Number of sets, or blank if not visible.")
    reps: str = Field(description="Number of reps, or blank if not visible.")
    weight: str = Field(description="Weight/load, or blank if not visible.")
    unit: str = Field(description="Weight unit such as lb or kg, or blank if not visible.")
    rir: str = Field(description="RIR value, or blank if not visible.")
    notes: str = Field(description="Any visible notes, or blank if none.")


class ImageResponse(BaseModel):
    rows: List[ImageWorkoutRow] = Field(
        description="Workout rows extracted from the image in generic CSV shape."
    )
    needs_review: bool = Field(
        description="True when any extracted value is uncertain or hard to read."
    )
    review_notes: List[str] = Field(
        description="Short notes for the human reviewer about ambiguity or missing data."
    )
    source_description: str = Field(
        description="Brief description of the visible workout log format."
    )


ImageWorkoutCsv = ImageResponse
