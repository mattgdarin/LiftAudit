from typing import Dict, List, Optional

from liftaudit.ingestion.schemas import ExerciseResolution, IngestionWarning


REQUIRED_FIELDS = ("exercise", "sets", "reps", "weight", "unit")
SUPPORTED_UNITS = {"lb", "lbs", "pound", "pounds", "kg", "kgs"}


def validate_required_fields(row: Dict[str, str], row_number: int) -> List[IngestionWarning]:
    warnings: List[IngestionWarning] = []
    for field in REQUIRED_FIELDS:
        if not _value(row.get(field)):
            warnings.append(
                IngestionWarning(
                    row_number=row_number,
                    field=field,
                    message=f"Missing required field: {field}",
                )
            )
    return warnings


def validate_parseable_values(row: Dict[str, str], row_number: int) -> List[IngestionWarning]:
    checks = {
        "sets": int,
        "reps": int,
        "weight": float,
        "rir": float,
    }
    warnings: List[IngestionWarning] = []
    for field, parser in checks.items():
        value = _value(row.get(field))
        if value is None:
            continue
        try:
            parser(value)
        except ValueError:
            warnings.append(
                IngestionWarning(
                    row_number=row_number,
                    field=field,
                    message=f"Could not parse {field}: {value}",
                )
            )
    return warnings


def validate_suspicious_entries(
    row: Dict[str, str],
    row_number: int,
    resolution: ExerciseResolution,
) -> List[IngestionWarning]:
    warnings: List[IngestionWarning] = []
    weight = _try_float(row.get("weight"))
    unit = (row.get("unit") or "").strip().lower()

    if (
        resolution.canonical_name == "lateral_raise"
        and unit in {"lb", "lbs", "pound", "pounds"}
        and weight is not None
        and weight >= 150
    ):
        warnings.append(
            IngestionWarning(
                row_number=row_number,
                field="weight",
                message="Suspicious lateral raise weight; keeping original value.",
                suggested_correction={"weight": weight / 10},
            )
        )

    return warnings


def should_skip_row(warnings: List[IngestionWarning]) -> bool:
    return any(
        warning.field in REQUIRED_FIELDS
        and (
            warning.message.startswith("Missing required field")
            or warning.message.startswith("Could not parse")
        )
        for warning in warnings
    )


def _value(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = str(value).strip()
    return stripped or None


def _try_float(value: Optional[str]) -> Optional[float]:
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None
