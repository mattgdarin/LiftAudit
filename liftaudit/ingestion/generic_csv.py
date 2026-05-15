import csv
from pathlib import Path
from typing import Dict, Iterable, List

from rapidfuzz import fuzz

from liftaudit.ingestion.llm_column_mapper import suggest_header_mapping


FUZZY_HEADER_THRESHOLD = 90
REQUIRED_CSV_COLUMNS = ["exercise", "sets", "reps", "weight", "unit", "rir", "notes"]
OPTIONAL_CSV_COLUMNS = ["date"]
CSV_COLUMNS = OPTIONAL_CSV_COLUMNS + REQUIRED_CSV_COLUMNS


def read_generic_csv(path: str | Path, use_llm_fallback: bool = True) -> List[Dict[str, str]]:
    csv_path = Path(path)
    with csv_path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            return []
        reader.fieldnames = _map_headers(reader.fieldnames, use_llm_fallback)
        return [{key: (row.get(key) or "") for key in CSV_COLUMNS} for row in reader]


def _map_headers(fieldnames: Iterable[str], use_llm_fallback: bool = True) -> List[str]:
    normalized = [field.strip() for field in fieldnames]
    mapped = _map_headers_deterministically(normalized)

    missing = [column for column in REQUIRED_CSV_COLUMNS if column not in mapped]
    if not missing:
        return mapped

    if use_llm_fallback:
        print("Using LLM")
        mapped = _map_headers_with_llm(normalized, mapped)
        missing = [column for column in REQUIRED_CSV_COLUMNS if column not in mapped]

    if missing:
        raise ValueError(f"CSV is missing columns: {', '.join(missing)}")

    return mapped


def _map_headers_deterministically(headers: List[str]) -> List[str]:
    mapped = headers[:]

    for column in CSV_COLUMNS:
        if column in mapped:
            continue

        match = _best_header_match(column, mapped)
        if match is not None:
            index, _score = match
            mapped[index] = column

    return mapped


def _best_header_match(column: str, headers: List[str]) -> tuple[int, float] | None:
    candidates = [
        (index, fuzz.token_sort_ratio(column, header))
        for index, header in enumerate(headers)
        if header not in CSV_COLUMNS
    ]
    if not candidates:
        return None

    index, score = max(candidates, key=lambda candidate: candidate[1])
    if score < FUZZY_HEADER_THRESHOLD:
        return None
    return index, score


def _map_headers_with_llm(original_headers: List[str], current_mapping: List[str]) -> List[str]:
    suggestion = suggest_header_mapping(
        headers=original_headers,
        required_columns=REQUIRED_CSV_COLUMNS,
        optional_columns=OPTIONAL_CSV_COLUMNS,
    )
    allowed_columns = set(CSV_COLUMNS)
    mapped = current_mapping[:]
    suggestions_by_header = {
        item.original_header: item.canonical_column for item in suggestion.mappings
    }

    for index, original_header in enumerate(original_headers):
        suggested_column = suggestions_by_header.get(original_header)
        if suggested_column not in allowed_columns:
            continue
        if suggested_column in mapped and mapped[index] != suggested_column:
            continue
        mapped[index] = suggested_column

    return mapped
