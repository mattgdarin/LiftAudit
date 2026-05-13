#!/usr/bin/env python3
"""Seed the database with synthetic workout data."""
from pathlib import Path

from liftaudit.dbstore.repository import save_ingestion_result
from liftaudit.ingestion.pipeline import ingest_csv

CSV_PATH = Path(__file__).parent.parent / "examples" / "synthetic_log.csv"


def main():
    print(f"Ingesting {CSV_PATH} ...")
    result = ingest_csv(CSV_PATH, use_llm_fallback=False)

    print(f"  Rows processed : {result.rows_processed}")
    print(f"  Sets loaded    : {len(result.sets)}")
    print(f"  Warnings       : {len(result.warnings)}")
    print(f"  Unresolved     : {len(result.unresolved_exercises)}")

    if result.warnings:
        for w in result.warnings:
            print(f"    [row {w.row_number}] {w.field}: {w.message}")

    batch_id = save_ingestion_result(result, source_type="csv", source_path=str(CSV_PATH))
    print(f"Saved as batch_id={batch_id}")


if __name__ == "__main__":
    main()
