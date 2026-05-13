import sqlite3

from liftaudit.dbstore import initialize_database, save_ingestion_result
from liftaudit.ingestion.pipeline import ingest_csv


def test_save_ingestion_result_persists_sets_and_warnings(tmp_path):
    db_path = tmp_path / "liftaudit.db"
    result = ingest_csv("examples/sample_bad_workout_log.csv", use_llm_fallback=False)

    batch_id = save_ingestion_result(
        result,
        db_path=db_path,
        source_type="csv",
        source_path="examples/sample_bad_workout_log.csv",
    )

    with sqlite3.connect(db_path) as connection:
        workout_count = connection.execute("SELECT COUNT(*) FROM workout_sets").fetchone()[0]
        warning_count = connection.execute("SELECT COUNT(*) FROM ingestion_warnings").fetchone()[0]
        unresolved_count = connection.execute("SELECT COUNT(*) FROM unresolved_exercises").fetchone()[0]

    assert batch_id == 1
    assert workout_count == len(result.sets)
    assert warning_count == len(result.warnings)
    assert unresolved_count == len(result.unresolved_exercises)


def test_initialize_database_creates_db_file(tmp_path):
    db_path = tmp_path / "liftaudit.db"

    initialize_database(db_path)

    assert db_path.exists()
