import sqlite3
from pathlib import Path


DEFAULT_DB_PATH = Path("data") / "liftaudit.db"


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    with connect(db_path) as connection:
        connection.executescript(SCHEMA_SQL)


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS ingestion_batches (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    source_path TEXT,
    rows_processed INTEGER NOT NULL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS workout_sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL,
    performed_on TEXT,
    raw_exercise_name TEXT NOT NULL,
    canonical_exercise TEXT,
    sets INTEGER NOT NULL,
    reps INTEGER NOT NULL,
    weight REAL NOT NULL,
    unit TEXT NOT NULL,
    rir REAL,
    notes TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id) REFERENCES ingestion_batches(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS ingestion_warnings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL,
    row_number INTEGER NOT NULL,
    field TEXT,
    message TEXT NOT NULL,
    suggested_correction_json TEXT,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id) REFERENCES ingestion_batches(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS unresolved_exercises (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    batch_id INTEGER NOT NULL,
    input_name TEXT NOT NULL,
    status TEXT NOT NULL,
    canonical_name TEXT,
    matched_alias TEXT,
    score REAL,
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (batch_id) REFERENCES ingestion_batches(id) ON DELETE CASCADE
);
"""

