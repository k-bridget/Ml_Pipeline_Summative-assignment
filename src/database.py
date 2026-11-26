import hashlib
import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Iterable, Optional


DEFAULT_DATABASE_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "retrain_uploads",
    "phishiris.db",
)


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def md5_file(file_path: str) -> str:
    digest = hashlib.md5()
    with open(file_path, "rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def get_database_path(upload_dir: Optional[str] = None) -> str:
    if upload_dir:
        return os.environ.get(
            "PHISHIRIS_DATABASE_PATH",
            os.path.join(upload_dir, "phishiris.db"),
        )
    return os.environ.get("PHISHIRIS_DATABASE_PATH", DEFAULT_DATABASE_PATH)


def initialize_database(database_path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(database_path)), exist_ok=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS retrain_upload_batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                class_label TEXT NOT NULL,
                numeric_label INTEGER NOT NULL,
                file_count INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS retrain_upload_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                file_name TEXT NOT NULL,
                file_path TEXT NOT NULL,
                class_label TEXT NOT NULL,
                md5_hash TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                FOREIGN KEY (batch_id) REFERENCES retrain_upload_batches(id)
            )
            """
        )
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS retraining_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                epochs INTEGER NOT NULL,
                learning_rate REAL NOT NULL,
                fine_tune_layers INTEGER NOT NULL,
                unique_samples_used INTEGER,
                model_path TEXT,
                backup_path TEXT,
                metrics_json TEXT,
                error_message TEXT
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_upload_files_batch ON retrain_upload_files(batch_id)"
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_upload_files_hash ON retrain_upload_files(md5_hash)"
        )


def record_retrain_upload(
    stored_files: Iterable[str],
    class_label: str,
    numeric_label: int,
    database_path: str,
) -> dict:
    initialize_database(database_path)
    timestamp = utc_timestamp()
    files = [path for path in stored_files if os.path.isfile(path)]

    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO retrain_upload_batches
                (timestamp, class_label, numeric_label, file_count)
            VALUES (?, ?, ?, ?)
            """,
            (timestamp, class_label, numeric_label, len(files)),
        )
        batch_id = cursor.lastrowid
        for file_path in files:
            connection.execute(
                """
                INSERT INTO retrain_upload_files
                    (batch_id, timestamp, file_name, file_path, class_label, md5_hash, file_size_bytes)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch_id,
                    timestamp,
                    os.path.basename(file_path),
                    os.path.abspath(file_path),
                    class_label,
                    md5_file(file_path),
                    os.path.getsize(file_path),
                ),
            )

    return {
        "timestamp": timestamp,
        "batch_id": batch_id,
        "file_count": len(files),
    }


def record_retraining_run(
    database_path: str,
    *,
    status: str,
    epochs: int,
    learning_rate: float,
    fine_tune_layers: int,
    unique_samples_used: Optional[int] = None,
    model_path: Optional[str] = None,
    backup_path: Optional[str] = None,
    metrics: Optional[dict] = None,
    error_message: Optional[str] = None,
) -> int:
    initialize_database(database_path)
    with sqlite3.connect(database_path) as connection:
        cursor = connection.execute(
            """
            INSERT INTO retraining_runs
                (timestamp, status, epochs, learning_rate, fine_tune_layers,
                 unique_samples_used, model_path, backup_path, metrics_json, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                utc_timestamp(),
                status,
                epochs,
                learning_rate,
                fine_tune_layers,
                unique_samples_used,
                model_path,
                backup_path,
                json.dumps(metrics or {}),
                error_message,
            ),
        )
        return int(cursor.lastrowid)


def get_database_status(database_path: str) -> dict:
    initialize_database(database_path)
    with sqlite3.connect(database_path) as connection:
        connection.row_factory = sqlite3.Row

        upload_counts = {
            row["class_label"]: row["file_count"]
            for row in connection.execute(
                """
                SELECT class_label, COUNT(*) AS file_count
                FROM retrain_upload_files
                GROUP BY class_label
                """
            )
        }
        batch_count = connection.execute(
            "SELECT COUNT(*) AS count FROM retrain_upload_batches"
        ).fetchone()["count"]
        file_count = connection.execute(
            "SELECT COUNT(*) AS count FROM retrain_upload_files"
        ).fetchone()["count"]

        recent_batches = [
            dict(row)
            for row in connection.execute(
                """
                SELECT id, timestamp, class_label, numeric_label, file_count
                FROM retrain_upload_batches
                ORDER BY id DESC
                LIMIT 10
                """
            )
        ]
        recent_runs = [
            dict(row)
            for row in connection.execute(
                """
                SELECT id, timestamp, status, epochs, learning_rate, fine_tune_layers,
                       unique_samples_used, model_path, backup_path, metrics_json, error_message
                FROM retraining_runs
                ORDER BY id DESC
                LIMIT 10
                """
            )
        ]

    for run in recent_runs:
        run["metrics"] = json.loads(run.pop("metrics_json") or "{}")

    return {
        "database_path": os.path.abspath(database_path),
        "batch_count": batch_count,
        "file_count": file_count,
        "upload_counts_by_class": upload_counts,
        "recent_batches": recent_batches,
        "recent_runs": recent_runs,
    }
