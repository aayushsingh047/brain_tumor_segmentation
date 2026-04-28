"""
database.py
-----------
SQLite persistence layer for Brain Tumor Segmentation reports.

Provides:
    init_db()           — Create tables on first run (idempotent)
    save_report(...)    — Insert a new report row
    get_reports()       — Return all rows as a list of dicts
    get_report_by_id()  — Single-row lookup
    delete_report()     — Remove a row by id
    get_stats()         — Aggregate statistics across all reports

The database file is created next to this module (../../database.db relative
to this file, i.e. in the project root).  You can override the path by setting
the environment variable  BRAIN_TUMOR_DB_PATH.
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional

# ── Locate database file ──────────────────────────────────────────────────────
_DEFAULT_DB_PATH = Path(__file__).parent.parent / "database.db"
DB_PATH: Path = Path(os.getenv("BRAIN_TUMOR_DB_PATH", str(_DEFAULT_DB_PATH)))


# ── Internal helpers ──────────────────────────────────────────────────────────

def _get_connection() -> sqlite3.Connection:
    """Return an open connection with row_factory set to dict-like Row."""
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row          # lets us access columns by name
    conn.execute("PRAGMA journal_mode=WAL") # better concurrency
    return conn


# ── Public API ────────────────────────────────────────────────────────────────

def init_db() -> None:
    """
    Create the `reports` table if it does not already exist.
    Safe to call on every app start-up — will not drop existing data.
    """
    sql = """
    CREATE TABLE IF NOT EXISTS reports (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_id       TEXT    NOT NULL,
        tumor_volume     REAL    NOT NULL DEFAULT 0.0,
        tumor_percentage REAL    NOT NULL DEFAULT 0.0,
        inference_time   REAL    NOT NULL DEFAULT 0.0,
        surface_area     REAL    NOT NULL DEFAULT 0.0,
        necrotic_vol     REAL    NOT NULL DEFAULT 0.0,
        edema_vol        REAL    NOT NULL DEFAULT 0.0,
        enhancing_vol    REAL    NOT NULL DEFAULT 0.0,
        model_type       TEXT    NOT NULL DEFAULT 'Unknown',
        tumor_detected   INTEGER NOT NULL DEFAULT 0,
        report_path      TEXT,
        notes            TEXT,
        created_at       TEXT    NOT NULL
    );
    """
    with _get_connection() as conn:
        conn.execute(sql)
        conn.commit()


def save_report(
    patient_id: str,
    tumor_volume: float,
    tumor_percentage: float,
    inference_time: float,
    surface_area: float = 0.0,
    necrotic_vol: float = 0.0,
    edema_vol: float = 0.0,
    enhancing_vol: float = 0.0,
    model_type: str = "3D U-Net",
    tumor_detected: bool = False,
    report_path: Optional[str] = None,
    notes: Optional[str] = None,
) -> int:
    """
    Insert a new report row and return the new row's id.

    Parameters
    ----------
    patient_id       : Patient identifier string (e.g. "PATIENT_001")
    tumor_volume     : Total tumour volume in cm³
    tumor_percentage : Tumour coverage as % of scan volume
    inference_time   : Model inference duration in seconds
    surface_area     : Estimated surface area in mm²
    necrotic_vol     : Necrotic core volume in cm³
    edema_vol        : Edema sub-region volume in cm³
    enhancing_vol    : Enhancing tumour volume in cm³
    model_type       : Architecture label (e.g. "3D U-Net")
    tumor_detected   : True if any tumour voxels were found
    report_path      : Absolute path to the saved .txt report file (optional)
    notes            : Free-text notes (optional)

    Returns
    -------
    int  The auto-incremented row id of the inserted record.
    """
    init_db()   # ensure table exists before writing

    sql = """
    INSERT INTO reports
        (patient_id, tumor_volume, tumor_percentage, inference_time,
         surface_area, necrotic_vol, edema_vol, enhancing_vol,
         model_type, tumor_detected, report_path, notes, created_at)
    VALUES
        (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    params = (
        patient_id,
        round(float(tumor_volume), 4),
        round(float(tumor_percentage), 4),
        round(float(inference_time), 4),
        round(float(surface_area), 2),
        round(float(necrotic_vol), 4),
        round(float(edema_vol), 4),
        round(float(enhancing_vol), 4),
        str(model_type),
        int(bool(tumor_detected)),
        report_path,
        notes,
        created_at,
    )
    with _get_connection() as conn:
        cursor = conn.execute(sql, params)
        conn.commit()
        return cursor.lastrowid


def get_reports(limit: int = 500) -> List[Dict[str, Any]]:
    """
    Return all report rows ordered by most-recent first.

    Parameters
    ----------
    limit : Maximum number of rows to return (default 500).

    Returns
    -------
    list of dicts, one per row.  Empty list if no records exist.
    """
    init_db()
    sql = "SELECT * FROM reports ORDER BY id DESC LIMIT ?"
    with _get_connection() as conn:
        rows = conn.execute(sql, (limit,)).fetchall()
    return [dict(row) for row in rows]


def get_report_by_id(report_id: int) -> Optional[Dict[str, Any]]:
    """
    Return a single report row as a dict, or None if not found.
    """
    init_db()
    sql = "SELECT * FROM reports WHERE id = ?"
    with _get_connection() as conn:
        row = conn.execute(sql, (report_id,)).fetchone()
    return dict(row) if row else None


def delete_report(report_id: int) -> bool:
    """
    Delete the report with the given id.

    Returns True if a row was deleted, False if id was not found.
    """
    init_db()
    sql = "DELETE FROM reports WHERE id = ?"
    with _get_connection() as conn:
        cursor = conn.execute(sql, (report_id,))
        conn.commit()
        return cursor.rowcount > 0


def get_stats() -> Dict[str, Any]:
    """
    Return aggregate statistics across all report rows.

    Keys
    ----
    total_reports       int
    tumor_detected_count int
    avg_tumor_volume    float
    avg_tumor_percentage float
    avg_inference_time  float
    max_tumor_volume    float
    min_tumor_volume    float
    """
    init_db()
    sql = """
    SELECT
        COUNT(*)                  AS total_reports,
        SUM(tumor_detected)       AS tumor_detected_count,
        AVG(tumor_volume)         AS avg_tumor_volume,
        AVG(tumor_percentage)     AS avg_tumor_percentage,
        AVG(inference_time)       AS avg_inference_time,
        MAX(tumor_volume)         AS max_tumor_volume,
        MIN(CASE WHEN tumor_volume > 0 THEN tumor_volume END) AS min_tumor_volume
    FROM reports
    """
    with _get_connection() as conn:
        row = conn.execute(sql).fetchone()

    if row is None or row["total_reports"] == 0:
        return {
            "total_reports": 0,
            "tumor_detected_count": 0,
            "avg_tumor_volume": 0.0,
            "avg_tumor_percentage": 0.0,
            "avg_inference_time": 0.0,
            "max_tumor_volume": 0.0,
            "min_tumor_volume": 0.0,
        }

    return {
        "total_reports": int(row["total_reports"] or 0),
        "tumor_detected_count": int(row["tumor_detected_count"] or 0),
        "avg_tumor_volume": round(float(row["avg_tumor_volume"] or 0), 3),
        "avg_tumor_percentage": round(float(row["avg_tumor_percentage"] or 0), 3),
        "avg_inference_time": round(float(row["avg_inference_time"] or 0), 3),
        "max_tumor_volume": round(float(row["max_tumor_volume"] or 0), 3),
        "min_tumor_volume": round(float(row["min_tumor_volume"] or 0), 3),
    }
