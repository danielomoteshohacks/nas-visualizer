# database.py
#
# SQLite database layer — stores every architecture evaluation result.
#
# SQLite is a file-based database built into Python's standard library.
# No server, no installation, no configuration. It's a single .db file
# that lives in your data/ folder and grows as you run experiments.
#
# Schema:
#   architectures table — one row per evaluated architecture
#     - arch_index:   unique integer ID (0-15624)
#     - arch_string:  NAS-Bench-201 format string
#     - operations:   JSON list of 6 operation names
#     - grad_norm:    zero-cost proxy score
#     - synflow:      zero-cost proxy score
#     - jacob_cov:    zero-cost proxy score
#     - proxy_score:  normalized ensemble of the three proxies
#     - val_accuracy: real validation accuracy after short training (if run)
#     - train_loss:   final training loss after short training (if run)
#     - epochs_trained: how many epochs this arch was trained for
#     - params:       number of trainable parameters
#     - flops:        estimated FLOPs
#     - created_at:   timestamp

import sqlite3
import json
import time
from pathlib import Path


DB_PATH = Path(__file__).parent.parent.parent / 'data' / 'nas_bench.db'


def get_connection() -> sqlite3.Connection:
    """
    Open a connection to the SQLite database.
    Creates the file if it doesn't exist yet.
    """
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    # Return rows as dict-like objects so we can access columns by name
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Create the database tables if they don't exist.
    Safe to call multiple times — IF NOT EXISTS prevents duplication.
    """
    conn = get_connection()
    conn.execute('''
        CREATE TABLE IF NOT EXISTS architectures (
            arch_index      INTEGER PRIMARY KEY,
            arch_string     TEXT NOT NULL,
            operations      TEXT NOT NULL,
            grad_norm       REAL,
            synflow         REAL,
            jacob_cov       REAL,
            proxy_score     REAL,
            val_accuracy    REAL,
            train_loss      REAL,
            epochs_trained  INTEGER DEFAULT 0,
            params          INTEGER,
            flops           REAL,
            created_at      REAL DEFAULT (strftime('%s','now'))
        )
    ''')
    # Index on proxy_score so sorting by score is fast
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_proxy_score
        ON architectures(proxy_score DESC)
    ''')
    # Index on val_accuracy for ranking by real accuracy
    conn.execute('''
        CREATE INDEX IF NOT EXISTS idx_val_accuracy
        ON architectures(val_accuracy DESC)
    ''')
    conn.commit()
    conn.close()


def upsert_proxy_scores(scores: dict):
    """
    Insert or update proxy scores for one architecture.
    'upsert' = insert if not exists, update if exists.
    """
    conn = get_connection()
    conn.execute('''
        INSERT INTO architectures
            (arch_index, arch_string, operations,
             grad_norm, synflow, jacob_cov, proxy_score)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(arch_index) DO UPDATE SET
            grad_norm    = excluded.grad_norm,
            synflow      = excluded.synflow,
            jacob_cov    = excluded.jacob_cov,
            proxy_score  = excluded.proxy_score
    ''', (
        scores['arch_index'],
        scores['arch_string'],
        json.dumps(scores['operations']),
        scores.get('grad_norm', 0.0),
        scores.get('synflow', 0.0),
        scores.get('jacob_cov', 0.0),
        scores.get('proxy_score', 0.0),
    ))
    conn.commit()
    conn.close()


def update_training_results(arch_index: int, val_accuracy: float,
                             train_loss: float, epochs: int,
                             params: int, flops: float):
    """
    Update an architecture's record with real training results.
    Called after short training runs complete.
    """
    conn = get_connection()
    conn.execute('''
        UPDATE architectures
        SET val_accuracy   = ?,
            train_loss     = ?,
            epochs_trained = ?,
            params         = ?,
            flops          = ?
        WHERE arch_index = ?
    ''', (val_accuracy, train_loss, epochs, params, flops, arch_index))
    conn.commit()
    conn.close()


def get_top_by_proxy(limit: int = 200) -> list:
    """
    Return the top architectures ranked by their proxy score.
    These are the candidates sent to real training.
    """
    conn = get_connection()
    rows = conn.execute('''
        SELECT * FROM architectures
        WHERE proxy_score IS NOT NULL
        ORDER BY proxy_score DESC
        LIMIT ?
    ''', (limit,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_top_by_accuracy(limit: int = 10) -> list:
    """
    Return top architectures ranked by real validation accuracy.
    Only returns architectures that have been trained.
    """
    conn = get_connection()
    rows = conn.execute('''
        SELECT * FROM architectures
        WHERE val_accuracy IS NOT NULL
        ORDER BY val_accuracy DESC
        LIMIT ?
    ''', (limit,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_arch(arch_index: int) -> dict:
    """Fetch a single architecture record by index."""
    conn = get_connection()
    row = conn.execute(
        'SELECT * FROM architectures WHERE arch_index = ?',
        (arch_index,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_all_scored() -> list:
    """Return all architectures that have proxy scores."""
    conn = get_connection()
    rows = conn.execute('''
        SELECT arch_index, arch_string, operations,
               proxy_score, val_accuracy, epochs_trained
        FROM architectures
        WHERE proxy_score IS NOT NULL
        ORDER BY proxy_score DESC
    ''').fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_stats() -> dict:
    """Return summary statistics about the current database state."""
    conn = get_connection()
    row = conn.execute('''
        SELECT
            COUNT(*)                                    AS total_scored,
            COUNT(CASE WHEN val_accuracy IS NOT NULL
                       THEN 1 END)                      AS total_trained,
            MAX(val_accuracy)                           AS best_accuracy,
            AVG(CASE WHEN val_accuracy IS NOT NULL
                     THEN val_accuracy END)             AS avg_accuracy,
            MAX(proxy_score)                            AS best_proxy_score
        FROM architectures
    ''').fetchone()
    conn.close()
    return dict(row)


def normalize_proxy_scores():
    """
    After scoring a batch of architectures, normalize each proxy to [0, 1]
    and compute the ensemble score as their average.

    Normalization is critical because the three proxies live on completely
    different scales. Without this, synflow (which can be millions) would
    completely dominate grad_norm (which is usually < 10).
    """
    conn = get_connection()

    # Get min/max for each proxy across all scored architectures
    stats = conn.execute('''
        SELECT
            MIN(grad_norm)  AS gn_min,  MAX(grad_norm)  AS gn_max,
            MIN(synflow)    AS sf_min,  MAX(synflow)    AS sf_max,
            MIN(jacob_cov)  AS jc_min,  MAX(jacob_cov)  AS jc_max
        FROM architectures
        WHERE grad_norm IS NOT NULL
    ''').fetchone()

    if not stats or stats['gn_max'] == stats['gn_min']:
        conn.close()
        return

    gn_min, gn_max = stats['gn_min'], stats['gn_max']
    sf_min, sf_max = stats['sf_min'], stats['sf_max']
    jc_min, jc_max = stats['jc_min'], stats['jc_max']

    # Avoid division by zero
    gn_range = max(gn_max - gn_min, 1e-8)
    sf_range = max(sf_max - sf_min, 1e-8)
    jc_range = max(jc_max - jc_min, 1e-8)

    # Update all rows with normalized ensemble score
    conn.execute('''
        UPDATE architectures
        SET proxy_score = (
            ((grad_norm - ?) / ?) +
            ((synflow   - ?) / ?) +
            ((jacob_cov - ?) / ?)
        ) / 3.0
        WHERE grad_norm IS NOT NULL
    ''', (gn_min, gn_range, sf_min, sf_range, jc_min, jc_range))

    conn.commit()
    conn.close()