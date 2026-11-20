"""Thread-safety of the SQLite backend's thread-local connection cache.

Hermetic: a temp DB, fake embeddings (no Ollama), seeded without Docling. Proves
each thread gets its own connection, concurrent reads are safe, and reads+writes
overlap without 'database is locked' (WAL + busy_timeout).
"""

import threading

import numpy as np
import pytest
from sqlite_vec import serialize_float32

from assistant.retrieval import sqlite as sq

DIM = sq.EMBED_DIM


def _unit_vec(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i % DIM] = 1.0
    return v


def _seed(conn, n: int = 6) -> None:
    for i in range(n):
        cur = conn.execute(
            "INSERT INTO chunks(chunk_id, source, text) VALUES (?,?,?)",
            (f"c{i}", f"file{i % 2}.txt", f"chunk {i} about vacation and dentist"),
        )
        rid = cur.lastrowid
        conn.execute(
            "INSERT INTO chunks_fts(rowid, chunk_id, text) VALUES (?,?,?)",
            (rid, f"c{i}", f"chunk {i} about vacation and dentist"),
        )
        conn.execute(
            "INSERT INTO vec_chunks(rowid, embedding) VALUES (?,?)",
            (rid, serialize_float32(_unit_vec(i))),
        )
    conn.commit()


@pytest.fixture
def backend(tmp_path, monkeypatch):
    # isolate: temp DB file + a fresh thread-local store so nothing touches data/rag.db
    monkeypatch.setattr(sq, "DB_PATH", tmp_path / "rag.db")
    monkeypatch.setattr(sq, "_local", threading.local())
    # fake embed so hybrid_retrieve never calls Ollama
    from assistant import rag

    qv = np.array(_unit_vec(0), dtype="float32")
    monkeypatch.setattr(
        rag, "embed", lambda texts, prefix: np.array([qv for _ in texts])
    )
    _seed(sq._get_conn())
    return sq.SQLiteBackend()


def test_thread_local_gives_each_thread_its_own_connection(backend):
    seen: dict[int, int] = {}

    def grab():
        seen[threading.get_ident()] = id(sq._get_conn())

    threads = [threading.Thread(target=grab) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # distinct connection object per thread...
    assert len(set(seen.values())) == 5
    # ...and the same connection is reused within one thread
    assert id(sq._get_conn()) == id(sq._get_conn())


def test_concurrent_retrieval_is_safe(backend):
    errors: list[str] = []
    counts: list[int] = []

    def worker():
        try:
            for _ in range(25):
                counts.append(len(backend.hybrid_retrieve("vacation dentist", 5)))
        except Exception as e:  # noqa: BLE001 - any threading error should fail the test
            errors.append(repr(e))

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors
    assert counts and all(c > 0 for c in counts)


def test_reads_and_writes_overlap_without_lock_errors(backend):
    errors: list[str] = []
    stop = threading.Event()

    def writer():
        i = 100
        try:
            while not stop.is_set():
                conn = sq._get_conn()
                cur = conn.execute(
                    "INSERT OR IGNORE INTO chunks(chunk_id, source, text) VALUES (?,?,?)",
                    (f"w{i}", "w.txt", f"written {i} vacation"),
                )
                if cur.rowcount == 1:
                    rid = cur.lastrowid
                    conn.execute(
                        "INSERT INTO chunks_fts(rowid, chunk_id, text) VALUES (?,?,?)",
                        (rid, f"w{i}", f"written {i} vacation"),
                    )
                    conn.execute(
                        "INSERT INTO vec_chunks(rowid, embedding) VALUES (?,?)",
                        (rid, serialize_float32(_unit_vec(1))),
                    )
                conn.commit()
                i += 1
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    def reader():
        try:
            for _ in range(50):
                backend.hybrid_retrieve("vacation", 5)
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    w = threading.Thread(target=writer)
    readers = [threading.Thread(target=reader) for _ in range(4)]
    w.start()
    for r in readers:
        r.start()
    for r in readers:
        r.join()
    stop.set()
    w.join()

    assert not errors, errors
