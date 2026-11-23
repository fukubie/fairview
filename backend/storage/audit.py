"""Tamper-evident decision audit log (SHA-256 hash chain, SQLite + JSONL).

Every recognition decision becomes one entry. Each entry carries the hash of
its predecessor and its own hash over the canonical JSON of all other fields:

    hash_n = SHA-256( canonical_json({...fields_n, "seq": n, "prev_hash": hash_{n-1}}) )

Editing, deleting or reordering any historical entry breaks every later hash.
The chain is written twice, to a SQLite table and to an append-only JSONL file,
and verification checks both chains and that they agree with each other, so an
attacker must rewrite two stores consistently.

Limit worth knowing: a hash chain without an external anchor cannot detect
truncation of the newest entries, nor a full rewrite by someone who can replace
both stores. ``head()`` exposes the latest hash so it can be anchored somewhere
the service cannot write (WORM storage, a ticket, another system's log).

One process owns the log at a time (an exclusive lock file enforces it): two
writers would race for the same sequence number and interleave the file.

Administrative actions (enrollment, erasure, renames, threshold changes) are
chained in with ``mode = "admin"``, as are repairs of the JSONL mirror, so a
template swap or a regenerated file cannot happen off the record.

No image data and no embeddings are ever written here. Subject labels are, and
they stay in the log after a subject is erased from the vault: the log's
integrity and a subject's erasure pull in opposite directions, so use
non-identifying labels where erasure requests must be honoured in full.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import logging
import math
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from backend.storage.vault import connect, restrict_permissions

GENESIS_HASH = "0" * 64
logger = logging.getLogger("fairface.audit")


class AuditLockError(RuntimeError):
    """Another process already owns this audit log."""


def _acquire_lock(path: Path):
    """Exclusive, non-blocking, process-lifetime lock on ``path``."""
    handle = open(path, "a+b")
    restrict_permissions(path)
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as exc:
        handle.close()
        raise AuditLockError(
            f"{path.parent} is in use by another FairFace Guard process; the audit log supports a "
            "single writer (do not run multiple workers against one data directory)"
        ) from exc
    return handle

# Fields required in every entry (the specification's hashed tuple) ...
CORE_FIELDS = (
    "timestamp",
    "mode",
    "quality_score",
    "liveness_score",
    "top1_subject",
    "top1_score",
    "top2_score",
    "margin",
    "thresholds",
    "decision_zone",
    "abstention_reason",
)
# ... plus context that is also covered by the hash.
EXTRA_FIELDS = ("engine", "source", "top2_subject", "asserted_subject", "claimed_subject", "face_count")
CSV_COLUMNS = ("seq", *CORE_FIELDS, *EXTRA_FIELDS, "prev_hash", "hash")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS audit_log (
    seq            INTEGER PRIMARY KEY,
    timestamp      TEXT NOT NULL,
    mode           TEXT NOT NULL,
    decision_zone  TEXT NOT NULL,
    top1_subject   TEXT,
    payload        TEXT NOT NULL,
    prev_hash      TEXT NOT NULL,
    hash           TEXT NOT NULL UNIQUE
);
"""


def canonical_json(payload: dict[str, Any]) -> str:
    """Deterministic serialisation: sorted keys, no whitespace, ASCII, no NaN."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def entry_hash(payload: dict[str, Any]) -> str:
    """Hash of an entry body (everything except the ``hash`` field itself)."""
    return hashlib.sha256(canonical_json(payload).encode("ascii")).hexdigest()


def _clean_number(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return round(number, 6) if math.isfinite(number) else None


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


class AuditLog:
    def __init__(self, db_path: Path, jsonl_path: Path) -> None:
        self.db_path = Path(db_path)
        self.jsonl_path = Path(jsonl_path)
        self._lock = threading.Lock()
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_handle = _acquire_lock(self.jsonl_path.with_name(self.jsonl_path.name + ".lock"))
        try:
            self._conn: sqlite3.Connection = connect(self.db_path)
            self._conn.executescript(_SCHEMA)
            self.jsonl_path.touch(mode=0o600, exist_ok=True)
            restrict_permissions(self.jsonl_path)
        except BaseException:
            self._lock_handle.close()
            raise
        # Set when an append to the JSONL mirror failed (disk full, ...); the
        # next write re-emits the missing rows from SQLite first.
        self._mirror_dirty = False
        self.recovered_entries = self._repair_mirror(startup=True)

    # -- writing ----------------------------------------------------------- #

    def _head_locked(self) -> tuple[int, str]:
        row = self._conn.execute("SELECT seq, hash FROM audit_log ORDER BY seq DESC LIMIT 1").fetchone()
        return (int(row[0]), str(row[1])) if row else (0, GENESIS_HASH)

    def record(
        self,
        *,
        mode: str,
        decision_zone: str,
        abstention_reason: str | None,
        quality_score: float | None,
        liveness_score: float | None,
        top1_subject: str | None,
        top1_score: float | None,
        top2_score: float | None,
        margin: float | None,
        thresholds: dict[str, float],
        engine: str = "fair",
        source: str = "api",
        top2_subject: str | None = None,
        asserted_subject: str | None = None,
        claimed_subject: str | None = None,
        face_count: int | None = None,
        timestamp: str | None = None,
    ) -> dict[str, Any]:
        """Append one entry. Returns the stored entry including its hash."""
        body: dict[str, Any] = {
            "timestamp": timestamp or _utc_now(),
            "mode": str(mode),
            "quality_score": _clean_number(quality_score),
            "liveness_score": _clean_number(liveness_score),
            "top1_subject": top1_subject,
            "top1_score": _clean_number(top1_score),
            "top2_score": _clean_number(top2_score),
            "margin": _clean_number(margin),
            "thresholds": {k: _clean_number(v) for k, v in sorted(thresholds.items())},
            "decision_zone": str(decision_zone),
            "abstention_reason": abstention_reason,
            "engine": str(engine),
            "source": str(source),
            "top2_subject": top2_subject,
            "asserted_subject": asserted_subject,
            "claimed_subject": claimed_subject,
            "face_count": None if face_count is None else int(face_count),
        }
        with self._lock:
            if self._mirror_dirty:
                self._repair_mirror_locked(startup=False)
            return self._record_locked(body)

    def _insert_locked(self, body: dict[str, Any]) -> dict[str, Any]:
        """Commit one entry to SQLite (the commit point). Returns it with its hash."""
        # Head read and insert are one transaction: nothing can take the same
        # sequence number in between.
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            last_seq, last_hash = self._head_locked()
            body["seq"] = last_seq + 1
            body["prev_hash"] = last_hash
            digest = entry_hash(body)
            self._conn.execute(
                "INSERT INTO audit_log(seq, timestamp, mode, decision_zone, top1_subject, payload, "
                "prev_hash, hash) VALUES(?,?,?,?,?,?,?,?)",
                (body["seq"], body["timestamp"], body["mode"], body["decision_zone"], body["top1_subject"],
                 canonical_json(body), last_hash, digest),
            )
            self._conn.execute("COMMIT")
        except BaseException:
            self._conn.execute("ROLLBACK")
            raise
        return {**body, "hash": digest}

    def _record_locked(self, body: dict[str, Any]) -> dict[str, Any]:
        entry = self._insert_locked(body)
        # The mirror line follows the commit. If the append fails the decision
        # is still durably recorded, and the mirror is brought back in step
        # before the next write.
        try:
            self._append_jsonl([entry])
        except OSError as exc:
            self._mirror_dirty = True
            logger.error("audit mirror append failed (%s); will resynchronise from SQLite", exc)
        return entry

    def _append_jsonl(self, entries: list[dict[str, Any]]) -> None:
        with open(self.jsonl_path, "a", encoding="ascii", newline="\n") as handle:
            for entry in entries:
                handle.write(canonical_json(entry) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def _repair_mirror(self, startup: bool) -> int:
        with self._lock:
            return self._repair_mirror_locked(startup)

    def _repair_mirror_locked(self, startup: bool) -> int:
        """Bring the JSONL mirror back in step with SQLite after an interruption.

        Repaired cases: a torn final line (crash mid-write), a final line that
        lost only its newline, and rows committed to SQLite whose line never
        reached the file. The file must otherwise be an exact prefix of the
        table; any other discrepancy (a line that differs, a blank line, extra
        lines) is left untouched for ``verify`` to report.

        After a crash at most one row can legitimately be missing, so at
        start-up nothing more than that is regenerated: a file that lost
        several entries (or vanished) while the table kept them was tampered
        with, and silently rebuilding it would launder that. A repair is
        recorded as a chained ``AUDIT_MIRROR_REPAIRED`` entry, and that entry
        is committed *before* the file is touched so it cannot be lost.

        All offsets are taken from the file's own bytes; the file is only ever
        cut back to the end of a line already known to be good.
        """
        try:
            raw = self.jsonl_path.read_bytes() if self.jsonl_path.exists() else b""
        except OSError:
            return 0
        rows = self._conn.execute("SELECT payload, hash FROM audit_log ORDER BY seq").fetchall()

        def matches(line: bytes, index: int) -> bool:
            if index >= len(rows):
                return False
            try:
                parsed = json.loads(line)
            except ValueError:
                return False
            return isinstance(parsed, dict) and parsed.get("hash") == rows[index][1]

        good_end = 0  # byte offset just past the last verified, newline-terminated line
        count = 0
        while True:
            newline = raw.find(b"\n", good_end)
            if newline < 0:
                break
            if not matches(raw[good_end:newline], count):
                return 0  # differs from the table, blank, or surplus: not ours to fix
            good_end, count = newline + 1, count + 1
        tail = raw[good_end:]
        unterminated = bool(tail) and matches(tail, count)  # whole line, newline missing
        torn = bool(tail) and not unterminated
        if unterminated:
            count += 1
        missing = rows[count:]
        if not missing and not torn and not unterminated:
            self._mirror_dirty = False
            return 0
        if startup and len(missing) > 1:
            logger.error(
                "audit mirror is %d entries behind the database; not regenerating (see /api/audit/verify)",
                len(missing),
            )
            return 0

        entries = [self._row_to_entry(payload, digest) for payload, digest in missing]
        if missing or torn:
            # Commit the record of this repair first; it is then simply one
            # more row for the file to catch up on.
            entries.append(self._insert_locked({
                "timestamp": _utc_now(), "mode": "admin", "quality_score": None, "liveness_score": None,
                "top1_subject": None, "top1_score": None, "top2_score": None, "margin": None, "thresholds": {},
                "decision_zone": "AUDIT_MIRROR_REPAIRED",
                "abstention_reason": f"entries_reemitted={len(missing)} torn_line_removed={str(torn).lower()}",
                "engine": "n/a", "source": "startup" if startup else "runtime", "top2_subject": None,
                "asserted_subject": None, "claimed_subject": None, "face_count": None,
            }))
        try:
            with open(self.jsonl_path, "r+b") as handle:
                if torn:
                    handle.truncate(good_end)  # good_end <= len(raw): only ever shortens
                handle.seek(0, os.SEEK_END)
                if unterminated:
                    handle.write(b"\n")
                for entry in entries:
                    handle.write(canonical_json(entry).encode("ascii") + b"\n")
                handle.flush()
                os.fsync(handle.fileno())
        except OSError as exc:
            self._mirror_dirty = True
            logger.error("audit mirror repair failed: %s", exc)
            return 0
        self._mirror_dirty = False
        return len(missing)

    # -- reading ----------------------------------------------------------- #

    @staticmethod
    def _row_to_entry(payload: str, digest: str) -> dict[str, Any]:
        entry = json.loads(payload)
        entry["hash"] = digest
        return entry

    def _iter_jsonl_raw(self) -> Iterator[str]:
        if not self.jsonl_path.exists():
            return
        with open(self.jsonl_path, "r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                stripped = line.strip()
                if stripped:
                    yield stripped

    def head(self) -> dict[str, Any]:
        with self._lock:
            seq, digest = self._head_locked()
        return {"entries": seq, "head_hash": digest}

    def entries(self, limit: int | None = None, newest_first: bool = False) -> list[dict[str, Any]]:
        query = "SELECT payload, hash FROM audit_log ORDER BY seq " + ("DESC" if newest_first else "ASC")
        params: tuple = ()
        if limit is not None:
            query += " LIMIT ?"
            params = (int(limit),)
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_entry(payload, digest) for payload, digest in rows]

    # -- verification ------------------------------------------------------ #

    @staticmethod
    def _check_chain(entries: Iterator[tuple[int, dict[str, Any] | None, str | None]], label: str):
        """Walk a chain. ``entries`` yields ``(position, entry_or_None, parse_error)``."""
        errors: list[dict[str, Any]] = []
        hashes: list[str] = []
        prev = GENESIS_HASH
        for position, entry, parse_error in entries:
            if entry is None:
                errors.append({"store": label, "seq": position, "error": parse_error or "unreadable entry"})
                hashes.append("")
                prev = ""
                continue
            stored_hash = entry.get("hash")
            body = {k: v for k, v in entry.items() if k != "hash"}
            if body.get("seq") != position:
                errors.append({"store": label, "seq": position, "error": f"sequence gap or reorder (found seq {body.get('seq')})"})
            if body.get("prev_hash") != prev:
                errors.append({"store": label, "seq": position, "error": "prev_hash does not match the preceding entry"})
            try:
                recomputed = entry_hash(body)
            except (TypeError, ValueError):
                recomputed = None
            if recomputed is None or recomputed != stored_hash:
                errors.append({"store": label, "seq": position, "error": "entry content does not match its hash"})
            hashes.append(str(stored_hash))
            prev = str(stored_hash)
        return hashes, errors

    def verify(self, max_errors: int = 20) -> dict[str, Any]:
        """Recompute both chains and cross-check them."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT seq, timestamp, mode, decision_zone, top1_subject, payload, prev_hash, hash "
                "FROM audit_log ORDER BY seq"
            ).fetchall()
            lines = list(self._iter_jsonl_raw())

        column_errors: list[dict[str, Any]] = []

        def db_entries():
            for position, row in enumerate(rows, start=1):
                seq, ts, mode, zone, top1, payload, prev_hash, digest = row
                try:
                    body = json.loads(payload)
                    if not isinstance(body, dict):
                        raise ValueError("payload is not an object")
                except (json.JSONDecodeError, ValueError) as exc:
                    yield position, None, f"payload is not valid JSON ({exc})"
                    continue
                # The indexed columns are a convenience copy; they must agree
                # with the hashed payload or queries could be made to lie.
                mirror = {"seq": seq, "timestamp": ts, "mode": mode, "decision_zone": zone,
                          "top1_subject": top1, "prev_hash": prev_hash}
                for key, value in mirror.items():
                    if body.get(key) != value:
                        column_errors.append({"store": "sqlite", "seq": position,
                                              "error": f"column {key} disagrees with the hashed payload"})
                yield position, {**body, "hash": digest}, None

        def jsonl_entries():
            for position, line in enumerate(lines, start=1):
                try:
                    entry = json.loads(line)
                    if not isinstance(entry, dict):
                        raise ValueError("line is not an object")
                except (json.JSONDecodeError, ValueError) as exc:
                    yield position, None, f"line is not valid JSON ({exc})"
                    continue
                yield position, entry, None

        db_hashes, db_errors = self._check_chain(db_entries(), "sqlite")
        db_errors.extend(column_errors)
        file_hashes, file_errors = self._check_chain(jsonl_entries(), "jsonl")

        cross_errors: list[dict[str, Any]] = []
        if len(db_hashes) != len(file_hashes):
            cross_errors.append({
                "store": "cross-check", "seq": min(len(db_hashes), len(file_hashes)) + 1,
                "error": f"entry count differs (sqlite={len(db_hashes)}, jsonl={len(file_hashes)})",
            })
        for position, (a, b) in enumerate(zip(db_hashes, file_hashes), start=1):
            if a != b:
                cross_errors.append({"store": "cross-check", "seq": position,
                                     "error": "sqlite and jsonl hold different entries"})
                break

        errors = db_errors + file_errors + cross_errors
        return {
            "ok": not errors,
            "entries": len(db_hashes),
            "head_hash": db_hashes[-1] if db_hashes else GENESIS_HASH,
            "sqlite_ok": not db_errors,
            "jsonl_ok": not file_errors,
            "stores_consistent": not cross_errors,
            "error_count": len(errors),
            "errors": errors[:max_errors],
            "verified_at": _utc_now(),
        }

    # -- export ------------------------------------------------------------ #

    def export_json(self) -> str:
        return json.dumps(
            {"exported_at": _utc_now(), "verification": self.verify(), "entries": self.entries()},
            indent=2,
            sort_keys=True,
        )

    def export_csv(self) -> str:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for entry in self.entries():
            row = []
            for column in CSV_COLUMNS:
                value = entry.get(column)
                if isinstance(value, dict):
                    value = canonical_json(value)
                elif isinstance(value, str):
                    value = _csv_safe(value)
                row.append("" if value is None else value)
            writer.writerow(row)
        return buffer.getvalue()

    def close(self) -> None:
        with self._lock:
            self._conn.close()
            self._lock_handle.close()  # releases the single-writer lock


def _csv_safe(text: str) -> str:
    """Neutralise spreadsheet formula injection in user-controlled labels."""
    return "'" + text if text[:1] in ("=", "+", "-", "@", "\t", "\r") else text
