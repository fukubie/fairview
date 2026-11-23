"""Encrypted biometric template store (SQLite; no enrollment images).

What is persisted: a subject label, per-sample face embeddings encrypted with
an authenticated cipher, a scalar quality score and timestamps per sample, and
the embedding dimensionality. Enrollment frames never reach this module.

Two narrow exceptions hold images, both encrypted with the same key: the
record photo of someone on the banned-from-premises list (added on purpose by
an administrator), and the face crops behind an *open* review case, which are
deleted as soon as a person decides the case, or after a retention limit.

Ciphers
-------
``aes256gcm`` (default)
    AES-256-GCM with a random 96-bit nonce. The subject's row id *and name*
    are bound as associated data. A ciphertext copied onto another subject's
    row, or a subject row relabelled with someone else's name directly in the
    database, fails authentication instead of silently impersonating them.
``fernet``
    ``cryptography.fernet.Fernet`` (AES-128-CBC + HMAC-SHA256). Offered for
    interoperability with existing Fernet tooling. Note that Fernet is *not*
    AES-256; choose the default when a 256-bit cipher is required.

Both use the same key format: 32 random bytes, url-safe base64 encoded, taken
from ``BIOMETRIC_ENCRYPTION_KEY`` or, failing that, generated once into
``<data>/biometric.key`` (mode 0600). Keeping the key file next to the database
only protects against the database leaking on its own; production deployments
should inject the key from a secret manager through the environment variable.
"""

from __future__ import annotations

import base64
import os
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
from cryptography.exceptions import InvalidTag
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

KEY_ENV = "BIOMETRIC_ENCRYPTION_KEY"
CIPHER_ENV = "BIOMETRIC_CIPHER"
KEY_FILE_NAME = "biometric.key"

_VERSION_AESGCM = b"\x01"
_VERSION_FERNET = b"\x02"
_PLAINTEXT_MAGIC = b"FFG1"
_KEY_CHECK_PLAINTEXT = b"fairface-guard key check"
_KEY_CHECK_AAD = b"fairface-guard/v1/key-check"

MAX_SUBJECT_NAME_LENGTH = 120


class VaultError(Exception):
    """Base class for vault failures."""


class VaultKeyError(VaultError):
    """The configured key is malformed or does not match the database."""


class SubjectNotFound(VaultError):
    pass


class SubjectExists(VaultError):
    pass


def generate_key() -> bytes:
    """Return a fresh key (url-safe base64 of 32 random bytes)."""
    return base64.urlsafe_b64encode(os.urandom(32))


def _decode_key(encoded: bytes | str) -> bytes:
    if isinstance(encoded, str):
        encoded = encoded.encode("ascii", errors="strict")
    try:
        raw = base64.urlsafe_b64decode(encoded.strip())
    except Exception as exc:  # binascii.Error / ValueError
        raise VaultKeyError(f"{KEY_ENV} is not valid url-safe base64") from exc
    if len(raw) != 32:
        raise VaultKeyError(f"{KEY_ENV} must decode to exactly 32 bytes, got {len(raw)}")
    return raw


def load_or_create_key(data_dir: Path) -> tuple[bytes, str]:
    """Resolve the encryption key. Returns ``(encoded_key, source)``."""
    env_value = os.environ.get(KEY_ENV)
    if env_value:
        _decode_key(env_value)
        return env_value.strip().encode("ascii"), "environment"
    key_path = data_dir / KEY_FILE_NAME
    if key_path.exists():
        encoded = key_path.read_bytes().strip()
        _decode_key(encoded)
        return encoded, "key file"
    data_dir.mkdir(parents=True, exist_ok=True)
    encoded = generate_key()
    # Write the complete key to a private temporary file first, then publish
    # it with a hard link: the link either appears with full content or not at
    # all (no empty key file after a crash), and it fails if another process
    # published a key in the meantime (never clobber). 0600: owner only.
    temporary = key_path.with_name(f"{KEY_FILE_NAME}.{os.getpid()}.{os.urandom(4).hex()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(encoded + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, key_path)
        except FileExistsError:
            encoded = key_path.read_bytes().strip()
            _decode_key(encoded)
            return encoded, "key file"
    finally:
        try:
            os.unlink(temporary)
        except OSError:
            pass
    return encoded, "generated key file"


class EmbeddingCipher:
    """Authenticated encryption of embedding blobs with a versioned envelope."""

    def __init__(self, encoded_key: bytes | str, default_cipher: str = "aes256gcm") -> None:
        raw = _decode_key(encoded_key)
        if default_cipher not in {"aes256gcm", "fernet"}:
            raise VaultKeyError(f"{CIPHER_ENV} must be 'aes256gcm' or 'fernet', got {default_cipher!r}")
        self.default_cipher = default_cipher
        self._aesgcm = AESGCM(raw)
        self._fernet = Fernet(base64.urlsafe_b64encode(raw))

    def encrypt(self, plaintext: bytes, aad: bytes) -> bytes:
        if self.default_cipher == "fernet":
            # Fernet has no associated-data channel; bind the context by
            # prefixing it inside the authenticated plaintext.
            return _VERSION_FERNET + self._fernet.encrypt(_pack_aad(aad) + plaintext)
        nonce = os.urandom(12)
        return _VERSION_AESGCM + nonce + self._aesgcm.encrypt(nonce, plaintext, aad)

    def decrypt(self, blob: bytes, aad: bytes) -> bytes:
        blob = bytes(blob)
        version, body = blob[:1], blob[1:]
        try:
            if version == _VERSION_AESGCM:
                if len(body) < 12 + 16:
                    raise VaultError("ciphertext is truncated")
                return self._aesgcm.decrypt(body[:12], body[12:], aad)
            if version == _VERSION_FERNET:
                plaintext = self._fernet.decrypt(body)
                prefix = _pack_aad(aad)
                if not plaintext.startswith(prefix):
                    raise VaultError("ciphertext is bound to a different record")
                return plaintext[len(prefix) :]
        except (InvalidTag, InvalidToken) as exc:
            raise VaultError(
                "embedding blob failed authentication (wrong key, corrupted or moved ciphertext)"
            ) from exc
        raise VaultError(f"unknown ciphertext version {version!r}")


def _pack_aad(aad: bytes) -> bytes:
    return struct.pack("<H", len(aad)) + aad


def _serialize_embedding(vector: np.ndarray) -> bytes:
    vec = np.ascontiguousarray(vector, dtype="<f4").reshape(-1)
    if vec.size == 0 or not np.all(np.isfinite(vec)):
        raise VaultError("embedding must be a non-empty finite vector")
    return _PLAINTEXT_MAGIC + struct.pack("<I", vec.size) + vec.tobytes()


def _deserialize_embedding(payload: bytes) -> np.ndarray:
    if payload[:4] != _PLAINTEXT_MAGIC or len(payload) < 8:
        raise VaultError("decrypted payload is not an embedding record")
    (dim,) = struct.unpack("<I", payload[4:8])
    body = payload[8:]
    if len(body) != dim * 4:
        raise VaultError("embedding record length mismatch")
    return np.frombuffer(body, dtype="<f4").astype(np.float32, copy=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def normalize_subject_name(name: str) -> str:
    if not isinstance(name, str):
        raise ValueError("subject name must be a string")
    cleaned = " ".join(name.split())
    if not cleaned:
        raise ValueError("subject name must not be empty")
    if len(cleaned) > MAX_SUBJECT_NAME_LENGTH:
        raise ValueError(f"subject name must be at most {MAX_SUBJECT_NAME_LENGTH} characters")
    if any(ord(ch) < 32 or ord(ch) == 127 for ch in cleaned):
        raise ValueError("subject name must not contain control characters")
    return cleaned


@dataclass(frozen=True)
class SubjectInfo:
    name: str
    samples: int
    mean_quality: float | None
    created_at: str
    updated_at: str
    check_in_days: int = 0

    def to_dict(self) -> dict:
        return {
            "subject": self.name,
            "samples": self.samples,
            "mean_quality": None if self.mean_quality is None else round(self.mean_quality, 4),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "check_in_days": self.check_in_days,
        }


@dataclass(frozen=True)
class Gallery:
    """Decrypted, in-memory enrollment templates (one unit vector per subject)."""

    names: tuple[str, ...]
    templates: np.ndarray  # (N, D) float32, rows L2-normalised
    sample_counts: tuple[int, ...]
    version: int

    def __len__(self) -> int:
        return len(self.names)

    def index_of(self, name: str) -> int | None:
        try:
            return self.names.index(name)
        except ValueError:
            return None


_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value BLOB NOT NULL
);
CREATE TABLE IF NOT EXISTS subjects (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    name       TEXT NOT NULL UNIQUE,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS samples (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    subject_id INTEGER NOT NULL REFERENCES subjects(id) ON DELETE CASCADE,
    embedding  BLOB NOT NULL,
    quality    REAL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_samples_subject ON samples(subject_id);
-- People banned from the premises. The optional record photo (e.g. from a
-- trespass notice) is shown to the guard next to the camera frame.
CREATE TABLE IF NOT EXISTS banned (
    subject_id INTEGER PRIMARY KEY REFERENCES subjects(id) ON DELETE CASCADE,
    note       TEXT,
    photo      BLOB,
    created_at TEXT NOT NULL
);
-- Face crops behind open review cases, deleted once a person decides the
-- case (or after a retention limit). Keyed by audit-log seq.
CREATE TABLE IF NOT EXISTS review_snapshots (
    seq        INTEGER PRIMARY KEY,
    image      BLOB NOT NULL,
    created    REAL NOT NULL
);
"""


def restrict_permissions(path: Path) -> None:
    """Owner-only access for files holding biometric or audit data (best effort)."""
    try:
        os.chmod(path, 0o600)
    except OSError:  # e.g. filesystems without POSIX permissions
        pass


def connect(db_path: Path) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if not db_path.exists():
        # Create with a restrictive mode up front; SQLite gives the WAL and
        # shared-memory side files the same permissions as the main file.
        os.close(os.open(db_path, os.O_WRONLY | os.O_CREAT, 0o600))
    restrict_permissions(db_path)
    conn = sqlite3.connect(db_path, check_same_thread=False, isolation_level=None, timeout=30.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=FULL")
    conn.execute("PRAGMA foreign_keys=ON")
    # Overwrite deleted pages so removed templates do not linger in free pages.
    conn.execute("PRAGMA secure_delete=ON")
    return conn


class BiometricVault:
    """Subject / embedding lifecycle on top of an encrypted SQLite table."""

    def __init__(
        self,
        db_path: Path,
        encoded_key: bytes | str | None = None,
        cipher: str | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        if encoded_key is None:
            encoded_key, self.key_source = load_or_create_key(self.db_path.parent)
        else:
            self.key_source = "explicit"
        self._cipher = EmbeddingCipher(
            encoded_key, cipher or os.environ.get(CIPHER_ENV, "aes256gcm").strip().lower()
        )
        self._lock = threading.RLock()
        self._conn = connect(self.db_path)
        self._conn.executescript(_SCHEMA)
        self._verify_key()
        self._gallery: Gallery | None = None
        self._version = 0

    # -- key handling ------------------------------------------------------ #

    def _verify_key(self) -> None:
        """Fail at startup, not at first match, when the key is wrong."""
        with self._lock:
            row = self._conn.execute("SELECT value FROM meta WHERE key='key_check'").fetchone()
            if row is None and self._conn.execute("SELECT 1 FROM samples LIMIT 1").fetchone():
                # Templates exist but the key check is gone: someone removed it.
                # Minting a new check with whatever key we were given would
                # hide a key mismatch until the first match attempt.
                raise VaultKeyError(
                    f"{self.db_path} holds templates but no key check record; refusing to open it"
                )
            if row is None:
                token = self._cipher.encrypt(_KEY_CHECK_PLAINTEXT, _KEY_CHECK_AAD)
                self._conn.execute(
                    "INSERT OR IGNORE INTO meta(key, value) VALUES('key_check', ?)", (token,)
                )
                row = self._conn.execute("SELECT value FROM meta WHERE key='key_check'").fetchone()
            try:
                ok = self._cipher.decrypt(row[0], _KEY_CHECK_AAD) == _KEY_CHECK_PLAINTEXT
            except VaultError:
                ok = False
            if not ok:
                raise VaultKeyError(
                    f"the encryption key does not match {self.db_path}. Restore the original "
                    f"{KEY_ENV} / {KEY_FILE_NAME}; enrolled templates cannot be read without it."
                )

    @property
    def cipher_name(self) -> str:
        return self._cipher.default_cipher

    @staticmethod
    def _aad(subject_id: int, name: str) -> bytes:
        return f"fairface-guard/v2/subject/{int(subject_id)}/{name}".encode("utf-8")

    # -- writes ------------------------------------------------------------ #

    def add_samples(
        self,
        name: str,
        embeddings: list[np.ndarray],
        qualities: list[float | None] | None = None,
        replace: bool = False,
        max_samples: int | None = None,
    ) -> SubjectInfo:
        """Encrypt and store enrollment samples, creating the subject if needed.

        With ``max_samples`` set, only the most recent ``max_samples`` samples
        of the subject are retained.
        """
        name = normalize_subject_name(name)
        if not embeddings:
            raise ValueError("at least one embedding is required")
        qualities = list(qualities) if qualities is not None else [None] * len(embeddings)
        if len(qualities) != len(embeddings):
            raise ValueError("qualities must align with embeddings")
        dims = {int(np.asarray(e).size) for e in embeddings}
        if len(dims) != 1:
            raise ValueError("all embeddings must share one dimensionality")
        payloads = [_serialize_embedding(np.asarray(e)) for e in embeddings]
        now = _now()
        with self._lock:
            self._check_dimension(dims.pop())
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                row = self._conn.execute("SELECT id FROM subjects WHERE name=?", (name,)).fetchone()
                if row is None:
                    cur = self._conn.execute(
                        "INSERT INTO subjects(name, created_at, updated_at) VALUES(?,?,?)",
                        (name, now, now),
                    )
                    subject_id = int(cur.lastrowid)
                else:
                    subject_id = int(row[0])
                    if replace:
                        self._conn.execute("DELETE FROM samples WHERE subject_id=?", (subject_id,))
                    self._conn.execute(
                        "UPDATE subjects SET updated_at=? WHERE id=?", (now, subject_id)
                    )
                aad = self._aad(subject_id, name)
                self._conn.executemany(
                    "INSERT INTO samples(subject_id, embedding, quality, created_at) VALUES(?,?,?,?)",
                    [
                        (
                            subject_id,
                            self._cipher.encrypt(payload, aad),
                            None if q is None else float(q),
                            now,
                        )
                        for payload, q in zip(payloads, qualities)
                    ],
                )
                if max_samples is not None and max_samples > 0:
                    self._conn.execute(
                        "DELETE FROM samples WHERE subject_id=? AND id NOT IN ("
                        "SELECT id FROM samples WHERE subject_id=? ORDER BY id DESC LIMIT ?)",
                        (subject_id, subject_id, int(max_samples)),
                    )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            if replace or max_samples:
                self._purge_wal()  # replaced / trimmed samples must not linger either
            self._invalidate()
            return self._subject_info(name)

    def _check_dimension(self, dim: int) -> None:
        row = self._conn.execute("SELECT value FROM meta WHERE key='embedding_dim'").fetchone()
        if row is None:
            self._conn.execute(
                "INSERT INTO meta(key, value) VALUES('embedding_dim', ?)", (str(dim).encode(),)
            )
            return
        stored = int(bytes(row[0]).decode())
        if stored != dim:
            raise VaultError(
                f"embedding dimension {dim} does not match the enrolled gallery ({stored}); "
                "the recognition models changed. Re-enroll into a fresh database."
            )

    def rename_subject(self, name: str, new_name: str) -> SubjectInfo:
        name, new_name = normalize_subject_name(name), normalize_subject_name(new_name)
        with self._lock:
            if self._conn.execute("SELECT 1 FROM subjects WHERE name=?", (name,)).fetchone() is None:
                raise SubjectNotFound(name)
            if name != new_name and self._conn.execute(
                "SELECT 1 FROM subjects WHERE name=?", (new_name,)
            ).fetchone():
                raise SubjectExists(new_name)
            subject_id = int(self._conn.execute("SELECT id FROM subjects WHERE name=?", (name,)).fetchone()[0])
            # The name is part of each blob's authenticated context, so a
            # rename re-encrypts the subject's samples, atomically.
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                old_aad, new_aad = self._aad(subject_id, name), self._aad(subject_id, new_name)
                for sample_id, blob in self._conn.execute(
                    "SELECT id, embedding FROM samples WHERE subject_id=?", (subject_id,)
                ).fetchall():
                    self._conn.execute(
                        "UPDATE samples SET embedding=? WHERE id=?",
                        (self._cipher.encrypt(self._cipher.decrypt(blob, old_aad), new_aad), sample_id),
                    )
                self._conn.execute(
                    "UPDATE subjects SET name=?, updated_at=? WHERE id=?", (new_name, _now(), subject_id)
                )
                self._conn.execute("COMMIT")
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._purge_wal()  # the superseded ciphertexts
            self._invalidate()
            return self._subject_info(new_name)

    def delete_subject(self, name: str) -> int:
        """Erase a subject and all of their templates. Returns samples removed."""
        name = normalize_subject_name(name)
        with self._lock:
            row = self._conn.execute("SELECT id FROM subjects WHERE name=?", (name,)).fetchone()
            if row is None:
                raise SubjectNotFound(name)
            removed = self._conn.execute(
                "SELECT COUNT(*) FROM samples WHERE subject_id=?", (row[0],)
            ).fetchone()[0]
            self._conn.execute("DELETE FROM subjects WHERE id=?", (row[0],))
            self._purge_wal()
            self._invalidate()
            return int(removed)

    def _purge_wal(self) -> None:
        """Fold the write-ahead log back into the database and empty it.

        ``secure_delete`` zeroes freed pages in the main file, but superseded
        copies of those pages would otherwise linger in the WAL. Best effort
        with a short wait: if another connection holds a read transaction (a
        backup tool, a database browser) the truncation is skipped rather than
        stalling recognition, and happens at a later purge or checkpoint.
        """
        try:
            self._conn.execute("PRAGMA busy_timeout=250")
            self._conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        except sqlite3.OperationalError:
            pass
        finally:
            self._conn.execute("PRAGMA busy_timeout=30000")

    def delete_all(self) -> int:
        with self._lock:
            count = self._conn.execute("SELECT COUNT(*) FROM subjects").fetchone()[0]
            self._conn.execute("DELETE FROM subjects")
            self._conn.execute("DELETE FROM meta WHERE key='embedding_dim'")
            self._purge_wal()
            self._invalidate()
            return int(count)

    # -- reads ------------------------------------------------------------- #

    def _subject_info(self, name: str) -> SubjectInfo:
        row = self._conn.execute(
            "SELECT s.name, COUNT(x.id), AVG(x.quality), s.created_at, s.updated_at, "
            "COUNT(DISTINCT substr(x.created_at, 1, 10)) "
            "FROM subjects s LEFT JOIN samples x ON x.subject_id = s.id "
            "WHERE s.name=? GROUP BY s.id",
            (name,),
        ).fetchone()
        if row is None:
            raise SubjectNotFound(name)
        return SubjectInfo(row[0], int(row[1]), row[2], row[3], row[4], int(row[5] or 0))

    def get_subject(self, name: str) -> SubjectInfo:
        with self._lock:
            return self._subject_info(normalize_subject_name(name))

    def list_subjects(self) -> list[SubjectInfo]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.name, COUNT(x.id), AVG(x.quality), s.created_at, s.updated_at, "
            "COUNT(DISTINCT substr(x.created_at, 1, 10)) "
                "FROM subjects s LEFT JOIN samples x ON x.subject_id = s.id "
                "GROUP BY s.id ORDER BY s.name COLLATE NOCASE"
            ).fetchall()
        return [SubjectInfo(r[0], int(r[1]), r[2], r[3], r[4], int(r[5] or 0)) for r in rows]

    def gallery(self) -> Gallery:
        """Decrypt all samples and build one mean template per subject (cached)."""
        with self._lock:
            if self._gallery is not None:
                return self._gallery
            rows = self._conn.execute(
                "SELECT s.id, s.name, x.embedding FROM subjects s "
                "JOIN samples x ON x.subject_id = s.id ORDER BY s.id, x.id"
            ).fetchall()
            grouped: dict[int, tuple[str, list[np.ndarray]]] = {}
            for subject_id, name, blob in rows:
                vector = _deserialize_embedding(self._cipher.decrypt(blob, self._aad(subject_id, name)))
                grouped.setdefault(subject_id, (name, []))[1].append(vector)
            names: list[str] = []
            templates: list[np.ndarray] = []
            counts: list[int] = []
            for name, vectors in grouped.values():
                template = build_template(vectors)
                if template is None:
                    continue
                names.append(name)
                templates.append(template)
                counts.append(len(vectors))
            matrix = (
                np.stack(templates).astype(np.float32)
                if templates
                else np.zeros((0, 0), dtype=np.float32)
            )
            self._gallery = Gallery(tuple(names), matrix, tuple(counts), self._version)
            return self._gallery

    # -- banned from premises ---------------------------------------------- #

    @staticmethod
    def _photo_aad(subject_id: int) -> bytes:
        return f"fairface-guard/v2/banned-photo/{int(subject_id)}".encode("utf-8")

    def _subject_id(self, name: str) -> int:
        row = self._conn.execute("SELECT id FROM subjects WHERE name=?", (normalize_subject_name(name),)).fetchone()
        if row is None:
            raise SubjectNotFound(name)
        return int(row[0])

    def set_banned(self, name: str, note: str | None, photo: bytes | None, keep_photo: bool = True) -> None:
        """Put an enrolled subject on the banned list (or update their entry)."""
        with self._lock:
            subject_id = self._subject_id(name)
            blob = self._cipher.encrypt(photo, self._photo_aad(subject_id)) if photo else None
            row = self._conn.execute("SELECT photo FROM banned WHERE subject_id=?", (subject_id,)).fetchone()
            if row is None:
                self._conn.execute("INSERT INTO banned(subject_id, note, photo, created_at) VALUES(?,?,?,?)",
                                   (subject_id, note, blob, _now()))
            else:
                if blob is None and keep_photo:
                    blob = row[0]
                self._conn.execute("UPDATE banned SET note=?, photo=? WHERE subject_id=?", (note, blob, subject_id))
                self._purge_wal()

    def clear_banned(self, name: str) -> bool:
        with self._lock:
            removed = self._conn.execute("DELETE FROM banned WHERE subject_id=?", (self._subject_id(name),)).rowcount
            self._purge_wal()
            return bool(removed)

    def banned(self) -> dict[str, dict]:
        """Name -> {note, has_photo, since} for everyone on the banned list."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT s.name, b.note, b.photo IS NOT NULL, b.created_at FROM banned b "
                "JOIN subjects s ON s.id = b.subject_id"
            ).fetchall()
        return {r[0]: {"note": r[1], "has_photo": bool(r[2]), "since": r[3]} for r in rows}

    def banned_photo(self, name: str) -> bytes | None:
        with self._lock:
            subject_id = self._subject_id(name)
            row = self._conn.execute("SELECT photo FROM banned WHERE subject_id=?", (subject_id,)).fetchone()
            if row is None or row[0] is None:
                return None
            return self._cipher.decrypt(row[0], self._photo_aad(subject_id))

    # -- review snapshots --------------------------------------------------- #

    @staticmethod
    def _snapshot_aad(seq: int) -> bytes:
        return f"fairface-guard/v2/review-snapshot/{int(seq)}".encode("utf-8")

    def add_snapshot(self, seq: int, jpeg: bytes) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO review_snapshots(seq, image, created) VALUES(?,?,?)",
                (int(seq), self._cipher.encrypt(jpeg, self._snapshot_aad(seq)), time.time()),
            )

    def snapshot(self, seq: int) -> bytes | None:
        with self._lock:
            row = self._conn.execute("SELECT image FROM review_snapshots WHERE seq=?", (int(seq),)).fetchone()
        return None if row is None else self._cipher.decrypt(row[0], self._snapshot_aad(seq))

    def snapshot_seqs(self) -> set[int]:
        with self._lock:
            return {int(r[0]) for r in self._conn.execute("SELECT seq FROM review_snapshots").fetchall()}

    def delete_snapshots(self, seqs: Iterable[int]) -> int:
        seqs = [int(s) for s in seqs]
        if not seqs:
            return 0
        with self._lock:
            removed = 0
            for start in range(0, len(seqs), 500):
                chunk = seqs[start:start + 500]
                removed += self._conn.execute(
                    f"DELETE FROM review_snapshots WHERE seq IN ({','.join('?' * len(chunk))})", chunk
                ).rowcount
            if removed:
                self._purge_wal()
            return removed

    def purge_snapshots(self, max_age_s: float, max_count: int) -> int:
        """Drop snapshots older than ``max_age_s`` and all but the newest ``max_count``."""
        with self._lock:
            removed = self._conn.execute(
                "DELETE FROM review_snapshots WHERE created < ?", (time.time() - max_age_s,)
            ).rowcount
            removed += self._conn.execute(
                "DELETE FROM review_snapshots WHERE seq NOT IN ("
                "SELECT seq FROM review_snapshots ORDER BY seq DESC LIMIT ?)", (int(max_count),)
            ).rowcount
            if removed:
                self._purge_wal()
            return removed

    def _invalidate(self) -> None:
        self._gallery = None
        self._version += 1

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def build_template(vectors: list[np.ndarray]) -> np.ndarray | None:
    """Average L2-normalised samples and re-normalise (multi-sample template).

    Returns ``None`` when the samples cancel out to a (near) zero vector, which
    would otherwise yield a meaningless direction.
    """
    if not vectors:
        return None
    stacked = np.stack([np.asarray(v, dtype=np.float64).reshape(-1) for v in vectors])
    norms = np.linalg.norm(stacked, axis=1, keepdims=True)
    if np.any(norms < 1e-9):
        return None
    mean = (stacked / norms).mean(axis=0)
    norm = float(np.linalg.norm(mean))
    if norm < 1e-6:
        return None
    return (mean / norm).astype(np.float32)
