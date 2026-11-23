"""Visitor numbers for faces that match nobody enrolled.

Someone who is not enrolled gets a short label ("Visitor 3") instead of a
name, and keeps it while they stay in view or step away and come back, so a
guard can tell two strangers apart on screen.

Everything here lives in memory only. Nothing about a visitor is written to
the database, the audit log or disk, and the numbers are forgotten when the
server stops or after ``ttl_s`` without a sighting. That is deliberate: the
system must not build a store of people who never agreed to be enrolled.
"""
from __future__ import annotations

import threading
import time

import numpy as np

# ArcFace cosine above which two sightings are treated as the same visitor.
# A little stricter than the comparison engine's match threshold: giving one
# person two numbers is a smaller mistake than giving two people one.
SAME_VISITOR_COSINE = 0.36
MAX_VISITORS = 200
TTL_SECONDS = 4 * 3600.0
# A borderline ("might be X") frame is settled as a visitor already seen this
# session when the face matches that visitor strongly AND clearly better than
# it matches the enrolled person it was confused with.
KNOWN_VISITOR_COSINE = 0.50
KNOWN_VISITOR_MARGIN = 0.15


def is_known_visitor(visitor_cosine: float, enrolled_cosine: float) -> bool:
    """True when a face is far more like a known visitor than like the enrolled candidate."""
    return (visitor_cosine >= KNOWN_VISITOR_COSINE
            and visitor_cosine - enrolled_cosine >= KNOWN_VISITOR_MARGIN)


class VisitorRegistry:
    def __init__(self, threshold: float = SAME_VISITOR_COSINE, max_visitors: int = MAX_VISITORS,
                 ttl_s: float = TTL_SECONDS) -> None:
        self.threshold = float(threshold)
        self.max_visitors = int(max_visitors)
        self.ttl_s = float(ttl_s)
        self._lock = threading.Lock()
        self._next = 1
        # number -> [unit template, sightings, last seen (monotonic seconds)]
        self._visitors: dict[int, list] = {}

    def __len__(self) -> int:
        with self._lock:
            return len(self._visitors)

    def assign(self, embedding: np.ndarray, now: float | None = None) -> int | None:
        """The visitor number for this face: an existing one if it has been seen, else a new one."""
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if vector.size == 0 or not np.isfinite(norm) or norm == 0.0:
            return None
        vector = vector / norm
        now = time.monotonic() if now is None else now
        with self._lock:
            for number in [n for n, v in self._visitors.items() if now - v[2] > self.ttl_s]:
                del self._visitors[number]
            best, best_cosine = None, self.threshold
            for number, (template, _count, _seen) in self._visitors.items():
                if template.shape != vector.shape:
                    continue
                cosine = float(template @ vector)
                if cosine >= best_cosine:
                    best, best_cosine = number, cosine
            if best is not None:
                template, count, _seen = self._visitors[best]
                # A running mean steadies the template as the person moves.
                merged = template * min(count, 20) + vector
                merged /= max(float(np.linalg.norm(merged)), 1e-9)
                self._visitors[best] = [merged, count + 1, now]
                return best
            if len(self._visitors) >= self.max_visitors:
                oldest = min(self._visitors, key=lambda n: self._visitors[n][2])
                del self._visitors[oldest]
            number, self._next = self._next, self._next + 1
            self._visitors[number] = [vector, 1, now]
            return number

    @staticmethod
    def _unit(embedding: np.ndarray) -> np.ndarray | None:
        vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if vector.size == 0 or not np.isfinite(norm) or norm == 0.0:
            return None
        return vector / norm

    def match(self, embedding: np.ndarray, now: float | None = None) -> tuple[int, float] | None:
        """The closest remembered visitor and its cosine, without changing anything."""
        vector = self._unit(embedding)
        if vector is None:
            return None
        now = time.monotonic() if now is None else now
        best: tuple[int, float] | None = None
        with self._lock:
            for number, (template, _count, seen) in self._visitors.items():
                if now - seen > self.ttl_s or template.shape != vector.shape:
                    continue
                cosine = float(template @ vector)
                if best is None or cosine > best[1]:
                    best = (number, cosine)
        return best

    def forget_like(self, embedding: np.ndarray) -> int:
        """Drop any visitor that is this face: someone confirmed as enrolled is not a visitor."""
        vector = self._unit(embedding)
        if vector is None:
            return 0
        with self._lock:
            gone = [n for n, (template, _c, _s) in self._visitors.items()
                    if template.shape == vector.shape and float(template @ vector) >= self.threshold]
            for number in gone:
                del self._visitors[number]
            return len(gone)

    def clear(self) -> None:
        with self._lock:
            self._visitors.clear()
            self._next = 1
