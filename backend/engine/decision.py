"""Three-zone decision policy: Match / Non-Match / Uncertain (defer to review).

The policy is a pure function of scores so it can be proven by unit tests and
replayed from the audit log. It never forces a top-1 answer: whenever image
quality is insufficient, the best score sits in the gray band, or two enrolled
candidates are too close to tell apart, the outcome is ``UNCERTAIN_REVIEW`` and
no identity is asserted.

Evaluation order (first hit wins as the primary reason):

1. liveness below ``l_min``                      -> SPOOF_DETECTED
2. quality below ``q_min``                       -> UNCERTAIN_REVIEW / LOW_IMAGE_QUALITY
3. top-1 score below ``tau_low``                 -> NO_MATCH
4. top-1 score at or above ``tau_high``:
     1:N and (s1 - s2) below ``margin``          -> UNCERTAIN_REVIEW / NARROW_CANDIDATE_MARGIN
     top-1 template not yet complete             -> UNCERTAIN_REVIEW / INCOMPLETE_ENROLLMENT
     otherwise                                   -> MATCH_CONFIRMED
5. otherwise (``tau_low <= s1 < tau_high``)      -> UNCERTAIN_REVIEW / GRAY_ZONE_SIMILARITY

The margin test is applied only once the top-1 score is no longer a clear
non-match: two unrelated candidates scoring 0.10 and 0.09 are a clean
``NO_MATCH``, not an ambiguity. Any non-finite input fails safe to
``UNCERTAIN_REVIEW``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Sequence

from backend.config import Thresholds


class Zone(str, Enum):
    MATCH_CONFIRMED = "MATCH_CONFIRMED"
    NO_MATCH = "NO_MATCH"
    UNCERTAIN_REVIEW = "UNCERTAIN_REVIEW"
    SPOOF_DETECTED = "SPOOF_DETECTED"


class Reason(str, Enum):
    LIVENESS_BELOW_MINIMUM = "LIVENESS_BELOW_MINIMUM"
    LOW_IMAGE_QUALITY = "LOW_IMAGE_QUALITY"
    GRAY_ZONE_SIMILARITY = "GRAY_ZONE_SIMILARITY"
    NARROW_CANDIDATE_MARGIN = "NARROW_CANDIDATE_MARGIN"
    INVALID_SCORE = "INVALID_SCORE"
    SCORE_ABOVE_HIGH_THRESHOLD = "SCORE_ABOVE_HIGH_THRESHOLD"
    SCORE_BELOW_LOW_THRESHOLD = "SCORE_BELOW_LOW_THRESHOLD"
    EMPTY_GALLERY = "EMPTY_GALLERY"
    INCOMPLETE_ENROLLMENT = "INCOMPLETE_ENROLLMENT"
    # A borderline score, settled as "not enrolled" because the face is clearly
    # a visitor already seen (and ruled out) earlier in this session.
    KNOWN_VISITOR = "KNOWN_VISITOR"


class Mode(str, Enum):
    IDENTIFY = "identify"  # 1:N open-set watchlist search
    VERIFY = "verify"  # 1:1 claimed-identity check


@dataclass(frozen=True)
class Candidate:
    subject: str
    score: float  # calibrated score s in [0, 1]
    cosine: float  # raw cosine similarity, kept for the audit trail
    # False for a template built from too few enrollment frames. Such a
    # candidate still takes part (it can make a margin narrow) but is never
    # asserted as an identity.
    eligible: bool = True

    def to_dict(self) -> dict:
        return {"subject": self.subject, "score": _round(self.score), "cosine": _round(self.cosine)}


@dataclass(frozen=True)
class Decision:
    zone: Zone
    reason: Reason
    mode: Mode
    # Identity asserted by the system. Set ONLY for MATCH_CONFIRMED; in every
    # other zone the caller must not present top1 as an identification.
    subject: str | None
    top1: Candidate | None
    top2: Candidate | None
    margin: float | None
    quality: float
    liveness: float | None
    thresholds: Thresholds
    # Every abstention condition that held, primary reason first. Lets the UI
    # show e.g. "gray zone AND narrow margin" without changing the zone logic.
    reasons: tuple[Reason, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return {
            "zone": self.zone.value,
            "reason": self.reason.value,
            "reasons": [r.value for r in self.reasons],
            "mode": self.mode.value,
            "subject": self.subject,
            "top1": self.top1.to_dict() if self.top1 else None,
            "top2": self.top2.to_dict() if self.top2 else None,
            "margin": _round(self.margin),
            "quality": _round(self.quality),
            "liveness": _round(self.liveness),
            "thresholds": self.thresholds.to_dict(),
        }


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None:
        return None
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


def _finite(value: float | None) -> bool:
    return value is not None and math.isfinite(float(value))


def decide(
    mode: Mode,
    candidates: Sequence[Candidate],
    quality: float,
    liveness: float | None,
    thresholds: Thresholds,
) -> Decision:
    """Apply the three-zone policy.

    ``candidates`` are distinct enrolled subjects with calibrated scores, in any
    order. In ``VERIFY`` mode pass exactly the claimed subject. ``liveness`` is
    ``None`` when the anti-spoofing stage is disabled, which skips step 1.
    """
    mode = Mode(mode)
    if mode is Mode.VERIFY and len(candidates) > 1:
        raise ValueError("verify mode takes exactly one claimed candidate")

    ranked = sorted(candidates, key=lambda c: (_finite(c.score), c.score if _finite(c.score) else 0.0), reverse=True)
    top1 = ranked[0] if ranked else None
    top2 = ranked[1] if len(ranked) > 1 else None
    margin: float | None = None
    if top1 is not None and top2 is not None and _finite(top1.score) and _finite(top2.score):
        margin = float(top1.score) - float(top2.score)

    def result(zone: Zone, primary: Reason, extra: Sequence[Reason] = ()) -> Decision:
        return Decision(
            zone=zone,
            reason=primary,
            reasons=(primary, *[r for r in extra if r is not primary]),
            mode=mode,
            subject=top1.subject if (zone is Zone.MATCH_CONFIRMED and top1) else None,
            top1=top1,
            top2=top2,
            margin=margin,
            quality=float(quality) if _finite(quality) else float("nan"),
            liveness=float(liveness) if _finite(liveness) else None,
            thresholds=thresholds,
        )

    # Fail safe on corrupt numerics before any threshold comparison: NaN
    # compares False against everything and would otherwise fall through to a
    # spurious NO_MATCH or MATCH.
    scores_ok = all(_finite(c.score) for c in ranked)
    if not _finite(quality) or not scores_ok or (liveness is not None and not _finite(liveness)):
        return result(Zone.UNCERTAIN_REVIEW, Reason.INVALID_SCORE)

    # 1. Presentation attack.
    if liveness is not None and liveness < thresholds.l_min:
        return result(Zone.SPOOF_DETECTED, Reason.LIVENESS_BELOW_MINIMUM)

    narrow = mode is Mode.IDENTIFY and margin is not None and margin < thresholds.margin
    gray = top1 is not None and thresholds.tau_low <= top1.score < thresholds.tau_high

    # 2. Degraded capture: no claim in either direction is trustworthy.
    if quality < thresholds.q_min:
        extra = []
        if gray:
            extra.append(Reason.GRAY_ZONE_SIMILARITY)
        if narrow and top1 is not None and top1.score >= thresholds.tau_low:
            extra.append(Reason.NARROW_CANDIDATE_MARGIN)
        return result(Zone.UNCERTAIN_REVIEW, Reason.LOW_IMAGE_QUALITY, extra)

    # 3. Verified non-match (an empty gallery cannot match anyone).
    if top1 is None:
        return result(Zone.NO_MATCH, Reason.EMPTY_GALLERY)
    if top1.score < thresholds.tau_low:
        return result(Zone.NO_MATCH, Reason.SCORE_BELOW_LOW_THRESHOLD)

    # 4. High score: confirm only if no second candidate is within the margin.
    if top1.score >= thresholds.tau_high:
        if narrow:
            return result(
                Zone.UNCERTAIN_REVIEW, Reason.NARROW_CANDIDATE_MARGIN,
                [] if top1.eligible else [Reason.INCOMPLETE_ENROLLMENT],
            )
        if not top1.eligible:
            return result(Zone.UNCERTAIN_REVIEW, Reason.INCOMPLETE_ENROLLMENT)
        return result(Zone.MATCH_CONFIRMED, Reason.SCORE_ABOVE_HIGH_THRESHOLD)

    # 5. Gray band.
    return result(
        Zone.UNCERTAIN_REVIEW,
        Reason.GRAY_ZONE_SIMILARITY,
        [Reason.NARROW_CANDIDATE_MARGIN] if narrow else [],
    )


def decide_baseline(
    mode: Mode,
    candidates: Sequence[tuple[str, float]],
    cosine_threshold: float,
    thresholds: Thresholds,
) -> Decision:
    """Naive comparison policy used by the A/B view.

    One raw cosine threshold, forced top-1, no quality gate, no liveness gate,
    no ambiguity margin and no abstention. ``candidates`` are
    ``(subject, raw_cosine)`` pairs. It exists to make the failure modes of the
    conventional design visible next to the fair engine; it is never used to
    produce an audited decision.
    """
    mode = Mode(mode)
    ranked = sorted(
        (Candidate(subject=s, score=float(c), cosine=float(c)) for s, c in candidates if _finite(c)),
        key=lambda c: c.score,
        reverse=True,
    )
    top1 = ranked[0] if ranked else None
    top2 = ranked[1] if len(ranked) > 1 else None
    margin = (top1.score - top2.score) if (top1 and top2) else None
    matched = top1 is not None and top1.score >= cosine_threshold
    if top1 is None:
        reason = Reason.EMPTY_GALLERY
    elif matched:
        reason = Reason.SCORE_ABOVE_HIGH_THRESHOLD
    else:
        reason = Reason.SCORE_BELOW_LOW_THRESHOLD
    return Decision(
        zone=Zone.MATCH_CONFIRMED if matched else Zone.NO_MATCH,
        reason=reason,
        reasons=(reason,),
        mode=mode,
        subject=top1.subject if matched and top1 else None,
        top1=top1,
        top2=top2,
        margin=margin,
        quality=float("nan"),
        liveness=None,
        thresholds=thresholds,
    )
