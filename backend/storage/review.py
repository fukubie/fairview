"""Human review queue built on top of the audit log.

Every ``UNCERTAIN_REVIEW`` or ``SPOOF_DETECTED`` decision, and every sighting
of someone on the banned-from-premises list, is something a person (e.g. a
security guard) should look at. A live stream logs a decision every couple of
seconds, so one moment in front of a camera produces many entries, often with
different scores as the person moves. All reviewable entries for the same
person on the same camera form ONE case for as long as they keep coming: a new
case starts only after ``GROUP_GAP_SECONDS`` (30 minutes) without one, because
by then enough could have happened to look again. Someone who stays in front
of the camera for an hour is one case, not hundreds. The guard sees the most
common concern and the clearest frames, and decides once; further sightings in
the same visit stay with that decided case.

A case is identified by the ``seq`` of its first audit entry. A reviewer's
decision is itself appended to the audit hash chain as an entry with
``mode = "review"``, so it is tamper-evident like every other decision.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Any, Iterable

REVIEWABLE_ZONES = ("UNCERTAIN_REVIEW", "SPOOF_DETECTED")
REVIEWABLE_MODES = ("identify", "verify")
# A banned person is worth a look even when the system is sure who it is.
BANNED_ZONES = ("MATCH_CONFIRMED", "UNCERTAIN_REVIEW")

OUTCOMES = {
    "same_person": "REVIEW_SAME_PERSON",
    "different_person": "REVIEW_DIFFERENT_PERSON",
    "spoof": "REVIEW_SPOOF_CONFIRMED",
}
OUTCOME_OF_ZONE = {zone: key for key, zone in OUTCOMES.items()}

# Sightings of one person on one camera closer together than this are one case.
GROUP_GAP_SECONDS = 30 * 60.0
# The clearest frames kept per case for the reviewer.
FRAMES_PER_CASE = 10
# Share of a moment's entries flagged as spoofs that makes the case a spoof case.
SPOOF_SHARE = 1 / 3
MAX_CANDIDATES = 5
# Unclear frames naming a banned person before that counts as a banned-list alert.
BANNED_MIN_SIGHTINGS = 3
CASE_REF_PREFIX = "case:"
# Admin audit event that dismisses every case still open at that point.
QUEUE_CLEARED = "REVIEW_QUEUE_CLEARED"


def _ts(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def case_ref(case_id: int) -> str:
    return f"{CASE_REF_PREFIX}{int(case_id)}"


def parse_case_ref(value: Any) -> int | None:
    if isinstance(value, str) and value.startswith(CASE_REF_PREFIX):
        try:
            return int(value[len(CASE_REF_PREFIX):])
        except ValueError:
            return None
    return None


def _banned_subject(entry: dict[str, Any], banned: frozenset[str] | set[str]) -> str | None:
    for key in ("asserted_subject", "top1_subject"):
        name = entry.get(key)
        if name and name in banned:
            return name
    return None


def is_reviewable(entry: dict[str, Any], banned: frozenset[str] | set[str] = frozenset()) -> bool:
    if entry.get("mode") not in REVIEWABLE_MODES:
        return False
    zone = entry.get("decision_zone")
    if zone in REVIEWABLE_ZONES:
        return True
    return zone in BANNED_ZONES and _banned_subject(entry, banned) is not None


def _quality_key(entry: dict[str, Any]) -> tuple:
    q = entry.get("quality_score")
    s = entry.get("top1_score")
    return (q if isinstance(q, (int, float)) else -1.0, s if isinstance(s, (int, float)) else -1.0)


def _frame(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "seq": int(entry["seq"]),
        "timestamp": entry.get("timestamp"),
        "zone": entry.get("decision_zone"),
        "reason": entry.get("abstention_reason"),
        "candidate": entry.get("top1_subject"),
        "score": entry.get("top1_score"),
        "quality": entry.get("quality_score"),
        "liveness": entry.get("liveness_score"),
    }


def _summarise(case: dict[str, Any], banned: frozenset[str] | set[str],
               with_image: frozenset[int] = frozenset()) -> dict[str, Any]:
    items: list[dict[str, Any]] = case.pop("_entries")
    zones = Counter(e.get("decision_zone") for e in items)
    spoofs = zones.get("SPOOF_DETECTED", 0)
    if spoofs and spoofs >= SPOOF_SHARE * len(items):
        zone = "SPOOF_DETECTED"
    elif zones.get("UNCERTAIN_REVIEW"):
        zone = "UNCERTAIN_REVIEW"
    else:
        zone = zones.most_common(1)[0][0]
    in_zone = [e for e in items if e.get("decision_zone") == zone] or items
    reasons = Counter(e.get("abstention_reason") for e in in_zone if e.get("abstention_reason"))

    names = Counter(e.get("top1_subject") for e in items if e.get("top1_subject"))
    ranked = [n for n, _ in names.most_common()]
    # A banned-list alert needs more than one unclear frame's best guess: the
    # system either confirmed the banned person, or named them in several
    # frames. Otherwise it is an ordinary "might be" case.
    confirmed = Counter(b for b in (_banned_subject(e, banned) for e in items
                                    if e.get("decision_zone") == "MATCH_CONFIRMED") if b)
    alerted = [n for n, _ in confirmed.most_common()]
    if not alerted and ranked and ranked[0] in banned and zone in BANNED_ZONES \
            and names[ranked[0]] >= BANNED_MIN_SIGHTINGS:
        alerted = [ranked[0]]
    ranked = alerted + [n for n in ranked if n not in alerted]
    candidates = [{"subject": n, "frames": names.get(n, 0) or confirmed.get(n, 0), "banned": n in banned}
                  for n in ranked[:MAX_CANDIDATES]]
    candidate = ranked[0] if ranked else None

    best = sorted(items, key=_quality_key, reverse=True)
    for_candidate = [e for e in best if e.get("top1_subject") == candidate] or best
    lead = for_candidate[0]
    # Frames a reviewer can actually look at come first, clearest first.
    shown = sorted(items, key=lambda e: (int(e["seq"]) in with_image, *_quality_key(e)), reverse=True)
    scores = [e.get("top1_score") for e in for_candidate if isinstance(e.get("top1_score"), (int, float))]

    case.update(
        zone=zone,
        reason=reasons.most_common(1)[0][0] if reasons else lead.get("abstention_reason"),
        candidate=candidate,
        candidates=candidates,
        alert="banned" if alerted else None,
        top1_score=max(scores) if scores else None,
        quality_score=lead.get("quality_score"),
        liveness_score=lead.get("liveness_score"),
        sightings=len(items),
        concerns=[{"zone": z, "count": c} for z, c in zones.most_common()],
        frames=[_frame(e) for e in shown[:FRAMES_PER_CASE]],
        seqs=[int(e["seq"]) for e in items],
    )
    return case


def build_cases(entries: Iterable[dict[str, Any]], banned: Iterable[str] = (),
                with_image: Iterable[int] = ()) -> list[dict[str, Any]]:
    """Group reviewable entries into one case per person and visit, with any review decision.

    ``entries`` may come in any order; the result is newest case first. Each
    case carries ``seqs`` (every audit entry in it), which the API uses to
    manage snapshots and strips before responding. ``with_image`` lists the
    entries that have a stored face crop, so those are offered as frames first.
    """
    banned = frozenset(banned)
    with_image = frozenset(int(s) for s in with_image)
    ordered = sorted(entries, key=lambda e: int(e.get("seq") or 0))
    cases: list[dict[str, Any]] = []
    current: dict[Any, dict[str, Any]] = {}
    by_seq: dict[int, dict[str, Any]] = {}

    for entry in ordered:
        if entry.get("mode") == "admin" and entry.get("decision_zone") == QUEUE_CLEARED:
            # An administrator cleared the queue: everything still undecided is
            # dismissed, and later sightings start fresh cases.
            for case in cases:
                if case["_review"] is None:
                    case["_dismissed"] = True
            current.clear()
            continue
        if entry.get("mode") == "review":
            ref = parse_case_ref(entry.get("abstention_reason"))
            case = by_seq.get(ref) if ref is not None else None
            if case is not None and not case.get("_dismissed"):
                case["_review"] = entry  # the latest decision on a case wins
                case["_decided_zones"] = {e.get("decision_zone") for e in case["_entries"]}
            continue
        if not is_reviewable(entry, banned):
            continue
        source = entry.get("source")
        when = _ts(entry.get("timestamp"))
        # One case per person per camera; faces nobody was close to share one.
        key = (source, entry.get("top1_subject") or entry.get("claimed_subject"))
        case = current.get(key)
        recent = (case is not None and when is not None and case["_last_ts"] is not None
                  and when - case["_last_ts"] <= GROUP_GAP_SECONDS)
        # After a decision the same visit stays with the decided case, unless
        # something new turns up that the reviewer did not see (e.g. a spoof).
        if recent and case.get("_review") is not None \
                and entry.get("decision_zone") not in case["_decided_zones"]:
            recent = False
        if recent:
            case["_entries"].append(entry)
            case["last_seq"] = int(entry["seq"])
            case["last_seen"] = entry.get("timestamp")
            case["_last_ts"] = when
        else:
            case = {
                "case_id": int(entry["seq"]),
                "first_seq": int(entry["seq"]),
                "last_seq": int(entry["seq"]),
                "first_seen": entry.get("timestamp"),
                "last_seen": entry.get("timestamp"),
                "mode": entry.get("mode"),
                "source": source,
                "claimed_subject": entry.get("claimed_subject"),
                "_entries": [entry],
                "_last_ts": when,
                "_review": None,
            }
            current[key] = case
            cases.append(case)
        by_seq[int(entry["seq"])] = case

    for case in cases:
        case.pop("_last_ts", None)
        case.pop("_decided_zones", None)
        review = case.pop("_review", None)
        dismissed = case.pop("_dismissed", False)
        _summarise(case, banned, with_image)
        if dismissed:
            case["status"] = "dismissed"
            case["review"] = None
        elif review is None:
            case["status"] = "open"
            case["review"] = None
        else:
            case["status"] = "resolved"
            case["review"] = {
                "outcome": OUTCOME_OF_ZONE.get(review.get("decision_zone"), "unknown"),
                "subject": review.get("top1_subject"),
                "timestamp": review.get("timestamp"),
                "seq": review.get("seq"),
                "hash": review.get("hash"),
            }
    cases.sort(key=lambda c: c["case_id"], reverse=True)
    return cases
