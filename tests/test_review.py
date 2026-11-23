"""Human review queue: case grouping and review decisions in the audit chain."""
from __future__ import annotations

from backend.storage.review import build_cases

THRESHOLDS = {"tau_high": 0.78, "tau_low": 0.48, "margin": 0.08, "q_min": 0.4, "l_min": 0.5}


def _record(audit, zone, ts, *, subject="Ana", source="ws:local", reason="GRAY_ZONE_SIMILARITY", mode="identify"):
    return audit.record(
        mode=mode, decision_zone=zone, abstention_reason=reason, quality_score=0.8, liveness_score=0.9,
        top1_subject=subject, top1_score=0.6, top2_score=None, margin=None, thresholds=THRESHOLDS,
        source=source, timestamp=ts,
    )


def _entry(seq, zone, ts, **kw):
    base = {"seq": seq, "mode": "identify", "decision_zone": zone, "timestamp": ts, "source": "ws:local",
            "top1_subject": "Ana", "claimed_subject": None, "abstention_reason": "GRAY_ZONE_SIMILARITY",
            "top1_score": 0.5, "quality_score": 0.8, "liveness_score": 0.9}
    base.update(kw)
    return base


class TestBuildCases:
    def test_repeated_stream_entries_form_one_case(self):
        entries = [
            _entry(1, "UNCERTAIN_REVIEW", "2026-11-14T10:00:00.000Z"),
            _entry(2, "UNCERTAIN_REVIEW", "2026-11-14T10:00:02.000Z", top1_score=0.7),
            _entry(3, "UNCERTAIN_REVIEW", "2026-11-14T10:00:04.000Z"),
        ]
        cases = build_cases(entries)
        assert len(cases) == 1
        assert cases[0]["case_id"] == 1 and cases[0]["last_seq"] == 3
        assert cases[0]["sightings"] == 3 and cases[0]["top1_score"] == 0.7
        assert cases[0]["status"] == "open"

    def test_a_long_visit_is_one_case_and_a_30_minute_gap_starts_another(self):
        entries = [
            _entry(1, "UNCERTAIN_REVIEW", "2026-11-14T13:10:00.000Z"),
            # Still there 25 minutes later: the same visit, whatever the outcome.
            _entry(2, "UNCERTAIN_REVIEW", "2026-11-14T13:35:00.000Z"),
            _entry(3, "SPOOF_DETECTED", "2026-11-14T13:35:01.000Z", abstention_reason="LIVENESS_BELOW_MINIMUM"),
            # Seen again more than 30 minutes after the last sighting: a new case.
            _entry(4, "UNCERTAIN_REVIEW", "2026-11-14T14:10:00.000Z"),
            # Another camera, and another person, each get their own case.
            _entry(5, "UNCERTAIN_REVIEW", "2026-11-14T14:10:01.000Z", source="ws:remote"),
            _entry(6, "UNCERTAIN_REVIEW", "2026-11-14T14:10:02.000Z", top1_subject="Dev"),
        ]
        cases = build_cases(entries)
        assert [c["case_id"] for c in cases] == [6, 5, 4, 1]
        visit = cases[-1]
        assert visit["sightings"] == 3 and visit["seqs"] == [1, 2, 3]
        assert (visit["first_seen"], visit["last_seen"]) == ("2026-11-14T13:10:00.000Z", "2026-11-14T13:35:01.000Z")
        # A third or more of the visit flagged as a spoof makes it a spoof case.
        assert visit["zone"] == "SPOOF_DETECTED" and visit["reason"] == "LIVENESS_BELOW_MINIMUM"
        assert {c["zone"]: c["count"] for c in visit["concerns"]} == {"UNCERTAIN_REVIEW": 2, "SPOOF_DETECTED": 1}

    def test_each_likely_person_gets_one_case(self):
        guesses = ["Dev", "Adam", "Dev", "Yousif", "Dev", "Adam"]
        entries = [_entry(i + 1, "UNCERTAIN_REVIEW", f"2026-11-14T10:00:{2 * i:02d}.000Z", top1_subject=name,
                          abstention_reason="LOW_IMAGE_QUALITY" if i else "GRAY_ZONE_SIMILARITY")
                   for i, name in enumerate(guesses)]
        cases = build_cases(entries)
        assert {c["candidate"]: c["sightings"] for c in cases} == {"Dev": 3, "Adam": 2, "Yousif": 1}
        dev = next(c for c in cases if c["candidate"] == "Dev")
        assert dev["reason"] == "LOW_IMAGE_QUALITY"  # the most common concern
        assert dev["alert"] is None and dev["candidates"] == [{"subject": "Dev", "frames": 3, "banned": False}]

    def test_frames_with_a_stored_picture_come_first(self):
        qualities = [0.9, 0.2, 0.8, 0.1]
        entries = [_entry(i + 1, "UNCERTAIN_REVIEW", f"2026-11-14T10:00:{i:02d}.000Z", quality_score=q)
                   for i, q in enumerate(qualities)]
        (case,) = build_cases(entries, with_image={2, 4})
        assert [f["seq"] for f in case["frames"]] == [2, 4, 1, 3]

    def test_frames_are_the_ten_clearest(self):
        qualities = [0.30, 0.90, 0.10, 0.55, 0.80, 0.20, 0.70, 0.60, 0.40, 0.95, 0.05, 0.50, 0.85, 0.15]
        entries = [_entry(i + 1, "UNCERTAIN_REVIEW", f"2026-11-14T10:00:{i:02d}.000Z", quality_score=q)
                   for i, q in enumerate(qualities)]
        (case,) = build_cases(entries)
        shown = [f["quality"] for f in case["frames"]]
        assert len(shown) == 10 and shown == sorted(qualities, reverse=True)[:10]
        assert case["frames"][0]["seq"] == 10 and case["quality_score"] == 0.95
        assert case["sightings"] == 14 and len(case["seqs"]) == 14

    def test_banned_person_raises_a_case_even_when_matched(self):
        entries = [
            _entry(1, "MATCH_CONFIRMED", "2026-11-14T10:00:00.000Z", top1_subject="Dev",
                   abstention_reason="SCORE_ABOVE_HIGH_THRESHOLD"),
            _entry(2, "MATCH_CONFIRMED", "2026-11-14T10:05:00.000Z", top1_subject="Yousif", asserted_subject="Yousif",
                   abstention_reason="SCORE_ABOVE_HIGH_THRESHOLD"),
            _entry(3, "NO_MATCH", "2026-11-14T10:05:02.000Z", top1_subject="Yousif"),
        ]
        assert build_cases(entries) == []  # nobody is banned: ordinary matches are not cases
        (case,) = build_cases(entries, banned={"Yousif"})
        assert case["case_id"] == 2 and case["alert"] == "banned" and case["zone"] == "MATCH_CONFIRMED"
        assert case["candidates"] == [{"subject": "Yousif", "frames": 1, "banned": True}]

    def test_one_or_two_unclear_guesses_of_a_banned_person_are_not_an_alert(self):
        entries = [
            _entry(1, "UNCERTAIN_REVIEW", "2026-11-14T10:00:00.000Z", top1_subject="Yousif"),
            _entry(2, "UNCERTAIN_REVIEW", "2026-11-14T10:00:02.000Z", top1_subject="Yousif"),
        ]
        (case,) = build_cases(entries, banned={"Yousif"})
        assert case["alert"] is None and case["candidate"] == "Yousif"
        assert case["candidates"] == [{"subject": "Yousif", "frames": 2, "banned": True}]

    def test_banned_alert_when_confirmed_or_named_in_several_frames(self):
        unsure = [_entry(i + 1, "UNCERTAIN_REVIEW", f"2026-11-14T10:00:0{2 * i}.000Z", top1_subject="Yousif")
                  for i in range(3)]
        (case,) = build_cases(unsure, banned={"Yousif"})
        assert case["alert"] == "banned" and case["candidate"] == "Yousif"

        # One confirmed match is enough.
        confirmed = [
            _entry(1, "UNCERTAIN_REVIEW", "2026-11-14T10:00:00.000Z", top1_subject="Yousif"),
            _entry(2, "MATCH_CONFIRMED", "2026-11-14T10:00:04.000Z", top1_subject="Yousif", asserted_subject="Yousif"),
        ]
        (case,) = build_cases(confirmed, banned={"Yousif"})
        assert case["alert"] == "banned" and case["zone"] == "UNCERTAIN_REVIEW"

    def test_sightings_after_a_decision_stay_with_the_decided_case(self):
        decision = {"seq": 3, "mode": "review", "decision_zone": "REVIEW_SAME_PERSON", "abstention_reason": "case:1",
                    "top1_subject": "Ana", "timestamp": "2026-11-14T10:00:03.000Z", "hash": "h"}
        entries = [
            _entry(1, "UNCERTAIN_REVIEW", "2026-11-14T10:00:00.000Z"),
            _entry(2, "UNCERTAIN_REVIEW", "2026-11-14T10:00:02.000Z"),
            decision,
            # Still at the camera: no new case for the guard.
            _entry(4, "UNCERTAIN_REVIEW", "2026-11-14T10:10:00.000Z"),
        ]
        (case,) = build_cases(entries)
        assert (case["case_id"], case["status"], case["seqs"]) == (1, "resolved", [1, 2, 4])

        # Something the reviewer did not see (a spoof), or a return after 30 minutes, is a new case.
        spoof = entries + [_entry(5, "SPOOF_DETECTED", "2026-11-14T10:12:00.000Z")]
        assert [(c["case_id"], c["status"]) for c in build_cases(spoof)] == [(5, "open"), (1, "resolved")]
        later = entries + [_entry(5, "UNCERTAIN_REVIEW", "2026-11-14T10:45:00.000Z")]
        assert [(c["case_id"], c["status"]) for c in build_cases(later)] == [(5, "open"), (1, "resolved")]

    def test_matches_no_matches_and_admin_entries_are_not_cases(self):
        entries = [
            _entry(1, "MATCH_CONFIRMED", "2026-11-14T10:00:00.000Z"),
            _entry(2, "NO_MATCH", "2026-11-14T10:00:01.000Z"),
            _entry(3, "SUBJECT_ENROLLED", "2026-11-14T10:00:02.000Z", mode="admin"),
        ]
        assert build_cases(entries) == []

    def test_review_entry_resolves_its_case(self):
        entries = [
            _entry(1, "UNCERTAIN_REVIEW", "2026-11-14T10:00:00.000Z"),
            {"seq": 2, "mode": "review", "decision_zone": "REVIEW_SAME_PERSON", "abstention_reason": "case:1",
             "top1_subject": "Ana", "timestamp": "2026-11-14T10:00:30.000Z", "hash": "h"},
        ]
        (case,) = build_cases(entries)
        assert case["status"] == "resolved"
        assert case["review"]["outcome"] == "same_person" and case["review"]["subject"] == "Ana"


class TestReviewApi:
    def test_list_and_resolve_a_case(self, client):
        audit = client.app.state.ffg.audit
        first = _record(audit, "UNCERTAIN_REVIEW", "2026-11-14T10:00:00.000Z")
        _record(audit, "UNCERTAIN_REVIEW", "2026-11-14T10:00:02.000Z")
        spoof = _record(audit, "SPOOF_DETECTED", "2026-11-14T10:05:00.000Z", subject="Ben",
                        reason="LIVENESS_BELOW_MINIMUM")

        listing = client.get("/api/review/cases?status=open").json()
        assert listing["open"] == 2
        assert [c["case_id"] for c in listing["cases"]] == [spoof["seq"], first["seq"]]

        decided = client.post(f"/api/review/cases/{first['seq']}", json={"outcome": "same_person"})
        assert decided.status_code == 200, decided.text
        assert decided.json()["review"]["subject"] == "Ana"

        again = client.post(f"/api/review/cases/{first['seq']}", json={"outcome": "different_person"})
        assert again.status_code == 409

        listing = client.get("/api/review/cases").json()
        assert listing["open"] == 1
        resolved = next(c for c in listing["cases"] if c["case_id"] == first["seq"])
        assert resolved["status"] == "resolved"

        # The decision is part of the tamper-evident chain.
        assert client.get("/api/audit/verify").json()["ok"] is True
        newest = client.get("/api/audit?limit=1").json()["entries"][0]
        assert newest["mode"] == "review" and newest["decision_zone"] == "REVIEW_SAME_PERSON"
        assert newest["abstention_reason"] == f"case:{first['seq']}"

    def test_snapshot_is_kept_until_the_case_is_decided(self, client):
        from tests.test_pipeline import enroll, identify, primary

        vault = client.app.state.ffg.vault
        enroll(client, "Alice", 11)
        assert primary(identify(client, 11))["zone"] == "MATCH_CONFIRMED"
        assert vault.snapshot_seqs() == set()  # an ordinary match keeps no picture

        unsure = identify(client, 11, quality=70).json()
        assert unsure["faces"][0]["decision"]["zone"] == "UNCERTAIN_REVIEW"
        seq = unsure["audit"][0]["seq"]
        assert vault.snapshot_seqs() == {seq}

        (case,) = client.get("/api/review/cases?status=open").json()["cases"]
        assert case["case_id"] == seq and "seqs" not in case
        assert case["frames"] == [dict(case["frames"][0], seq=seq, image=True)]
        frame = client.get(f"/api/review/frames/{seq}")
        assert frame.status_code == 200 and frame.headers["content-type"] == "image/jpeg"
        assert frame.content[:2] == b"\xff\xd8"

        decided = client.post(f"/api/review/cases/{seq}", json={"outcome": "same_person"})
        assert decided.status_code == 200 and "seqs" not in decided.json()
        assert vault.snapshot_seqs() == set()
        assert client.get(f"/api/review/frames/{seq}").status_code == 404
        (resolved,) = client.get("/api/review/cases").json()["cases"]
        assert resolved["status"] == "resolved" and resolved["frames"][0]["image"] is False

    def test_snapshots_expire_and_can_be_turned_off(self, client, monkeypatch):
        from tests.test_pipeline import enroll, identify

        state = client.app.state.ffg
        enroll(client, "Alice", 11)
        seq = identify(client, 11, liveness=40).json()["audit"][0]["seq"]
        assert state.vault.snapshot_seqs() == {seq}
        # Listing the queue drops snapshots older than the retention limit.
        monkeypatch.setattr(state, "settings", state.settings.__class__(
            **{**state.settings.__dict__, "review_snapshot_hours": 0.0}))
        (case,) = client.get("/api/review/cases?status=open").json()["cases"]
        assert case["frames"][0]["image"] is False and state.vault.snapshot_seqs() == set()
        # With retention at zero no new snapshot is kept either.
        identify(client, 11, liveness=40)
        assert state.vault.snapshot_seqs() == set()

    def test_ban_list_and_banned_sighting(self, client):
        from tests.conftest import frame_bytes
        from tests.test_pipeline import enroll, identify, primary

        enroll(client, "Dev", 11)
        enroll(client, "Yousif", 42)
        assert client.post("/api/subjects/Nobody/ban", data={"note": "x"}).status_code == 404

        banned = client.post("/api/subjects/Yousif/ban", data={"note": "  Trespass notice,   14 Nov "},
                             files={"photo": ("record.png", frame_bytes(42), "image/png")})
        assert banned.status_code == 200, banned.text
        subjects = {s["subject"]: s for s in client.get("/api/subjects").json()["subjects"]}
        assert subjects["Yousif"]["banned"] is True and subjects["Yousif"]["banned_photo"] is True
        assert subjects["Yousif"]["banned_note"] == "Trespass notice, 14 Nov"
        assert subjects["Dev"]["banned"] is False and subjects["Dev"]["banned_note"] is None
        photo = client.get("/api/subjects/Yousif/record-photo")
        assert photo.status_code == 200 and photo.content[:2] == b"\xff\xd8"
        assert client.get("/api/subjects/Dev/record-photo").status_code == 404

        # Updating the note keeps the record photo.
        assert client.post("/api/subjects/Yousif/ban", data={"note": "Updated"}).status_code == 200
        assert client.get("/api/subjects/Yousif/record-photo").status_code == 200

        # An employee walking in is not a case; the banned person is, with a snapshot.
        assert primary(identify(client, 11))["zone"] == "MATCH_CONFIRMED"
        sighting = identify(client, 42).json()
        assert sighting["faces"][0]["decision"]["zone"] == "MATCH_CONFIRMED"
        (case,) = client.get("/api/review/cases?status=open").json()["cases"]
        assert case["alert"] == "banned" and case["candidate"] == "Yousif"
        assert case["banned"] == [{"subject": "Yousif", "note": "Updated", "photo": True}]
        assert case["frames"][0]["image"] is True
        assert client.get(f"/api/review/frames/{sighting['audit'][0]['seq']}").status_code == 200

        # Banning and lifting are part of the tamper-evident log.
        assert client.delete("/api/subjects/Yousif/ban").json() == {"subject": "Yousif", "banned": False}
        assert client.get("/api/subjects/Yousif/record-photo").status_code == 404
        assert client.get("/api/review/cases?status=open").json()["cases"] == []
        zones = [e["decision_zone"] for e in client.get("/api/audit?limit=50").json()["entries"]]
        assert zones.count("SUBJECT_BANNED") == 2 and zones.count("SUBJECT_UNBANNED") == 1
        assert client.get("/api/audit/verify").json()["ok"] is True

    def test_removing_a_person_removes_their_ban(self, client):
        from tests.test_pipeline import enroll

        enroll(client, "Yousif", 42)
        assert client.post("/api/subjects/Yousif/ban", data={"note": ""}).status_code == 200
        assert client.app.state.ffg.vault.banned() == {"Yousif": {"note": None, "has_photo": False,
                                                                  "since": client.app.state.ffg.vault.banned()["Yousif"]["since"]}}
        assert client.delete("/api/subjects/Yousif").status_code == 200
        assert client.app.state.ffg.vault.banned() == {}

    def test_unknown_case_and_bad_outcome_are_rejected(self, client):
        assert client.post("/api/review/cases/999", json={"outcome": "spoof"}).status_code == 404
        audit = client.app.state.ffg.audit
        case = _record(audit, "SPOOF_DETECTED", "2026-11-14T11:00:00.000Z")
        assert client.post(f"/api/review/cases/{case['seq']}", json={"outcome": "maybe"}).status_code == 422

    def test_clearing_the_queue_dismisses_open_cases_and_is_logged(self, client):
        audit = client.app.state.ffg.audit
        decided = _record(audit, "UNCERTAIN_REVIEW", "2026-11-14T10:00:00.000Z", subject="Ana")
        open_one = _record(audit, "UNCERTAIN_REVIEW", "2026-11-14T10:00:02.000Z", subject="Ben")
        _record(audit, "SPOOF_DETECTED", "2026-11-14T10:00:04.000Z", subject="Cy", reason="LIVENESS_BELOW_MINIMUM")
        assert client.post(f"/api/review/cases/{decided['seq']}", json={"outcome": "same_person"}).status_code == 200

        cleared = client.post("/api/review/clear")
        assert cleared.status_code == 200 and cleared.json() == {"cleared": 2}
        listing = client.get("/api/review/cases").json()
        assert listing["open"] == 0
        # The decided case is still there; the cleared ones are gone from the queue.
        assert [(c["case_id"], c["status"]) for c in listing["cases"]] == [(decided["seq"], "resolved")]
        # A cleared case can no longer be decided, and clearing an empty queue logs nothing.
        assert client.post(f"/api/review/cases/{open_one['seq']}", json={"outcome": "spoof"}).status_code == 409
        before = client.get("/api/audit?limit=1").json()
        assert before["entries"][0]["mode"] == "admin" and before["entries"][0]["decision_zone"] == "REVIEW_QUEUE_CLEARED"
        assert client.post("/api/review/clear").json() == {"cleared": 0}
        assert client.get("/api/audit?limit=1").json()["total"] == before["total"]
        # Nothing was deleted from the tamper-evident log.
        assert client.get("/api/audit/verify").json()["ok"] is True

        # The same person seen again right after the clear is a new case.
        again = _record(audit, "UNCERTAIN_REVIEW", "2026-11-14T10:00:06.000Z", subject="Ben")
        listing = client.get("/api/review/cases?status=open").json()
        assert listing["open"] == 1 and listing["cases"][0]["case_id"] == again["seq"]


class TestClearingTheQueue:
    CLEAR = {"mode": "admin", "decision_zone": "REVIEW_QUEUE_CLEARED", "abstention_reason": "cases=1",
             "timestamp": "2026-11-14T10:00:10.000Z"}

    def test_clear_dismisses_open_cases_but_not_resolved_ones(self):
        entries = [
            _entry(1, "UNCERTAIN_REVIEW", "2026-11-14T10:00:00.000Z", top1_subject="Ana"),
            {"seq": 2, "mode": "review", "decision_zone": "REVIEW_SAME_PERSON", "abstention_reason": "case:1",
             "top1_subject": "Ana", "timestamp": "2026-11-14T10:00:03.000Z", "hash": "h"},
            _entry(3, "UNCERTAIN_REVIEW", "2026-11-14T10:00:04.000Z", top1_subject="Ben"),
            dict(self.CLEAR, seq=4),
        ]
        assert [(c["case_id"], c["status"]) for c in build_cases(entries)] == [(3, "dismissed"), (1, "resolved")]

    def test_sightings_after_a_clear_start_a_new_open_case(self):
        entries = [
            _entry(1, "UNCERTAIN_REVIEW", "2026-11-14T10:00:00.000Z"),
            _entry(2, "UNCERTAIN_REVIEW", "2026-11-14T10:00:02.000Z"),
            dict(self.CLEAR, seq=3),
            # Two seconds later, well inside the 30-minute window: still a fresh case.
            _entry(4, "UNCERTAIN_REVIEW", "2026-11-14T10:00:12.000Z"),
            # A late decision aimed at the cleared case changes nothing.
            {"seq": 5, "mode": "review", "decision_zone": "REVIEW_SPOOF_CONFIRMED", "abstention_reason": "case:1",
             "top1_subject": None, "timestamp": "2026-11-14T10:00:13.000Z", "hash": "h"},
        ]
        new, old = build_cases(entries)
        assert (old["case_id"], old["status"], old["seqs"]) == (1, "dismissed", [1, 2])
        assert (new["case_id"], new["status"], new["seqs"]) == (4, "open", [4])
