"""Employee onboarding: check-ins on separate days build up one template."""
from __future__ import annotations

import numpy as np

from backend.storage import vault as vault_module
from backend.storage.vault import BiometricVault


def unit(seed: int, dim: int = 512) -> np.ndarray:
    v = np.random.default_rng(seed).standard_normal(dim).astype(np.float32)
    return v / np.linalg.norm(v)


def test_check_in_days_count_distinct_capture_dates(tmp_path, monkeypatch):
    vault = BiometricVault(tmp_path / "b.db")
    days = iter(["2026-11-09T09:00:00.000Z", "2026-11-09T17:30:00.000Z", "2026-11-10T09:05:00.000Z",
                 "2026-11-12T08:55:00.000Z"])
    for seed in range(4):
        monkeypatch.setattr(vault_module, "_now", lambda value=next(days): value)
        info = vault.add_samples("Jordan", [unit(seed), unit(seed + 10), unit(seed + 20)], max_samples=15)
    assert info.samples == 12
    assert info.check_in_days == 3  # two captures on the 9th count as one day
    assert vault.list_subjects()[0].to_dict()["check_in_days"] == 3


def test_subjects_api_reports_onboarding_progress(client):
    from tests.test_pipeline import enroll

    enroll(client, "Alice", 11, frames=3)
    config = client.get("/api/config").json()["enrollment"]
    assert config["onboarding_days"] == 5 and config["retain_samples"] == 15
    subject = client.get("/api/subjects").json()["subjects"][0]
    assert subject["check_in_days"] == 1 and subject["onboarding_complete"] is False
