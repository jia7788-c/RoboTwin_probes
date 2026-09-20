"""Task-event detectors. Outputs evidence, not causal claims."""
from __future__ import annotations
from dataclasses import dataclass
from math import ceil
from typing import Iterable, Mapping, Any

@dataclass(frozen=True)
class Event:
    name: str
    onset_s: float | None
    confirmed_s: float | None
    confidence: str
    evidence: dict[str, Any]

def seconds_to_samples(seconds: float, sample_hz: float) -> int:
    if seconds < 0 or sample_hz <= 0: raise ValueError("seconds must be nonnegative and sample_hz positive")
    return max(1, ceil(seconds * sample_hz))

def first_stable_grasp(samples: Iterable[Mapping[str, Any]], *, sample_hz: float, stable_seconds: float, max_relative_motion_m: float) -> Event | None:
    window = seconds_to_samples(stable_seconds, sample_hz); run: list[Mapping[str, Any]] = []
    for sample in samples:
        if sample.get("contact") is None or sample.get("relative_motion_m") is None: run = []; continue
        if bool(sample["contact"]) and float(sample["relative_motion_m"]) <= max_relative_motion_m: run.append(sample)
        else: run = []
        if len(run) >= window:
            onset, confirmed = run[-window], run[-1]
            return Event("stable_grasp", float(onset["time_s"]), float(confirmed["time_s"]), "rule_based", {"window_samples": window})
    return None

def handover_timing(donor_release: Event | None, receiver_grasp: Event | None, *, early_margin_s: float) -> dict[str, Any]:
    if donor_release is None or receiver_grasp is None or donor_release.onset_s is None or receiver_grasp.onset_s is None:
        return {"delta_t_s": None, "early_release_candidate": None, "availability": False}
    delta = donor_release.onset_s - receiver_grasp.onset_s
    return {"delta_t_s": delta, "early_release_candidate": delta < -early_margin_s, "availability": True}
