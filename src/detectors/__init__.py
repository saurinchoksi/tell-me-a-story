"""Failure-mode detector registry.

Detectors are read-only monitors over session transcripts — they emit flags,
never transcript edits. Register new detectors here explicitly.
"""

from detectors.family_names import FamilyNameDetector
from detectors.name_consistency import NameConsistencyDetector

DETECTORS = [FamilyNameDetector(), NameConsistencyDetector()]


def get_detector(detector_id: str):
    for det in DETECTORS:
        if det.id == detector_id:
            return det
    raise ValueError(
        f"Unknown detector: {detector_id!r}. Registered: {[d.id for d in DETECTORS]}"
    )
