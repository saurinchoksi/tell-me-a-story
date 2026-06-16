"""Offline per-story name auditor — M9b (improvised inconsistency) + M9c (sourced
canon mistranscription), one local-Gemma pass per story. Graduated from the sealed
EMP Stage-1 v2 probe; the validated logic is ported verbatim into the private
_segment/_audit/_names modules (the family_names.py precedent)."""
from detectors.story_names.detector import StoryNameDetector

__all__ = ["StoryNameDetector"]
