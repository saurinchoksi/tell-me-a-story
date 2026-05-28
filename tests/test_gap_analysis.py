"""Unit tests for the pure interval logic in scripts/gap_analysis.py.

The Moon Story reproduction (all 90 original gaps; >=1.0s floor of 4 / 9.7s) is
the integration test, run via the script itself. These cover the interval math
that underpins it — the part most likely to break under a future edit.

scripts/ isn't on pytest's pythonpath (which is [".", "src"]), so add it here.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from gap_analysis import merge_intervals, subtract_intervals, mmss  # noqa: E402


def test_merge_intervals():
    # overlapping spans collapse
    assert merge_intervals([(0, 1), (0.5, 2)]) == [(0, 2)]
    # touching spans (end == next start) merge into one
    assert merge_intervals([(0, 1), (1, 2)]) == [(0, 2)]
    # a clear gap is preserved; input order doesn't matter
    assert merge_intervals([(3, 4), (0, 1)]) == [(0, 1), (3, 4)]
    # zero-length and inverted spans are dropped
    assert merge_intervals([(1, 1), (2, 1), (0, 0.5)]) == [(0, 0.5)]
    assert merge_intervals([]) == []


def test_subtract_intervals_basic():
    # carve covered regions out of a segment, clipping at both ends
    assert subtract_intervals((0, 5), [(1, 2), (3, 3.5)]) == [(0, 1), (2, 3), (3.5, 5)]
    # fully covered -> nothing left
    assert subtract_intervals((1, 2), [(0, 3)]) == []
    # no coverage -> whole segment
    assert subtract_intervals((1, 2), []) == [(1, 2)]


def test_subtract_intervals_edges():
    # coverage that only touches the segment edges leaves the middle
    assert subtract_intervals((1, 4), [(0, 1), (4, 5)]) == [(1, 4)]
    # coverage starting before and ending inside clips the front
    assert subtract_intervals((2, 6), [(0, 3)]) == [(3, 6)]
    # the Moon Story shape: a diarization span ending past the last covered
    # segment leaves exactly the uncovered tail (e.g. 6.10-6.58 = 0.48s)
    tail = subtract_intervals((4.82, 6.58), merge_intervals([(4.48, 6.10), (7.72, 16.48)]))
    assert tail == [(6.10, 6.58)]
    assert round(tail[0][1] - tail[0][0], 2) == 0.48


def test_mmss():
    assert mmss(0) == "0:00.00"
    assert mmss(394.14) == "6:34.14"
    assert mmss(9.7) == "0:09.70"
