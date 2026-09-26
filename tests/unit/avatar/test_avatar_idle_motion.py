"""IdleMotion: deterministic, bounded, breathing, saccades, suppressed while speaking."""

from __future__ import annotations

import itertools
import math

from aivtube.avatar import HEAD_PARAMS, IdleMotion


def _series(motion: IdleMotion, key: str, *, speaking: float = 0.0, seconds: float = 20.0):
    return [motion.sample(i / 60, 0.0, speaking=speaking)[key] for i in range(int(seconds * 60))]


def _rms(xs: list[float]) -> float:
    mean = sum(xs) / len(xs)
    return math.sqrt(sum((x - mean) ** 2 for x in xs) / len(xs))


def test_deterministic_per_seed_and_complete() -> None:
    a, b, c = IdleMotion(1), IdleMotion(1), IdleMotion(2)
    sa = [a.sample(i / 60) for i in range(300)]
    assert sa == [b.sample(i / 60) for i in range(300)]
    assert sa != [c.sample(i / 60) for i in range(300)]
    assert set(sa[0]) == set(HEAD_PARAMS)


def test_values_stay_in_gentle_ranges() -> None:
    m = IdleMotion(3)
    for i in range(60 * 60):
        v = m.sample(i / 60, speech_level=1.0 if i % 120 < 60 else 0.0)
        assert abs(v["FaceAngleX"]) <= 8.5
        assert abs(v["FaceAngleY"]) <= 7.0
        assert abs(v["FaceAngleZ"]) <= 4.5
        assert abs(v["FacePositionY"]) <= 0.61
        for eye in ("EyeLeftX", "EyeLeftY", "EyeRightX", "EyeRightY"):
            assert -0.36 <= v[eye] <= 0.36
        assert v["EyeLeftX"] == v["EyeRightX"]


def test_breathing_bob_is_periodic() -> None:
    ys = _series(IdleMotion(0), "FacePositionY", seconds=20.0)
    crossings = sum(1 for a, b in itertools.pairwise(ys) if a < 0 <= b)
    assert 3 <= crossings <= 6  # ~0.22 Hz over 20 s


def test_saccades_move_the_gaze() -> None:
    xs = _series(IdleMotion(0), "EyeLeftX", seconds=30.0)
    assert len({round(x, 6) for x in xs}) >= 8


def test_sway_is_suppressed_while_speaking() -> None:
    quiet = _rms(_series(IdleMotion(5), "FaceAngleX", speaking=0.0))
    talking = _rms(_series(IdleMotion(5), "FaceAngleX", speaking=1.0))
    assert talking < 0.5 * quiet


def test_speech_level_adds_small_nods() -> None:
    m1, m2 = IdleMotion(9), IdleMotion(9)
    diffs = [
        abs(
            m1.sample(i / 60, 1.0, speaking=1.0)["FaceAngleY"]
            - m2.sample(i / 60, 0.0, speaking=1.0)["FaceAngleY"]
        )
        for i in range(120)
    ]
    assert 0.5 < max(diffs) <= 3.0 + 1e-9
