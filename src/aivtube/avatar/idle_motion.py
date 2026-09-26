"""Procedural "alive" motion for a model with no face tracker (avatar brief §2).

Slow sums of sines for head sway, a gentle breathing bob, random saccades and small nods that
follow the speech level. Sway, breathing and saccades are suppressed while she speaks so the
head settles and looks at the viewer. Values are VTS default input parameters (degrees for
``FaceAngle*``, VTS units for ``FacePosition*``, -1..1 for eye gaze). Deterministic per seed.
"""

from __future__ import annotations

import math
import random

__all__ = ["HEAD_PARAMS", "IdleMotion"]

HEAD_PARAMS: tuple[str, ...] = (
    "FaceAngleX",
    "FaceAngleY",
    "FaceAngleZ",
    "FacePositionX",
    "FacePositionY",
    "EyeLeftX",
    "EyeLeftY",
    "EyeRightX",
    "EyeRightY",
)

_TAU = 2.0 * math.pi


class IdleMotion:
    """``sample(t, speech_level)`` returns the idle head/eye parameters at time ``t`` (s).

    ``speaking`` (0..1, smoothed by the caller) suppresses sway, breathing and saccades;
    ``speech_level`` (the current mouth opening) drives small nods.
    """

    def __init__(
        self,
        seed: int = 0,
        *,
        amplitude: float = 1.0,
        breath_hz: float = 0.22,
        speaking_suppression: float = 0.7,
    ) -> None:
        self._rng = random.Random(seed)
        self._ph = [self._rng.uniform(0.0, _TAU) for _ in range(8)]
        self.amplitude = amplitude
        self.breath_hz = breath_hz
        self.speaking_suppression = min(1.0, max(0.0, speaking_suppression))
        self._gaze = (0.0, 0.0)
        self._next_saccade = 0.0

    def sample(
        self, t: float, speech_level: float = 0.0, *, speaking: float = 0.0
    ) -> dict[str, float]:
        s = math.sin
        ph = self._ph
        speaking = min(1.0, max(0.0, speaking))
        level = min(1.0, max(0.0, speech_level))
        calm = self.amplitude * (1.0 - self.speaking_suppression * speaking)
        if t >= self._next_saccade or t < self._next_saccade - 10.0:
            reach = 0.35 * (1.0 - 0.6 * speaking)
            self._gaze = (self._rng.uniform(-reach, reach), self._rng.uniform(-reach, reach))
            hold = self._rng.uniform(0.8, 3.0) * (1.0 + speaking)
            self._next_saccade = t + hold
        nod = 3.0 * self.amplitude * level * s(_TAU * 2.2 * t + ph[0])
        breath = s(_TAU * self.breath_hz * t + ph[6])
        gx, gy = self._gaze
        yaw = 6.0 * s(_TAU * 0.11 * t + ph[1]) + 2.0 * s(_TAU * 0.37 * t + ph[2])
        return {
            "FaceAngleX": calm * yaw,
            "FaceAngleY": calm * (3.0 * s(_TAU * 0.09 * t + ph[3]) + 0.8 * breath) + nod,
            "FaceAngleZ": calm * 4.0 * s(_TAU * 0.07 * t + ph[4]),
            "FacePositionX": calm * 0.8 * s(_TAU * 0.05 * t + ph[5]),
            "FacePositionY": calm * 0.6 * breath,
            "EyeLeftX": gx,
            "EyeRightX": gx,
            "EyeLeftY": gy,
            "EyeRightY": gy,
        }
