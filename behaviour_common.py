"""Shared timing helpers for body gestures and facial expressions.

Rules use True (observed), False (observed absent), or None (unobservable).
This module keeps brief positives from immediately becoming video labels.
"""

import math


class TemporalGate:
    """Require continuous positive evidence; missing data immediately resets it."""

    def __init__(self, duration_s=0.25):
        if not math.isfinite(duration_s) or duration_s < 0:
            raise ValueError("duration_s must be finite and nonnegative")
        self.duration_s = duration_s
        self.since = {}

    def update(self, states, timestamp_s):
        output = {}
        for label, value in states.items():
            if value is True:
                start = self.since.setdefault(label, timestamp_s)
                output[label] = timestamp_s - start + 1e-9 >= self.duration_s
            else:
                self.since.pop(label, None)
                output[label] = value
        return output
