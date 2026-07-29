"""Static-frame shortcut check - video/image tasks only, not numeric/audio.

Compares a model's metric on the real video/image against the same metric
when the model is given a single repeated frame instead, per the report
template (Section 5):
  - drop < ~5%   -> shortcut suspected (model may be reading static
                    appearance/color/shape, not real motion)
  - drop > ~20-30% -> good sign, model needs real motion to succeed
Callers compute both scores themselves (run the same eval twice, once with
a static-frame-substituted dataset variant) - this module just applies the
report's decision rule consistently.
"""
from __future__ import annotations

SHORTCUT_SUSPECTED_THRESHOLD = 0.05
LIKELY_USES_MOTION_THRESHOLD = 0.20


def compute_shortcut_gap(real_score: float, static_frame_score: float) -> dict:
    if real_score == 0:
        relative_drop = float("nan")
    else:
        relative_drop = (real_score - static_frame_score) / real_score

    if relative_drop < SHORTCUT_SUSPECTED_THRESHOLD:
        verdict = "shortcut_suspected"
    elif relative_drop > LIKELY_USES_MOTION_THRESHOLD:
        verdict = "likely_uses_motion"
    else:
        verdict = "inconclusive"

    return {
        "real_score": real_score,
        "static_frame_score": static_frame_score,
        "relative_drop": relative_drop,
        "verdict": verdict,
    }
