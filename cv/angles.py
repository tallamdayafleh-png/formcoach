"""Joint-angle math."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def joint_angle(a: ArrayLike, b: ArrayLike, c: ArrayLike) -> np.ndarray:
    """Angle in degrees at point b, between the segments b->a and b->c.

    Points can be 2D or 3D, and can be stacked (shape (..., 2) or (..., 3)) to
    compute many frames at once. Returns NaN where a segment has zero length
    or any input is NaN. 180 = straight limb, smaller = more bent.

        a (hip)
         \\
          b (knee)   angle at b
         /
        c (ankle)
    """
    a, b, c = (np.asarray(p, dtype=float) for p in (a, b, c))
    v1 = a - b
    v2 = c - b
    n1 = np.linalg.norm(v1, axis=-1)
    n2 = np.linalg.norm(v2, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.sum(v1 * v2, axis=-1) / (n1 * n2)
    # clip guards against values like 1.0000000002 from float rounding
    angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return np.where((n1 > 0) & (n2 > 0), angle, np.nan)
