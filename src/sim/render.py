"""Deterministic software renderer for double-pendulum trajectories.

The renderer draws the two rods plus the two joints as anti-aliased
capsules on a grayscale canvas. It deliberately has no graphics-library
dependency: every pixel value is produced by plain NumPy arithmetic, so
the mapping from world coordinates to bytes is fixed by this file alone
and cannot drift with a plotting backend version.

Intensity model
---------------
Each rod is a line segment. For every pixel we compute the Euclidean
distance ``d`` to the segment and map it to intensity with a linear ramp
over the anti-aliasing band of width ``aa_px``::

    intensity = clip((r + aa - d) / aa, 0, 1)

with ``r`` the rod radius in pixels. Overlapping elements combine with a
pixelwise maximum.
"""

from __future__ import annotations

import numpy as np


def _segment_distance(
    px: np.ndarray,
    py: np.ndarray,
    ax: np.ndarray,
    ay: np.ndarray,
    bx: np.ndarray,
    by: np.ndarray,
) -> np.ndarray:
    """Distance from each pixel to segment a->b, broadcast over leading dims."""
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    denom = vx * vx + vy * vy
    denom = np.where(denom < 1e-12, 1e-12, denom)
    t = np.clip((wx * vx + wy * vy) / denom, 0.0, 1.0)
    dx = wx - t * vx
    dy = wy - t * vy
    return np.sqrt(dx * dx + dy * dy)


def render_positions(
    positions: np.ndarray,
    height: int,
    width: int,
    view_half_extent: float,
    rod_radius_px: float,
    aa_px: float,
    joint_radius_scale: float = 1.6,
) -> np.ndarray:
    """Render a single trajectory.

    Parameters
    ----------
    positions : (T, 2, 2)
        Trajectory of joint positions in metres, as produced by
        :meth:`DoublePendulum.positions`.
    view_half_extent : float
        World-space half-width of the square viewport, centred on the pivot.

    Returns
    -------
    (T, height, width) uint8 array.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim != 3 or positions.shape[1:] != (2, 2):
        raise ValueError(f"expected positions of shape (T, 2, 2), got {positions.shape}")

    t_count = positions.shape[0]

    xs = np.linspace(-view_half_extent, view_half_extent, width)
    ys = np.linspace(view_half_extent, -view_half_extent, height)  # row 0 = top
    pixel_x, pixel_y = np.meshgrid(xs, ys)  # (H, W)

    units_per_px = (2.0 * view_half_extent) / float(max(width, height) - 1)
    radius = rod_radius_px * units_per_px
    aa = aa_px * units_per_px

    px = pixel_x[None, :, :]
    py = pixel_y[None, :, :]

    joint1 = positions[:, 0, :]
    joint2 = positions[:, 1, :]
    pivot = np.zeros_like(joint1)

    intensity = np.zeros((t_count, height, width), dtype=np.float64)

    for start, end in ((pivot, joint1), (joint1, joint2)):
        dist = _segment_distance(
            px,
            py,
            start[:, 0, None, None],
            start[:, 1, None, None],
            end[:, 0, None, None],
            end[:, 1, None, None],
        )
        intensity = np.maximum(intensity, np.clip((radius + aa - dist) / aa, 0.0, 1.0))

    joint_radius = radius * joint_radius_scale
    for joint in (joint1, joint2):
        dist = np.sqrt(
            (px - joint[:, 0, None, None]) ** 2 + (py - joint[:, 1, None, None]) ** 2
        )
        intensity = np.maximum(
            intensity, np.clip((joint_radius + aa - dist) / aa, 0.0, 1.0)
        )

    return np.clip(intensity * 255.0, 0.0, 255.0).astype(np.uint8)


def mean_pool(frames: np.ndarray, factor: int) -> np.ndarray:
    """Mean-pool frames by an integer factor.

    Input (..., H, W) in [0, 255] -> output (..., H//factor, W//factor)
    float32 in [0, 1].
    """
    if factor < 1:
        raise ValueError(f"factor must be >= 1, got {factor}")
    arr = np.asarray(frames, dtype=np.float32) / 255.0
    if factor == 1:
        return arr
    h, w = arr.shape[-2], arr.shape[-1]
    if h % factor or w % factor:
        raise ValueError(f"shape {(h, w)} not divisible by factor {factor}")
    arr = arr.reshape(*arr.shape[:-2], h // factor, factor, w // factor, factor)
    return arr.mean(axis=(-3, -1), dtype=np.float32)
