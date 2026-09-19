"""Seeded initial-condition sampling for the frozen train/val/test regions.

Reproducibility rule
--------------------
Each trajectory draws its own dedicated generator, seeded as::

    trajectory_seed = split_seed * 1_000_000 + trajectory_index

so that trajectory ``i`` of a split is byte-identical regardless of how many
other trajectories are generated, in what order, or on how many workers.
Draw order within a trajectory is fixed and is part of the frozen contract:

    1. ``theta1`` magnitude  ~ Uniform(theta1_0_deg[0], theta1_0_deg[1])
    2. ``theta1`` sign       ~ Uniform{-1, +1}
    3. ``theta2``            ~ Uniform(theta2_0_deg[0], theta2_0_deg[1])

Angles are converted from degrees to radians at the end. Velocities are
constant across every split (released from rest).
"""

from __future__ import annotations

import numpy as np

SEED_STRIDE = 1_000_000


def trajectory_seed(split_seed: int, index: int) -> int:
    return int(split_seed) * SEED_STRIDE + int(index)


def sample_initial_conditions(spec: dict, index: int) -> tuple[np.ndarray, int]:
    """Draw the initial state for one trajectory of a split.

    Parameters
    ----------
    spec : dict
        A split block from ``experiment_contract.json``.
    index : int
        Trajectory index within the split.

    Returns
    -------
    (state0, seed) where ``state0`` has shape (4,) = [th1, th2, w1, w2].
    """
    seed = trajectory_seed(spec["seed"], index)
    rng = np.random.default_rng(seed)

    lo, hi = spec["theta1_0_deg"]
    magnitude = rng.uniform(lo, hi)
    sign = 1.0 if rng.random() < 0.5 else -1.0
    theta1 = np.deg2rad(magnitude * sign)

    t2_lo, t2_hi = spec["theta2_0_deg"]
    theta2 = np.deg2rad(rng.uniform(t2_lo, t2_hi))

    omega1 = float(spec["theta1_dot_0"])
    omega2 = float(spec["theta2_dot_0"])

    state0 = np.array([theta1, theta2, omega1, omega2], dtype=np.float64)
    return state0, seed


def split_initial_conditions(spec: dict) -> tuple[np.ndarray, np.ndarray]:
    """All initial conditions for a split.

    Returns
    -------
    (states, seeds) with shapes (n, 4) and (n,).
    """
    n = int(spec["n"])
    states = np.empty((n, 4), dtype=np.float64)
    seeds = np.empty(n, dtype=np.int64)
    for index in range(n):
        states[index], seeds[index] = sample_initial_conditions(spec, index)
    return states, seeds
