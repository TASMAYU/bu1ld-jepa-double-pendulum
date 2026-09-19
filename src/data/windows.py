"""Sliding-window assembly for context/target video-prediction pairs.

A window is a contiguous run of frames. The first ``context_frames`` are
given to a model; the following ``target_frames`` are what it must predict.

Windows never cross a trajectory boundary, so every window belongs to exactly
one split.
"""

from __future__ import annotations

import numpy as np


def window_starts(
    n_frames: int, context_frames: int, target_frames: int, stride: int
) -> list[int]:
    span = context_frames + target_frames
    if n_frames < span:
        return []
    return list(range(0, n_frames - span + 1, stride))


def build_windows(
    trajectories: list[np.ndarray],
    context_frames: int,
    target_frames: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack windows from a list of per-trajectory representation arrays.

    Each input array has shape (T, h, w). Returns
    ``(context, target, trajectory_index)`` with shapes
    ``(N, context_frames, h, w)``, ``(N, target_frames, h, w)`` and ``(N,)``.
    """
    contexts: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    owners: list[int] = []

    for traj_index, frames in enumerate(trajectories):
        starts = window_starts(
            frames.shape[0], context_frames, target_frames, stride
        )
        for start in starts:
            contexts.append(frames[start : start + context_frames])
            targets.append(
                frames[start + context_frames : start + context_frames + target_frames]
            )
            owners.append(traj_index)

    if not contexts:
        raise ValueError("no windows produced; check n_frames against the contract")

    return (
        np.stack(contexts).astype(np.float32),
        np.stack(targets).astype(np.float32),
        np.asarray(owners, dtype=np.int64),
    )
