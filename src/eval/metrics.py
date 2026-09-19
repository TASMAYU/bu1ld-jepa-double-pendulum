"""Evaluation metrics.

Everything is computed on the *representation* grid declared in the contract
(mean-pooled grayscale), not on raw pixels. Fixing the evaluation resolution
in the contract is what keeps the numbers comparable across models.

Reported quantities
-------------------
``mse``
    Mean squared error over (window, horizon step, pixel).
``rmse``
    ``sqrt(mse)``, in the same units as the representation (0-1 intensity).
``mse_over_target_variance``
    ``mse`` divided by the variance of the ground-truth targets on the same
    split. This is the "skill" number: **1.0 means the predictor is exactly as
    good as always emitting the split mean**, below 1.0 means it carries real
    information, above 1.0 means it is actively harmful. Raw MSE alone is not
    interpretable across splits because the three splits have very different
    motion energy.
``per_horizon_mse``
    MSE at each target step k = 0..target_frames-1. This is the error-growth
    curve and is the most informative single plot for a chaotic system.
"""

from __future__ import annotations

import numpy as np


def per_horizon_mse(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    """MSE at each horizon step.

    Both inputs have shape (N, horizon, h, w).
    """
    if prediction.shape != target.shape:
        raise ValueError(
            f"shape mismatch: prediction {prediction.shape} vs target {target.shape}"
        )
    diff = prediction.astype(np.float64) - target.astype(np.float64)
    return np.mean(diff * diff, axis=(0, 2, 3))


def summarize(
    prediction: np.ndarray,
    target: np.ndarray,
    reference_variance: float | None = None,
) -> dict:
    """Full metric block for one baseline on one split."""
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    per_horizon = per_horizon_mse(prediction, target)
    mse = float(np.mean(per_horizon))

    if reference_variance is None:
        reference_variance = float(np.var(target))

    return {
        "n_windows": int(target.shape[0]),
        "horizon": int(target.shape[1]),
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mse_over_target_variance": (
            float(mse / reference_variance) if reference_variance > 0 else None
        ),
        "per_horizon_mse": [float(value) for value in per_horizon],
    }
