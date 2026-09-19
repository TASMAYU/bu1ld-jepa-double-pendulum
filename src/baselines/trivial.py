"""Trivial baselines.

These are deliberately weak and deliberately fixed. They exist to establish
the score a real model has to beat before any result is worth reporting. The
baseline *family* is frozen in ``experiment_contract.json``: adding a
stronger baseline after seeing a JEPA result would invalidate the comparison.

All three consume a context window ``(N, context_frames, h, w)`` and emit a
prediction ``(N, horizon, h, w)``.
"""

from __future__ import annotations

import numpy as np


class Persistence:
    """Repeat the last observed context frame for the whole horizon.

    The strongest trivial baseline for short horizons in smooth video.
    """

    name = "persistence"
    learned = False

    def fit(self, context: np.ndarray, target: np.ndarray) -> "Persistence":
        return self

    def predict(self, context: np.ndarray) -> np.ndarray:
        last = context[:, -1]  # (N, h, w)
        horizon = getattr(self, "horizon", None)
        if horizon is None:
            raise ValueError("horizon must be set before predict()")
        return np.repeat(last[:, None], horizon, axis=1)


class TemporalMean:
    """Repeat the mean of the context frames for the whole horizon.

    A blurry, motion-free prediction. It cannot track the pendulum but it is
    a surprisingly hard baseline to beat on pixel MSE because it hedges.
    """

    name = "temporal_mean"
    learned = False

    def fit(self, context: np.ndarray, target: np.ndarray) -> "TemporalMean":
        return self

    def predict(self, context: np.ndarray) -> np.ndarray:
        mean_frame = context.mean(axis=1)  # (N, h, w)
        return np.repeat(mean_frame[:, None], self.horizon, axis=1)


class LinearAR:
    """Ridge-regression one-step predictor, rolled out autoregressively.

    This is the only baseline that actually learns anything, which makes it
    the honest "is deep learning even needed?" control. It fits a single
    linear map from the flattened context window to the next frame, then
    feeds its own predictions back in to reach the full horizon.

    Solved in closed form: ``W = (X^T X + lambda I)^-1 X^T Y``.
    """

    name = "linear_ar"
    learned = True

    def __init__(self, ridge_lambda: float = 1.0) -> None:
        self.ridge_lambda = float(ridge_lambda)
        self.W: np.ndarray | None = None
        self.b: np.ndarray | None = None
        self.context_frames: int | None = None
        self.horizon: int | None = None
        self.n_constant_features: int | None = None

    def fit(self, context: np.ndarray, target: np.ndarray) -> "LinearAR":
        n, k, h, w = context.shape
        self.context_frames = k
        self.horizon = target.shape[1]

        X = context.reshape(n, k * h * w).astype(np.float64)
        Y = target[:, 0].reshape(n, h * w).astype(np.float64)

        # Pixels that never light up anywhere in the training split have zero
        # variance, which makes X^T X exactly rank-deficient. This is a real
        # property of a low-energy training split rather than a bug: a pendulum
        # released from a small angle never swings into the corners of the
        # frame. The ridge term is what keeps the solve well posed, and the
        # resulting inability to represent motion in those pixels is a genuine
        # limitation of this baseline rather than an artefact.
        self.n_constant_features = int(np.sum(X.std(axis=0) == 0.0))

        X = np.concatenate([X, np.ones((n, 1), dtype=np.float64)], axis=1)
        d = X.shape[1]

        # macOS Accelerate BLAS raises spurious divide-by-zero / overflow /
        # invalid floating-point flags on these products even though every
        # input and every output is finite (verified below). Suppress the
        # noise and assert finiteness explicitly instead of trusting the flags.
        with np.errstate(all="ignore"):
            gram = X.T @ X
            gram.flat[:: d + 1] += self.ridge_lambda
            gram[-1, -1] -= self.ridge_lambda  # the bias term is not penalised
            rhs = X.T @ Y
            solution = np.linalg.solve(gram, rhs)

        if not np.isfinite(solution).all():
            raise FloatingPointError("ridge solve produced non-finite coefficients")

        self.W = solution[:-1]
        self.b = solution[-1]
        return self

    def predict(self, context: np.ndarray) -> np.ndarray:
        if self.W is None or self.b is None:
            raise ValueError("fit() must be called before predict()")

        window = context.astype(np.float64).copy()
        n = window.shape[0]
        h, w = window.shape[-2:]

        predictions: list[np.ndarray] = []
        with np.errstate(all="ignore"):
            for _ in range(int(self.horizon)):
                flat = window.reshape(n, -1)
                nxt = flat @ self.W + self.b
                frame = nxt.reshape(n, h, w)
                predictions.append(frame)
                window = np.concatenate([window[:, 1:], frame[:, None, :, :]], axis=1)

        return np.stack(predictions, axis=1).astype(np.float32)

    def fit_diagnostics(self) -> dict:
        return {
            "ridge_lambda": self.ridge_lambda,
            "context_frames": self.context_frames,
            "n_design_features": (
                None
                if self.context_frames is None
                else self.context_frames * 16 * 16
            ),
            "n_constant_features": self.n_constant_features,
            "note": "n_constant_features counts training-split pixels that never light up and therefore have exactly zero variance. These make the unregularised normal equations singular; the ridge term is what makes the solve well posed.",
        }


def build_baselines(contract: dict) -> list:
    """Instantiate the frozen baseline family declared in the contract."""
    specs = contract["baselines"]
    model = specs["linear_ar"]

    baselines: list = []
    if specs["persistence"]["enabled"]:
        baselines.append(Persistence())
    if specs["temporal_mean"]["enabled"]:
        baselines.append(TemporalMean())
    if model["enabled"]:
        baselines.append(LinearAR(ridge_lambda=model["ridge_lambda"]))

    horizon = int(contract["windows"]["target_frames"])
    for baseline in baselines:
        baseline.horizon = horizon
    return baselines
