"""Deterministic planar double pendulum.

Conventions
-----------
* ``theta1`` and ``theta2`` are the rod angles measured from the DOWNWARD
  vertical, in radians, positive counter-clockwise.
* Rods are rigid and massless. ``m1`` is a point mass at the end of rod 1,
  ``m2`` a point mass at the end of rod 2.
* No damping, no driving force. The system is conservative, so total
  mechanical energy is a conserved quantity and is used as an integrator
  sanity check.

Equations of motion are the standard Lagrangian form for the planar double
pendulum with point masses (see e.g. the "Double pendulum" article on
Wikipedia, Lagrangian derivation).
"""

from __future__ import annotations

import numpy as np

STATE_DIM = 4  # [theta1, theta2, omega1, omega2]


class DoublePendulum:
    """Planar double pendulum with point masses and rigid massless rods."""

    def __init__(
        self,
        m1: float = 1.0,
        m2: float = 1.0,
        l1: float = 1.0,
        l2: float = 1.0,
        g: float = 9.81,
    ) -> None:
        for name, value in (("m1", m1), ("m2", m2), ("l1", l1), ("l2", l2)):
            if value <= 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        self.m1, self.m2, self.l1, self.l2, self.g = m1, m2, l1, l2, g

    # -- dynamics ---------------------------------------------------------

    def derivatives(self, state: np.ndarray) -> np.ndarray:
        """Time derivative of the state. Accepts shape (..., 4)."""
        th1 = state[..., 0]
        th2 = state[..., 1]
        w1 = state[..., 2]
        w2 = state[..., 3]

        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g

        d = th1 - th2
        sin_d = np.sin(d)
        cos_d = np.cos(d)
        denom = 2.0 * m1 + m2 - m2 * np.cos(2.0 * d)

        acc1 = (
            -g * (2.0 * m1 + m2) * np.sin(th1)
            - m2 * g * np.sin(th1 - 2.0 * th2)
            - 2.0 * sin_d * m2 * (w2 * w2 * l2 + w1 * w1 * l1 * cos_d)
        ) / (l1 * denom)

        acc2 = (
            2.0
            * sin_d
            * (
                w1 * w1 * l1 * (m1 + m2)
                + g * (m1 + m2) * np.cos(th1)
                + w2 * w2 * l2 * m2 * cos_d
            )
        ) / (l2 * denom)

        return np.stack([w1, w2, acc1, acc2], axis=-1)

    def rk4_step(self, state: np.ndarray, dt: float) -> np.ndarray:
        """One classical Runge-Kutta 4 step."""
        k1 = self.derivatives(state)
        k2 = self.derivatives(state + 0.5 * dt * k1)
        k3 = self.derivatives(state + 0.5 * dt * k2)
        k4 = self.derivatives(state + dt * k3)
        return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def simulate(
        self, state0: np.ndarray, dt: float, n_steps: int
    ) -> np.ndarray:
        """Integrate a batch of initial states.

        Parameters
        ----------
        state0 : (B, 4)
        dt : integration step in seconds
        n_steps : number of steps to take

        Returns
        -------
        (B, n_steps + 1, 4) array of states including the initial state.
        """
        state0 = np.atleast_2d(np.asarray(state0, dtype=np.float64))
        if state0.shape[-1] != STATE_DIM:
            raise ValueError(
                f"expected state0 with last dim {STATE_DIM}, got {state0.shape}"
            )
        out = np.empty((state0.shape[0], n_steps + 1, STATE_DIM), dtype=np.float64)
        out[:, 0] = state0
        state = state0.copy()
        for step in range(n_steps):
            state = self.rk4_step(state, dt)
            out[:, step + 1] = state
        return out

    # -- diagnostics ------------------------------------------------------

    def energy(self, state: np.ndarray) -> np.ndarray:
        """Total mechanical energy. Conserved for the undamped system."""
        th1, th2 = state[..., 0], state[..., 1]
        w1, w2 = state[..., 2], state[..., 3]
        m1, m2, l1, l2, g = self.m1, self.m2, self.l1, self.l2, self.g

        kinetic = 0.5 * m1 * (l1 * w1) ** 2 + 0.5 * m2 * (
            (l1 * w1) ** 2
            + (l2 * w2) ** 2
            + 2.0 * l1 * l2 * w1 * w2 * np.cos(th1 - th2)
        )
        potential = -(m1 + m2) * g * l1 * np.cos(th1) - m2 * g * l2 * np.cos(th2)
        return kinetic + potential

    def positions(self, state: np.ndarray) -> np.ndarray:
        """Cartesian joint positions in metres.

        Returns shape (..., 2, 2): ``[..., 0, :]`` is joint 1, ``[..., 1, :]``
        is joint 2 (the tip). The pivot is the origin.
        """
        th1, th2 = state[..., 0], state[..., 1]
        x1 = self.l1 * np.sin(th1)
        y1 = -self.l1 * np.cos(th1)
        x2 = x1 + self.l2 * np.sin(th2)
        y2 = y1 - self.l2 * np.cos(th2)
        joint1 = np.stack([x1, y1], axis=-1)
        joint2 = np.stack([x2, y2], axis=-1)
        return np.stack([joint1, joint2], axis=-2)
