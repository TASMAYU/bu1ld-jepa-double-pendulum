"""Deterministic data manifest: what was generated, from what, and its hash.

Layout produced by :func:`build_split`
--------------------------------------
::

    data/raw/<split>/<split>_<index:04d>.frames.npy   uint8  (T, H, W)
    data/raw/<split>/<split>_<index:04d>.state.npy    f64    (T, 4)
    data/raw/<split>/<split>_<index:04d>.ic.npy       f64    (4,)

Plain ``.npy`` is used rather than ``.npz`` on purpose: ``np.savez`` writes a
zip container that embeds file modification times, which would make the bytes
differ between runs. ``np.save`` output is a fixed header followed by raw
array bytes and is byte-stable across runs, which is what makes the SHA-256
hashes in the manifest meaningful.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from ..sim.double_pendulum import DoublePendulum
from ..sim.render import mean_pool, render_positions
from .ic_sampler import split_initial_conditions


def energy_scale(sim_cfg: dict) -> float:
    """Fixed physical energy scale used to normalise integrator drift.

    ``(m1+m2)*g*l1 + m2*g*l2`` is the energy difference between the fully
    inverted configuration and the hanging configuration at rest, i.e. the
    total energy range available to the system. It never vanishes, unlike a
    trajectory's own ``|E_initial|``.
    """
    g = sim_cfg["g_m_s2"]
    return (sim_cfg["m1_kg"] + sim_cfg["m2_kg"]) * g * sim_cfg["l1_m"] + sim_cfg[
        "m2_kg"
    ] * g * sim_cfg["l2_m"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_split(
    split_name: str,
    spec: dict,
    contract: dict,
    raw_root: Path,
) -> list[dict]:
    """Simulate, render and persist every trajectory of one split."""
    sim_cfg = contract["simulator"]
    render_cfg = contract["renderer"]
    repr_cfg = contract["representation"]

    pendulum = DoublePendulum(
        m1=sim_cfg["m1_kg"],
        m2=sim_cfg["m2_kg"],
        l1=sim_cfg["l1_m"],
        l2=sim_cfg["l2_m"],
        g=sim_cfg["g_m_s2"],
    )

    dt = float(sim_cfg["dt_s"])
    n_steps = int(sim_cfg["n_steps"])
    stride = int(render_cfg["frame_stride"])
    if n_steps % stride:
        raise ValueError(f"n_steps {n_steps} not divisible by frame_stride {stride}")

    states0, seeds = split_initial_conditions(spec)

    out_dir = raw_root / split_name
    out_dir.mkdir(parents=True, exist_ok=True)

    entries: list[dict] = []
    for index in range(states0.shape[0]):
        state0 = states0[index]

        states = pendulum.simulate(state0, dt, n_steps)[0]  # (n_steps+1, 4)
        sampled = states[::stride]  # (n_frames, 4)
        frames = render_positions(
            pendulum.positions(sampled),
            height=int(render_cfg["height"]),
            width=int(render_cfg["width"]),
            view_half_extent=float(render_cfg["view_half_extent_m"]),
            rod_radius_px=float(render_cfg["rod_radius_px"]),
            aa_px=float(render_cfg["aa_px"]),
        )
        representation = mean_pool(frames, int(repr_cfg["mean_pool_factor"]))

        stem = f"{split_name}_{index:04d}"
        frames_path = out_dir / f"{stem}.frames.npy"
        state_path = out_dir / f"{stem}.state.npy"
        ic_path = out_dir / f"{stem}.ic.npy"
        repr_path = out_dir / f"{stem}.repr.npy"

        np.save(frames_path, frames)
        np.save(state_path, sampled.astype(np.float64))
        np.save(ic_path, state0.astype(np.float64))
        np.save(repr_path, representation)

        energy = pendulum.energy(sampled)
        drift = np.abs(energy - energy[0])
        scale = energy_scale(sim_cfg)
        entries.append(
            {
                "split": split_name,
                "index": index,
                "seed": int(seeds[index]),
                "ic": {
                    "theta1_rad": float(state0[0]),
                    "theta2_rad": float(state0[1]),
                    "omega1_rad_s": float(state0[2]),
                    "omega2_rad_s": float(state0[3]),
                    "theta1_deg": float(np.rad2deg(state0[0])),
                },
                "n_frames": int(frames.shape[0]),
                "energy": {
                    "initial_j": float(energy[0]),
                    "max_abs_drift_j": float(np.max(drift)),
                    "max_rel_drift": float(np.max(drift) / scale),
                },
                "files": {
                    "frames": str(frames_path.relative_to(raw_root.parent.parent)),
                    "state": str(state_path.relative_to(raw_root.parent.parent)),
                    "ic": str(ic_path.relative_to(raw_root.parent.parent)),
                    "representation": str(repr_path.relative_to(raw_root.parent.parent)),
                },
                "sha256": {
                    "frames": sha256_file(frames_path),
                    "state": sha256_file(state_path),
                    "ic": sha256_file(ic_path),
                    "representation": sha256_file(repr_path),
                },
            }
        )
    return entries


def write_manifest(entries: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, sort_keys=True) + "\n")


def read_manifest(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
