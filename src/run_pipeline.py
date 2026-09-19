"""Single entry point for the M1 pipeline.

Stages
------
``data``
    Simulate, render and hash every trajectory of all three splits, then
    write the deterministic manifest. Generating test data is not the same
    as evaluating on it: generation is required for the artifact to be
    complete and is fully specified by the frozen contract.

``baselines``
    Fit the frozen baseline family on the train split and score it on train
    and validation. The test split is **not** evaluated unless ``--allow-test``
    is passed explicitly, which prints a warning to stderr. Under the M1
    evidence gate that flag must not be used.

``verify``
    Re-derive every array from the contract and compare its SHA-256 against
    the manifest. This is the determinism proof: any mismatch means the
    pipeline is not reproducible.

``all``
    ``data`` followed by ``baselines``.

Usage
-----
::

    python -m src.run_pipeline                 # data + baselines (train/val only)
    python -m src.run_pipeline --stage data
    python -m src.run_pipeline --stage verify
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np

from .baselines.trivial import build_baselines
from .data.manifest import generate_split, read_manifest, write_manifest
from .data.windows import build_windows
from .eval.metrics import summarize
from .sim.double_pendulum import DoublePendulum
from .sim.render import mean_pool, render_positions

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLIT_ORDER = ("train", "val", "test")


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def load_contract(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def build_pendulum(contract: dict) -> DoublePendulum:
    sim = contract["simulator"]
    return DoublePendulum(
        m1=sim["m1_kg"], m2=sim["m2_kg"], l1=sim["l1_m"], l2=sim["l2_m"], g=sim["g_m_s2"]
    )


def render_trajectory(pendulum: DoublePendulum, contract: dict, state0: np.ndarray):
    """Derive (sampled_states, frames, representation) for one trajectory."""
    sim_cfg = contract["simulator"]
    render_cfg = contract["renderer"]
    repr_cfg = contract["representation"]

    stride = int(render_cfg["frame_stride"])
    states = pendulum.simulate(state0, float(sim_cfg["dt_s"]), int(sim_cfg["n_steps"]))[0]
    sampled = states[::stride]
    frames = render_positions(
        pendulum.positions(sampled),
        height=int(render_cfg["height"]),
        width=int(render_cfg["width"]),
        view_half_extent=float(render_cfg["view_half_extent_m"]),
        rod_radius_px=float(render_cfg["rod_radius_px"]),
        aa_px=float(render_cfg["aa_px"]),
        joint_radius_scale=float(render_cfg["joint_radius_scale"]),
    )
    representation = mean_pool(frames, int(repr_cfg["mean_pool_factor"]))
    return sampled, frames, representation


def npy_sha256(array: np.ndarray) -> str:
    """SHA-256 of the exact bytes ``np.save`` would write for this array."""
    buffer = io.BytesIO()
    np.save(buffer, array)
    return hashlib.sha256(buffer.getvalue()).hexdigest()


def load_representations(entries: list[dict], split: str) -> list[np.ndarray]:
    ordered = sorted(
        (e for e in entries if e["split"] == split), key=lambda e: e["index"]
    )
    return [np.load(REPO_ROOT / e["files"]["representation"]) for e in ordered]


# --------------------------------------------------------------------------
# stages
# --------------------------------------------------------------------------


def stage_data(contract: dict, raw_root: Path, manifest_path: Path) -> list[dict]:
    entries: list[dict] = []
    tolerance = float(contract["energy_drift_tolerance_rel"])

    for split in SPLIT_ORDER:
        spec = contract["splits"][split]
        split_entries = generate_split(split, spec, contract, raw_root)
        entries.extend(split_entries)

        drifts = [e["energy"]["max_rel_drift"] for e in split_entries]
        worst = max(drifts)
        status = "ok" if worst <= tolerance else "FAIL"
        print(
            f"[data] {split:5s} n={len(split_entries):4d} "
            f"frames={split_entries[0]['n_frames']} "
            f"max rel|dE|={worst:.3e} (tol {tolerance:.0e}) -> {status}"
        )
        if worst > tolerance:
            raise RuntimeError(
                f"relative energy drift {worst:.3e} exceeds tolerance "
                f"{tolerance:.3e} on split '{split}'; the integrator is not "
                f"conserving energy"
            )

    write_manifest(entries, manifest_path)
    print(f"[data] manifest written: {manifest_path.relative_to(REPO_ROOT)} "
          f"({len(entries)} trajectories)")
    return entries


def stage_baselines(contract: dict, manifest_path: Path, allow_test: bool) -> dict:
    entries = read_manifest(manifest_path)
    windows_cfg = contract["windows"]
    k = int(windows_cfg["context_frames"])
    horizon = int(windows_cfg["target_frames"])
    stride = int(windows_cfg["stride"])

    splits = ["train", "val"]
    if allow_test:
        print(
            "[baselines] WARNING: --allow-test passed. The test split is being "
            "evaluated. Under the M1 evidence gate this must not be done, and the "
            "resulting numbers may not be used for any model-selection decision.",
            file=sys.stderr,
        )
        splits.append("test")

    windows: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split in splits:
        context, target, _ = build_windows(
            load_representations(entries, split), k, horizon, stride
        )
        windows[split] = (context, target)
        print(f"[baselines] {split:5s} windows={context.shape[0]}")

    # Reference variance is taken from the split's own targets so the
    # no-skill baseline sits at exactly 1.0 on every split.
    reference_variance = {
        split: float(np.var(target)) for split, (_, target) in windows.items()
    }

    train_context, train_target = windows["train"]
    results: dict[str, dict] = {split: {} for split in splits}

    for baseline in build_baselines(contract):
        baseline.fit(train_context, train_target)
        diagnostics = getattr(baseline, "fit_diagnostics", lambda: {})()
        for split in splits:
            context, target = windows[split]
            prediction = baseline.predict(context)
            results[split][baseline.name] = summarize(
                prediction, target, reference_variance=reference_variance[split]
            )
            if diagnostics:
                results[split][baseline.name]["fit_diagnostics"] = diagnostics
            metrics = results[split][baseline.name]
            print(
                f"[baselines] {split:5s} {baseline.name:14s} "
                f"mse={metrics['mse']:.6e} "
                f"nrmse={metrics['mse_over_target_variance']:.4f}"
            )

    return {
        "contract_version": contract["contract_version"],
        "splits_evaluated": splits,
        "test_evaluated": bool(allow_test),
        "windows": {
            "context_frames": k,
            "target_frames": horizon,
            "stride": stride,
        },
        "evaluation_grid": contract["metrics"]["evaluation_grid"],
        "target_variance_reference": reference_variance,
        "results": results,
    }


def stage_verify(contract: dict, manifest_path: Path) -> bool:
    entries = read_manifest(manifest_path)
    pendulum = build_pendulum(contract)

    checked = 0
    failures: list[str] = []

    for entry in entries:
        state0 = np.load(REPO_ROOT / entry["files"]["ic"])
        sampled, frames, representation = render_trajectory(pendulum, contract, state0)

        recomputed = {
            "frames": npy_sha256(frames),
            "state": npy_sha256(sampled.astype(np.float64)),
            "ic": npy_sha256(state0.astype(np.float64)),
            "representation": npy_sha256(representation),
        }
        for key, digest in recomputed.items():
            if digest != entry["sha256"][key]:
                failures.append(
                    f"{entry['split']}/{entry['index']:04d}:{key} "
                    f"manifest={entry['sha256'][key][:16]} recomputed={digest[:16]}"
                )
        checked += 1

    print(f"[verify] recomputed {checked} trajectories ({checked * 4} hashes)")
    if failures:
        print(f"[verify] {len(failures)} MISMATCHES", file=sys.stderr)
        for failure in failures[:20]:
            print(f"[verify]   {failure}", file=sys.stderr)
        return False
    print("[verify] all hashes match the manifest: pipeline is deterministic")
    return True


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="bu1ld-jepa-double-pendulum M1 pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=("all", "data", "baselines", "verify"),
        default="all",
        help="which stage to run (default: all = data + baselines)",
    )
    parser.add_argument(
        "--contract",
        default=str(REPO_ROOT / "experiment_contract.json"),
        help="path to the frozen experiment contract",
    )
    parser.add_argument(
        "--allow-test",
        action="store_true",
        help="evaluate the held-out test split. NOT used in M1; see the contract.",
    )
    args = parser.parse_args(argv)

    contract = load_contract(Path(args.contract))
    raw_root = REPO_ROOT / "data" / "raw"
    manifest_path = REPO_ROOT / "data" / "manifest.jsonl"
    results_path = REPO_ROOT / "results" / "baselines_m1.json"

    if args.stage in ("all", "data"):
        stage_data(contract, raw_root, manifest_path)

    if args.stage in ("all", "baselines"):
        if not manifest_path.exists():
            print(
                "[baselines] manifest missing; run --stage data first", file=sys.stderr
            )
            return 2
        results = stage_baselines(contract, manifest_path, args.allow_test)
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2, sort_keys=True)
            handle.write("\n")
        print(f"[baselines] results written: {results_path.relative_to(REPO_ROOT)}")

    if args.stage == "verify":
        if not manifest_path.exists():
            print("[verify] manifest missing; run --stage data first", file=sys.stderr)
            return 2
        if not stage_verify(contract, manifest_path):
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
