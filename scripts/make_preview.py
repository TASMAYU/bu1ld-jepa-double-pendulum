"""Render a contact sheet of sample trajectories to results/preview.png.

Deliberately dependency-free: the PNG is written by hand with ``zlib`` and
``struct`` so that the repository's only runtime dependency stays NumPy, and
so that the preview cannot drift with a plotting-library version.

Layout: one row per split, one column per sampled frame. The frame index runs
left to right, showing how each regime evolves over the 4 second clip.

Usage::

    python -m scripts.make_preview
"""

from __future__ import annotations

import json
import struct
import zlib
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")
FRAME_COLUMNS = 6
GAP = 2
SEPARATOR = 255


def write_png_gray(path: Path, image: np.ndarray) -> None:
    """Write a single-channel 8-bit PNG. ``image`` has shape (H, W)."""
    image = np.ascontiguousarray(image, dtype=np.uint8)
    height, width = image.shape

    raw = bytearray()
    for row in range(height):
        raw.append(0)  # filter type 0 (None) for every scanline
        raw.extend(image[row].tobytes())

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (
            struct.pack(">I", len(payload))
            + tag
            + payload
            + struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF)
        )

    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)  # 8-bit grayscale
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", header)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(png)


def main() -> int:
    manifest_path = REPO_ROOT / "data" / "manifest.jsonl"
    if not manifest_path.exists():
        raise SystemExit("data/manifest.jsonl missing; run `make data` first")

    with open(manifest_path, "r", encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip()]

    rows: list[np.ndarray] = []
    for split in SPLITS:
        candidates = sorted(
            (e for e in entries if e["split"] == split), key=lambda e: e["index"]
        )
        entry = candidates[0]
        frames = np.load(REPO_ROOT / entry["files"]["frames"])  # (T, H, W)
        picks = np.linspace(0, frames.shape[0] - 1, FRAME_COLUMNS).astype(int)

        cells: list[np.ndarray] = []
        for column, frame_index in enumerate(picks):
            cells.append(frames[frame_index])
            if column != FRAME_COLUMNS - 1:
                cells.append(
                    np.full((frames.shape[1], GAP), SEPARATOR, dtype=np.uint8)
                )
        rows.append(np.concatenate(cells, axis=1))

    width = max(row.shape[1] for row in rows)
    padded = [
        np.pad(row, ((0, 0), (0, width - row.shape[1])), constant_values=SEPARATOR)
        if row.shape[1] != width
        else row
        for row in rows
    ]

    separator = np.full((GAP, width), SEPARATOR, dtype=np.uint8)
    sheet_parts: list[np.ndarray] = []
    for index, row in enumerate(padded):
        sheet_parts.append(row)
        if index != len(padded) - 1:
            sheet_parts.append(separator)
    sheet = np.concatenate(sheet_parts, axis=0)

    out_path = REPO_ROOT / "results" / "preview.png"
    write_png_gray(out_path, sheet)
    print(
        f"wrote {out_path.relative_to(REPO_ROOT)} "
        f"({sheet.shape[1]}x{sheet.shape[0]}) "
        f"rows={SPLITS} cols={FRAME_COLUMNS} evenly spaced frames"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
