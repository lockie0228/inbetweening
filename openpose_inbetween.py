#!/usr/bin/env python3
"""
Generate in-between OpenPose JSONs for all pairs inside a folder.

Usage:
    python openpose_inbetween.py path/to/json_dir --out-dir inb_res --frames 25

The script expects OpenPose-style files (version=1.3, pose/face/hands). For each
sorted pair (file[i], file[i+1]) it produces an output directory named:
    <out-dir>/<start_stem>_to_<end_stem>/
containing `frame_000.json` ... `frame_{n-1}.json`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


KEYPOINT_FIELDS = (
    "pose_keypoints_2d",
    "face_keypoints_2d",
    "hand_left_keypoints_2d",
    "hand_right_keypoints_2d",
)


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fin:
        return json.load(fin)


def validate_people(frame: Dict[str, Any]) -> List[Dict[str, Any]]:
    people = frame.get("people")
    if not isinstance(people, list) or not people:
        raise ValueError(f"{frame!r} missing people array.")
    return people


def interp_flat(start: Iterable[Any], end: Iterable[Any], alpha: float) -> List[float]:
    start_list = list(start)
    end_list = list(end)

    if len(start_list) != len(end_list):
        if len(start_list) == 0 and len(end_list) == 0:
            return []
        raise ValueError("Keypoint arrays must have identical length.")

    return [
        (1.0 - alpha) * float(a) + alpha * float(b)
        for a, b in zip(start_list, end_list)
    ]


def interpolate_person(
    p0: Dict[str, Any],
    p1: Dict[str, Any],
    alpha: float,
) -> Dict[str, Any]:
    result = {}
    for key, value in p0.items():
        if key in KEYPOINT_FIELDS:
            result[key] = interp_flat(value, p1.get(key, []), alpha)
        else:
            result[key] = value

    for key in KEYPOINT_FIELDS:
        if key not in result and key in p1:
            result[key] = p1[key]

    return result


def build_frame(
    version: Any,
    people0: Sequence[Dict[str, Any]],
    people1: Sequence[Dict[str, Any]],
    alpha: float,
) -> Dict[str, Any]:
    if len(people0) != len(people1):
        raise ValueError("Mismatch in number of people between frames.")

    people_out = [
        interpolate_person(p0, p1, alpha) for p0, p1 in zip(people0, people1)
    ]
    return {"version": version, "people": people_out}


def write_frame(path: Path, frame: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fout:
        json.dump(frame, fout, ensure_ascii=False, separators=(",", ":"))


def collect_pairs(json_dir: Path) -> List[Tuple[Path, Path]]:
    files = sorted(json_dir.glob("*.json"))
    if len(files) < 2:
        raise ValueError(f"Need at least two JSON files in {json_dir}.")

    return list(zip(files[:-1], files[1:]))


def interpolate_pair(
    start_path: Path,
    end_path: Path,
    out_dir: Path,
    n_frames: int,
) -> None:
    if n_frames < 2:
        raise ValueError("Number of frames must be >= 2.")

    frame0 = load_json(start_path)
    frame1 = load_json(end_path)

    version0 = frame0.get("version", 1.0)
    if frame1.get("version", version0) != version0:
        raise ValueError("Start/end frames must share the same OpenPose version.")

    people0 = validate_people(frame0)
    people1 = validate_people(frame1)

    for idx in range(n_frames):
        alpha = idx / (n_frames - 1)
        frame = build_frame(version0, people0, people1, alpha)
        write_frame(out_dir / f"frame_{idx:03d}.json", frame)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create in-between OpenPose JSONs for each pair inside a folder.",
    )
    parser.add_argument(
        "json_dir",
        type=Path,
        help="Directory containing OpenPose JSON files (at least two).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("inb_res"),
        help="Root directory to store interpolation results.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=25,
        help="Total frames per interpolation sequence (start & end inclusive).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=16.0,
        help="Only for logging; does not affect interpolation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = collect_pairs(args.json_dir)

    print(f"[info] Found {len(pairs)} pair(s) in {args.json_dir}")
    for start_path, end_path in pairs:
        pair_name = f"{start_path.stem}_to_{end_path.stem}"
        out_dir = args.out_dir / pair_name
        print(
            f"[pair] {start_path.name} -> {end_path.name} | "
            f"{args.frames} frames @ {args.fps} fps -> {out_dir}"
        )
        interpolate_pair(start_path, end_path, out_dir, args.frames)


if __name__ == "__main__":
    main()
