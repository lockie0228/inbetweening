#!/usr/bin/env python3
"""
Render interpolated OpenPose sequences to mp4 with optional frame dumps.

Example:
    python visualize_sequence.py inb_res/000_keypoints_to_100_keypoints \
        --out-video videos_out/sequence.mp4 --fps 16
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import cv2
import numpy as np

from draw import draw_skeleton


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize OpenPose JSON sequence and export video.",
    )
    parser.add_argument(
        "json_dir",
        type=Path,
        help="Directory containing frame_XXX.json files.",
    )
    parser.add_argument(
        "--out-video",
        type=Path,
        default=Path("videos_out/sequence.mp4"),
        help="Output mp4 path.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=16.0,
        help="Frames per second for the output video.",
    )
    parser.add_argument(
        "--score-thr",
        type=float,
        default=0.4,
        help="Minimum confidence for drawing joints/links.",
    )
    parser.add_argument(
        "--save-frames",
        action="store_true",
        help="Save each rendered frame as PNG alongside the video.",
    )
    parser.add_argument(
        "--canvas",
        type=str,
        default=None,
        help="Optional canvas size formatted as WIDTHxHEIGHT. "
        "If omitted the size is inferred from keypoints.",
    )
    return parser.parse_args()


def list_json_files(folder: Path) -> List[Path]:
    files = sorted(folder.glob("*.json"))
    if not files:
        raise ValueError(f"No JSON files found in {folder}.")
    return files


def unpack_keypoints(data: List[float]) -> Tuple[np.ndarray, np.ndarray]:
    arr = np.array(data, dtype=np.float32)
    if arr.size % 3 != 0:
        raise ValueError("Expected flattened [x, y, score] triplets.")
    arr = arr.reshape(-1, 3)
    xy = arr[:, :2]
    scores = arr[:, 2]
    return xy, scores


def load_openpose_json(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as fin:
        obj = json.load(fin)

    people = obj.get("people", [])
    if not people:
        raise ValueError(f"{path} does not contain any people.")
    person = people[0]

    xy_parts: List[np.ndarray] = []
    score_parts: List[np.ndarray] = []

    for field in (
        "pose_keypoints_2d",
        "face_keypoints_2d",
        "hand_left_keypoints_2d",
        "hand_right_keypoints_2d",
    ):
        xy, score = unpack_keypoints(person.get(field, []))
        xy_parts.append(xy)
        score_parts.append(score)

    xy_full = np.concatenate(xy_parts, axis=0)
    score_full = np.concatenate(score_parts, axis=0)
    return xy_full, score_full


def infer_canvas_size(keypoints: Iterable[np.ndarray], margin: int = 40) -> Tuple[int, int]:
    xs: List[float] = []
    ys: List[float] = []
    for points in keypoints:
        xs.extend(points[:, 0].tolist())
        ys.extend(points[:, 1].tolist())

    if not xs or not ys:
        raise ValueError("Unable to infer canvas size: no valid keypoints found.")

    max_x = max(xs)
    max_y = max(ys)
    width = int(np.ceil(max_x)) + margin
    height = int(np.ceil(max_y)) + margin
    return width, height


def ensure_parent(path: Path) -> None:
    if path.parent:
        path.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    json_files = list_json_files(args.json_dir)

    keypoints_cache: List[Tuple[np.ndarray, np.ndarray]] = [
        load_openpose_json(path) for path in json_files
    ]

    if args.canvas:
        w_str, h_str = args.canvas.lower().split("x")
        canvas_size = (int(w_str), int(h_str))
    else:
        canvas_size = infer_canvas_size(points for points, _ in keypoints_cache)

    width, height = canvas_size
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    ensure_parent(args.out_video)
    video_writer = cv2.VideoWriter(str(args.out_video), fourcc, args.fps, (width, height))
    if not video_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {args.out_video}")

    frame_dir: Path | None = None
    if args.save_frames:
        frame_dir = args.out_video.with_suffix("").parent / f"{args.out_video.stem}_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)

    for idx, (points, scores) in enumerate(keypoints_cache):
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        draw_skeleton(
            canvas,
            keypoints=points[None, :, :],
            scores=scores[None, :],
            openpose_skeleton=True,
            kpt_thr=args.score_thr,
        )
        video_writer.write(canvas)

        if frame_dir is not None:
            out_path = frame_dir / f"frame_{idx:03d}.png"
            cv2.imwrite(str(out_path), canvas)

    video_writer.release()
    print(f"[ok] wrote {len(keypoints_cache)} frames to {args.out_video}")


if __name__ == "__main__":
    main()
