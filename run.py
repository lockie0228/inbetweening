#!/usr/bin/env python3
"""
Extract OpenPose134 keypoints for the first and last frame in a folder.

The script runs rtmlib.Wholebody on the selected images and saves two outputs per
frame:
  1. `<frame>_keypoints.json`  – OpenPose-compatible JSON (pose/face/hands).
  2. `<frame>_skeleton.png`    – Visualization overlay for quick inspection.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np
from rtmlib import Wholebody

from draw import draw_skeleton


# Keypoint index ranges in rtmlib Wholebody output (OpenPose134 ordering).
BODY_RANGE = (0, 18)
FEET_RANGE = (18, 24)
FACE_RANGE = (24, 92)
LEFT_HAND_RANGE = (92, 113)
RIGHT_HAND_RANGE = (113, 134)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Wholebody on the first/last frame of a folder.",
    )
    parser.add_argument(
        "image_dir",
        type=Path,
        help="Directory containing source images.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("keypoints_out"),
        help="Directory to store JSON/visualization files.",
    )
    parser.add_argument(
        "--frames",
        default="first,last",
        help=(
            "Comma-separated list of frame selectors. "
            "Accepted values: 'first', 'last', integer indices, or negative indices."
        ),
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for rtmlib (cpu/cuda).",
    )
    parser.add_argument(
        "--backend",
        default="onnxruntime",
        help="Inference backend passed to rtmlib.Wholebody.",
    )
    parser.add_argument(
        "--kpt-thr",
        type=float,
        default=0.5,
        help="Confidence threshold when rendering skeleton overlays.",
    )
    return parser.parse_args()


def list_images(folder: Path) -> List[Path]:
    patterns = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp")
    files: List[Path] = []
    for pattern in patterns:
        files.extend(folder.glob(pattern))
    files.sort()
    return files


def parse_frame_indices(spec: str, total: int) -> List[int]:
    if total == 0:
        raise ValueError("No images found in the specified directory.")

    selections: List[int] = []
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        if item == "first":
            idx = 0
        elif item == "last":
            idx = total - 1
        else:
            idx = int(item)
            if idx < 0:
                idx = total + idx
        if not 0 <= idx < total:
            raise IndexError(f"Frame index {item} resolved to {idx}, "
                             f"but total images = {total}.")
        if idx not in selections:
            selections.append(idx)
    return selections


def flatten_xyc(xy: np.ndarray, scores: np.ndarray) -> List[float]:
    data = np.concatenate([xy, scores[..., None]], axis=-1).reshape(-1)
    return data.astype(float).tolist()


def select_slice(range_tuple: tuple[int, int], length: int) -> slice:
    start, end = range_tuple
    start = max(0, min(start, length))
    end = max(0, min(end, length))
    return slice(start, end)


def run_inference(
    model: Wholebody,
    image_path: Path,
    out_dir: Path,
    kpt_thr: float,
) -> None:
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    keypoints, scores = model(img)  # (N, 134, 2), (N, 134)
    if keypoints.size == 0:
        raise RuntimeError(f"No people detected in image: {image_path}")

    os.makedirs(out_dir, exist_ok=True)

    for person_idx, (kp, sc) in enumerate(zip(keypoints, scores)):
        length = kp.shape[0]

        pose_xy = np.concatenate(
            [
                kp[select_slice(BODY_RANGE, length)],
                kp[select_slice(FEET_RANGE, length)],
            ],
            axis=0,
        )
        pose_sc = np.concatenate(
            [
                sc[select_slice(BODY_RANGE, length)],
                sc[select_slice(FEET_RANGE, length)],
            ],
            axis=0,
        )

        person_json = {
            "person_id": [-1],
            "pose_keypoints_2d": flatten_xyc(pose_xy, pose_sc),
            "face_keypoints_2d": flatten_xyc(
                kp[select_slice(FACE_RANGE, length)],
                sc[select_slice(FACE_RANGE, length)],
            ),
            "hand_left_keypoints_2d": flatten_xyc(
                kp[select_slice(LEFT_HAND_RANGE, length)],
                sc[select_slice(LEFT_HAND_RANGE, length)],
            ),
            "hand_right_keypoints_2d": flatten_xyc(
                kp[select_slice(RIGHT_HAND_RANGE, length)],
                sc[select_slice(RIGHT_HAND_RANGE, length)],
            ),
            "pose_keypoints_3d": [],
            "face_keypoints_3d": [],
            "hand_left_keypoints_3d": [],
            "hand_right_keypoints_3d": [],
        }

        record = {"version": 1.3, "people": [person_json]}

        stem = image_path.stem
        if len(keypoints) > 1:
            stem = f"{stem}_p{person_idx}"

        json_path = out_dir / f"{stem}_keypoints.json"
        with json_path.open("w", encoding="utf-8") as fout:
            json.dump(record, fout, ensure_ascii=False)

        canvas = img.copy()
        draw_skeleton(
            canvas,
            keypoints=np.expand_dims(kp, axis=0),
            scores=np.expand_dims(sc, axis=0),
            openpose_skeleton=True,
            kpt_thr=kpt_thr,
        )

        vis_path = out_dir / f"{stem}_skeleton.png"
        cv2.imwrite(str(vis_path), canvas)

        print(f"[ok] {image_path.name} -> {json_path.name}, {vis_path.name}")


def main() -> None:
    args = parse_args()

    images = list_images(args.image_dir)
    if len(images) < 2:
        raise ValueError("Need at least two images in the folder.")

    indices = parse_frame_indices(args.frames, len(images))
    if not indices:
        indices = [0, len(images) - 1]

    model = Wholebody(
        to_openpose=True,
        mode="performance",
        backend=args.backend,
        device=args.device,
    )

    for idx in indices:
        run_inference(
            model=model,
            image_path=images[idx],
            out_dir=args.output_dir,
            kpt_thr=args.kpt_thr,
        )


if __name__ == "__main__":
    main()
