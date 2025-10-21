#!/usr/bin/env python3
"""
End-to-end OpenPose134 workflow for two-frame sequences.

Given an input image folder containing pairs named `<name>_1.png` and
`<name>_2.png`, the script:

1. Detects Wholebody keypoints on both frames (rtmlib) and saves OpenPose-style
   JSON along with pose-overlaid PNGs on the originals.
2. Linearly interpolates the keypoints (default 25 frames, inclusive).
3. Renders the sequence on a fixed 1024x1024 black canvas and exports an mp4.

Outputs are stored under `--output-root/<name>/`.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from rtmlib import Wholebody

from draw import draw_skeleton


POSE_COUNT = 24  # body 18 + feet 6
FACE_COUNT = 68
HAND_COUNT = 21  # per hand
TOTAL_KPTS = POSE_COUNT + FACE_COUNT + 2 * HAND_COUNT  # 134

# Index ranges in concatenated keypoint arrays.
POSE_SLICE = slice(0, POSE_COUNT)
FACE_SLICE = slice(POSE_COUNT, POSE_COUNT + FACE_COUNT)
L_HAND_SLICE = slice(POSE_COUNT + FACE_COUNT, POSE_COUNT + FACE_COUNT + HAND_COUNT)
R_HAND_SLICE = slice(POSE_COUNT + FACE_COUNT + HAND_COUNT, TOTAL_KPTS)


@dataclass
class PoseResult:
    record: Dict[str, object]
    keypoints: np.ndarray  # (134, 2)
    scores: np.ndarray  # (134,)
    json_path: Path
    overlay_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenPose134 end-to-end workflow.")
    parser.add_argument("image_dir", type=Path, help="Folder with frames (x_1.png, x_2.png).")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("workflow_out"),
        help="Base directory for all outputs.",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=25,
        help="Number of frames to interpolate (inclusive of start/end).",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=16.0,
        help="FPS for exported videos.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device string passed to rtmlib Wholebody.",
    )
    parser.add_argument(
        "--backend",
        default="onnxruntime",
        help="Backend string passed to rtmlib Wholebody.",
    )
    parser.add_argument(
        "--kpt-thr",
        type=float,
        default=0.2,
        help="Confidence threshold for skeleton rendering.",
    )
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Additionally save intermediate interpolated JSON frames.",
    )
    return parser.parse_args()


def list_image_pairs(folder: Path) -> List[Tuple[str, Path, Path]]:
    pattern = re.compile(r"^(?P<name>.+)_(?P<idx>[12])\.(png|jpg|jpeg|bmp|webp)$", re.IGNORECASE)
    files = sorted(p for p in folder.iterdir() if p.is_file())

    sequences: Dict[str, Dict[str, Path]] = {}
    for path in files:
        match = pattern.match(path.name)
        if not match:
            continue
        name = match.group("name")
        idx = match.group("idx")
        sequences.setdefault(name, {})[idx] = path

    pairs: List[Tuple[str, Path, Path]] = []
    for name, mapping in sequences.items():
        if "1" in mapping and "2" in mapping:
            pairs.append((name, mapping["1"], mapping["2"]))

    if not pairs:
        raise ValueError(f"No valid *_1 / *_2 pairs found in {folder}")

    pairs.sort(key=lambda item: item[0])
    return pairs


def slice_range(bounds: Tuple[int, int], length: int) -> slice:
    start, end = bounds
    start = max(0, min(start, length))
    end = max(0, min(end, length))
    return slice(start, end)


def flatten_xyc(xy: np.ndarray, scores: np.ndarray) -> List[float]:
    xyc = np.concatenate([xy, scores[..., None]], axis=-1).reshape(-1)
    return xyc.astype(float).tolist()


def build_openpose_record(points: np.ndarray, scores: np.ndarray) -> Dict[str, object]:
    pose_xy = points[POSE_SLICE]
    face_xy = points[FACE_SLICE]
    lhand_xy = points[L_HAND_SLICE]
    rhand_xy = points[R_HAND_SLICE]

    pose_sc = scores[POSE_SLICE]
    face_sc = scores[FACE_SLICE]
    lhand_sc = scores[L_HAND_SLICE]
    rhand_sc = scores[R_HAND_SLICE]

    person = {
        "person_id": [-1],
        "pose_keypoints_2d": flatten_xyc(pose_xy, pose_sc),
        "face_keypoints_2d": flatten_xyc(face_xy, face_sc),
        "hand_left_keypoints_2d": flatten_xyc(lhand_xy, lhand_sc),
        "hand_right_keypoints_2d": flatten_xyc(rhand_xy, rhand_sc),
        "pose_keypoints_3d": [],
        "face_keypoints_3d": [],
        "hand_left_keypoints_3d": [],
        "hand_right_keypoints_3d": [],
    }
    return {"version": 1.3, "people": [person]}


def extract_segments(kp: np.ndarray, sc: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    for slc in (POSE_SLICE, FACE_SLICE, L_HAND_SLICE, R_HAND_SLICE):
        if slc.stop > kp.shape[0]:
            raise ValueError("Keypoint array shorter than expected for OpenPose134.")
    return kp[:TOTAL_KPTS], sc[:TOTAL_KPTS]


def run_wholebody(
    model: Wholebody,
    image_path: Path,
    out_dir: Path,
    kpt_thr: float,
) -> PoseResult:
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    keypoints, scores = model(img)
    if keypoints.size == 0:
        raise RuntimeError(f"No people detected in {image_path}")

    # Assume first person.
    kp = keypoints[0]
    sc = scores[0]

    kp_concat = np.concatenate(
        [
            kp[slice_range((0, 18), kp.shape[0])],
            kp[slice_range((18, 24), kp.shape[0])],
            kp[slice_range((24, 92), kp.shape[0])],
            kp[slice_range((92, 113), kp.shape[0])],
            kp[slice_range((113, 134), kp.shape[0])],
        ],
        axis=0,
    )
    sc_concat = np.concatenate(
        [
            sc[slice_range((0, 18), sc.shape[0])],
            sc[slice_range((18, 24), sc.shape[0])],
            sc[slice_range((24, 92), sc.shape[0])],
            sc[slice_range((92, 113), sc.shape[0])],
            sc[slice_range((113, 134), sc.shape[0])],
        ],
        axis=0,
    )

    kp_concat, sc_concat = extract_segments(kp_concat, sc_concat)

    record = build_openpose_record(kp_concat, sc_concat)

    json_path = out_dir / f"{image_path.stem}_keypoints.json"
    with json_path.open("w", encoding="utf-8") as fout:
        json.dump(record, fout, ensure_ascii=False)

    overlay = draw_skeleton(
        img.copy(),
        keypoints=kp_concat[None, :, :],
        scores=sc_concat[None, :],
        openpose_skeleton=True,
        kpt_thr=kpt_thr,
    )
    overlay_path = out_dir / f"{image_path.stem}_overlay.png"
    cv2.imwrite(str(overlay_path), overlay)

    return PoseResult(
        record=record,
        keypoints=kp_concat,
        scores=sc_concat,
        json_path=json_path,
        overlay_path=overlay_path,
    )


def interpolate_frames(
    start: PoseResult,
    end: PoseResult,
    n_frames: int,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    if n_frames < 2:
        raise ValueError("Number of frames must be at least 2.")

    frames: List[Tuple[np.ndarray, np.ndarray]] = []
    for idx in range(n_frames):
        alpha = idx / (n_frames - 1)
        pts = (1.0 - alpha) * start.keypoints + alpha * end.keypoints
        sc = (1.0 - alpha) * start.scores + alpha * end.scores
        frames.append((pts, sc))
    return frames


def compute_canvas_transform(
    points_list: Sequence[np.ndarray],
    canvas_size: Tuple[int, int],
    margin: int = 40,
) -> Tuple[float, np.ndarray]:
    combined = np.vstack(points_list)
    min_xy = combined.min(axis=0)
    max_xy = combined.max(axis=0)
    span = np.maximum(max_xy - min_xy, 1e-6)

    width, height = canvas_size
    usable_w = width - 2 * margin
    usable_h = height - 2 * margin

    scale = min(usable_w / span[0], usable_h / span[1])
    scale = max(scale, 1e-6)

    offset = np.array([margin, margin]) - min_xy * scale
    return scale, offset


def transform_points(points: np.ndarray, scale: float, offset: np.ndarray) -> np.ndarray:
    transformed = points * scale + offset
    return transformed


def save_intermediate_json(
    frames: List[Tuple[np.ndarray, np.ndarray]],
    out_dir: Path,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for idx, (pts, sc) in enumerate(frames):
        record = build_openpose_record(pts, sc)
        path = out_dir / f"frame_{idx:03d}.json"
        with path.open("w", encoding="utf-8") as fout:
            json.dump(record, fout, ensure_ascii=False)


def reencode_video(input_path: Path, output_path: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        if input_path != output_path:
            input_path.replace(output_path)
        print("[warn] ffmpeg not found; keeping raw mp4v file.")
        return

    cmd = [
        ffmpeg,
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    input_path.unlink(missing_ok=True)


def render_skeleton_video(
    frames: List[Tuple[np.ndarray, np.ndarray]],
    out_path: Path,
    fps: float,
    canvas_size: Tuple[int, int],
    kpt_thr: float,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    scale, offset = compute_canvas_transform([pts for pts, _ in frames], canvas_size)

    width, height = canvas_size
    raw_path = out_path.with_suffix(".raw.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(raw_path), fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {raw_path}")

    for pts, sc in frames:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        transformed = transform_points(pts, scale, offset)
        rendered = draw_skeleton(
            canvas,
            keypoints=transformed[None, :, :],
            scores=sc[None, :],
            openpose_skeleton=True,
            kpt_thr=kpt_thr,
        )
        writer.write(rendered.astype(np.uint8))

    writer.release()
    reencode_video(raw_path, out_path, fps)


def main() -> None:
    args = parse_args()
    pairs = list_image_pairs(args.image_dir)

    model = Wholebody(
        to_openpose=True,
        mode="performance",
        backend=args.backend,
        device=args.device,
    )

    for name, start_path, end_path in pairs:
        seq_dir = args.output_root / name
        seq_dir.mkdir(parents=True, exist_ok=True)

        print(f"[seq] Processing {name}: {start_path.name} -> {end_path.name}")
        start_result = run_wholebody(model, start_path, seq_dir, args.kpt_thr)
        end_result = run_wholebody(model, end_path, seq_dir, args.kpt_thr)

        frames = interpolate_frames(start_result, end_result, args.frames)

        if args.save_json:
            save_intermediate_json(frames, seq_dir / "interpolated_json")

        video_path = seq_dir / f"{name}_skeleton.mp4"
        render_skeleton_video(
            frames,
            out_path=video_path,
            fps=args.fps,
            canvas_size=(1024, 1024),
            kpt_thr=args.kpt_thr,
        )

        print(
            f"[done] {name}: JSONs -> {seq_dir}, overlay PNGs -> {seq_dir}, "
            f"video -> {video_path}"
        )


if __name__ == "__main__":
    main()
