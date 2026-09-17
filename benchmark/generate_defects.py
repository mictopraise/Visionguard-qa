from __future__ import annotations

import json
from pathlib import Path
import cv2
import numpy as np


def _fourcc_for_output(output: Path) -> int:
    if output.suffix.lower() == ".avi":
        return cv2.VideoWriter_fourcc(*"MJPG")
    return cv2.VideoWriter_fourcc(*"mp4v")


def _open_video_for_rewrite(source: Path, output: Path):
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Unable to open source video: {source}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Source video has invalid metadata")

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        _fourcc_for_output(output),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Unable to create output video: {output}")

    return cap, writer, fps, width, height


def _append_manifest(record: dict, manifest_path: Path | None) -> None:
    if manifest_path is None:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def inject_freeze(
    source: Path,
    output: Path,
    start_frame: int,
    freeze_frames: int,
    manifest_path: Path | None = None,
) -> dict:
    cap, writer, fps, _, _ = _open_video_for_rewrite(source, output)

    frame_index = 0
    freeze_source = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index == start_frame:
            freeze_source = frame.copy()

        if freeze_source is not None and start_frame <= frame_index < start_frame + freeze_frames:
            writer.write(freeze_source)
        else:
            writer.write(frame)

        frame_index += 1

    cap.release()
    writer.release()

    record = {
        "source": source.name,
        "output": output.name,
        "defect": "frozen_frames",
        "start_frame": start_frame,
        "end_frame": start_frame + freeze_frames - 1,
        "fps": fps,
        "start_time": start_frame / fps,
        "end_time": (start_frame + freeze_frames - 1) / fps,
    }
    _append_manifest(record, manifest_path)
    return record


def inject_motion_jump(
    source: Path,
    output: Path,
    jump_frame: int,
    shift_pixels: int = 90,
    manifest_path: Path | None = None,
) -> dict:
    cap, writer, fps, width, height = _open_video_for_rewrite(source, output)

    frame_index = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if frame_index == jump_frame:
            transform = np.float32([[1, 0, shift_pixels], [0, 1, 0]])
            frame = cv2.warpAffine(
                frame,
                transform,
                (width, height),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )

        writer.write(frame)
        frame_index += 1

    cap.release()
    writer.release()

    record = {
        "source": source.name,
        "output": output.name,
        "defect": "motion_jump",
        "start_frame": jump_frame,
        "end_frame": jump_frame,
        "fps": fps,
        "start_time": jump_frame / fps,
        "end_time": jump_frame / fps,
        "shift_pixels": shift_pixels,
    }
    _append_manifest(record, manifest_path)
    return record
