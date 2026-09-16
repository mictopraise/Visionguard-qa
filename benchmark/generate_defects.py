from __future__ import annotations

import json
from pathlib import Path
import cv2

def inject_freeze(source: Path, output: Path, start_frame: int, freeze_frames: int, manifest_path: Path | None = None) -> dict:
    cap = cv2.VideoCapture(str(source))
    if not cap.isOpened():
        raise ValueError(f"Unable to open source video: {source}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if fps <= 0 or width <= 0 or height <= 0:
        cap.release()
        raise ValueError("Source video has invalid metadata")

    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise ValueError(f"Unable to create output video: {output}")

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

    if manifest_path is not None:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        with manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

    return record
