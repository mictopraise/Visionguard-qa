from __future__ import annotations

from pathlib import Path

import cv2


def export_evidence_frame(
    video_path: Path,
    output_dir: Path,
    *,
    timestamp: float,
    label: str,
) -> str:
    """Export one representative JPEG frame at or near a timestamp."""
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if fps <= 0 or frame_count <= 0:
        cap.release()
        raise ValueError(f"Invalid video metadata: {video_path}")

    target_frame = min(frame_count - 1, max(0, int(round(timestamp * fps))))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise ValueError(f"Unable to read evidence frame {target_frame} from {video_path.name}")

    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in label)
    filename = f"{safe_label}_f{target_frame:06d}.jpg"
    destination = output_dir / filename

    if not cv2.imwrite(str(destination), frame):
        raise ValueError(f"Unable to write evidence frame: {destination}")

    return str(destination)
