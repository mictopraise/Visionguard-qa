from pathlib import Path

import cv2

from app.core.models import VideoMetadata


class VideoProbeError(ValueError):
    pass


def probe_video(path: Path, original_name: str | None = None) -> VideoMetadata:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise VideoProbeError(f"Unable to open video: {path.name}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    capture.release()

    if fps <= 0 or frame_count <= 0:
        raise VideoProbeError(f"Invalid video metadata: {path.name}")

    return VideoMetadata(
        filename=original_name or path.name,
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_seconds=frame_count / fps,
        opencv_version=cv2.__version__,
    )
