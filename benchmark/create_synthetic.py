from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def create_moving_square_video(
    output: Path,
    *,
    fps: float = 24.0,
    seconds: float = 6.0,
    width: int = 320,
    height: int = 180,
) -> Path:
    """Create a deterministic clip for detector benchmarks."""
    output.parent.mkdir(parents=True, exist_ok=True)

    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise ValueError(f"Unable to create synthetic video: {output}")

    frame_count = int(round(fps * seconds))
    square_size = 36

    for frame_index in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        x = 10 + ((frame_index * 3) % max(1, width - square_size - 20))
        y = int(height * 0.45)

        cv2.rectangle(
            frame,
            (x, y),
            (x + square_size, y + square_size),
            (255, 255, 255),
            thickness=-1,
        )

        cv2.putText(
            frame,
            f"{frame_index:03d}",
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (180, 180, 180),
            1,
            cv2.LINE_AA,
        )

        writer.write(frame)

    writer.release()
    return output
