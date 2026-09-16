import cv2
from fastapi import APIRouter

router = APIRouter(tags=["health"])


@router.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "VisionGuard QA",
        "opencv": cv2.__version__,
    }
