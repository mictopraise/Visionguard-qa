import shutil
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.config import settings
from app.core.models import ComparisonJob
from app.video.probe import VideoProbeError, probe_video

router = APIRouter(prefix="/compare", tags=["comparison"])

ALLOWED_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def _save_upload(upload: UploadFile, job_dir: Path, label: str) -> Path:
    suffix = Path(upload.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(status_code=415, detail=f"Unsupported video format: {suffix or 'unknown'}")

    destination = job_dir / f"{label}{suffix}"
    with destination.open("wb") as output:
        shutil.copyfileobj(upload.file, output)

    max_bytes = settings.max_upload_mb * 1024 * 1024
    if destination.stat().st_size > max_bytes:
        destination.unlink(missing_ok=True)
        raise HTTPException(status_code=413, detail="Video exceeds upload size limit")
    return destination


@router.post("", response_model=ComparisonJob)
def create_comparison(
    video_a: UploadFile = File(...),
    video_b: UploadFile = File(...),
) -> ComparisonJob:
    job_id = uuid4().hex
    job_dir = settings.upload_dir / job_id
    job_dir.mkdir(parents=True, exist_ok=False)

    try:
        path_a = _save_upload(video_a, job_dir, "video_a")
        path_b = _save_upload(video_b, job_dir, "video_b")
        meta_a = probe_video(path_a, video_a.filename)
        meta_b = probe_video(path_b, video_b.filename)
    except (VideoProbeError, HTTPException):
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    return ComparisonJob(
        job_id=job_id,
        video_a=meta_a,
        video_b=meta_b,
        status="INGESTED",
        notes=["Foundation milestone: upload and OpenCV metadata probing complete."],
    )
