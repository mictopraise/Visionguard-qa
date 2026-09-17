from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes_compare import router as comparison_router
from app.api.routes_health import router as health_router
from app.core.config import settings

FRONTEND_DIR = Path("frontend")

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Agentic video QA powered by OpenCV 5.",
)

app.mount("/ui", StaticFiles(directory=FRONTEND_DIR), name="ui")
app.mount("/results", StaticFiles(directory=settings.result_dir), name="results")

app.include_router(health_router)
app.include_router(comparison_router)


@app.get("/", include_in_schema=False)
def root() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")
