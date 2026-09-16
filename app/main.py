from fastapi import FastAPI

from app.api.routes_compare import router as comparison_router
from app.api.routes_health import router as health_router
from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Agentic video QA powered by OpenCV 5.",
)

app.include_router(health_router)
app.include_router(comparison_router)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "status": "foundation-ready",
        "docs": "/docs",
    }
