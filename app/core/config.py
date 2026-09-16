from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "VisionGuard QA"
    app_env: str = "development"
    upload_dir: Path = Path("uploads")
    result_dir: Path = Path("results")
    max_upload_mb: int = 100

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
settings.upload_dir.mkdir(parents=True, exist_ok=True)
settings.result_dir.mkdir(parents=True, exist_ok=True)
