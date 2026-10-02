"""Application configuration using pydantic-settings.

Configuration is anchored to the ``backend/`` directory instead of the current
working directory, so ``uvicorn`` behaves identically whether it is launched
from ``backend/`` or from the repository root.
"""
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# backend/  (this file lives at backend/app/core/config.py)
BACKEND_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BACKEND_DIR / ".env"

# Load backend/.env explicitly (a missing file is not an error).
load_dotenv(ENV_FILE)


class Settings(BaseSettings):
    """Application settings loaded from environment variables / backend/.env."""

    model_config = SettingsConfigDict(env_file=str(ENV_FILE), extra="ignore")

    # App
    APP_NAME: str = "AutoML-Lens"
    APP_VERSION: str = "1.0.0"
    APP_DESCRIPTION: str = "LLM-Powered Automated Machine Learning Framework"
    DEBUG: bool = True
    APP_HOST: str = "0.0.0.0"
    APP_PORT: int = 8000
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000"

    # Database
    DATABASE_URL: str = "sqlite:///./automl_lens.db"

    # Storage
    STORAGE_PATH: str = "./storage"

    # LLM
    LLM_PROVIDER: str = "fallback"
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-flash-lite-latest"
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def resolved_database_url(self) -> str:
        """Return DATABASE_URL with relative SQLite paths anchored to backend/."""
        url = (self.DATABASE_URL or "").strip()
        if not url.startswith("sqlite:///"):
            return url
        raw = url[len("sqlite:///"):]
        if not raw or raw == ":memory:":
            return url
        path = Path(raw)
        if not path.is_absolute():
            path = BACKEND_DIR / path
        return f"sqlite:///{path.as_posix()}"

    @property
    def storage_path(self) -> Path:
        """Root storage directory (relative values resolve inside backend/)."""
        p = Path(self.STORAGE_PATH)
        if not p.is_absolute():
            p = BACKEND_DIR / p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def datasets_path(self) -> Path:
        p = self.storage_path / "datasets"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def models_path(self) -> Path:
        p = self.storage_path / "models"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def reports_path(self) -> Path:
        p = self.storage_path / "reports"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def predictions_path(self) -> Path:
        p = self.storage_path / "predictions"
        p.mkdir(parents=True, exist_ok=True)
        return p


settings = Settings()
