"""Application configuration using pydantic-settings."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parent.parent.parent.parent
load_dotenv(_project_root / ".env")


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

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
    GEMINI_MODEL: str = "gemini-2.0-flash"
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = "https://api.openai.com/v1"
    OPENAI_MODEL: str = "gpt-4o-mini"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]

    @property
    def storage_path(self) -> Path:
        p = Path(self.STORAGE_PATH)
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

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
