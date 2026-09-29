"""AutoML-Lens FastAPI Application Entry Point."""
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from pathlib import Path

from app.core.config import settings
from app.core.database import init_db
from app.api.datasets import router as datasets_router
from app.api.experiments import router as experiments_router
from app.api.chat import router as chat_router
from app.llm.manager import LLMManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global LLM manager
llm_manager: LLMManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown."""
    global llm_manager
    logger.info("Starting AutoML-Lens...")
    init_db()
    logger.info("Database initialized.")

    # Initialize LLM
    llm_config = {
        "LLM_PROVIDER": settings.LLM_PROVIDER,
        "GEMINI_API_KEY": settings.GEMINI_API_KEY,
        "GEMINI_MODEL": settings.GEMINI_MODEL,
        "OPENAI_API_KEY": settings.OPENAI_API_KEY,
        "OPENAI_BASE_URL": settings.OPENAI_BASE_URL,
        "OPENAI_MODEL": settings.OPENAI_MODEL,
    }
    llm_manager = LLMManager(llm_config)
    provider_info = llm_manager.get_provider_info()
    logger.info(f"LLM Provider: {provider_info['name']} (available: {provider_info['available']})")

    # Ensure storage directories
    settings.datasets_path
    settings.models_path
    settings.reports_path
    settings.predictions_path

    logger.info("AutoML-Lens ready!")
    yield
    logger.info("AutoML-Lens shutting down.")


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.APP_NAME,
        description=settings.APP_DESCRIPTION,
        version=settings.APP_VERSION,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins_list,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # API routes
    app.include_router(datasets_router, prefix="/api/datasets", tags=["Datasets"])
    app.include_router(experiments_router, prefix="/api/experiments", tags=["Experiments"])
    app.include_router(chat_router, prefix="/api", tags=["Chat"])

    # Health check
    @app.get("/api/health", tags=["Health"])
    async def health():
        provider_info = llm_manager.get_provider_info() if llm_manager else {"name": "Not initialized", "available": False}
        return {
            "status": "healthy",
            "version": settings.APP_VERSION,
            "llm_provider": provider_info["name"],
            "llm_available": provider_info["available"],
        }

    # Serve reports as static files
    reports_dir = settings.reports_path
    if reports_dir.exists():
        app.mount("/reports", StaticFiles(directory=str(reports_dir)), name="reports")

    # Serve predictions
    preds_dir = settings.predictions_path
    if preds_dir.exists():
        app.mount("/predictions", StaticFiles(directory=str(preds_dir)), name="predictions")

    return app


app = create_app()
