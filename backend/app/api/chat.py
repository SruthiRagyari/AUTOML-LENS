"""Chat API route for AI assistant."""
import json
import logging
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.database import get_db, Experiment
from app.models.schemas import ChatRequest

logger = logging.getLogger(__name__)
router = APIRouter()


def _get_llm_manager():
    from app.main import llm_manager as active_manager
    if active_manager is not None:
        return active_manager
    from app.llm.manager import LLMManager
    from app.core.config import settings
    return LLMManager(
        {
            "LLM_PROVIDER": settings.LLM_PROVIDER,
            "GEMINI_API_KEY": settings.GEMINI_API_KEY,
            "GEMINI_MODEL": settings.GEMINI_MODEL,
            "OPENAI_API_KEY": settings.OPENAI_API_KEY,
            "OPENAI_BASE_URL": settings.OPENAI_BASE_URL,
            "OPENAI_MODEL": settings.OPENAI_MODEL,
        }
    )


@router.post("/chat")
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    """AI assistant endpoint."""
    llm = _get_llm_manager()

    history = req.history or []
    if len(history) > 10:
        history = history[-10:]

    experiment_context = None
    if req.experiment_id:
        exp = db.query(Experiment).filter(Experiment.id == req.experiment_id).first()
        if exp:
            experiment_context = {
                "target_column": exp.target_column,
                "problem_type": exp.problem_type,
                "primary_metric": exp.primary_metric,
                "mode": exp.mode,
                "status": exp.status,
                "best_model_name": exp.best_model_name,
                "best_score": exp.best_score,
            }
            if exp.preprocessing_json:
                try:
                    experiment_context["preprocessing_summary"] = json.loads(exp.preprocessing_json)
                except (TypeError, ValueError):
                    pass
            if exp.explainability_json:
                try:
                    expl = json.loads(exp.explainability_json)
                    experiment_context["top_features"] = expl.get("top_features", [])
                except (TypeError, ValueError, AttributeError):
                    pass
            if exp.results_json:
                try:
                    results = json.loads(exp.results_json)
                    experiment_context["models_trained"] = len(results)
                    experiment_context["models_completed"] = sum(1 for r in results if r.get("status") == "COMPLETED")
                except (TypeError, ValueError, AttributeError):
                    pass
    try:
        detail = await llm.chat_detail(
            req.message,
            experiment_context,
            history=history,
            context_mode=req.context_mode or "general",
        )
    except Exception as e:
        logger.warning("Chat provider failed; served honest fallback. error=%s", type(e).__name__)
        from app.llm.fallback import FallbackLLMProvider
        text = await FallbackLLMProvider().chat(
            req.message, experiment_context, history=history,
            context_mode=req.context_mode or "general",
        )
        detail = {"text": text, "provider_used": "Fallback (Deterministic)", "is_fallback": True}

    return {
        "response": detail["text"],
        "provider": detail["provider_used"],
        "is_fallback": detail["is_fallback"],
        "context_mode": req.context_mode or "general",
    }
