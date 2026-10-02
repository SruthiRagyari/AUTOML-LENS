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
    from app.main import llm_manager
    return llm_manager


@router.post("/chat")
async def chat(req: ChatRequest, db: Session = Depends(get_db)):
    """AI assistant endpoint."""
    llm = _get_llm_manager()

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
                experiment_context["preprocessing_summary"] = json.loads(exp.preprocessing_json)
            if exp.explainability_json:
                expl = json.loads(exp.explainability_json)
                experiment_context["top_features"] = expl.get("top_features", [])
            if exp.results_json:
                results = json.loads(exp.results_json)
                experiment_context["models_trained"] = len(results)
                experiment_context["models_completed"] = sum(1 for r in results if r.get("status") == "COMPLETED")

    response = await llm.chat(req.message, experiment_context)
    provider = llm.get_provider_info()

    return {
        "response": response,
        "provider": provider["name"],
    }
