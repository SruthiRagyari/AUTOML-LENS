"""OpenAI-compatible LLM provider using httpx."""
import json
import logging
from typing import Optional
import httpx
from app.llm.base import LLMProvider, DatasetAnalysisResult
from app.services.model_registry import get_model_catalog

logger = logging.getLogger(__name__)


class OpenAICompatProvider(LLMProvider):
    """LLM provider for OpenAI-compatible APIs (OpenAI, Azure, local)."""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com/v1",
                 model: str = "gpt-4o-mini"):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._available: Optional[bool] = None

    def get_provider_name(self) -> str:
        return "OpenAI Compatible"

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        if not self.api_key:
            self._available = False
            return False
        try:
            with httpx.Client(timeout=10) as client:
                resp = client.get(
                    f"{self.base_url}/models",
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                self._available = resp.status_code == 200
        except Exception:
            self._available = False
        return self._available

    async def _call(self, messages: list[dict], temperature: float = 0.1) -> str:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    def _parse_json(self, text: str) -> dict:
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())

    async def analyze_dataset(self, dataset_context: dict) -> DatasetAnalysisResult:
        system = (
            "You are an expert data scientist. Analyze the dataset and return ONLY "
            "a valid JSON object matching the specified schema. No markdown fences."
        )
        description_text = dataset_context.get("description_text") or json.dumps(
            dataset_context, default=str
        )
        catalog = json.dumps(
            dataset_context.get("feature_operation_catalog") or {}, default=str
        )
        clf_models = ", ".join(
            m["name"] for m in get_model_catalog("classification"))
        reg_models = ", ".join(
            m["name"] for m in get_model_catalog("regression"))
        user = (
            f"Dataset:\n{description_text}\n\n"
            "Return JSON with keys: problem_type, target_column, reasoning, "
            "problem_understanding, "
            "preprocessing (list of {{column, action, reason}}), "
            "feature_engineering (list of {{name, description, type}}), "
            "suggested_operations (list of {{column, operation, params, reason}}), "
            "useful_feature_candidates, potentially_irrelevant_columns, "
            "leakage_warnings, modelling_considerations, "
            "candidate_models (list of registry model names), "
            "model_recommendations (list of {{model_id, reason, suitability, strengths, limitations}}), recommended_metric, "
            "optimization_strategy, warnings (list), confidence (0-1).\n\n"
            "Feature-engineering rules: suggested_operations may ONLY use operation "
            f"names allowed per column in this catalog {catalog}; never propose an "
            "operation for the target column; do not write code - the registry "
            "validates every entry and rejects anything unknown. "
            "\n\nModel-selection rules: model_recommendations may ONLY use "
            "registry model names for this problem type - "
            f"classification: {clf_models}; regression: {reg_models} - "
            "recommending 2-6 models. Never invent names and never write "
            "code; unknown or incompatible model_ids are rejected with a reason."
        )
        for attempt in range(2):
            try:
                text = await self._call([
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ])
                raw = self._parse_json(text)
                return self.validate_analysis(raw, dataset_context)
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"OpenAI analysis attempt 1 failed: {e}")
                    user += "\nReturn ONLY pure JSON."
                else:
                    raise RuntimeError(f"OpenAI analysis failed: {e}")

    async def explain_results(self, results_context: dict) -> str:
        messages = [
            {"role": "system", "content": "You are an ML expert. Explain real experiment results. Do not invent numbers."},
            {"role": "user", "content": f"Results:\n{json.dumps(results_context, default=str)}"},
        ]
        return await self._call(messages, temperature=0.3)

    async def chat(self, message: str, experiment_context: Optional[dict] = None) -> str:
        ctx = json.dumps(experiment_context, default=str) if experiment_context else "No experiment context."
        messages = [
            {"role": "system", "content": f"You are AutoML-Lens assistant. Context: {ctx}"},
            {"role": "user", "content": message},
        ]
        return await self._call(messages, temperature=0.3)
