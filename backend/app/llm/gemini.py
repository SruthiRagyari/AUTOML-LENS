"""Google Gemini LLM provider."""
import json
import logging
from typing import Optional
from app.llm.base import LLMProvider, DatasetAnalysisResult
from app.services.model_registry import get_model_catalog
from app.services.pipeline_planner import AutoMLPipelinePlan, PipelinePlanValidator

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """LLM provider using Google Gemini API."""

    def __init__(self, api_key: str, model_name: str = "gemini-flash-lite-latest"):
        self.api_key = api_key
        self.model_name = model_name
        self._available: Optional[bool] = None
        self.client = None
        try:
            import google.generativeai as genai
            if self.api_key:
                genai.configure(api_key=self.api_key)
                self.client = genai.GenerativeModel(self.model_name)
        except ImportError:
            logger.warning("google-generativeai package not installed.")

    def get_provider_name(self) -> str:
        return "Google Gemini"

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available
        if not self.api_key or not self.client:
            self._available = False
            return False
        try:
            self.client.generate_content("ping")
            self._available = True
        except Exception as e:
            logger.warning(f"Gemini availability check failed: {e}")
            self._available = False
        return self._available

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from potentially markdown-wrapped response."""
        text = text.strip()
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
        return json.loads(text.strip())

    def _build_analysis_prompt(self, dataset_context: dict) -> str:
        """Build the analysis prompt from the compact description when available."""
        description_text = dataset_context.get("description_text")
        if not description_text:
            description_text = (
                "Dataset Context:\n"
                f"- Shape: {dataset_context.get('shape')}\n"
                f"- Columns: {dataset_context.get('columns')}\n"
                f"- Data Types: {json.dumps(dataset_context.get('dtypes', {}))}\n"
                f"- Missing Percentages: {json.dumps(dataset_context.get('missing_percentages', {}))}\n"
                f"- Unique Value Counts: {json.dumps(dataset_context.get('unique_counts', {}))}\n"
                f"- Target Column: {dataset_context.get('target_column', 'unknown')}\n"
                f"- Target Statistics: {json.dumps(dataset_context.get('target_stats', {}))}\n"
                f"- Sample Rows: {json.dumps(dataset_context.get('sample_rows', []))}\n"
                f"- Task Description: {dataset_context.get('task_description', 'Not provided')}"
            )
        catalog = json.dumps(
            dataset_context.get("feature_operation_catalog") or {}, default=str
        )
        clf_models = ", ".join(
            m["name"] for m in get_model_catalog("classification"))
        reg_models = ", ".join(
            m["name"] for m in get_model_catalog("regression"))
        return f"""You are an expert data scientist. Analyze this dataset and return ONLY a valid JSON object.

{description_text}

Return this exact JSON structure (no markdown fences, no extra text). Never invent numbers or statistics; if you cannot justify a confidence value, use null for it.
{{
    "problem_type": "classification" or "regression",
    "target_column": "column_name",
    "reasoning": "Detailed explanation of your analysis",
    "problem_understanding": "1-2 sentences describing the task and what success looks like",
    "preprocessing": [{{"column": "col", "action": "action", "reason": "why"}}],
    "feature_engineering": [{{"name": "feature", "description": "what", "type": "type"}}],
    "suggested_operations": [{{"column": "col", "operation": "operation_name", "params": {{}}, "reason": "why"}}],
    "model_recommendations": [{{"model_id": "registry_model_name", "reason": "why this model fits", "suitability": "fit for this dataset", "strengths": "expected strengths", "limitations": "expected limitations"}}],
    "useful_feature_candidates": ["column_with_signal"],
    "potentially_irrelevant_columns": ["redundant_or_noisy_column"],
    "leakage_warnings": ["risk_of_target_leakage"],
    "modelling_considerations": ["class_imbalance", "small_sample", "scaling_needed"],
    "candidate_models": ["model_name_1", "model_name_2"],
    "recommended_metric": "metric_name",
    "optimization_strategy": "optuna",
    "warnings": ["warning1"],
    "confidence": null or a number between 0 and 1 you can justify
}}

Feature-engineering rules (hard requirements):
- "suggested_operations" may ONLY use operation names allowed for each column in this per-column catalog: {catalog}
- Never suggest an operation for the target column; the label is never an input.
- Entry shape: {{"column": "...", "operation": "...", "params": {{}}, "reason": "..."}}. "interaction" requires params.with_column (a numeric column); "rare_category_grouping" accepts params.threshold (0-0.5) and params.group_label.
- Do not write code. The pipeline validates every entry against a fixed operation registry and executes only what passes; rejected entries are reported back, so stay within the catalog.
- Prefer at most 12 operations, only where they plausibly add signal.
Model-selection rules (hard requirements):
- "model_recommendations" and "candidate_models" may ONLY use model_id values from the registry lists below, and only models compatible with this problem type. Recommend 2-6 models ordered by expected suitability.
- Never invent model names and never write Python or model code; the pipeline validates every model_id against the registry and reports rejections.

Available model names (registry-validated - use ONLY these): classification: {clf_models} | regression: {reg_models}

Available classification metrics: accuracy, f1_weighted, precision_weighted, recall_weighted, balanced_accuracy, roc_auc
Available regression metrics: r2, neg_mean_squared_error, neg_mean_absolute_error, neg_root_mean_squared_error"""

    async def plan_pipeline(self, dataset_context: dict) -> AutoMLPipelinePlan:
        """Generate a structured, registry-validated AutoML pipeline plan using Gemini."""
        if not self.client:
            raise RuntimeError("Gemini client not initialized")

        prompt = self._build_pipeline_plan_prompt(dataset_context)
        for attempt in range(2):
            try:
                response = await self.client.generate_content_async(prompt)
                raw = self._parse_json_response(response.text)
                return PipelinePlanValidator.validate_and_sanitize(
                    raw, dataset_context, provider_used=self.get_provider_name(), model_used=self.model_name
                )
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"Gemini pipeline planning attempt 1 failed: {e}, retrying...")
                    prompt += "\n\nIMPORTANT: Return ONLY pure JSON. No markdown fences. No code."
                else:
                    raise RuntimeError(f"Gemini pipeline planning failed after 2 attempts: {e}")

    def _build_pipeline_plan_prompt(self, dataset_context: dict) -> str:
        """Build the structured pipeline planning prompt."""
        description_text = dataset_context.get("description_text")
        if not description_text:
            description_text = (
                "Dataset Context:\n"
                f"- Shape: {dataset_context.get('shape')}\n"
                f"- Columns: {dataset_context.get('columns')}\n"
                f"- Data Types: {json.dumps(dataset_context.get('dtypes', {}))}\n"
                f"- Missing Percentages: {json.dumps(dataset_context.get('missing_percentages', {}))}\n"
                f"- Unique Value Counts: {json.dumps(dataset_context.get('unique_counts', {}))}\n"
                f"- Target Column: {dataset_context.get('target_column', 'unknown')}\n"
                f"- Target Statistics: {json.dumps(dataset_context.get('target_stats', {}))}\n"
                f"- Sample Rows: {json.dumps(dataset_context.get('sample_rows', []))}\n"
                f"- Task Description: {dataset_context.get('task_description', 'Not provided')}"
            )
        catalog = json.dumps(
            dataset_context.get("feature_operation_catalog") or {}, default=str
        )
        clf_models = ", ".join(m["name"] for m in get_model_catalog("classification"))
        reg_models = ", ".join(m["name"] for m in get_model_catalog("regression"))
        target_col = dataset_context.get("target_column", "target")

        return f"""You are an expert AutoML architect. Design a structured AutoML Pipeline Plan for this dataset.
Return ONLY a valid JSON object (no markdown fences, no text outside JSON).

{description_text}

Required JSON structure:
{{
    "problem_type": "classification" or "regression",
    "target_column": "{target_col}",
    "primary_metric": "f1_weighted" or "neg_root_mean_squared_error",
    "preprocessing_strategy": {{
        "numeric_imputation": "median",
        "categorical_imputation": "most_frequent",
        "scaling": "standard",
        "max_categories": 20
    }},
    "feature_operations": [
        {{"column": "col_name", "operation": "op_name", "params": {{}}, "reason": "why"}}
    ],
    "candidate_models": ["model_1", "model_2", "model_3"],
    "hyperparameter_strategy": {{
        "n_trials": 20,
        "n_folds": 5,
        "mode": "automl"
    }},
    "ensemble_strategy": {{
        "enabled": true,
        "strategies": ["weighted_average", "soft_voting", "stacking"]
    }},
    "rationale": "High-level architectural rationale for this plan"
}}

REGISTRY CONSTRAINTS:
1. NEVER generate Python code. The pipeline will strictly reject arbitrary code.
2. "candidate_models" MUST only be chosen from:
   Classification: {clf_models}
   Regression: {reg_models}
3. "feature_operations" MUST only use allowed operations from this catalog: {catalog}.
   NEVER propose an operation on the target column ('{target_col}').
4. Valid classification metrics: accuracy, f1_weighted, precision_weighted, recall_weighted, balanced_accuracy, roc_auc
   Valid regression metrics: r2, neg_mean_squared_error, neg_mean_absolute_error, neg_root_mean_squared_error"""

    async def analyze_dataset(self, dataset_context: dict) -> DatasetAnalysisResult:
        if not self.client:
            raise RuntimeError("Gemini client not initialized")

        prompt = self._build_analysis_prompt(dataset_context)

        for attempt in range(2):
            try:
                response = await self.client.generate_content_async(prompt)
                raw = self._parse_json_response(response.text)
                return self.validate_analysis(raw, dataset_context)
            except Exception as e:
                if attempt == 0:
                    logger.warning(f"Gemini analysis attempt 1 failed: {e}, retrying...")
                    prompt += "\n\nIMPORTANT: Return ONLY pure JSON. No markdown. No explanation."
                else:
                    raise RuntimeError(f"Gemini dataset analysis failed after 2 attempts: {e}")

    async def explain_results(self, results_context: dict) -> str:
        if not self.client:
            raise RuntimeError("Gemini client not initialized")

        prompt = f"""You are an ML expert explaining real experiment results.
IMPORTANT: Do NOT invent any numbers. Only reference the data provided below.

Experiment Results:
{json.dumps(results_context, indent=2, default=str)}

Provide a clear, professional explanation covering:
1. Which model won and its score
2. How it compared to other models
3. Key hyperparameters that contributed to performance
4. Any observations or recommendations
Keep it concise (3-5 paragraphs)."""

        response = await self.client.generate_content_async(prompt)
        return response.text

    async def chat(self, message: str, experiment_context: Optional[dict] = None,
                   history: Optional[list] = None, context_mode: str = "general", **kwargs) -> str:
        if not self.client:
            raise RuntimeError("Gemini client not initialized")

        system_instruction = (
            "You are AutoML-Lens AI, an intelligent, helpful, and versatile general-purpose AI assistant. "
            "You can answer general knowledge questions, write and debug code, explain concepts across science, "
            "engineering, mathematics, philosophy, literature, and everyday life, as well as assist with machine learning and data science. "
            "When specific AutoML-Lens project/experiment context is provided, use it accurately to explain the user's models, "
            "datasets, metrics, and results. When no project context is provided, answer helpfully as a general-purpose AI. "
            "Format your responses cleanly in Markdown with bold headers, bullet lists, and syntax-highlighted code blocks."
        )

        ctx_str = ""
        if context_mode == "project" and experiment_context:
            ctx_str = f"\n\n[Active AutoML-Lens Project Context]:\n{json.dumps(experiment_context, indent=2, default=str)}"

        convo_lines = []
        if history:
            for item in history[-10:]:
                role = "User" if item.get("role") == "user" else "Assistant"
                convo_lines.append(f"{role}: {item.get('content', '')}")

        history_str = ""
        if convo_lines:
            history_str = "\n\nConversation History:\n" + "\n".join(convo_lines)

        prompt = f"{system_instruction}{ctx_str}{history_str}\n\nUser Question: {message}"

        response = await self.client.generate_content_async(prompt)
        return response.text
