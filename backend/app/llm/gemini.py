"""Google Gemini LLM provider."""
import json
import logging
from typing import Optional
from app.llm.base import LLMProvider, DatasetAnalysisResult

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    """LLM provider using Google Gemini API."""

    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
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

    async def analyze_dataset(self, dataset_context: dict) -> DatasetAnalysisResult:
        if not self.client:
            raise RuntimeError("Gemini client not initialized")

        prompt = f"""You are an expert data scientist. Analyze this dataset and return ONLY a valid JSON object.

Dataset Context:
- Shape: {dataset_context.get('shape')}
- Columns: {dataset_context.get('columns')}
- Data Types: {json.dumps(dataset_context.get('dtypes', {}))}
- Missing Percentages: {json.dumps(dataset_context.get('missing_percentages', {}))}
- Unique Value Counts: {json.dumps(dataset_context.get('unique_counts', {}))}
- Target Column: {dataset_context.get('target_column', 'unknown')}
- Target Statistics: {json.dumps(dataset_context.get('target_stats', {}))}
- Sample Rows: {json.dumps(dataset_context.get('sample_rows', []))}
- Task Description: {dataset_context.get('task_description', 'Not provided')}

Return this exact JSON structure (no markdown fences, no extra text):
{{
    "problem_type": "classification" or "regression",
    "target_column": "column_name",
    "reasoning": "Detailed explanation of your analysis",
    "preprocessing": [{{"column": "col", "action": "action", "reason": "why"}}],
    "feature_engineering": [{{"name": "feature", "description": "what", "type": "type"}}],
    "candidate_models": ["model_name_1", "model_name_2"],
    "recommended_metric": "metric_name",
    "optimization_strategy": "optuna",
    "warnings": ["warning1"],
    "confidence": 0.85
}}

Available model names: logistic_regression, decision_tree_clf, random_forest_clf, gradient_boosting_clf, hist_gradient_boosting_clf, knn_clf, svm_clf, naive_bayes, linear_regression, ridge, lasso, decision_tree_reg, random_forest_reg, gradient_boosting_reg, hist_gradient_boosting_reg, knn_reg, svr

Available classification metrics: accuracy, f1_weighted, precision_weighted, recall_weighted, balanced_accuracy, roc_auc
Available regression metrics: r2, neg_mean_squared_error, neg_mean_absolute_error, neg_root_mean_squared_error"""

        for attempt in range(2):
            try:
                response = await self.client.generate_content_async(prompt)
                raw = self._parse_json_response(response.text)
                return self.validate_analysis(raw)
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

    async def chat(self, message: str, experiment_context: Optional[dict] = None) -> str:
        if not self.client:
            raise RuntimeError("Gemini client not initialized")

        ctx_str = ""
        if experiment_context:
            ctx_str = f"\nExperiment Context:\n{json.dumps(experiment_context, indent=2, default=str)}"

        prompt = f"""You are the AutoML-Lens AI assistant. Help users understand their ML experiments.
Rules:
- Answer based on the experiment data provided
- If data is not available, explain concepts in general ML terms
- Never invent metrics or results
- Be concise and helpful
{ctx_str}

User Question: {message}"""

        response = await self.client.generate_content_async(prompt)
        return response.text
