"""LLM Manager - handles provider selection and fallback chain."""
import logging
from typing import Any, Optional
from app.llm.base import LLMProvider, DatasetAnalysisResult
from app.llm.fallback import FallbackLLMProvider
from app.llm.gemini import GeminiProvider
from app.llm.openai_compat import OpenAICompatProvider

logger = logging.getLogger(__name__)


class LLMManager:
    """Manages LLM provider lifecycle with automatic fallback.

    Every call reports which provider actually produced the answer, so the UI
    can label real-LLM output versus the deterministic fallback instead of
    guessing. API keys are never logged or returned.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.provider_type = config.get("LLM_PROVIDER", "fallback").lower()
        self._fallback = FallbackLLMProvider()
        self.active_provider = self._init_provider()
        logger.info(f"LLM Provider: {self.active_provider.get_provider_name()}")

    def _init_provider(self) -> LLMProvider:
        if self.provider_type == "gemini":
            key = self.config.get("GEMINI_API_KEY", "")
            model = self.config.get("GEMINI_MODEL", "gemini-flash-lite-latest")
            if key:
                provider = GeminiProvider(api_key=key, model_name=model)
                if provider.is_available():
                    return provider
                logger.warning("Gemini API key provided but not available.")
        elif self.provider_type == "openai":
            key = self.config.get("OPENAI_API_KEY", "")
            url = self.config.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
            model = self.config.get("OPENAI_MODEL", "gpt-4o-mini")
            if key:
                provider = OpenAICompatProvider(api_key=key, base_url=url, model=model)
                if provider.is_available():
                    return provider
                logger.warning("OpenAI API key provided but not available.")
        return self._fallback

    def get_provider(self) -> LLMProvider:
        return self.active_provider

    def get_provider_info(self) -> dict[str, Any]:
        return {
            "name": self.active_provider.get_provider_name(),
            "available": self.active_provider.is_available(),
            "type": type(self.active_provider).__name__,
            "is_fallback": self.active_provider is self._fallback,
            "label": self._display_label(self.active_provider),
        }

    def _display_label(self, provider: LLMProvider) -> str:
        """Human-readable, honest label for the provider that answered."""
        if provider is self._fallback:
            return "Rule-based (no LLM configured)"
        return provider.get_provider_name()

    def _payload(self, provider: LLMProvider,
                 fallback_reason: Optional[str] = None) -> dict[str, Any]:
        return {
            "provider_used": provider.get_provider_name(),
            "provider_label": self._display_label(provider),
            "is_fallback": provider is self._fallback,
            "fallback_reason": fallback_reason,
        }

    async def analyze_dataset(self, dataset_context: dict) -> dict[str, Any]:
        provider = self.active_provider
        fallback_reason: Optional[str] = None
        try:
            result = await self.active_provider.analyze_dataset(dataset_context)
        except Exception as e:
            logger.warning(f"Provider failed analysis: {e}. Using fallback.")
            provider = self._fallback
            fallback_reason = str(e)
            result = await self._fallback.analyze_dataset(dataset_context)

        payload = self._payload(provider, fallback_reason)
        payload["result"] = result.model_dump()
        # Provenance travels inside the result too, so a stored analysis stays
        # self-describing even when the payload wrapper is stripped away.
        payload["result"]["provider_used"] = payload["provider_used"]
        payload["result"]["is_fallback"] = payload["is_fallback"]
        return payload

    async def plan_pipeline(self, dataset_context: dict) -> dict[str, Any]:
        """Generate a structured AutoML pipeline plan with automatic fallback.

        Always produces a valid AutoMLPipelinePlan. On provider failure or timeout,
        falls back cleanly to deterministic pipeline planning with audit provenance.
        """
        provider = self.active_provider
        fallback_reason: Optional[str] = None
        try:
            plan = await self.active_provider.plan_pipeline(dataset_context)
        except Exception as e:
            logger.warning(f"Provider failed pipeline planning: {e}. Using deterministic fallback.")
            provider = self._fallback
            fallback_reason = str(e)
            plan = await self._fallback.plan_pipeline(dataset_context)
            plan.validation_status = "fallback"
            plan.source = "deterministic_fallback"
            plan.fallback_reason = str(e)

        payload = self._payload(provider, fallback_reason)
        payload["plan"] = plan.model_dump()
        payload["plan"]["provider_used"] = payload["provider_used"]
        payload["plan"]["is_fallback"] = payload["is_fallback"]
        if fallback_reason:
            payload["plan"]["fallback_reason"] = fallback_reason
        return payload

    async def explain_results(self, results_context: dict) -> str:
        """Explanation text only (backwards-compatible helper)."""
        detail = await self.explain_results_detail(results_context)
        return detail["text"]

    async def explain_results_detail(self, results_context: dict) -> dict[str, Any]:
        provider = self.active_provider
        fallback_reason: Optional[str] = None
        try:
            text = await self.active_provider.explain_results(results_context)
        except Exception as e:
            logger.warning(f"Provider failed explanation: {e}. Using fallback.")
            provider = self._fallback
            fallback_reason = str(e)
            text = await self._fallback.explain_results(results_context)

        payload = self._payload(provider, fallback_reason)
        payload["text"] = text
        return payload

    async def chat(self, message: str, experiment_context: Optional[dict] = None) -> str:
        """Chat text only (backwards-compatible helper)."""
        detail = await self.chat_detail(message, experiment_context)
        return detail["text"]

    async def chat_detail(self, message: str,
                          experiment_context: Optional[dict] = None) -> dict[str, Any]:
        provider = self.active_provider
        fallback_reason: Optional[str] = None
        try:
            text = await self.active_provider.chat(message, experiment_context)
        except Exception as e:
            logger.warning(f"Provider failed chat: {e}. Using fallback.")
            provider = self._fallback
            fallback_reason = str(e)
            text = await self._fallback.chat(message, experiment_context)

        payload = self._payload(provider, fallback_reason)
        payload["text"] = text
        return payload
