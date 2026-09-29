"""LLM Manager - handles provider selection and fallback chain."""
import logging
from typing import Any, Optional
from app.llm.base import LLMProvider, DatasetAnalysisResult
from app.llm.fallback import FallbackLLMProvider
from app.llm.gemini import GeminiProvider
from app.llm.openai_compat import OpenAICompatProvider

logger = logging.getLogger(__name__)


class LLMManager:
    """Manages LLM provider lifecycle with automatic fallback."""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.provider_type = config.get("LLM_PROVIDER", "fallback").lower()
        self._fallback = FallbackLLMProvider()
        self.active_provider = self._init_provider()
        logger.info(f"LLM Provider: {self.active_provider.get_provider_name()}")

    def _init_provider(self) -> LLMProvider:
        if self.provider_type == "gemini":
            key = self.config.get("GEMINI_API_KEY", "")
            model = self.config.get("GEMINI_MODEL", "gemini-2.0-flash")
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
        }

    async def analyze_dataset(self, dataset_context: dict) -> dict[str, Any]:
        try:
            result = await self.active_provider.analyze_dataset(dataset_context)
            return {"result": result.model_dump(), "provider_used": self.active_provider.get_provider_name()}
        except Exception as e:
            logger.warning(f"Provider failed analysis: {e}. Using fallback.")
            result = await self._fallback.analyze_dataset(dataset_context)
            return {"result": result.model_dump(), "provider_used": self._fallback.get_provider_name()}

    async def explain_results(self, results_context: dict) -> str:
        try:
            return await self.active_provider.explain_results(results_context)
        except Exception as e:
            logger.warning(f"Provider failed explanation: {e}. Using fallback.")
            return await self._fallback.explain_results(results_context)

    async def chat(self, message: str, experiment_context: Optional[dict] = None) -> str:
        try:
            return await self.active_provider.chat(message, experiment_context)
        except Exception as e:
            logger.warning(f"Provider failed chat: {e}. Using fallback.")
            return await self._fallback.chat(message, experiment_context)
