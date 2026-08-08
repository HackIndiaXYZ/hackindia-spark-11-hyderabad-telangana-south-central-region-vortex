import logging
import os
from typing import Any, Dict, List, Optional
import httpx
from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Dynamic Gemini Model Manager.

    Handles loading primary/fallback models, dynamic endpoint construction,
    model caching, automatic discovery/failover, and startup validation.
    """

    def __init__(self):
        self._cached_active_model: Optional[str] = None
        self.last_failure_info: Optional[str] = None
        self.model_cache_status: str = "Uninitialized"

    @property
    def primary_model(self) -> str:
        return settings.gemini_primary_model or os.getenv("GEMINI_PRIMARY_MODEL", "gemini-3.5-flash")

    @property
    def fallback_models(self) -> List[str]:
        fallback_str = settings.gemini_fallback_models or os.getenv("GEMINI_FALLBACK_MODELS", "")
        if not fallback_str:
            return []
        return [m.strip() for m in fallback_str.split(",") if m.strip()]

    def get_all_candidate_models(self) -> List[str]:
        """Returns the list of all configured models in preference order (primary first)."""
        candidates = [self.primary_model]
        for m in self.fallback_models:
            if m not in candidates:
                candidates.append(m)
        return candidates

    def get_active_model(self) -> str:
        """Returns the currently cached successful model, or the primary model if none is cached."""
        if self._cached_active_model:
            return self._cached_active_model
        return self.primary_model

    def build_endpoint(self, model_name: str) -> str:
        """Dynamically constructs the Gemini API endpoint for a specific model."""
        return f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"


    def cache_model(self, model_name: str):
        """Caches a successful model name for future requests."""
        if self._cached_active_model != model_name:
            if settings.debug_ai:
                logger.info(f"[MODEL MANAGER] Caching successful model: '{model_name}'")
            self._cached_active_model = model_name
            self.model_cache_status = "Cached"

    def invalidate_cache(self, failed_model: str):
        """Invalidates the cache if the currently cached model fails."""
        if self._cached_active_model == failed_model:
            if settings.debug_ai:
                logger.info(f"[MODEL MANAGER] Invalidating cached model: '{failed_model}' due to failure")
            self._cached_active_model = None
            self.model_cache_status = "Invalidated"

    async def verify_models_startup(self) -> Dict[str, Any]:
        """Probes all configured models at startup, caches the first working one, and prints status report."""
        api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            logger.warning("[MODEL MANAGER] Cannot verify models at startup: GEMINI_API_KEY is missing.")
            self.model_cache_status = "Missing API Key"
            return {"gemini_connected": False, "active_model": self.primary_model, "available_models": []}

        candidates = self.get_all_candidate_models()
        available = []
        first_working = None

        logger.info(f"[MODEL MANAGER] Starting verification of models: {candidates}")

        for model in candidates:
            endpoint = f"{self.build_endpoint(model)}?key={api_key}"
            try:
                # Probe request with light prompt and short timeout
                async with httpx.AsyncClient(timeout=5.0) as client:
                    resp = await client.post(
                        endpoint,
                        json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
                        headers={"Content-Type": "application/json"},
                    )
                    if resp.status_code == 200:
                        available.append(model)
                        if not first_working:
                            first_working = model
                        if settings.debug_ai:
                            logger.info(f"[MODEL MANAGER] Probe successful for model '{model}' (200 OK)")
                    else:
                        if settings.debug_ai:
                            logger.warning(f"[MODEL MANAGER] Probe failed for model '{model}' with HTTP {resp.status_code}")
            except Exception as e:
                if settings.debug_ai:
                    logger.warning(f"[MODEL MANAGER] Probe exception for model '{model}': {e}")

        # Cache the first working model discovered
        if first_working:
            self.cache_model(first_working)
            print("[+] Gemini Connected")
            print(f"[+] Active Model: {first_working}")
            print(f"[+] Fallback Ready: {', '.join([m for m in candidates if m != first_working])}")
            print("[+] Cache Initialized")
        else:
            self.model_cache_status = "No working model found"
            logger.error("[MODEL MANAGER] No configured Gemini model was successfully connected.")

        return {
            "gemini_connected": len(available) > 0,
            "active_model": self.get_active_model(),
            "available_models": available,
        }


# Singleton instance
model_manager = ModelManager()
