import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
import httpx
from fastapi import HTTPException, status
from pydantic import ValidationError

from app.core.config import settings
from app.schemas.ai import (
    ActivityItem,
    BudgetItem,
    ChatRequest,
    ChatResponse,
    GeneratedItinerary,
    IntelligenceContext,
    ItineraryDay,
    PackingCategory,
    PackingItem,
    PlanRequest,
    TripPlanInput,
)
from app.services.intelligence.intelligence_service import IntelligenceService
from app.services.memory_service import SessionMemory, memory_service
from app.services.model_manager import model_manager

logger = logging.getLogger(__name__)


class AIService:
    """Production AI Service orchestrating Gemini calls via ModelManager, JSON repair, Memory, and Live Intelligence."""

    def __init__(self):
        self.intelligence_service = IntelligenceService()

        # Telemetry & Status attributes
        self.last_gemini_request: Optional[Dict[str, Any]] = None
        self.last_http_status: Optional[int] = None
        self.last_response_time_ms: Optional[float] = None
        self.last_error: Optional[str] = None
        self.gemini_connected: bool = False

    def _get_api_key(self) -> str:
        """Loads and verifies GEMINI_API_KEY. Throws descriptive error if missing and not in DEMO_MODE."""
        api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        if not api_key:
            err_msg = (
                "GEMINI_API_KEY is not configured in backend environment variables. "
                "Please set GEMINI_API_KEY in backend/.env or set DEMO_MODE=true."
            )
            self.last_error = err_msg
            if not settings.demo_mode:
                logger.error(err_msg)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=err_msg,
                )
        return api_key

    def _log_pre_request(self, prompt: str, session_id: Optional[str], model: str, endpoint: str) -> str:
        api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        api_key_loaded = bool(api_key.strip())
        now_iso = datetime.now(timezone.utc).isoformat()

        req_info = {
            "timestamp": now_iso,
            "api_key_loaded": api_key_loaded,
            "model": model,
            "endpoint": endpoint,
            "prompt_length": len(prompt),
            "session_id": session_id or "default",
        }
        self.last_gemini_request = req_info

        log_msg = (
            f"[GEMINI PRE-REQUEST] Time: {now_iso} | Key Loaded: {api_key_loaded} | "
            f"Model: {model} | Endpoint: {endpoint} | Prompt Len: {len(prompt)} | Session: {session_id}"
        )
        if settings.debug_ai:
            logger.info(log_msg)
        else:
            logger.debug(log_msg)

        return now_iso

    def _log_post_request(self, http_status: int, response_text: str, response_time_ms: float, retry_count: int):
        now_iso = datetime.now(timezone.utc).isoformat()
        self.last_http_status = http_status
        self.last_response_time_ms = response_time_ms
        self.gemini_connected = (http_status == 200)

        body_snippet = response_text if settings.debug_ai else response_text[:300]
        log_msg = (
            f"[GEMINI POST-REQUEST] Time: {now_iso} | HTTP Status: {http_status} | "
            f"Response Time: {response_time_ms:.2f}ms | Retry Count: {retry_count} | Body: {body_snippet}"
        )
        if settings.debug_ai:
            logger.info(log_msg)
        else:
            logger.debug(log_msg)

    def _handle_http_error(self, http_status: int, body_text: str, current_model: str):
        """Differentiates HTTP status errors from Gemini."""
        self.last_http_status = http_status
        self.gemini_connected = False

        if http_status in (400, 401):
            err = f"Gemini API authentication failed ({http_status}): Invalid API key or request parameters. Details: {body_text}"
        elif http_status == 403:
            err = f"Gemini API permission denied (403 Forbidden): Access restricted. Details: {body_text}"
        elif http_status == 404:
            err = f"Gemini API model endpoint not found (404 Not Found): Check model name '{current_model}'. Details: {body_text}"
        elif http_status == 429:
            err = f"Gemini API rate limit exceeded (429 Too Many Requests): Quota limit reached. Details: {body_text}"
        elif http_status >= 500:
            err = f"Gemini API server error ({http_status}): Service down or overloaded. Details: {body_text}"
        else:
            err = f"Gemini API error ({http_status}): Request failed. Details: {body_text}"

        self.last_error = err
        logger.error(err)

        if not settings.demo_mode:
            raise HTTPException(
                status_code=http_status if http_status in [400, 401, 403, 404, 429] else status.HTTP_502_BAD_GATEWAY,
                detail=err,
            )

    def _get_candidate_models(self) -> List[str]:
        """Gets ordered candidate models starting from active cached model."""
        active = model_manager.get_active_model()
        all_candidates = model_manager.get_all_candidate_models()
        ordered = [active]
        for m in all_candidates:
            if m not in ordered:
                ordered.append(m)
        return ordered

    def _is_404_error(self, http_status: int, body_text: str) -> bool:
        """Determines if error represents a model not found / 404 error."""
        if http_status == 404:
            return True
        text_upper = body_text.upper()
        return "MODEL_NOT_FOUND" in text_upper or "NOT_FOUND" in text_upper

    async def generate_trip_plan(self, req: PlanRequest) -> Dict[str, Any]:
        """Generates dynamic, structured JSON trip itineraries via Gemini with auto repair and ModelManager fallback."""
        api_key = self._get_api_key()

        if settings.demo_mode or not api_key:
            logger.info("Demo Mode active. Returning structured fallback itinerary.")
            return self._build_demo_plan_response(req.prompt)

        session = memory_service.get_or_create_session(req.session_id)
        remembered_ctx = session.context.model_dump()

        system_instruction = """You are SAHA, the world's best AI Travel Companion and Journey Planner.
Generate a structured JSON travel itinerary tailored specifically for the destination and request.

STRICT REQUIREMENTS:
1. Respond ONLY with valid, raw JSON (no markdown fences, no extra commentary).
2. The JSON structure MUST match this exact schema:
{
  "id": "saha-plan-unique",
  "destination": "Destination Name",
  "coverImage": "https://images.unsplash.com/photo-1506744038136-46273834b3fb",
  "travelDays": 3,
  "totalCostEstimate": "₹18,000 (₹4,500 / person)",
  "summary": "High-level summary of the trip...",
  "days": [
    {
      "dayNumber": 1,
      "title": "Day 1 Title",
      "dateStr": "Day 1",
      "summary": "Overview of day 1",
      "activities": [
        {
          "id": "act-1",
          "time": "08:00 AM",
          "period": "morning",
          "title": "Activity title",
          "description": "Activity description",
          "location": "Location name",
          "duration": "2 hours",
          "estimatedCost": "₹500",
          "category": "transit",
          "companionAdvice": "Saha co-driver tip for safety or route optimization"
        }
      ]
    }
  ],
  "budgetBreakdown": [
    {"category": "Accommodation", "amount": 7000, "percentage": 38.8, "color": "var(--color-pastel-purple)"},
    {"category": "Dining & Food", "amount": 5000, "percentage": 27.7, "color": "var(--color-pastel-teal)"},
    {"category": "Transit & Fuel", "amount": 4000, "percentage": 22.2, "color": "var(--color-pastel-amber)"},
    {"category": "Activities & Sightseeing", "amount": 1200, "percentage": 6.6, "color": "var(--color-pastel-rose)"},
    {"category": "Emergency Buffer", "amount": 800, "percentage": 4.4, "color": "var(--color-pastel-blue)"}
  ],
  "packingCategories": [
    {
      "name": "Driving & Travel Essentials",
      "iconName": "Car",
      "items": [
        {"id": "p1", "label": "Driver's license & vehicle docs", "checked": true},
        {"id": "p2", "label": "Tire pressure gauge", "checked": false}
      ]
    }
  ],
  "safetyNotes": ["Check mountain road fog before 8 AM", "Carry tire repair kit"],
  "weatherNotes": ["Expect cool temperatures at altitude", "Clear afternoon roads"],
  "drivingTips": ["Keep distance on winding ghats", "Refuel at main highway stations"]
}

Create a unique, realistic, and detailed itinerary tailored specifically for the destination and context requested!"""

        user_content = f"User Request: '{req.prompt}'\nSession Context: {remembered_ctx}\nAdditional Request Context: {req.context}"
        full_prompt = f"{system_instruction}\n\n{user_content}"

        candidate_models = self._get_candidate_models()
        last_exception: Optional[Exception] = None

        for idx, model in enumerate(candidate_models):
            endpoint_url = model_manager.build_endpoint(model)
            full_endpoint = f"{endpoint_url}?key={api_key}"

            if settings.debug_ai:
                logger.info(f"Attempting model:\n{model}")

            self._log_pre_request(
                prompt=full_prompt,
                session_id=req.session_id,
                model=model,
                endpoint=endpoint_url,
            )

            start_time = time.time()
            try:
                async with httpx.AsyncClient(timeout=30.0) as client:
                    resp = await client.post(
                        full_endpoint,
                        json={"contents": [{"role": "user", "parts": [{"text": full_prompt}]}]},
                        headers={"Content-Type": "application/json"},
                    )

                    elapsed_ms = (time.time() - start_time) * 1000
                    self._log_post_request(resp.status_code, resp.text, elapsed_ms, retry_count=idx)

                    if resp.status_code == 200:
                        if settings.debug_ai:
                            logger.info(f"200\nModel cached.")
                        model_manager.cache_model(model)

                        data = resp.json()
                        raw_text = data["candidates"][0]["content"]["parts"][0]["text"]

                        repaired_data = self._clean_and_repair_json(raw_text)
                        if repaired_data:
                            try:
                                itinerary = GeneratedItinerary.model_validate(repaired_data)
                                logger.info(f"Successfully generated dynamic trip plan for '{itinerary.destination}' using model '{model}'!")
                                self.last_error = None
                                return itinerary.model_dump()
                            except ValidationError as val_err:
                                err_msg = f"JSON schema validation error on Gemini response: {val_err}"
                                logger.warning(err_msg)
                                self.last_error = err_msg
                                if not settings.demo_mode:
                                    raise HTTPException(
                                        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                        detail=f"Gemini returned invalid JSON structure: {val_err}",
                                    )
                        else:
                            err_msg = f"Failed to repair JSON output from Gemini response: {raw_text[:200]}"
                            logger.warning(err_msg)
                            self.last_error = err_msg
                            if not settings.demo_mode:
                                raise HTTPException(
                                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                    detail=f"Malformed JSON returned by Gemini: {raw_text[:300]}",
                                )
                    elif self._is_404_error(resp.status_code, resp.text):
                        if settings.debug_ai:
                            logger.info(f"{resp.status_code}")
                        model_manager.invalidate_cache(model)
                        model_manager.last_failure_info = f"Model '{model}' returned {resp.status_code} NOT_FOUND."
                        if idx < len(candidate_models) - 1:
                            next_model = candidate_models[idx + 1]
                            if settings.debug_ai:
                                logger.info(f"Switching to:\n{next_model}")
                            continue
                        else:
                            self._handle_http_error(resp.status_code, resp.text, model)
                    else:
                        # Non-404 HTTP errors (401, 403, 429, 500) do NOT trigger fallback per Objective 5
                        self._handle_http_error(resp.status_code, resp.text, model)

            except (httpx.ConnectError, httpx.NetworkError) as net_err:
                self.gemini_connected = False
                err_msg = f"Network failure connecting to Gemini model '{model}': {net_err}"
                self.last_error = err_msg
                logger.error(err_msg)
                last_exception = net_err
                if not settings.demo_mode:
                    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=err_msg)
            except httpx.TimeoutException as time_err:
                self.gemini_connected = False
                err_msg = f"Timeout error waiting for Gemini model '{model}': {time_err}"
                self.last_error = err_msg
                logger.error(err_msg)
                last_exception = time_err
                if not settings.demo_mode:
                    raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=err_msg)
            except HTTPException:
                raise
            except Exception as e:
                err_msg = f"Unexpected error during Gemini trip plan generation: {e}"
                self.last_error = err_msg
                logger.error(err_msg)
                last_exception = e
                if not settings.demo_mode:
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=err_msg)

        if settings.demo_mode:
            logger.warning("Gemini generation failed or disabled. Returning demo plan fallback (DEMO_MODE=true).")
            return self._build_demo_plan_response(req.prompt)

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Failed to generate trip plan from Gemini: {last_exception}",
        )

    async def chat_with_saha(self, req: ChatRequest) -> ChatResponse:
        """Processes chat messages with Saha AI co-driver personality, memory, and ModelManager fallback."""
        api_key = self._get_api_key()

        session = memory_service.get_or_create_session(req.session_id)
        if req.context:
            session.update_context(req.context)

        # Store user message in memory
        session.add_message("user", req.message)

        # Fetch Live Intelligence Layer (Weather & Nearby POIs)
        intel: IntelligenceContext = await self.intelligence_service.get_live_context(
            query=req.message, context=req.context
        )

        if settings.demo_mode or not api_key:
            logger.info("Demo Mode active. Returning structured chat fallback.")
            response = self._build_demo_chat_response(req.message, req.persona, session, intel)
            session.add_message("model", response.response)
            return response

        # Construct Saha Co-Driver System Prompt
        system_prompt = self._build_saha_system_prompt(req.persona or "companion", session, intel)
        history_str = "\n".join([f"{m.role.capitalize()}: {m.content}" for m in session.messages[-8:]])
        full_prompt = f"{system_prompt}\n\nRecent Conversation:\n{history_str}\n\nUser Question: {req.message}\n\nSaha Co-Driver Response:"

        candidate_models = self._get_candidate_models()

        for idx, model in enumerate(candidate_models):
            endpoint_url = model_manager.build_endpoint(model)
            full_endpoint = f"{endpoint_url}?key={api_key}"

            if settings.debug_ai:
                logger.info(f"Attempting model:\n{model}")

            self._log_pre_request(
                prompt=full_prompt,
                session_id=session.session_id,
                model=model,
                endpoint=endpoint_url,
            )

            start_time = time.time()
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    resp = await client.post(
                        full_endpoint,
                        json={"contents": [{"role": "user", "parts": [{"text": full_prompt}]}]},
                        headers={"Content-Type": "application/json"},
                    )

                    elapsed_ms = (time.time() - start_time) * 1000
                    self._log_post_request(resp.status_code, resp.text, elapsed_ms, retry_count=idx)

                    if resp.status_code == 200:
                        if settings.debug_ai:
                            logger.info(f"200\nModel cached.")
                        model_manager.cache_model(model)

                        data = resp.json()
                        text = data["candidates"][0]["content"]["parts"][0]["text"]

                        # Post-process text to enforce identity constraint (never mention Gemini, Google, LLM, etc.)
                        text = self._sanitize_identity_mentions(text)

                        cards = self._generate_response_cards(req.message, intel)
                        suggestions = self._generate_response_suggestions(req.message, req.persona)

                        chat_res = ChatResponse(
                            response=text,
                            persona=req.persona or "companion",
                            is_demo=False,
                            suggestions=suggestions,
                            cards=cards,
                            session_id=session.session_id,
                        )
                        session.add_message("model", text)
                        self.last_error = None
                        return chat_res
                    elif self._is_404_error(resp.status_code, resp.text):
                        if settings.debug_ai:
                            logger.info(f"{resp.status_code}")
                        model_manager.invalidate_cache(model)
                        model_manager.last_failure_info = f"Model '{model}' returned {resp.status_code} NOT_FOUND."
                        if idx < len(candidate_models) - 1:
                            next_model = candidate_models[idx + 1]
                            if settings.debug_ai:
                                logger.info(f"Switching to:\n{next_model}")
                            continue
                        else:
                            self._handle_http_error(resp.status_code, resp.text, model)
                    else:
                        self._handle_http_error(resp.status_code, resp.text, model)

            except (httpx.ConnectError, httpx.NetworkError) as net_err:
                self.gemini_connected = False
                err_msg = f"Network error during chat with Gemini model '{model}': {net_err}"
                self.last_error = err_msg
                logger.error(err_msg)
                if not settings.demo_mode:
                    raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=err_msg)
            except httpx.TimeoutException as time_err:
                self.gemini_connected = False
                err_msg = f"Timeout error during chat with Gemini model '{model}': {time_err}"
                self.last_error = err_msg
                logger.error(err_msg)
                if not settings.demo_mode:
                    raise HTTPException(status_code=status.HTTP_504_GATEWAY_TIMEOUT, detail=err_msg)
            except HTTPException:
                raise
            except Exception as e:
                err_msg = f"Unexpected error calling Gemini Chat API with model '{model}': {e}"
                self.last_error = err_msg
                logger.error(err_msg)
                if not settings.demo_mode:
                    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=err_msg)

        if settings.demo_mode:
            fallback = self._build_demo_chat_response(req.message, req.persona, session, intel)
            session.add_message("model", fallback.response)
            return fallback

        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail="Chat service failed to respond.")

    async def test_gemini_raw(self, test_prompt: str = "Hello Gemini") -> Dict[str, Any]:
        """Sends a raw test prompt directly to Gemini API and returns unparsed JSON response."""
        api_key = self._get_api_key()
        active_model = model_manager.get_active_model()
        endpoint_url = model_manager.build_endpoint(active_model)
        endpoint = f"{endpoint_url}?key={api_key}"

        start_time = time.time()
        self._log_pre_request(
            prompt=test_prompt,
            session_id="test-endpoint",
            model=active_model,
            endpoint=endpoint_url,
        )

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.post(
                    endpoint,
                    json={"contents": [{"role": "user", "parts": [{"text": test_prompt}]}]},
                    headers={"Content-Type": "application/json"},
                )

                elapsed_ms = (time.time() - start_time) * 1000
                self._log_post_request(resp.status_code, resp.text, elapsed_ms, retry_count=0)

                try:
                    raw_json = resp.json()
                except Exception:
                    raw_json = {"raw_text": resp.text}

                return {
                    "http_status": resp.status_code,
                    "response_time_ms": round(elapsed_ms, 2),
                    "model": active_model,
                    "endpoint": endpoint_url,
                    "gemini_raw_response": raw_json,
                }
        except Exception as e:
            logger.error(f"Error testing Gemini raw endpoint: {e}")
            raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=f"Gemini Test Failed: {str(e)}")

    def get_ai_status(self) -> Dict[str, Any]:
        """Returns comprehensive status telemetry for GET /api/v1/ai/status."""
        api_key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
        api_key_loaded = bool(api_key.strip())
        active_model = model_manager.get_active_model()
        active_endpoint = model_manager.build_endpoint(active_model)

        return {
            "backend_running": True,
            "gemini_connected": self.gemini_connected,
            "gemini_api_key_loaded": api_key_loaded,
            "current_model": active_model,
            "current_endpoint": active_endpoint,
            "active_model": active_model,
            "primary_model": model_manager.primary_model,
            "fallback_models": model_manager.fallback_models,
            "last_successful_model": model_manager._cached_active_model,
            "last_failure": model_manager.last_failure_info or self.last_error,
            "model_cache_status": model_manager.model_cache_status,
            "memory_service_running": True,
            "weather_service_running": True,
            "openstreetmap_running": True,
            "demo_mode": settings.demo_mode,
            "debug_ai": settings.debug_ai,
            "last_gemini_request": self.last_gemini_request,
            "last_http_status": self.last_http_status,
            "last_response_time_ms": self.last_response_time_ms,
            "last_error": self.last_error,

            # Capitalized human-readable format
            "Backend Running": True,
            "Gemini Connected": self.gemini_connected,
            "Gemini API Key Loaded": api_key_loaded,
            "Current Active Model": active_model,
            "Configured Primary Model": model_manager.primary_model,
            "Fallback Models": model_manager.fallback_models,
            "Last Successful Model": model_manager._cached_active_model,
            "Last Failure": model_manager.last_failure_info or self.last_error,
            "Model Cache Status": model_manager.model_cache_status,
            "Memory Service Running": True,
            "Weather Service Running": True,
            "OpenStreetMap Running": True,
            "Demo Mode": settings.demo_mode,
            "Last Gemini Request": self.last_gemini_request,
            "Last HTTP Status": self.last_http_status,
        }

    def _build_saha_system_prompt(
        self, persona: str, session: SessionMemory, intel: IntelligenceContext
    ) -> str:
        ctx = session.context
        weather_desc = f"{intel.weather.temp_c}°C, {intel.weather.condition}" if intel.weather else "Clear"
        if intel.weather and intel.weather.warning:
            weather_desc += f" (ALERT: {intel.weather.warning})"

        nearby_str = ", ".join([f"{p.name} ({p.distance_km} km)" for p in intel.nearby_places[:3]]) or "None nearby"

        return f"""You are SAHA, an intelligent, calm, reassuring, professional, and confident travel co-driver for Sahachaara.

CRITICAL IDENTITY RULES:
- You are SAHA, a trusted driving partner and travel co-driver.
- NEVER mention "Gemini", "Google", "LLM", "AI model", "Chatbot", or "ChatGPT".
- Respond strictly as a live, human-like travel co-driver navigating the journey together.

SESSION MEMORY & TRAVELER STATE:
- Destination: {ctx.destination or 'En route'}
- Vehicle: {ctx.vehicle or 'Car'}
- Travelers: {ctx.travelers or 'Not specified'}
- Battery/Fuel: {ctx.battery_level or 'Normal'}
- Food Preference: {ctx.food_preference or 'Anything'}

LIVE REAL-TIME INTELLIGENCE LAYER:
- Location: {intel.location_name}
- Live Weather: {weather_desc}
- Nearby Recommended Places/Services: {nearby_str}

RESPONSE STRUCTURE PRIORITIES:
1. Short & Reassuring Summary
2. Key Recommendations / Driving Advice
3. Safety Warning (if relevant)
4. Next Action steps

Keep responses practical, concise, and structured with bullet points or short paragraphs for safe driving readability."""

    def _sanitize_identity_mentions(self, text: str) -> str:
        """Sanitizes any accidental leaks of LLM / Gemini identity in text response."""
        replacements = [
            (r"\bGemini\b", "Saha Engine"),
            (r"\bGoogle\b", "Sahachaara"),
            (r"\bAI model\b", "co-driver system"),
            (r"\bLLM\b", "co-driver assistant"),
            (r"\bChatbot\b", "co-driver"),
        ]
        for pattern, repl in replacements:
            text = re.sub(pattern, repl, text, flags=re.IGNORECASE)
        return text

    def _clean_and_repair_json(self, raw_text: str) -> Optional[Dict[str, Any]]:
        """Cleans markdown fence blocks, repairs trailing commas, and extracts JSON objects."""
        if not raw_text:
            return None

        cleaned = raw_text.strip()

        # Remove markdown ```json and ``` fences
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

        # Repair trailing commas inside arrays and objects: e.g. ", ]" -> "]" or ", }" -> "}"
        cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)

        # Attempt direct json loads
        try:
            return json.loads(cleaned)
        except Exception:
            pass

        # Substring JSON search heuristic
        start_idx = cleaned.find("{")
        end_idx = cleaned.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            json_str = cleaned[start_idx : end_idx + 1]
            json_str = re.sub(r",\s*([\]}])", r"\1", json_str)
            try:
                return json.loads(json_str)
            except Exception as ex:
                logger.warning(f"JSON substring repair attempt failed: {ex}")

        return None

    def _generate_response_cards(self, query: str, intel: IntelligenceContext) -> List[Dict[str, Any]]:
        """Generates dynamic UI cards based on query and live intelligence."""
        cards = []
        if intel.nearby_places:
            top_place = intel.nearby_places[0]
            cards.append(
                {
                    "type": "recommendation",
                    "title": top_place.name,
                    "category": top_place.category.replace("_", " ").title(),
                    "rating": top_place.rating,
                    "distance": f"{top_place.distance_km} km ahead",
                    "description": f"Verified live recommendation near {intel.location_name}.",
                    "action": "Set Waypoint",
                }
            )

        if intel.weather and intel.weather.warning:
            cards.append(
                {
                    "type": "safety_alert",
                    "severity": "warning",
                    "title": f"Live Weather: {intel.weather.condition}",
                    "description": intel.weather.warning,
                    "recommendation": "Reduce speed and maintain 4-second trailing distance.",
                }
            )
        return cards

    def _generate_response_suggestions(self, query: str, persona: Optional[str]) -> List[str]:
        q = query.lower()
        if "food" in q or "eat" in q or "restaurant" in q:
            return ["Find coffee stops", "Show vegetarian options", "Navigate to restaurant"]
        elif "ev" in q or "battery" in q:
            return ["Show fast chargers", "Check range to destination", "Find food while charging"]
        elif "weather" in q or "rain" in q:
            return ["Show safe detour", "Check destination weather", "Set rest stop alert"]
        return ["Plan a trip", "Find nearby food", "Check weather ahead", "Emergency support"]

    def _build_demo_chat_response(
        self, query: str, persona: Optional[str], session: SessionMemory, intel: IntelligenceContext
    ) -> ChatResponse:
        """Demo response enriched with real live intelligence."""
        q = query.lower()
        p = persona or "companion"
        ctx = session.context

        ctx_summary = ""
        if ctx.travelers or ctx.vehicle or ctx.battery_level:
            bits = [f"Vehicle: {ctx.vehicle}" if ctx.vehicle else None, f"Travelers: {ctx.travelers}" if ctx.travelers else None, f"Battery: {ctx.battery_level}" if ctx.battery_level else None]
            ctx_summary = f" (Remembered Context: {', '.join(filter(None, bits))})"

        weather_line = f"Live Weather: {intel.weather.temp_c}°C, {intel.weather.condition}." if intel.weather else ""

        if "food" in q or "restaurant" in q or "lunch" in q or "breakfast" in q:
            place = intel.nearby_places[0] if intel.nearby_places else None
            place_name = place.name if place else "The Highway Gourmet Café"
            place_dist = f"{place.distance_km} km ahead" if place else "6.2 km ahead"
            return ChatResponse(
                response=f"I've found great dining options for your group{ctx_summary}. {place_name} ({place_dist}) has a 4.8 ★ rating with fresh meals, EV charging, and clean washrooms.\n\n• Location: {intel.location_name}\n• {weather_line}",
                persona="guide",
                is_demo=True,
                suggestions=self._generate_response_suggestions(query, persona),
                cards=self._generate_response_cards(query, intel),
                session_id=session.session_id,
            )
        elif "rain" in q or "weather" in q or "fog" in q:
            warn = intel.weather.warning if intel.weather and intel.weather.warning else "Road surface may be wet. Maintain 50 km/h."
            return ChatResponse(
                response=f"Weather update for {intel.location_name}:\n{weather_line}\n\n• Safety Advice: {warn}\n• Co-Driver Tip: Keep fog lights on and maintain a 4-second trailing distance.",
                persona="safety",
                is_demo=True,
                suggestions=self._generate_response_suggestions(query, persona),
                cards=self._generate_response_cards(query, intel),
                session_id=session.session_id,
            )
        else:
            return ChatResponse(
                response=f"Hi, I'm Saha! I'm tracking your journey to {ctx.destination or 'your destination'}{ctx_summary}. {weather_line} Road traffic is light and your safety score is 95/100. How can I assist your drive?",
                persona=p,
                is_demo=True,
                suggestions=self._generate_response_suggestions(query, persona),
                cards=self._generate_response_cards(query, intel),
                session_id=session.session_id,
            )

    def _build_demo_plan_response(self, prompt: str) -> Dict[str, Any]:
        """Structured fallback plan matching GeneratedItinerary model."""
        dest = prompt.strip().title() if prompt else "Araku Valley & Visakhapatnam"
        return {
            "id": f"saha-plan-demo-{abs(hash(prompt)) % 10000}",
            "destination": dest,
            "coverImage": "https://images.unsplash.com/photo-1506744038136-46273834b3fb",
            "travelDays": 3,
            "totalCostEstimate": "₹21,000 (₹4,200 / person)",
            "summary": f"Customized trip plan for '{prompt}': A scenic 3-day adventure with mountain drives, viewpoint stops, local cuisine, and rest stops.",
            "days": [
                {
                    "dayNumber": 1,
                    "title": "Scenic Highway Drive & Plantation Break",
                    "dateStr": "Day 1",
                    "summary": "Morning drive via main highway with scenic breaks.",
                    "activities": [
                        {
                            "id": "act-1",
                            "time": "07:00 AM",
                            "period": "morning",
                            "title": "Departure & Scenic Drive",
                            "description": "Smooth morning departure to beat city traffic.",
                            "location": "Main Highway Interchange",
                            "duration": "3 hours",
                            "estimatedCost": "₹1,200 Fuel",
                            "category": "transit",
                            "companionAdvice": "Saha Tip: Departure before 07:30 AM avoids heavy truck congestion.",
                        },
                        {
                            "id": "act-2",
                            "time": "10:30 AM",
                            "period": "morning",
                            "title": "Coffee Plantation & Rest Stop",
                            "description": "Artisanal coffee and local breakfast.",
                            "location": "Hilltop Organic Estate",
                            "duration": "1.5 hours",
                            "estimatedCost": "₹400",
                            "category": "food",
                            "companionAdvice": "Saha Tip: Great washroom facility and EV fast charger available.",
                        },
                    ],
                },
                {
                    "dayNumber": 2,
                    "title": "Viewpoint Sunrise & Cave Exploration",
                    "dateStr": "Day 2",
                    "summary": "Sunrise at mountain peak followed by local heritage sights.",
                    "activities": [
                        {
                            "id": "act-3",
                            "time": "06:00 AM",
                            "period": "morning",
                            "title": "Peak Sunrise Viewpoint",
                            "description": "Panoramic valley view from 3,500 ft elevation.",
                            "location": "Valley View Point",
                            "duration": "1.5 hours",
                            "estimatedCost": "Free",
                            "category": "nature",
                            "companionAdvice": "Saha Tip: Early morning mist present. Keep headlights on low beam.",
                        }
                    ],
                },
            ],
            "budgetBreakdown": [
                {"category": "Accommodation", "amount": 8000, "percentage": 38.0, "color": "var(--color-pastel-purple)"},
                {"category": "Dining & Food", "amount": 5500, "percentage": 26.2, "color": "var(--color-pastel-teal)"},
                {"category": "Transit & Fuel", "amount": 5000, "percentage": 23.8, "color": "var(--color-pastel-amber)"},
                {"category": "Activities & Sightseeing", "amount": 1500, "percentage": 7.1, "color": "var(--color-pastel-rose)"},
                {"category": "Emergency Buffer", "amount": 1000, "percentage": 4.9, "color": "var(--color-pastel-blue)"},
            ],
            "packingCategories": [
                {
                    "name": "Driving & Travel Essentials",
                    "iconName": "Car",
                    "items": [
                        {"id": "p1", "label": "Vehicle registration & Driving License", "checked": True},
                        {"id": "p2", "label": "Tire pressure gauge & repair kit", "checked": False},
                        {"id": "p3", "label": "Fastag top-up (₹1,000)", "checked": True},
                    ],
                }
            ],
            "safetyNotes": ["Check weather for mountain fog", "Keep emergency numbers saved"],
            "weatherNotes": ["Cool morning breeze", "Sunny afternoon"],
            "drivingTips": ["Maintain safe trailing distance", "Use engine braking on downhill ghats"],
        }


# Global singleton instance
ai_service = AIService()
