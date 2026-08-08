import logging
from typing import Any, Dict, Optional
from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from app.schemas.ai import ChatRequest, ChatResponse, PlanRequest
from app.services.ai_service import ai_service
from app.services.memory_service import memory_service

logger = logging.getLogger(__name__)

router = APIRouter()


class TestRequest(BaseModel):
    prompt: Optional[str] = Field(default="Hello Gemini", description="Raw prompt to send to Gemini for test")


@router.get("/status", summary="Get AI System & Gemini Telemetry Status")
async def get_ai_status() -> Dict[str, Any]:
    """Returns AI backend running state, Gemini connection status, loaded model, endpoint, and telemetry."""
    return ai_service.get_ai_status()


@router.post("/test", summary="Test Raw Gemini API Connection")
async def test_gemini_raw(req: Optional[TestRequest] = None) -> Dict[str, Any]:
    """Sends a raw test prompt directly to Gemini API and returns unparsed JSON response without filtering."""
    prompt_str = req.prompt if req and req.prompt else "Hello Gemini"
    return await ai_service.test_gemini_raw(prompt_str)


@router.post("/chat", response_model=ChatResponse, summary="Chat with Saha AI Co-Driver Companion")
async def chat_with_saha(req: ChatRequest) -> ChatResponse:
    """Intelligent, session-aware chat endpoint with Live Intelligence Layer (Weather & OSM POIs)."""
    try:
        return await ai_service.chat_with_saha(req)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in chat_with_saha route: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process chat request: {str(e)}",
        )


@router.post("/plan", summary="Generate AI Trip Plan")
async def plan_trip_with_saha(req: PlanRequest) -> Dict[str, Any]:
    """Generates structured, dynamic JSON trip itineraries powered by Gemini AI with auto repair & validation."""
    try:
        return await ai_service.generate_trip_plan(req)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in plan_trip_with_saha route: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate trip plan: {str(e)}",
        )


@router.get("/memory/{session_id}", summary="Get Session Memory Context")
async def get_session_memory(session_id: str) -> Dict[str, Any]:
    """Retrieves remembered session context and conversation history for debugging and frontend state inspection."""
    session = memory_service.get_or_create_session(session_id)
    return {
        "session_id": session.session_id,
        "context": session.context.model_dump(),
        "messages": [m.model_dump() for m in session.messages],
    }


@router.delete("/memory/{session_id}", summary="Clear Session Memory")
async def clear_session_memory(session_id: str) -> Dict[str, str]:
    """Clears conversation memory and session state for the given session ID."""
    success = memory_service.clear_session(session_id)
    if success:
        return {"message": f"Session memory cleared for session_id '{session_id}'"}
    return {"message": f"Session '{session_id}' not found"}
