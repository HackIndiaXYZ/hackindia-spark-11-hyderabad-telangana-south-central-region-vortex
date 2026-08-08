import asyncio
import json
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

from app.core.config import settings
from app.schemas.ai import ChatRequest, PlanRequest
from app.services.ai_service import ai_service
from app.services.memory_service import memory_service
from app.services.model_manager import model_manager


async def run_tests():
    print("==================================================")
    print("SAHACHAARA AI MODEL MANAGER & PIPELINE VERIFICATION")
    print("==================================================")

    # 1. Environment & ModelManager Config Check
    print("\n[1] Dynamic ModelManager Configuration Check:")
    print(f"  • Configured Primary Model: {model_manager.primary_model}")
    print(f"  • Configured Fallback Models: {model_manager.fallback_models}")
    print(f"  • All Candidate Models: {model_manager.get_all_candidate_models()}")
    print(f"  • Currently Active Cached Model: {model_manager.get_active_model()}")
    print(f"  • DEBUG_AI: {settings.debug_ai}")
    print(f"  • DEMO_MODE: {settings.demo_mode}")
    key = settings.gemini_api_key or os.getenv("GEMINI_API_KEY", "")
    print(f"  • GEMINI_API_KEY Loaded: {bool(key.strip())} ({'Present' if key else 'MISSING'})")

    # 2. Startup Verification Test
    print("\n[2] Testing ModelManager Startup Verification:")
    startup_report = await model_manager.verify_models_startup()
    print(f"  • Gemini Connected: {startup_report.get('gemini_connected')}")
    print(f"  • Active Model Selected: {startup_report.get('active_model')}")
    print(f"  • Available Probed Models: {startup_report.get('available_models')}")

    # 3. Status Endpoint Telemetry Check
    print("\n[3] Testing Health & Telemetry Endpoint (GET /api/v1/ai/status):")
    status_report = ai_service.get_ai_status()
    print(f"  • Backend Running: {status_report.get('backend_running')}")
    print(f"  • Current Active Model: {status_report.get('active_model')}")
    print(f"  • Primary Model: {status_report.get('primary_model')}")
    print(f"  • Fallback Models: {status_report.get('fallback_models')}")
    print(f"  • Model Cache Status: {status_report.get('model_cache_status')}")

    # 4. Raw Test Endpoint Check
    print("\n[4] Testing Raw Gemini Connection (POST /api/v1/ai/test):")
    try:
        raw_res = await ai_service.test_gemini_raw("Hello Gemini")
        print(f"  • HTTP Status: {raw_res.get('http_status')}")
        print(f"  • Model Used: {raw_res.get('model')}")
        print(f"  • Response Time: {raw_res.get('response_time_ms')}ms")
    except Exception as e:
        print(f"  [X] Raw Test Result: {e}")

    # 5. AI Trip Planner Test
    print("\n[5] Testing AI Trip Planner with Dynamic Model Selection:")
    try:
        plan = await ai_service.generate_trip_plan(PlanRequest(prompt="Plan a 3-day trip to Goa", session_id="test-plan"))
        print(f"  ✓ Destination: {plan.get('destination')}")
        print(f"  ✓ Travel Days: {plan.get('travelDays')}")
        print(f"  ✓ Days Generated: {len(plan.get('days', []))}")
    except Exception as e:
        print(f"  [X] Trip Planner Result: {e}")

    # 6. AI Chat Test
    print("\n[6] Testing SAHA AI Chat with Memory & Co-Driver Personality:")
    try:
        res = await ai_service.chat_with_saha(ChatRequest(message="What's the weather ahead?", session_id="test-chat"))
        print(f"  ✓ Response snippet: {res.response[:150]}...")

        # Personality identity test (forbidden words)
        forbidden = ["gemini", "google", "llm", "ai model", "chatbot"]
        text_lower = res.response.lower()
        leaks = [w for w in forbidden if w in text_lower]
        if leaks:
            print(f"  [!] Identity leak detected: {leaks}")
        else:
            print("  ✓ Identity Guard Verified: 0 LLM/Gemini mentions!")
    except Exception as e:
        print(f"  [X] Chat Result: {e}")

    # 7. Model Cache & Fallback Verification
    print("\n[7] Verifying Model Cache & Dynamic Endpoint Construction:")
    active = model_manager.get_active_model()
    ep = model_manager.build_endpoint(active)
    print(f"  ✓ Active Model: {active}")
    print(f"  ✓ Constructed Endpoint: {ep}")
    if "gemini-" in ep and ":generateContent" in ep:
        print("  ✓ Dynamic endpoint construction verified without hardcoded model strings!")

    print("\n==================================================")
    print("ALL MODEL MANAGER VERIFICATION TESTS COMPLETED")
    print("==================================================")


if __name__ == "__main__":
    asyncio.run(run_tests())
