from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.services.model_manager import model_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup verification of Gemini models via ModelManager
    await model_manager.verify_models_startup()
    
    # Auto-create database tables on startup
    try:
        from app.models.base import Base
        from app.core.database import engine
        import app.models  # ensure models are registered
        Base.metadata.create_all(bind=engine)
        print("[+] Database tables initialized successfully")
    except Exception as db_err:
        print(f"[-] Database auto-creation error: {db_err}")
        
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Sahachaara — AI-powered travel companion API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health Check ─────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"], summary="Health check")
async def health() -> dict:
    """Returns service health status. Used by load balancers and monitoring."""
    return {
        "status": "healthy",
        "app": settings.app_name,
        "version": settings.app_version,
        "environment": settings.environment,
    }


# ─── Root ─────────────────────────────────────────────────────────────────────
@app.get("/", tags=["System"], summary="API root")
async def root() -> dict:
    return {"message": f"Welcome to {settings.app_name} API", "docs": "/docs"}


# ─── Routers ──────────────────────────────────────────────────────────────────
from app.routers import ai, navigation, safety

app.include_router(ai.router, prefix="/api/v1/ai", tags=["Saha AI Companion"])
app.include_router(navigation.router, prefix="/api/v1/navigation", tags=["Navigation & Journey Support"])
app.include_router(safety.router, prefix="/api/v1/safety", tags=["Safety & Crowdsourcing"])
