from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="Sahachaara — AI-powered travel companion API",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
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


# ─── Routers (add as modules are built) ───────────────────────────────────────
# from app.routers import auth, trips, companion
# app.include_router(auth.router, prefix="/api/v1/auth", tags=["Auth"])
# app.include_router(trips.router, prefix="/api/v1/trips", tags=["Trips"])
