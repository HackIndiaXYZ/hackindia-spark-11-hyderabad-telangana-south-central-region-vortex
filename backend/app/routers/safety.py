from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.observation import RoadObservation
from app.services.safety_service import safety_service

router = APIRouter()

# ─── Pydantic Validation Models ────────────────────────────────────────────────
class LatLngModel(BaseModel):
    lat: float
    lng: float

class CreateReportRequest(BaseModel):
    latitude: float
    longitude: float
    category: str = Field(..., description="lighting, damage, flooding, accident, blocked, connectivity, isolated, other")
    severity: int = Field(..., ge=1, le=5)
    description: Optional[str] = None

class SafetyConfidenceRequest(BaseModel):
    routeId: str
    currentPosition: LatLngModel
    routeGeometry: List[LatLngModel]

# ─── Endpoints ──────────────────────────────────────────────────────────────────
@router.post("/reports", summary="Submit a crowdsourced road condition report")
def create_report(payload: CreateReportRequest, db: Session = Depends(get_db)):
    try:
        report = RoadObservation(
            latitude=payload.latitude,
            longitude=payload.longitude,
            category=payload.category,
            severity=payload.severity,
            description=payload.description
        )
        db.add(report)
        db.commit()
        db.refresh(report)
        return {
            "status": "success",
            "report": {
                "id": report.id,
                "latitude": report.latitude,
                "longitude": report.longitude,
                "category": report.category,
                "severity": report.severity,
                "description": report.description,
                "timestamp": report.timestamp.isoformat() + "Z" if report.timestamp else None
            }
        }
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to save report: {e}")

@router.post("/confidence", summary="Calculate route safety confidence segments and score")
async def get_safety_confidence(payload: SafetyConfidenceRequest, db: Session = Depends(get_db)):
    current_pos = {"lat": payload.currentPosition.lat, "lng": payload.currentPosition.lng}
    geom = [{"lat": pt.lat, "lng": pt.lng} for pt in payload.routeGeometry]
    
    result = await safety_service.get_route_safety_confidence(
        route_id=payload.routeId,
        current_position=current_pos,
        route_geometry=geom,
        db=db
    )
    return result
