from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List
from app.services.journey_support import journey_support_service

router = APIRouter()

class LatLngModel(BaseModel):
    lat: float
    lng: float

class JourneySupportRequest(BaseModel):
    routeId: str
    currentPosition: LatLngModel
    routeGeometry: List[LatLngModel]

@router.post("/journey-support", summary="Get dynamic journey support POIs")
async def get_journey_support(payload: JourneySupportRequest):
    current_pos = {"lat": payload.currentPosition.lat, "lng": payload.currentPosition.lng}
    route_geom = [{"lat": pt.lat, "lng": pt.lng} for pt in payload.routeGeometry]
    
    result = await journey_support_service.get_journey_support(
        route_id=payload.routeId,
        current_position=current_pos,
        route_geometry=route_geom
    )
    return result
