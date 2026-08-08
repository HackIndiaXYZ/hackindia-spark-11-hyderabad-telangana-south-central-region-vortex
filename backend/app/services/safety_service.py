import math
import logging
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from app.models.observation import RoadObservation
from app.services.journey_support import journey_support_service
from app.services.intelligence.weather_provider import OpenMeteoWeatherProvider

logger = logging.getLogger(__name__)

class SafetyService:
    def __init__(self):
        self.weather_provider = OpenMeteoWeatherProvider()

    def calculate_haversine_km(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        R = 6371.0
        d_lat = math.radians(lat2 - lat1)
        d_lng = math.radians(lng2 - lng1)
        a = (math.sin(d_lat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lng / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def segment_route(self, route_coords: list, target_segment_len_km: float = 0.8) -> list:
        segments = []
        if len(route_coords) < 2:
            return []
        
        current_segment_coords = [route_coords[0]]
        current_len = 0.0
        segment_idx = 0
        
        for i in range(1, len(route_coords)):
            p1 = route_coords[i - 1]
            p2 = route_coords[i]
            dist = self.calculate_haversine_km(p1['lat'], p1['lng'], p2['lat'], p2['lng'])
            current_len += dist
            current_segment_coords.append(p2)
            
            if current_len >= target_segment_len_km or i == len(route_coords) - 1:
                segments.append({
                    "segment_id": f"seg-{segment_idx}",
                    "geometry": current_segment_coords,
                    "start_coordinate": current_segment_coords[0],
                    "end_coordinate": current_segment_coords[-1],
                    "length_km": current_len
                })
                current_segment_coords = [p2]
                current_len = 0.0
                segment_idx += 1
                
        return segments

    async def get_route_safety_confidence(
        self,
        route_id: str,
        current_position: dict,
        route_geometry: list,
        db: Session
    ) -> dict:
        logger.info(f"[SafetyService] Calculating safety confidence for route {route_id}")
        
        # 1. Fetch weather at user current position (homogeneous for route context)
        weather_data = None
        try:
            weather_data = await self.weather_provider.get_weather(current_position["lat"], current_position["lng"])
        except Exception as we:
            logger.warning(f"[SafetyService] Weather fetch failed: {we}")

        # 2. Fetch Journey Support POIs (using server cache or Overpass)
        support_res = await journey_support_service.get_journey_support(route_id, current_position, route_geometry)
        support_data = support_res.get("data") or {}
        
        hospitals = support_data.get("hospitals", [])
        police_stations = support_data.get("policeStations", [])
        fuel_stations = support_data.get("fuelStations", [])

        # 3. Query all crowdsourced reports from database
        all_observations = db.query(RoadObservation).all()

        # 4. Segment the route geometry
        segments = self.segment_route(route_geometry, target_segment_len_km=0.8)
        if not segments:
            # If no segments, return fallback default
            return {
                "routeId": route_id,
                "overallConfidence": "INSUFFICIENT DATA",
                "overallScore": 0,
                "reasons": ["• Limited observations available", "• Connectivity information unavailable"],
                "segments": []
            }

        scored_segments = []
        overall_actual_sum = 0.0
        overall_max_sum = 0.0

        for seg in segments:
            # Determine center coordinate of the segment
            geom = seg["geometry"]
            mid_idx = len(geom) // 2
            center_coord = geom[mid_idx]

            # ─── A. Support infrastructure score (max 35) ───
            has_hosp = any(self.calculate_haversine_km(center_coord["lat"], center_coord["lng"], h["latitude"], h["longitude"]) <= 2.5 for h in hospitals)
            has_police = any(self.calculate_haversine_km(center_coord["lat"], center_coord["lng"], p["latitude"], p["longitude"]) <= 2.5 for p in police_stations)
            has_fuel = any(self.calculate_haversine_km(center_coord["lat"], center_coord["lng"], f["latitude"], f["longitude"]) <= 2.5 for f in fuel_stations)

            support_score = 0.0
            if has_hosp:
                support_score += 15.0
            if has_police:
                support_score += 10.0
            if has_fuel:
                support_score += 10.0

            # Find observations near this segment (within 500 meters of any coordinate in segment)
            seg_observations = []
            for obs in all_observations:
                is_near = False
                for pt in geom:
                    if self.calculate_haversine_km(pt["lat"], pt["lng"], obs.latitude, obs.longitude) <= 0.5:
                        is_near = True
                        break
                if is_near:
                    seg_observations.append(obs)

            # ─── B. Connectivity score (max 25) ───
            connectivity_score = 25.0
            conn_reports = [o for o in seg_observations if o.category == "connectivity"]
            for r in conn_reports:
                age_hours = (datetime.now(timezone.utc) - r.timestamp.replace(tzinfo=timezone.utc)).total_seconds() / 3600.0
                decay = max(0.0, 1.0 - (age_hours / 24.0))
                connectivity_score -= (r.severity * 5.0) * decay
            connectivity_score = max(0.0, connectivity_score)

            # ─── C. Environment score (max 25) ───
            environment_score = 25.0
            env_reports = [o for o in seg_observations if o.category == "isolated"]
            for r in env_reports:
                age_hours = (datetime.now(timezone.utc) - r.timestamp.replace(tzinfo=timezone.utc)).total_seconds() / 3600.0
                decay = max(0.0, 1.0 - (age_hours / 24.0))
                environment_score -= (r.severity * 5.0) * decay
            environment_score = max(0.0, environment_score)

            # ─── D. Weather score (max 20) ───
            has_weather = weather_data is not None
            weather_score = 20.0
            if has_weather:
                precip = weather_data.precipitation_mm
                cond = weather_data.condition.lower()
                if precip > 2.0 or any(w in cond for w in ["rain", "storm", "snow", "fog"]):
                    weather_score = 10.0

            # ─── E. Crowdsourced observations score (max 20) ───
            crowd_score = 20.0
            hazard_reports = [o for o in seg_observations if o.category not in ["connectivity", "isolated"]]
            for r in hazard_reports:
                age_hours = (datetime.now(timezone.utc) - r.timestamp.replace(tzinfo=timezone.utc)).total_seconds() / 3600.0
                decay = max(0.0, 1.0 - (age_hours / 24.0))
                crowd_score -= (r.severity * 4.0) * decay
            crowd_score = max(0.0, crowd_score)

            # ─── Summation & Normalization ───
            seg_actual = support_score + connectivity_score + environment_score + crowd_score
            seg_max = 35.0 + 25.0 + 25.0 + 20.0
            
            if has_weather:
                seg_actual += weather_score
                seg_max += 20.0

            seg_pct = (seg_actual / seg_max) * 100.0

            if seg_pct >= 80.0:
                conf = "HIGH"
            elif seg_pct >= 50.0:
                conf = "MODERATE"
            else:
                conf = "LOW"

            # Explainable reasons
            reasons = []
            if support_score >= 25.0:
                reasons.append("✓ Good emergency support availability")
                reasons.append("✓ Multiple support facilities along route")
            else:
                reasons.append("⚠ Limited nearby emergency support")

            if has_weather:
                if weather_score == 20.0:
                    reasons.append("✓ Current weather conditions favorable")
                else:
                    reasons.append("⚠ Favorable weather conditions not present (Rain/Heavy)")
            else:
                reasons.append("• Weather information unavailable")

            if crowd_score >= 18.0:
                reasons.append("✓ No recent reported hazards")
            else:
                reasons.append("⚠ Recent road-condition reports")

            if connectivity_score >= 20.0:
                reasons.append("✓ Good road connectivity")
            else:
                reasons.append("⚠ Reports of low network/connectivity")

            scored_segments.append({
                "segment_id": seg["segment_id"],
                "geometry": seg["geometry"],
                "start_coordinate": seg["start_coordinate"],
                "end_coordinate": seg["end_coordinate"],
                "support_score": round(support_score, 1),
                "connectivity_score": round(connectivity_score, 1),
                "environment_score": round(environment_score, 1),
                "current_condition_score": round(weather_score, 1) if has_weather else None,
                "crowd_observation_score": round(crowd_score, 1),
                "safety_confidence": conf,
                "reasons": reasons
            })

            overall_actual_sum += seg_actual
            overall_max_sum += seg_max

        # Overall average score
        overall_pct = (overall_actual_sum / overall_max_sum) * 100.0 if overall_max_sum > 0 else 0.0

        if len(hospitals) == 0 and len(police_stations) == 0 and not has_weather:
            overall_conf = "INSUFFICIENT DATA"
        elif overall_pct >= 80.0:
            overall_conf = "HIGH"
        elif overall_pct >= 50.0:
            overall_conf = "MODERATE"
        else:
            overall_conf = "LOW"

        # Overall explanation reasons
        overall_reasons = []
        if overall_conf == "INSUFFICIENT DATA":
            overall_reasons = [
                "• Limited observations available",
                "• Connectivity/weather information unavailable"
            ]
        else:
            avg_support = sum(s["support_score"] for s in scored_segments) / len(scored_segments)
            avg_crowd = sum(s["crowd_observation_score"] for s in scored_segments) / len(scored_segments)
            avg_conn = sum(s["connectivity_score"] for s in scored_segments) / len(scored_segments)

            if avg_support >= 25.0:
                overall_reasons.append("✓ Good emergency support availability")
            else:
                overall_reasons.append("⚠ Limited nearby emergency support")

            if has_weather:
                avg_weather = sum(s["current_condition_score"] for s in scored_segments if s["current_condition_score"] is not None) / len(scored_segments)
                if avg_weather >= 18.0:
                    overall_reasons.append("✓ Current weather conditions favorable")
                else:
                    overall_reasons.append("⚠ Favorable weather conditions not present (Rain/Heavy)")
            else:
                overall_reasons.append("• Weather information unavailable")

            if avg_crowd >= 18.0:
                overall_reasons.append("✓ No recent reported hazards")
            else:
                overall_reasons.append("⚠ Recent road-condition reports")

            if avg_conn >= 20.0:
                overall_reasons.append("✓ Good road connectivity")
            else:
                overall_reasons.append("⚠ Reports of low network/connectivity")

        return {
            "routeId": route_id,
            "overallConfidence": overall_conf,
            "overallScore": round(overall_pct, 1),
            "reasons": overall_reasons,
            "segments": scored_segments
        }

safety_service = SafetyService()
