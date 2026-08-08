import math
import logging
from datetime import datetime
import httpx
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)

class JourneySupportService:
    def __init__(self):
        # Memory cache: route_id -> list of raw OSM element dicts
        self._poi_cache = {}

    def calculate_haversine_km(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        R = 6371.0
        d_lat = math.radians(lat2 - lat1)
        d_lng = math.radians(lng2 - lng1)
        a = (math.sin(d_lat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lng / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return R * c

    def check_is_ahead(self, current_pos: dict, poi_pos: dict, route_coords: list) -> bool:
        if len(route_coords) < 2:
            return True
        
        curr_idx = 0
        curr_min_dist = float('inf')
        poi_idx = 0
        poi_min_dist = float('inf')
        
        for i, pt in enumerate(route_coords):
            d_curr = self.calculate_haversine_km(current_pos['lat'], current_pos['lng'], pt['lat'], pt['lng'])
            if d_curr < curr_min_dist:
                curr_min_dist = d_curr
                curr_idx = i
                
            d_poi = self.calculate_haversine_km(poi_pos['lat'], poi_pos['lng'], pt['lat'], pt['lng'])
            if d_poi < poi_min_dist:
                poi_min_dist = d_poi
                poi_idx = i
                
        return poi_idx >= curr_idx

    def sample_route_points(self, coords: list, n: int) -> list:
        if len(coords) <= n:
            return coords
        points = []
        step = (len(coords) - 1) / (n - 1)
        for i in range(n):
            idx = round(i * step)
            points.append(coords[idx])
        return points

    async def get_journey_support(
        self,
        route_id: str,
        current_position: dict,
        route_geometry: list
    ) -> dict:
        is_cached = route_id in self._poi_cache
        raw_elements = []
        status = "live"
        source = "overpass"

        logger.info(f"[JourneySupport] Received journey support request for lat={current_position['lat']}, lng={current_position['lng']}")

        if is_cached:
            raw_elements = self._poi_cache[route_id]
            status = "cached"
            source = "cache"
            logger.info(f"[JourneySupport] Reusing cached POIs for route {route_id}")
        else:
            # Query Overpass
            if len(route_geometry) > 0:
                sample_pts = self.sample_route_points(route_geometry, 4)
                
                radius_meters = 1500
                query_parts = []
                for c in sample_pts:
                    query_parts.append(f'node["amenity"~"hospital|police|fuel"](around:{radius_meters},{c["lat"]},{c["lng"]});')
                    query_parts.append(f'way["amenity"~"hospital|police|fuel"](around:{radius_meters},{c["lat"]},{c["lng"]});')
                    
                union_query = "\n".join(query_parts)
                overpass_query = f"""
                [out:json][timeout:8];
                (
                  {union_query}
                );
                out center 40;
                """
                
                overpass_urls = [
                    "https://overpass.openstreetmap.fr/api/interpreter",
                    "https://overpass.kumi.systems/api/interpreter",
                    "https://overpass-api.de/api/interpreter",
                    "https://lz4.overpass-api.de/api/interpreter",
                    "https://z.overpass-api.de/api/interpreter"
                ]
                headers = {
                    "User-Agent": "SahachaaraNavigationApp/1.0 (contact: simplicitsports@gmail.com)"
                }
                
                success = False
                for overpass_url in overpass_urls:
                    logger.info(f"[JourneySupport] Querying Overpass endpoint: {overpass_url}")
                    try:
                        async with httpx.AsyncClient() as client:
                            resp = await client.post(overpass_url, data={"data": overpass_query}, headers=headers, timeout=10.0)
                            if resp.status_code == 200:
                                # Validate JSON response structure
                                data = resp.json()
                                if isinstance(data, dict) and "elements" in data:
                                    raw_elements = data.get("elements", [])
                                    self._poi_cache[route_id] = raw_elements
                                    status = "live"
                                    source = "overpass"
                                    success = True
                                    logger.info(f"[JourneySupport] Overpass query succeeded on {overpass_url}, found {len(raw_elements)} elements")
                                    break
                                else:
                                    logger.warning(
                                        f"[JourneySupport] Overpass request failed\n"
                                        f"Endpoint: {overpass_url}\n"
                                        f"Status: {resp.status_code}\n"
                                        f"Exception: MalformedResponse\n"
                                        f"Message: Response JSON is missing 'elements' list\n"
                                        f"Response: {resp.text[:300]}"
                                    )
                            else:
                                logger.warning(
                                    f"[JourneySupport] Overpass request failed\n"
                                    f"Endpoint: {overpass_url}\n"
                                    f"Status: {resp.status_code}\n"
                                    f"Response: {resp.text[:300]}"
                                )
                    except httpx.TimeoutException:
                        logger.warning(
                            f"[JourneySupport] Overpass request failed\n"
                            f"Endpoint: {overpass_url}\n"
                            f"Exception: TimeoutException\n"
                            f"Message: Request timed out (limit: 6.0s)"
                        )
                    except Exception as e:
                        logger.warning(
                            f"[JourneySupport] Overpass request failed\n"
                            f"Endpoint: {overpass_url}\n"
                            f"Exception: {type(e).__name__}\n"
                            f"Message: {str(e)}"
                        )
                
                if not success:
                    status = "unavailable"
                    source = "overpass"
            else:
                status = "unavailable"
                source = "overpass"

        # If it failed/unavailable and we have no elements, return structured unavailable state
        if status == "unavailable" and not raw_elements:
            return {
                "status": "unavailable",
                "source": "overpass",
                "message": "Live support information temporarily unavailable",
                "data": {
                    "routeId": route_id,
                    "cachedAt": datetime.utcnow().isoformat() + "Z",
                    "isOfflineMode": False,
                    "pois": [],
                    "hospitals": [],
                    "policeStations": [],
                    "fuelStations": []
                }
            }

        # Process and construct SupportPOI structures
        pois = []
        for el in raw_elements:
            tags = el.get("tags", {})
            raw_amenity = tags.get("amenity", "")
            category = None
            if raw_amenity == "hospital":
                category = "hospital"
            elif raw_amenity == "police":
                category = "police"
            elif raw_amenity == "fuel":
                category = "fuel"

            if not category:
                continue

            lat = el.get("lat") or el.get("center", {}).get("lat")
            lng = el.get("lon") or el.get("center", {}).get("lon")
            if lat is None or lng is None:
                continue

            poi_coords = {"lat": lat, "lng": lng}
            dist_km = self.calculate_haversine_km(current_position["lat"], current_position["lng"], lat, lng)
            is_ahead = self.check_is_ahead(current_position, poi_coords, route_geometry)

            dist_str = f"{dist_km:.1f}"
            formatted_dist = f"{dist_str} km ahead" if is_ahead else f"{dist_str} km away"

            poi_id = f"osm-{el.get('id')}"
            brand = tags.get("brand")
            name = tags.get("name") or (f"{brand} Station" if brand else f"Highway {category.capitalize()} Station")

            pois.append({
                "id": poi_id,
                "name": name,
                "category": category,
                "latitude": lat,
                "longitude": lng,
                "distanceKm": dist_km,
                "routeSegment": el.get("routeSegment", 0),
                "source": "live" if status == "live" else "cache",
                "cachedAt": datetime.utcnow().isoformat() + "Z",
                "isAhead": is_ahead,
                "formattedDistance": formatted_dist
            })

        # Sort POIs: ahead first, then by distance
        pois.sort(key=lambda p: (not p["isAhead"], p["distanceKm"]))

        hospitals = [p for p in pois if p["category"] == "hospital"]
        police = [p for p in pois if p["category"] == "police"]
        fuel = [p for p in pois if p["category"] == "fuel"]

        support_data = {
            "routeId": route_id,
            "cachedAt": datetime.utcnow().isoformat() + "Z",
            "isOfflineMode": False,
            "pois": pois,
            "hospitals": hospitals,
            "policeStations": police,
            "fuelStations": fuel
        }

        return {
            "status": "cached" if status == "cached" else "live",
            "source": source,
            "data": support_data
        }

journey_support_service = JourneySupportService()
