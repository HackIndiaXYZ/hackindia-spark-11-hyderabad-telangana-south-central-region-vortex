import logging
from typing import Any, Dict, Optional

from app.schemas.ai import IntelligenceContext, WeatherData
from app.services.intelligence.places_provider import OSMPlacesProvider
from app.services.intelligence.weather_provider import OpenMeteoWeatherProvider

logger = logging.getLogger(__name__)


class IntelligenceService:
    """Service layer combining Weather, POI, and Live context for Saha AI."""

    def __init__(self):
        self.weather_provider = OpenMeteoWeatherProvider()
        self.places_provider = OSMPlacesProvider()

    async def get_live_context(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> IntelligenceContext:
        """Assembles live intelligence (weather + POIs) for a user query & context."""
        ctx = context or {}
        lat = ctx.get("lat") or 17.3850  # Default: Hyderabad coordinates if not provided
        lon = ctx.get("lon") or 78.4867
        location_name = ctx.get("destination") or ctx.get("location") or "Current Location"

        # Attempt to geocode destination if provided and lat/lon are default
        if (lat == 17.3850 and lon == 78.4867) and location_name and location_name != "Current Location":
            coords = await self.places_provider.geocode_location(location_name)
            if coords:
                lat, lon = coords

        # Concurrent weather & POI fetching
        weather = await self.weather_provider.get_weather(lat, lon)

        # Detect category interest from user query
        category = self._detect_category(query)
        nearby_places = await self.places_provider.search_nearby(lat, lon, category=category, radius_km=15.0)

        return IntelligenceContext(
            location_name=location_name,
            lat=lat,
            lon=lon,
            weather=weather,
            nearby_places=nearby_places,
        )

    def _detect_category(self, query: str) -> str:
        q = query.lower()
        if any(w in q for w in ["food", "restaurant", "lunch", "breakfast", "dinner", "eat"]):
            return "restaurant"
        elif any(w in q for w in ["ev", "charger", "battery", "charging"]):
            return "ev_charger"
        elif any(w in q for w in ["fuel", "gas", "petrol", "diesel", "cng"]):
            return "fuel"
        elif any(w in q for w in ["emergency", "hospital", "doctor", "breakdown", "police", "help"]):
            return "hospital"
        elif any(w in q for w in ["cafe", "coffee", "tea"]):
            return "cafe"
        return "general"
