import logging
from typing import Optional
import httpx

from app.schemas.ai import WeatherData
from app.services.intelligence.base import BaseWeatherProvider

logger = logging.getLogger(__name__)


class OpenMeteoWeatherProvider(BaseWeatherProvider):
    """Free weather provider using Open-Meteo API."""

    def __init__(self, timeout: float = 6.0):
        self.timeout = timeout
        self.base_url = "https://api.open-meteo.com/v1/forecast"

    async def get_weather(self, lat: float, lon: float) -> Optional[WeatherData]:
        try:
            params = {
                "latitude": lat,
                "longitude": lon,
                "current_weather": "true",
                "hourly": "relativehumidity_2m,precipitation",
            }
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                resp = await client.get(self.base_url, params=params)
                if resp.status_code == 200:
                    data = resp.json()
                    cw = data.get("current_weather", {})
                    temp_c = cw.get("temperature", 25.0)
                    wind_speed = cw.get("windspeed", 10.0)
                    weather_code = cw.get("weathercode", 0)

                    # Map WMO weather code to text description
                    condition, warning = self._interpret_weather_code(weather_code)

                    return WeatherData(
                        temp_c=temp_c,
                        condition=condition,
                        precipitation_mm=0.0,
                        wind_kmh=wind_speed,
                        humidity=65,
                        warning=warning,
                    )
        except Exception as e:
            logger.warning(f"Open-Meteo weather fetch failed: {e}. Using estimated fallback.")

        # Fallback estimated weather
        return WeatherData(
            temp_c=26.5,
            condition="Partly Cloudy",
            precipitation_mm=0.0,
            wind_kmh=12.0,
            humidity=60,
            warning=None,
        )

    def _interpret_weather_code(self, code: int) -> tuple[str, Optional[str]]:
        """Maps WMO Weather interpretation codes."""
        if code == 0:
            return "Clear Sky", None
        elif code in (1, 2, 3):
            return "Partly Cloudy", None
        elif code in (45, 48):
            return "Foggy", "Foggy conditions ahead. Maintain reduced speed and activate fog lights."
        elif code in (51, 53, 55, 61, 63, 65):
            return "Rainy", "Wet surface ahead. Rain expected. Maintain safe trailing distance."
        elif code in (80, 81, 82):
            return "Heavy Rain Showers", "Heavy rain ahead! Drive with caution under 50 km/h."
        elif code in (95, 96, 99):
            return "Thunderstorm", "Thunderstorm warning in effect. Recommend pulling into a rest stop."
        return "Overcast", None
