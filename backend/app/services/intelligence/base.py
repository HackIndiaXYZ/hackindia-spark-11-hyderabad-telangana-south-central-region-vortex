from abc import ABC, abstractmethod
from typing import List, Optional
from app.schemas.ai import WeatherData, NearbyPlace


class BaseWeatherProvider(ABC):
    """Abstract interface for weather providers."""

    @abstractmethod
    async def get_weather(self, lat: float, lon: float) -> Optional[WeatherData]:
        """Fetch current weather for coordinates."""
        pass


class BasePlacesProvider(ABC):
    """Abstract interface for nearby place/POI providers."""

    @abstractmethod
    async def search_nearby(
        self, lat: float, lon: float, category: str, radius_km: float = 10.0, limit: int = 5
    ) -> List[NearbyPlace]:
        """Search nearby points of interest."""
        pass

    @abstractmethod
    async def geocode_location(self, location_name: str) -> Optional[tuple[float, float]]:
        """Geocode place name to (latitude, longitude)."""
        pass
