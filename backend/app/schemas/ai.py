from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class ActivityItem(BaseModel):
    id: str = Field(default="act-1")
    time: str = Field(default="08:00 AM")
    period: str = Field(default="morning", description="morning, afternoon, or evening")
    title: str = Field(..., description="Short activity title")
    description: str = Field(..., description="Detailed description")
    location: str = Field(default="Local Point")
    duration: str = Field(default="1.5 hours")
    estimatedCost: str = Field(default="Free")
    category: str = Field(default="sightseeing", description="culture, food, nature, relaxation, transit, sightseeing")
    companionAdvice: Optional[str] = Field(default=None, description="Co-driver Saha advice tip")
    completed: bool = Field(default=False)


class ItineraryDay(BaseModel):
    dayNumber: int = Field(..., ge=1)
    title: str = Field(..., description="Day title")
    dateStr: str = Field(default="Day 1")
    summary: str = Field(..., description="Day summary")
    activities: List[ActivityItem] = Field(default_factory=list)


class BudgetItem(BaseModel):
    category: str = Field(..., description="Accommodation, Dining & Food, Activities & Sightseeing, Transit & Fuel, Emergency Buffer")
    amount: float = Field(..., ge=0)
    percentage: float = Field(..., ge=0, le=100)
    color: str = Field(default="var(--color-pastel-purple)")


class PackingItem(BaseModel):
    id: str
    label: str
    checked: bool = False


class PackingCategory(BaseModel):
    name: str
    iconName: str = Field(default="Briefcase")
    items: List[PackingItem] = Field(default_factory=list)


class TripPlanInput(BaseModel):
    destination: str = Field(default="Unknown Destination")
    budgetLevel: str = Field(default="moderate")
    customBudgetValue: Optional[str] = None
    vehicle: str = Field(default="car")
    travelers: str = Field(default="couple")
    foodPreference: str = Field(default="anything")
    interests: List[str] = Field(default_factory=list)
    travelDays: int = Field(default=3)


class GeneratedItinerary(BaseModel):
    id: str = Field(default="saha-plan-ai")
    destination: str
    coverImage: str = Field(default="https://images.unsplash.com/photo-1506744038136-46273834b3fb")
    travelDays: int
    totalCostEstimate: str
    summary: str
    days: List[ItineraryDay] = Field(default_factory=list)
    budgetBreakdown: List[BudgetItem] = Field(default_factory=list)
    packingCategories: List[PackingCategory] = Field(default_factory=list)
    input: Optional[TripPlanInput] = None
    
    # Extended Travel Intelligence attributes
    safetyNotes: List[str] = Field(default_factory=list)
    weatherNotes: List[str] = Field(default_factory=list)
    bestDepartureTime: Optional[str] = None
    drivingTips: List[str] = Field(default_factory=list)
    recommendedStops: List[Dict[str, Any]] = Field(default_factory=list)


class ChatMessage(BaseModel):
    role: str = Field(..., description="'user', 'model' or 'system'")
    content: str = Field(..., description="Message text")


class ChatRequest(BaseModel):
    message: str = Field(..., description="User query or input")
    persona: Optional[str] = Field("companion", description="companion, guide, safety, emergency")
    session_id: Optional[str] = Field(None, description="Session ID for persistent conversation memory")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Journey context payload")
    history: Optional[List[ChatMessage]] = Field(default_factory=list, description="Recent message history")


class ChatResponse(BaseModel):
    response: str
    persona: str
    is_demo: bool = False
    suggestions: List[str] = Field(default_factory=list)
    cards: List[Dict[str, Any]] = Field(default_factory=list)
    session_id: Optional[str] = Field(None)


class PlanRequest(BaseModel):
    prompt: str = Field(..., description="Natural language prompt describing the trip")
    session_id: Optional[str] = Field(None)
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)


class WeatherData(BaseModel):
    temp_c: float
    condition: str
    precipitation_mm: float = 0.0
    wind_kmh: float = 0.0
    humidity: int = 50
    warning: Optional[str] = None


class NearbyPlace(BaseModel):
    name: str
    category: str
    lat: float
    lon: float
    distance_km: float
    rating: float = 4.5
    open_now: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IntelligenceContext(BaseModel):
    location_name: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    weather: Optional[WeatherData] = None
    nearby_places: List[NearbyPlace] = Field(default_factory=list)
