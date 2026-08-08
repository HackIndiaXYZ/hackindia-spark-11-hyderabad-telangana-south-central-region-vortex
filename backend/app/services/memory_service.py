import logging
import re
import uuid
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from app.schemas.ai import ChatMessage

logger = logging.getLogger(__name__)


class SessionContext(BaseModel):
    session_id: str
    destination: Optional[str] = None
    vehicle: Optional[str] = None
    travelers: Optional[str] = None
    budget: Optional[str] = None
    battery_level: Optional[str] = None
    driving_style: Optional[str] = None
    food_preference: Optional[str] = None
    interests: List[str] = Field(default_factory=list)
    custom_state: Dict[str, Any] = Field(default_factory=dict)


class SessionMemory:
    """In-memory session container storing message history and extracted journey context."""

    def __init__(self, session_id: str, max_history: int = 15):
        self.session_id = session_id
        self.max_history = max_history
        self.messages: List[ChatMessage] = []
        self.context = SessionContext(session_id=session_id)

    def add_message(self, role: str, content: str):
        self.messages.append(ChatMessage(role=role, content=content))
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history :]
        
        if role == "user":
            self.extract_entities(content)

    def update_context(self, incoming_context: Dict[str, Any]):
        """Updates session state with explicitly passed context."""
        if not incoming_context:
            return
        if incoming_context.get("destination"):
            self.context.destination = incoming_context["destination"]
        if incoming_context.get("vehicle"):
            self.context.vehicle = incoming_context["vehicle"]
        if incoming_context.get("travelers"):
            self.context.travelers = str(incoming_context["travelers"])
        if incoming_context.get("budget"):
            self.context.budget = str(incoming_context["budget"])
        if incoming_context.get("battery_level"):
            self.context.battery_level = str(incoming_context["battery_level"])
        if incoming_context.get("driving_style"):
            self.context.driving_style = str(incoming_context["driving_style"])

    def extract_entities(self, text: str):
        """Intelligent regex pattern entity extractor from user chat text."""
        t_lower = text.lower()

        # EV Battery level detection
        battery_match = re.search(r'(\b\d{1,3}%\b|\b\d{1,3}\s*percent\b|\bbattery\s*is\s*\d{1,3}%)', t_lower)
        if battery_match:
            self.context.battery_level = battery_match.group(0)

        # Travelers count detection
        travelers_match = re.search(r'(\b\d+\s*(?:friends|people|family|passengers|travelers|adults|kids)\b|solo|couple|family)', t_lower)
        if travelers_match:
            self.context.travelers = travelers_match.group(0)

        # Vehicle type detection
        if any(v in t_lower for v in ["ev", "electric vehicle", "tata nexon ev", "tesla"]):
            self.context.vehicle = "EV / Electric Vehicle"
        elif any(v in t_lower for v in ["bike", "motorcycle", "scooter"]):
            self.context.vehicle = "Motorcycle"
        elif any(v in t_lower for v in ["car", "suv", "sedan", "car driving"]):
            self.context.vehicle = "Car"

        # Food preferences
        if "veg" in t_lower or "vegetarian" in t_lower:
            self.context.food_preference = "Vegetarian"
        elif "vegan" in t_lower:
            self.context.food_preference = "Vegan"


class MemoryService:
    """Store for managing active user session memories. Designed for easy Redis swap."""

    def __init__(self):
        self._sessions: Dict[str, SessionMemory] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> SessionMemory:
        sid = session_id or str(uuid.uuid4())
        if sid not in self._sessions:
            self._sessions[sid] = SessionMemory(session_id=sid)
        return self._sessions[sid]

    def clear_session(self, session_id: str) -> bool:
        if session_id in self._sessions:
            del self._sessions[session_id]
            return True
        return False


# Global singleton instance for in-memory session management
memory_service = MemoryService()
