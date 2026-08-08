from sqlalchemy import Column, Float, String, Integer, DateTime
from app.models.base import BaseModel
from datetime import datetime

class RoadObservation(BaseModel):
    __tablename__ = "road_observations"

    latitude = Column(Float, nullable=False)
    longitude = Column(Float, nullable=False)
    category = Column(String, nullable=False)  # lighting, damage, flooding, accident, blocked, connectivity, isolated, other
    severity = Column(Integer, nullable=False)  # 1 to 5
    description = Column(String, nullable=True)
    timestamp = Column(DateTime(timezone=True), default=datetime.utcnow, nullable=False)
