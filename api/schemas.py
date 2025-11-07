from pydantic import BaseModel
from typing import Dict, List, Any, Optional

class ForecastRequest(BaseModel):
    historical_data: Dict[str, Any]
    weather_forecast: Dict[str, Any]
    hours_ahead: Optional[int] = 24

class ForecastResponse(BaseModel):
    demand_forecast: List[float]
    confidence_intervals: Dict[str, List[float]]
    optimization_recommendations: Dict[str, Any]
    timestamp: str
