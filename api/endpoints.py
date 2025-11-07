from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
from typing import Dict, List, Any

class EnergyForecastAPI:
    def __init__(self, forecast_system):
        self.app = FastAPI(title="EnergyForecast AI API",
                          description="Smart Grid Optimization System",
                          version="1.0.0")
        self.forecast_system = forecast_system
        self.setup_routes()
    
    def setup_routes(self):
        @self.app.post("/forecast-demand/")
        async def forecast_demand(request: Dict[str, Any]):
            try:
                historical_data = request.get('historical_data')
                weather_forecast = request.get('weather_forecast')
                hours_ahead = request.get('hours_ahead', 24)
                
                forecast = self.forecast_system.forecast_energy_demand(
                    historical_data, weather_forecast, hours_ahead
                )
                
                return JSONResponse(content=forecast)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/optimize-grid/")
        async def optimize_grid(request: Dict[str, Any]):
            try:
                demand_forecast = request.get('demand_forecast')
                renewable_forecast = request.get('renewable_forecast')
                grid_constraints = request.get('grid_constraints', {})
                
                optimization = self.forecast_system.optimize_grid_operations(
                    demand_forecast, renewable_forecast, grid_constraints
                )
                
                return JSONResponse(content=optimization)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/grid-health/")
        async def grid_health():
            try:
                health_status = self.forecast_system.get_grid_health()
                return JSONResponse(content=health_status)
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health/")
        async def health_check():
            return {"status": "healthy", "service": "EnergyForecast AI"}
    
    def run(self, host: str = "0.0.0.0", port: int = 8000):
        uvicorn.run(self.app, host=host, port=port)
