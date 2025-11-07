import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import time
import pandas as pd
from core.demand_forecaster import DemandForecaster
from core.grid_optimizer import GridOptimizer
from core.renewable_integrator import RenewableIntegrator
from data.energy_processor import EnergyDataProcessor
from data.weather_integration import WeatherIntegration
from optimization.dispatch_optimizer import DispatchOptimizer
from optimization.constraint_handler import ConstraintHandler
from api.endpoints import EnergyForecastAPI
from utils.config import Config
import torch

class EnergyForecastSystem:
    def __init__(self, config_path: str = "configs/default.yaml"):
        self.config = Config(config_path)
        
        self.demand_forecaster = DemandForecaster(self.config.get('forecasting', {}))
        self.grid_optimizer = GridOptimizer(self.config.get('optimization', {}))
        self.renewable_integrator = RenewableIntegrator(self.config.get('renewable', {}))
        self.energy_processor = EnergyDataProcessor(self.config.get('energy', {}))
        self.weather_integration = WeatherIntegration(self.config.get('weather', {}))
        self.dispatch_optimizer = DispatchOptimizer(self.config.get('dispatch', {}))
        self.constraint_handler = ConstraintHandler(self.config.get('constraints', {}))
        
        self.initialize_system()
    
    def initialize_system(self):
        print("Initializing EnergyForecast AI System...")
        
        try:
            self.demand_forecaster.model.load_state_dict(
                torch.load('models/demand_forecaster.pth')
            )
            print("Demand forecasting model loaded successfully")
        except FileNotFoundError:
            print("Trained model not found. Please run train_models.py first.")
        
        sample_energy_data = self.energy_processor.generate_sample_energy_data('2024-01-01', '2024-02-01')
        sample_weather_data = self.weather_integration.generate_sample_weather_data('2024-01-01', '2024-02-01')
        
        combined_data = self.weather_integration.create_combined_dataset(
            sample_energy_data, sample_weather_data
        )
        combined_data['solar_generation'] = np.random.exponential(50, len(combined_data))
        combined_data['wind_generation'] = np.random.weibull(2, len(combined_data)) * 30
        
        self.renewable_integrator.train_renewable_models(combined_data)
        
        print("EnergyForecast AI System initialized successfully!")
    
    def forecast_energy_demand(self, historical_data: pd.DataFrame,
                             weather_forecast: pd.DataFrame,
                             hours_ahead: int = 24) -> Dict[str, Any]:
        
        demand_forecast = self.demand_forecaster.forecast_demand(
            historical_data, weather_forecast, hours_ahead
        )
        
        renewable_forecast = self.renewable_integrator.forecast_renewable_generation(
            weather_forecast
        )
        
        grid_optimization = self.grid_optimizer.optimize_dispatch(
            demand_forecast['demand_forecast'],
            renewable_forecast['total_renewable'],
            self.config.get('generator_costs', {}),
            self.config.get('storage_status', {})
        )
        
        return {
            'demand_forecast': demand_forecast,
            'renewable_forecast': renewable_forecast,
            'grid_optimization': grid_optimization,
            'timestamp': pd.Timestamp.now(),
            'forecast_horizon': hours_ahead
        }
    
    def optimize_grid_operations(self, demand_forecast: List[float],
                               renewable_forecast: List[float],
                               grid_constraints: Dict[str, Any]) -> Dict[str, Any]:
        
        dispatch_optimization = self.dispatch_optimizer.optimize_economic_dispatch(
            demand_forecast,
            self.config.get('generator_costs', {}),
            self.config.get('generator_limits', {})
        )
        
        constraint_check = self.constraint_handler.check_grid_constraints(
            dispatch_optimization['dispatch_schedule'],
            self.config.get('line_limits', {})
        )
        
        reserve_requirements = self.constraint_handler.calculate_reserve_requirements(
            demand_forecast, renewable_forecast
        )
        
        return {
            'dispatch_optimization': dispatch_optimization,
            'constraint_analysis': constraint_check,
            'reserve_requirements': reserve_requirements,
            'optimization_timestamp': pd.Timestamp.now()
        }
    
    def get_grid_health(self) -> Dict[str, Any]:
        sample_energy = self.energy_processor.generate_sample_energy_data(
            pd.Timestamp.now() - pd.Timedelta(days=7), pd.Timestamp.now()
        )
        
        metrics = self.energy_processor.calculate_energy_metrics(sample_energy)
        
        return {
            'grid_metrics': metrics,
            'system_status': 'operational',
            'last_updated': pd.Timestamp.now(),
            'anomaly_count': 0,
            'renewable_penetration': 25.5
        }

def main():
    system = EnergyForecastSystem()
    
    print("Starting EnergyForecast AI System...")
    
    api = EnergyForecastAPI(system)
    
    print("EnergyForecast AI System is running!")
    print("API available at http://localhost:8000")
    print("Health check: http://localhost:8000/health")
    
    api.run(host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()