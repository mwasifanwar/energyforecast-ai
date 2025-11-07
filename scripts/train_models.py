import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from core.demand_forecaster import DemandForecaster
from core.renewable_integrator import RenewableIntegrator
from data.energy_processor import EnergyDataProcessor
from data.weather_integration import WeatherIntegration
from utils.config import Config
import torch

def main():
    config = Config()
    
    print("Initializing EnergyForecast AI Training...")
    
    energy_processor = EnergyDataProcessor(config.get('energy', {}))
    weather_integration = WeatherIntegration(config.get('weather', {}))
    
    print("Generating sample energy data...")
    energy_data = energy_processor.generate_sample_energy_data('2023-01-01', '2024-01-01')
    energy_data_with_anomalies = energy_processor.add_anomalies(energy_data)
    
    print("Training demand forecasting model...")
    demand_forecaster = DemandForecaster(config.get('forecasting', {}))
    demand_forecaster.train(energy_data_with_anomalies, epochs=50)
    
    torch.save(demand_forecaster.model.state_dict(), 'models/demand_forecaster.pth')
    
    print("Training renewable integration models...")
    weather_data = weather_integration.generate_sample_weather_data('2023-01-01', '2024-01-01')
    
    combined_data = weather_integration.create_combined_dataset(energy_data, weather_data)
    combined_data['solar_generation'] = np.random.exponential(50, len(combined_data))
    combined_data['wind_generation'] = np.random.weibull(2, len(combined_data)) * 30
    
    renewable_integrator = RenewableIntegrator(config.get('renewable', {}))
    renewable_integrator.train_renewable_models(combined_data)
    
    print("EnergyForecast AI models trained successfully!")
    
    return demand_forecaster, renewable_integrator

if __name__ == "__main__":
    demand_forecaster, renewable_integrator = main()
