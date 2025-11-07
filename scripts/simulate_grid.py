import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from data.energy_processor import EnergyDataProcessor
from data.weather_integration import WeatherIntegration
from utils.config import Config

def simulate_grid_operations(days: int = 30):
    config = Config()
    
    energy_processor = EnergyDataProcessor(config.get('energy', {}))
    weather_integration = WeatherIntegration(config.get('weather', {}))
    
    print(f"Simulating {days} days of grid operations...")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    energy_data = energy_processor.generate_sample_energy_data(
        start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    )
    weather_data = weather_integration.generate_sample_weather_data(
        start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    )
    
    combined_data = weather_integration.create_combined_dataset(energy_data, weather_data)
    
    correlations = weather_integration.correlate_weather_energy(energy_data, weather_data)
    
    metrics = energy_processor.calculate_energy_metrics(energy_data)
    
    print("\n--- Grid Simulation Summary ---")
    print(f"Total energy demand: {metrics['total_energy']:,.0f} MWh")
    print(f"Peak demand: {metrics['peak_demand']:.0f} MW")
    print(f"Load factor: {metrics['load_factor']:.2%}")
    print(f"Price volatility: {metrics['price_volatility']:.2f}")
    
    print("\nWeather-Energy Correlations:")
    for factor, correlation in correlations.items():
        print(f"  {factor}: {correlation:.3f}")
    
    daily_energy = energy_data['energy_demand'].resample('D').sum()
    peak_days = daily_energy.nlargest(3)
    
    print(f"\nPeak demand days:")
    for date, demand in peak_days.items():
        print(f"  {date.strftime('%Y-%m-%d')}: {demand:,.0f} MWh")
    
    return {
        'energy_data': energy_data,
        'weather_data': weather_data,
        'combined_data': combined_data,
        'metrics': metrics,
        'correlations': correlations
    }

if __name__ == "__main__":
    simulation_results = simulate_grid_operations(90)