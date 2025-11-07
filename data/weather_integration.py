import pandas as pd
import numpy as np
from typing import Dict, List, Any

class WeatherIntegration:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def correlate_weather_energy(self, energy_data: pd.DataFrame, 
                               weather_data: pd.DataFrame) -> Dict[str, float]:
        merged_data = pd.merge(energy_data, weather_data, left_index=True, right_index=True)
        
        correlations = {}
        weather_cols = ['temperature', 'humidity', 'wind_speed', 'cloud_cover']
        
        for col in weather_cols:
            if col in merged_data.columns:
                corr = merged_data['energy_demand'].corr(merged_data[col])
                correlations[col] = float(corr)
        
        return correlations
    
    def create_combined_dataset(self, energy_data: pd.DataFrame,
                              weather_data: pd.DataFrame) -> pd.DataFrame:
        combined = pd.merge(energy_data, weather_data, left_index=True, right_index=True)
        
        combined['heating_degree_days'] = np.maximum(18 - combined['temperature'], 0)
        combined['cooling_degree_days'] = np.maximum(combined['temperature'] - 24, 0)
        
        combined['weather_impact_score'] = (
            combined['heating_degree_days'] * 0.4 +
            combined['cooling_degree_days'] * 0.4 +
            combined['humidity'] * 0.1 +
            combined['cloud_cover'] * 0.1
        )
        
        return combined