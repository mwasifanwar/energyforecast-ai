import pandas as pd
import numpy as np
from typing import Dict, List, Any

class WeatherProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def generate_sample_weather_data(self, start_date: str, end_date: str, 
                                   freq: str = 'H') -> pd.DataFrame:
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        np.random.seed(42)
        data = {
            'temperature': self.generate_temperature(date_range),
            'humidity': np.random.uniform(30, 90, len(date_range)),
            'wind_speed': np.random.weibull(2, len(date_range)) * 10,
            'wind_direction': np.random.uniform(0, 360, len(date_range)),
            'cloud_cover': np.random.uniform(0, 1, len(date_range)),
            'pressure': np.random.normal(1013, 10, len(date_range)),
            'precipitation': np.random.exponential(0.1, len(date_range))
        }
        
        df = pd.DataFrame(data, index=date_range)
        return df
    
    def generate_temperature(self, dates: pd.DatetimeIndex) -> np.ndarray:
        base_temp = 15
        seasonal_variation = 10 * np.sin(2 * np.pi * (dates.dayofyear - 80) / 365)
        daily_variation = 8 * np.sin(2 * np.pi * (dates.hour - 6) / 24)
        noise = np.random.normal(0, 2, len(dates))
        
        return base_temp + seasonal_variation + daily_variation + noise
    
    def process_weather_forecast(self, raw_forecast: pd.DataFrame) -> pd.DataFrame:
        df = raw_forecast.copy()
        
        for column in df.columns:
            if df[column].isna().any():
                df[column] = df[column].interpolate()
        
        df['apparent_temperature'] = self.calculate_apparent_temperature(
            df['temperature'], df['humidity'], df['wind_speed']
        )
        
        df['wind_chill'] = self.calculate_wind_chill(
            df['temperature'], df['wind_speed']
        )
        
        df['comfort_index'] = self.calculate_comfort_index(
            df['temperature'], df['humidity']
        )
        
        return df
    
    def calculate_apparent_temperature(self, temperature: pd.Series, 
                                     humidity: pd.Series, 
                                     wind_speed: pd.Series) -> pd.Series:
        return temperature + 0.33 * humidity / 100 * 6.105 * np.exp(17.27 * temperature / (237.7 + temperature)) - 0.7 * wind_speed - 4
    
    def calculate_wind_chill(self, temperature: pd.Series, wind_speed: pd.Series) -> pd.Series:
        mask = temperature <= 10
        wind_chill = temperature.copy()
        wind_chill[mask] = 13.12 + 0.6215 * temperature[mask] - 11.37 * wind_speed[mask] ** 0.16 + 0.3965 * temperature[mask] * wind_speed[mask] ** 0.16
        return wind_chill
    
    def calculate_comfort_index(self, temperature: pd.Series, humidity: pd.Series) -> pd.Series:
        return 0.5 * (temperature + 61.0 + (temperature - 68.0) * 1.2 + humidity * 0.094)