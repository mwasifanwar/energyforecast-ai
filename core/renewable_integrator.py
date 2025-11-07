import numpy as np
import pandas as pd
from typing import Dict, List, Any
from sklearn.ensemble import RandomForestRegressor

class RenewableIntegrator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.solar_forecaster = RandomForestRegressor(n_estimators=100, random_state=42)
        self.wind_forecaster = RandomForestRegressor(n_estimators=100, random_state=42)
        self.is_trained = False
        
    def prepare_renewable_features(self, weather_data: pd.DataFrame) -> pd.DataFrame:
        df = weather_data.copy()
        
        df['hour'] = df.index.hour
        df['day_of_year'] = df.index.dayofyear
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        df['solar_zenith'] = self.calculate_solar_zenith(df.index)
        df['wind_power_law'] = df['wind_speed'] * (50 / 10) ** 0.14
        
        return df
    
    def calculate_solar_zenith(self, timestamps: pd.DatetimeIndex) -> pd.Series:
        latitudes = np.full(len(timestamps), 40.0)
        day_of_year = timestamps.dayofyear
        hour_decimal = timestamps.hour + timestamps.minute / 60
        
        declination = 23.45 * np.sin(2 * np.pi * (284 + day_of_year) / 365)
        hour_angle = 15 * (hour_decimal - 12)
        
        zenith = np.arccos(
            np.sin(np.radians(latitudes)) * np.sin(np.radians(declination)) +
            np.cos(np.radians(latitudes)) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle))
        )
        
        return np.degrees(zenith)
    
    def train_renewable_models(self, historical_data: pd.DataFrame):
        feature_df = self.prepare_renewable_features(historical_data)
        
        solar_features = ['temperature', 'humidity', 'cloud_cover', 'solar_zenith', 
                         'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        wind_features = ['wind_speed', 'wind_direction', 'temperature', 'pressure',
                        'wind_power_law', 'hour_sin', 'hour_cos']
        
        if 'solar_generation' in historical_data.columns:
            self.solar_forecaster.fit(feature_df[solar_features], historical_data['solar_generation'])
        
        if 'wind_generation' in historical_data.columns:
            self.wind_forecaster.fit(feature_df[wind_features], historical_data['wind_generation'])
        
        self.is_trained = True
    
    def forecast_renewable_generation(self, weather_forecast: pd.DataFrame) -> Dict[str, Any]:
        if not self.is_trained:
            raise ValueError("Renewable models not trained. Call train_renewable_models() first.")
        
        feature_df = self.prepare_renewable_features(weather_forecast)
        
        solar_features = ['temperature', 'humidity', 'cloud_cover', 'solar_zenith', 
                         'hour_sin', 'hour_cos', 'day_sin', 'day_cos']
        wind_features = ['wind_speed', 'wind_direction', 'temperature', 'pressure',
                        'wind_power_law', 'hour_sin', 'hour_cos']
        
        solar_forecast = self.solar_forecaster.predict(feature_df[solar_features])
        wind_forecast = self.wind_forecaster.predict(feature_df[wind_features])
        
        solar_capacity = self.config.get('solar_capacity', 100)
        wind_capacity = self.config.get('wind_capacity', 50)
        
        solar_forecast = np.clip(solar_forecast, 0, solar_capacity)
        wind_forecast = np.clip(wind_forecast, 0, wind_capacity)
        
        total_renewable = solar_forecast + wind_forecast
        
        return {
            'solar_forecast': solar_forecast.tolist(),
            'wind_forecast': wind_forecast.tolist(),
            'total_renewable': total_renewable.tolist(),
            'renewable_percentage': (np.sum(total_renewable) / 
                                   (np.sum(total_renewable) + self.config.get('base_demand', 1000)) * 100),
            'peak_renewable': float(np.max(total_renewable)),
            'forecast_horizon': len(weather_forecast)
        }
    
    def optimize_renewable_integration(self, renewable_forecast: Dict[str, Any],
                                     demand_forecast: List[float],
                                     grid_constraints: Dict[str, float]) -> Dict[str, Any]:
        
        solar_forecast = np.array(renewable_forecast['solar_forecast'])
        wind_forecast = np.array(renewable_forecast['wind_forecast'])
        total_renewable = solar_forecast + wind_forecast
        
        net_demand = np.array(demand_forecast) - total_renewable
        net_demand = np.maximum(net_demand, 0)
        
        curtailment = np.maximum(total_renewable - np.array(demand_forecast), 0)
        
        renewable_penetration = np.sum(total_renewable) / np.sum(demand_forecast) * 100
        
        storage_requirements = self.calculate_storage_requirements(net_demand, total_renewable)
        
        return {
            'net_demand': net_demand.tolist(),
            'renewable_curtailment': curtailment.tolist(),
            'renewable_penetration': float(renewable_penetration),
            'storage_requirements': storage_requirements,
            'integration_efficiency': (1 - np.sum(curtailment) / np.sum(total_renewable)) * 100,
            'carbon_savings': self.calculate_carbon_savings(total_renewable)
        }
    
    def calculate_storage_requirements(self, net_demand: np.ndarray, 
                                     renewable_generation: np.ndarray) -> Dict[str, float]:
        energy_deficit = np.sum(np.maximum(-net_demand, 0))
        energy_surplus = np.sum(np.maximum(net_demand - renewable_generation, 0))
        
        return {
            'required_capacity': float(np.max(np.abs(net_demand))),
            'energy_storage': float(energy_deficit),
            'peak_charging': float(np.max(net_demand)),
            'peak_discharging': float(np.max(-net_demand)),
            'round_trip_efficiency': 0.85
        }
    
    def calculate_carbon_savings(self, renewable_generation: np.ndarray) -> Dict[str, float]:
        total_renewable_energy = np.sum(renewable_generation)
        carbon_intensity = 0.5
        carbon_savings = total_renewable_energy * carbon_intensity
        
        return {
            'carbon_savings_kg': float(carbon_savings),
            'equivalent_trees': int(carbon_savings / 21.77),
            'equivalent_cars': int(carbon_savings / 4600)
        }
