import pandas as pd
import numpy as np
from typing import Dict, List, Any

class EnergyDataProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def generate_sample_energy_data(self, start_date: str, end_date: str, 
                                  freq: str = 'H') -> pd.DataFrame:
        date_range = pd.date_range(start=start_date, end=end_date, freq=freq)
        
        base_demand = 1000
        seasonal_pattern = 200 * np.sin(2 * np.pi * (date_range.dayofyear - 80) / 365)
        weekly_pattern = 100 * (date_range.dayofweek >= 5)
        daily_pattern = 300 * np.sin(2 * np.pi * (date_range.hour - 6) / 24)
        noise = np.random.normal(0, 50, len(date_range))
        
        energy_demand = base_demand + seasonal_pattern + weekly_pattern + daily_pattern + noise
        energy_demand = np.maximum(energy_demand, 200)
        
        data = {
            'energy_demand': energy_demand,
            'energy_price': np.random.lognormal(3, 0.5, len(date_range)),
            'grid_frequency': np.random.normal(60, 0.05, len(date_range)),
            'voltage': np.random.normal(120, 2, len(date_range))
        }
        
        df = pd.DataFrame(data, index=date_range)
        return df
    
    def add_anomalies(self, df: pd.DataFrame, anomaly_rate: float = 0.01) -> pd.DataFrame:
        df_anomalous = df.copy()
        n_anomalies = int(len(df) * anomaly_rate)
        
        anomaly_indices = np.random.choice(len(df), n_anomalies, replace=False)
        
        for idx in anomaly_indices:
            anomaly_type = np.random.choice(['spike', 'dip', 'noise'])
            
            if anomaly_type == 'spike':
                df_anomalous.iloc[idx, 0] *= 1.5
            elif anomaly_type == 'dip':
                df_anomalous.iloc[idx, 0] *= 0.5
            else:
                df_anomalous.iloc[idx, 0] += np.random.normal(0, 200)
        
        return df_anomalous
    
    def calculate_energy_metrics(self, df: pd.DataFrame) -> Dict[str, float]:
        return {
            'total_energy': float(df['energy_demand'].sum()),
            'average_demand': float(df['energy_demand'].mean()),
            'peak_demand': float(df['energy_demand'].max()),
            'load_factor': float(df['energy_demand'].mean() / df['energy_demand'].max()),
            'daily_variation': float(df['energy_demand'].std() / df['energy_demand'].mean()),
            'price_volatility': float(df['energy_price'].std())
        }