import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
import holidays

class TemporalFusionTransformer(nn.Module):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.encoder_lstm = nn.LSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            batch_first=True,
            dropout=config.get('dropout', 0.1)
        )
        self.decoder_lstm = nn.LSTM(
            input_size=config['input_size'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            batch_first=True,
            dropout=config.get('dropout', 0.1)
        )
        self.attention = nn.MultiheadAttention(
            embed_dim=config['hidden_size'],
            num_heads=config.get('num_heads', 4),
            dropout=config.get('attention_dropout', 0.1)
        )
        self.forecast_head = nn.Sequential(
            nn.Linear(config['hidden_size'] * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, config['output_size'])
        )
    
    def forward(self, encoder_input, decoder_input):
        encoder_output, (hidden, cell) = self.encoder_lstm(encoder_input)
        decoder_output, _ = self.decoder_lstm(decoder_input, (hidden, cell))
        
        attended_output, _ = self.attention(
            decoder_output.transpose(0, 1),
            encoder_output.transpose(0, 1),
            encoder_output.transpose(0, 1)
        )
        
        combined = torch.cat([decoder_output, attended_output.transpose(0, 1)], dim=-1)
        forecast = self.forecast_head(combined[:, -1, :])
        return forecast

class DemandForecaster:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.scalers = {}
        self.us_holidays = holidays.US()
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_holiday'] = df.index.map(lambda x: x in self.us_holidays).astype(int)
        
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        for window in [24, 168, 720]:
            df[f'demand_rolling_mean_{window}'] = df['energy_demand'].rolling(window=window, min_periods=1).mean()
            df[f'demand_rolling_std_{window}'] = df['energy_demand'].rolling(window=window, min_periods=1).std()
        
        df['demand_lag_1'] = df['energy_demand'].shift(1)
        df['demand_lag_24'] = df['energy_demand'].shift(24)
        df['demand_lag_168'] = df['energy_demand'].shift(168)
        
        return df.dropna()
    
    def prepare_sequences(self, df: pd.DataFrame, target_col: str, 
                         sequence_length: int, forecast_horizon: int) -> Tuple[np.ndarray, np.ndarray]:
        feature_cols = [col for col in df.columns if col != target_col]
        
        X, y = [], []
        for i in range(len(df) - sequence_length - forecast_horizon + 1):
            X.append(df[feature_cols].iloc[i:i+sequence_length].values)
            y.append(df[target_col].iloc[i+sequence_length:i+sequence_length+forecast_horizon].values)
        
        return np.array(X), np.array(y)
    
    def build_model(self, input_size: int, output_size: int) -> nn.Module:
        config = {
            'input_size': input_size,
            'hidden_size': self.config.get('hidden_size', 128),
            'num_layers': self.config.get('num_layers', 2),
            'output_size': output_size,
            'num_heads': self.config.get('num_heads', 4),
            'dropout': self.config.get('dropout', 0.1)
        }
        return TemporalFusionTransformer(config).to(self.device)
    
    def train(self, train_data: pd.DataFrame, epochs: int = 100):
        feature_df = self.create_features(train_data)
        sequence_length = self.config.get('sequence_length', 168)
        forecast_horizon = self.config.get('forecast_horizon', 24)
        
        X_train, y_train = self.prepare_sequences(feature_df, 'energy_demand', 
                                                 sequence_length, forecast_horizon)
        
        input_size = X_train.shape[2]
        self.model = self.build_model(input_size, forecast_horizon)
        
        criterion = nn.HuberLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), 
                                   lr=self.config.get('learning_rate', 0.001))
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
        
        X_train_tensor = torch.FloatTensor(X_train).to(self.device)
        y_train_tensor = torch.FloatTensor(y_train).to(self.device)
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            predictions = self.model(X_train_tensor, X_train_tensor[:, -24:])
            loss = criterion(predictions, y_train_tensor)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(loss)
            
            if epoch % 20 == 0:
                print(f'Epoch {epoch}, Loss: {loss.item():.4f}')
    
    def forecast_demand(self, historical_data: pd.DataFrame, 
                       weather_forecast: pd.DataFrame,
                       hours_ahead: int = 24) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        with torch.no_grad():
            feature_df = self.create_features(historical_data)
            sequence_length = self.config.get('sequence_length', 168)
            
            if len(feature_df) < sequence_length:
                raise ValueError(f"Need at least {sequence_length} historical data points")
            
            latest_sequence = feature_df.iloc[-sequence_length:]
            input_tensor = torch.FloatTensor(latest_sequence.values).unsqueeze(0).to(self.device)
            
            predictions = self.model(input_tensor, input_tensor[:, -24:])
            forecast = predictions.cpu().numpy()[0]
            
            confidence_intervals = self.calculate_confidence_intervals(forecast)
            
            return {
                'demand_forecast': forecast.tolist(),
                'confidence_intervals': confidence_intervals,
                'timestamp': pd.Timestamp.now(),
                'forecast_horizon': hours_ahead,
                'peak_demand': float(np.max(forecast)),
                'peak_time': int(np.argmax(forecast))
            }
    
    def calculate_confidence_intervals(self, forecast: np.ndarray, 
                                     confidence_level: float = 0.95) -> Dict[str, List[float]]:
        z_score = 1.96
        std_dev = forecast * 0.1
        
        upper = forecast + z_score * std_dev
        lower = forecast - z_score * std_dev
        
        return {
            'upper_bound': upper.tolist(),
            'lower_bound': lower.tolist(),
            'confidence_level': confidence_level
        }