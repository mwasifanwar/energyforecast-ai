import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Any

class EnergyVisualizer:
    def __init__(self):
        self.color_scheme = {
            'demand': '#1f77b4',
            'forecast': '#ff7f0e',
            'renewable': '#2ca02c',
            'conventional': '#d62728',
            'storage': '#9467bd'
        }
    
    def create_forecast_dashboard(self, historical_data: pd.DataFrame,
                                forecast_results: Dict[str, Any],
                                weather_data: pd.DataFrame) -> go.Figure:
        
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=('Energy Demand Forecast', 'Weather Correlation',
                          'Renewable Generation', 'Grid Stability',
                          'Price Forecast', 'Anomaly Detection'),
            specs=[[{"secondary_y": False}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        forecast_dates = pd.date_range(start=historical_data.index[-1] + pd.Timedelta(hours=1),
                                     periods=len(forecast_results['demand_forecast']), freq='H')
        
        fig.add_trace(
            go.Scatter(x=historical_data.index, y=historical_data['energy_demand'],
                      name='Historical Demand', line=dict(color=self.color_scheme['demand'])),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=forecast_dates, y=forecast_results['demand_forecast'],
                      name='Demand Forecast', line=dict(color=self.color_scheme['forecast'])),
            row=1, col=1
        )
        
        if 'confidence_intervals' in forecast_results:
            fig.add_trace(
                go.Scatter(x=forecast_dates, y=forecast_results['confidence_intervals']['upper_bound'],
                          name='Upper Bound', line=dict(dash='dash', color='gray'),
                          showlegend=False),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(x=forecast_dates, y=forecast_results['confidence_intervals']['lower_bound'],
                          name='Lower Bound', line=dict(dash='dash', color='gray'),
                          fill='tonexty', showlegend=False),
                row=1, col=1
            )
        
        fig.add_trace(
            go.Scatter(x=weather_data.index, y=weather_data['temperature'],
                      name='Temperature', line=dict(color='red')),
            row=1, col=2, secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=historical_data.index, y=historical_data['energy_demand'],
                      name='Demand', line=dict(color='blue')),
            row=1, col=2, secondary_y=True
        )
        
        if 'renewable_forecast' in forecast_results:
            fig.add_trace(
                go.Bar(x=forecast_dates, y=forecast_results['renewable_forecast']['solar_forecast'],
                      name='Solar Forecast', marker_color='yellow'),
                row=2, col=1
            )
            fig.add_trace(
                go.Bar(x=forecast_dates, y=forecast_results['renewable_forecast']['wind_forecast'],
                      name='Wind Forecast', marker_color='green'),
                row=2, col=1
            )
        
        fig.update_layout(height=900, title_text="Energy Forecasting Dashboard")
        return fig