import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import IsolationForest

class GridAnomalyDetector:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.isolation_forest = IsolationForest(
            contamination=config.get('contamination', 0.01),
            random_state=42
        )
        self.autoencoder = None
        self.is_trained = False
        
    def detect_voltage_anomalies(self, voltage_data: np.ndarray) -> Dict[str, Any]:
        if voltage_data.ndim == 1:
            voltage_data = voltage_data.reshape(-1, 1)
        
        self.isolation_forest.fit(voltage_data)
        anomalies = self.isolation_forest.predict(voltage_data)
        anomaly_scores = self.isolation_forest.decision_function(voltage_data)
        
        anomaly_indices = np.where(anomalies == -1)[0]
        
        return {
            'anomaly_indices': anomaly_indices.tolist(),
            'anomaly_scores': anomaly_scores.tolist(),
            'anomaly_count': len(anomaly_indices),
            'voltage_stats': {
                'mean': float(np.mean(voltage_data)),
                'std': float(np.std(voltage_data)),
                'min': float(np.min(voltage_data)),
                'max': float(np.max(voltage_data))
            }
        }
    
    def detect_frequency_deviations(self, frequency_data: np.ndarray, 
                                  nominal_frequency: float = 60.0) -> Dict[str, Any]:
        deviations = np.abs(frequency_data - nominal_frequency)
        threshold = self.config.get('frequency_threshold', 0.2)
        
        anomaly_indices = np.where(deviations > threshold)[0]
        max_deviation = np.max(deviations)
        
        return {
            'anomaly_indices': anomaly_indices.tolist(),
            'deviation_magnitudes': deviations[anomaly_indices].tolist(),
            'max_deviation': float(max_deviation),
            'anomaly_count': len(anomaly_indices),
            'stability_index': max(0, 100 - max_deviation * 100)
        }