import numpy as np
from typing import Dict, List, Any

class ConstraintHandler:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def check_grid_constraints(self, power_flows: Dict[str, List[float]],
                             line_limits: Dict[str, float]) -> Dict[str, Any]:
        
        violations = {}
        constraint_checks = {}
        
        for line, flow in power_flows.items():
            limit = line_limits.get(line, float('inf'))
            max_flow = max(flow)
            min_flow = min(flow)
            
            overload = max_flow > limit
            underload = min_flow < -limit
            
            if overload or underload:
                violations[line] = {
                    'max_flow': max_flow,
                    'limit': limit,
                    'overload_percentage': (max_flow - limit) / limit * 100 if overload else 0,
                    'violation_type': 'overload' if overload else 'underload'
                }
            
            constraint_checks[line] = {
                'within_limits': not (overload or underload),
                'utilization': abs(max_flow) / limit * 100 if limit > 0 else 0
            }
        
        return {
            'constraint_violations': violations,
            'constraint_checks': constraint_checks,
            'total_violations': len(violations),
            'grid_stability': len(violations) == 0
        }
    
    def calculate_reserve_requirements(self, demand_forecast: List[float],
                                    renewable_forecast: List[float]) -> Dict[str, float]:
        
        peak_demand = max(demand_forecast)
        total_renewable = sum(renewable_forecast)
        
        spinning_reserve = peak_demand * 0.05
        non_spinning_reserve = peak_demand * 0.03
        regulation_reserve = peak_demand * 0.01
        
        return {
            'spinning_reserve': spinning_reserve,
            'non_spinning_reserve': non_spinning_reserve,
            'regulation_reserve': regulation_reserve,
            'total_reserve': spinning_reserve + non_spinning_reserve + regulation_reserve,
            'reserve_margin': (spinning_reserve + non_spinning_reserve) / peak_demand * 100
        }