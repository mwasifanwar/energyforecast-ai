import numpy as np
from typing import Dict, List, Any

class DispatchOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def optimize_economic_dispatch(self, demand_forecast: List[float],
                                 generator_costs: Dict[str, float],
                                 generator_limits: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        
        time_horizon = len(demand_forecast)
        generators = list(generator_costs.keys())
        
        dispatch = {}
        total_cost = 0
        
        for t in range(time_horizon):
            hour_demand = demand_forecast[t]
            remaining_demand = hour_demand
            
            hour_dispatch = {}
            hour_cost = 0
            
            sorted_generators = sorted(generators, key=lambda x: generator_costs[x])
            
            for gen in sorted_generators:
                if remaining_demand <= 0:
                    hour_dispatch[gen] = 0
                    continue
                
                min_power = generator_limits[gen]['min_power']
                max_power = generator_limits[gen]['max_power']
                
                if remaining_demand < min_power:
                    hour_dispatch[gen] = 0
                    continue
                
                gen_output = min(max_power, remaining_demand)
                hour_dispatch[gen] = gen_output
                hour_cost += gen_output * generator_costs[gen]
                remaining_demand -= gen_output
            
            if remaining_demand > 0:
                raise ValueError(f"Insufficient generation capacity at hour {t}")
            
            dispatch[t] = hour_dispatch
            total_cost += hour_cost
        
        return {
            'dispatch_schedule': dispatch,
            'total_cost': total_cost,
            'marginal_cost': max(generator_costs.values()),
            'utilization_rates': self.calculate_utilization(dispatch, generator_limits)
        }
    
    def calculate_utilization(self, dispatch: Dict[int, Dict[str, float]],
                            generator_limits: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        
        utilization = {}
        for gen in generator_limits.keys():
            total_output = sum(dispatch[t].get(gen, 0) for t in dispatch)
            total_capacity = sum(generator_limits[gen]['max_power'] for t in dispatch)
            utilization[gen] = total_output / total_capacity if total_capacity > 0 else 0
        
        return utilization