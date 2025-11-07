import numpy as np
import pandas as pd
from typing import Dict, List, Any, Tuple
from scipy.optimize import minimize
import cvxpy as cp

class GridOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.generators = config.get('generators', {})
        self.storage_systems = config.get('storage_systems', {})
        self.grid_constraints = config.get('grid_constraints', {})
        
    def optimize_dispatch(self, demand_forecast: List[float],
                         renewable_forecast: List[float],
                         generator_costs: Dict[str, float],
                         storage_status: Dict[str, float]) -> Dict[str, Any]:
        
        time_horizon = len(demand_forecast)
        generators = list(self.generators.keys())
        storage_units = list(self.storage_systems.keys())
        
        generator_power = cp.Variable((len(generators), time_horizon))
        storage_charge = cp.Variable((len(storage_units), time_horizon))
        storage_discharge = cp.Variable((len(storage_units), time_horizon))
        storage_soc = cp.Variable((len(storage_units), time_horizon))
        renewable_curtailment = cp.Variable(time_horizon)
        
        objective = 0
        constraints = []
        
        for t in range(time_horizon):
            total_generation = cp.sum(generator_power[:, t])
            total_storage_discharge = cp.sum(storage_discharge[:, t])
            total_storage_charge = cp.sum(storage_charge[:, t])
            
            power_balance = (total_generation + total_storage_discharge + 
                           renewable_forecast[t] - renewable_curtailment[t] - 
                           total_storage_charge - demand_forecast[t])
            
            constraints.append(power_balance == 0)
            constraints.append(renewable_curtailment[t] >= 0)
            constraints.append(renewable_curtailment[t] <= renewable_forecast[t])
            
            for i, gen in enumerate(generators):
                min_power = self.generators[gen]['min_power']
                max_power = self.generators[gen]['max_power']
                ramp_up = self.generators[gen]['ramp_up']
                ramp_down = self.generators[gen]['ramp_down']
                
                constraints.append(generator_power[i, t] >= min_power)
                constraints.append(generator_power[i, t] <= max_power)
                
                if t > 0:
                    constraints.append(generator_power[i, t] - generator_power[i, t-1] <= ramp_up)
                    constraints.append(generator_power[i, t-1] - generator_power[i, t] <= ramp_down)
                
                cost = generator_costs.get(gen, 100)
                objective += cost * generator_power[i, t]
            
            for j, storage in enumerate(storage_units):
                max_charge = self.storage_systems[storage]['max_charge']
                max_discharge = self.storage_systems[storage]['max_discharge']
                capacity = self.storage_systems[storage]['capacity']
                efficiency = self.storage_systems[storage]['efficiency']
                initial_soc = storage_status.get(storage, 0.5) * capacity
                
                constraints.append(storage_charge[j, t] >= 0)
                constraints.append(storage_charge[j, t] <= max_charge)
                constraints.append(storage_discharge[j, t] >= 0)
                constraints.append(storage_discharge[j, t] <= max_discharge)
                
                if t == 0:
                    constraints.append(storage_soc[j, t] == initial_soc - 
                                     storage_discharge[j, t] / efficiency + 
                                     storage_charge[j, t] * efficiency)
                else:
                    constraints.append(storage_soc[j, t] == storage_soc[j, t-1] - 
                                     storage_discharge[j, t] / efficiency + 
                                     storage_charge[j, t] * efficiency)
                
                constraints.append(storage_soc[j, t] >= 0)
                constraints.append(storage_soc[j, t] <= capacity)
        
        problem = cp.Problem(cp.Minimize(objective), constraints)
        problem.solve(solver=cp.ECOS)
        
        if problem.status not in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
            raise ValueError("Optimization failed to converge")
        
        return self.format_solution(generator_power, storage_charge, storage_discharge, 
                                  storage_soc, renewable_curtailment, generators, storage_units)
    
    def format_solution(self, generator_power, storage_charge, storage_discharge,
                       storage_soc, renewable_curtailment, generators, storage_units):
        
        solution = {
            'generator_dispatch': {},
            'storage_operations': {},
            'renewable_curtailment': renewable_curtailment.value.tolist(),
            'total_cost': float(np.sum([np.sum(generator_power[i].value) * 100 
                                      for i in range(len(generators))])),
            'optimization_status': 'optimal'
        }
        
        for i, gen in enumerate(generators):
            solution['generator_dispatch'][gen] = generator_power[i].value.tolist()
        
        for j, storage in enumerate(storage_units):
            solution['storage_operations'][storage] = {
                'charge': storage_charge[j].value.tolist(),
                'discharge': storage_discharge[j].value.tolist(),
                'state_of_charge': storage_soc[j].value.tolist()
            }
        
        return solution
    
    def calculate_grid_stability(self, dispatch_solution: Dict[str, Any],
                               demand_forecast: List[float]) -> Dict[str, float]:
        
        total_generation = np.zeros(len(demand_forecast))
        
        for gen_dispatch in dispatch_solution['generator_dispatch'].values():
            total_generation += np.array(gen_dispatch)
        
        for storage_ops in dispatch_solution['storage_operations'].values():
            total_generation += np.array(storage_ops['discharge'])
        
        total_generation += np.array(dispatch_solution.get('renewable_generation', [0] * len(demand_forecast)))
        
        balance_errors = total_generation - np.array(demand_forecast)
        max_imbalance = np.max(np.abs(balance_errors))
        avg_imbalance = np.mean(np.abs(balance_errors))
        
        reserve_margin = self.calculate_reserve_margin(total_generation, demand_forecast)
        
        return {
            'max_power_imbalance': float(max_imbalance),
            'average_power_imbalance': float(avg_imbalance),
            'reserve_margin': float(reserve_margin),
            'grid_stability_score': max(0, 100 - max_imbalance * 10),
            'voltage_stability_index': self.calculate_vsi(total_generation, demand_forecast)
        }
    
    def calculate_reserve_margin(self, generation: np.ndarray, demand: np.ndarray) -> float:
        peak_demand = np.max(demand)
        available_capacity = np.max(generation)
        return (available_capacity - peak_demand) / peak_demand * 100
    
    def calculate_vsi(self, generation: np.ndarray, demand: np.ndarray) -> float:
        load_ratio = demand / (generation + 1e-6)
        return float(np.exp(-np.std(load_ratio)))
