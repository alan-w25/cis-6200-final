import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import fsolve, minimize
from market_core import DuopolyEnv

def calculate_instantaneous_benchmarks(demand_shock, costs, env_config):
    """
    Calculates NE and Collusive prices for a specific state (demand shock + costs).
    """
    q1, q2 = env_config['quality']
    mu = env_config['price_sensitivity']
    
    # Effective Quality = Base Quality + Shock
    # In AR drift mode, shock is additive to utility
    eff_q1 = q1 + demand_shock
    eff_q2 = q2 + demand_shock
    
    c1, c2 = costs
    
    # --- Helper: Profit Function ---
    def get_profits(p):
        p1, p2 = p
        exp_u1 = np.exp(eff_q1 - mu * p1)
        exp_u2 = np.exp(eff_q2 - mu * p2)
        denom = 1.0 + exp_u1 + exp_u2
        
        s1 = exp_u1 / denom
        s2 = exp_u2 / denom
        
        pi1 = (p1 - c1) * s1
        pi2 = (p2 - c2) * s2
        return pi1, pi2

    # --- 1. Nash Equilibrium (FOCs) ---
    def focs(p):
        p1, p2 = p
        exp_u1 = np.exp(eff_q1 - mu * p1)
        exp_u2 = np.exp(eff_q2 - mu * p2)
        denom = 1.0 + exp_u1 + exp_u2
        
        s1 = exp_u1 / denom
        s2 = exp_u2 / denom
        
        # FOC: 1 - mu * (p - c) * (1 - s) = 0
        foc1 = 1 - mu * (p1 - c1) * (1 - s1)
        foc2 = 1 - mu * (p2 - c2) * (1 - s2)
        return [foc1, foc2]

    # Solve NE
    ne_prices = fsolve(focs, x0=[c1+1.0, c2+1.0])
    
    # --- 2. Collusive Optimum (Joint Profit Max) ---
    def neg_joint_profit(p):
        pi1, pi2 = get_profits(p)
        return -(pi1 + pi2)
    
    # Solve Collusive
    # Bounds to keep optimization stable
    bounds = [(c1, 20), (c2, 20)]
    res = minimize(neg_joint_profit, x0=ne_prices, bounds=bounds)
    collusive_prices = res.x
    
    return ne_prices, collusive_prices