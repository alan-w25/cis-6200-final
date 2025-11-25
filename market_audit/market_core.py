import gymnasium as gym
import numpy as np
from gymnasium import spaces

class DuopolyEnv(gym.Env):
    """
    Duopoly Market Environment for auditing pricing behaviors.
    
    State Space: [p1_t-1, p2_t-1, demand_shock_signal, c1, c2]
    Action Space: Continuous prices for both agents (or single agent if training one against fixed).
    Here we assume a joint action space or we can step with a tuple. 
    For standard Gym, it's usually single agent perspective, but for multi-agent simulation, 
    we might want to accept a tuple of actions. 
    However, to keep it simple and compatible with standard RL loops, 
    we can design it as a multi-agent env or just take a list of prices.
    Let's design it to take a tuple/list of 2 prices: [p1, p2].
    """
    
    def __init__(self, config=None):
        super(DuopolyEnv, self).__init__()
        
        self.config = config if config else {}
        self.market_mode = self.config.get('market_mode', 'ar_drift') # Options: 'static', 'ar_drift', 'regime_switch'
        
        # Market Parameters (Base)
        self.base_quality = np.array(self.config.get('quality', [2.0, 2.0]))
        self.base_price_sensitivity = self.config.get('price_sensitivity', 2.0)
        self.max_price = self.config.get('max_price', 10.0)
        self.production_costs_mean = np.array(self.config.get('production_costs', [1.0, 1.0]))
        
        # AR Drift Parameters
        self.demand_shock_mean = self.config.get('demand_shock_mean', 0.0)
        self.demand_shock_std = self.config.get('demand_shock_std', 0.1)
        self.demand_shock_phi = self.config.get('demand_shock_phi', 0.5)
        
        # Regime Switch Parameters
        # Boom: High Quality, Low Sensitivity
        self.boom_quality = np.array(self.config.get('boom_quality', [2.5, 2.5]))
        self.boom_sensitivity = self.config.get('boom_sensitivity', 0.8)
        # Recession: Low Quality, High Sensitivity
        self.recession_quality = np.array(self.config.get('recession_quality', [1.5, 1.5]))
        self.recession_sensitivity = self.config.get('recession_sensitivity', 1.5)
        self.regime_transition_prob = self.config.get('regime_transition_prob', 0.05)
        
        # Cost Dynamics (AR(1) for all modes unless disabled via config)
        self.cost_std = self.config.get('cost_std', 0.05)
        self.cost_phi = self.config.get('cost_phi', 0.9)
        
        # Action Space
        self.action_space = spaces.Box(low=0, high=self.max_price, shape=(2,), dtype=np.float32)
        
        # Observation Space: [p1_t-1, p2_t-1, demand_signal, c1, c2]
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.inf, 0, 0]),
            high=np.array([self.max_price, self.max_price, np.inf, np.inf, np.inf]),
            dtype=np.float32
        )
        
        self.state = None
        self.current_step = 0
        self.max_steps = self.config.get('max_steps', 100)
        
        # Internal state for dynamics
        self.current_demand_shock = 0.0
        self.current_regime = 0 # 0: Boom, 1: Recession
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        p_init = np.zeros(2)
        
        # Initialize Demand State based on Mode
        if self.market_mode == 'static':
            self.current_demand_shock = 0.0
            self.current_regime = 0
            
        elif self.market_mode == 'ar_drift':
            # Stationary initialization
            if abs(self.demand_shock_phi) < 1.0 - 1e-6:
                std = self.demand_shock_std / np.sqrt(1 - self.demand_shock_phi**2)
            else:
                std = self.demand_shock_std
            self.current_demand_shock = self.np_random.normal(self.demand_shock_mean, std)
            
        elif self.market_mode == 'regime_switch':
            # Randomly pick initial regime
            self.current_regime = 0 if self.np_random.random() < 0.5 else 1
            self.current_demand_shock = float(self.current_regime) # Signal is regime index
            
        # Initialize Costs (AR process)
        if abs(self.cost_phi) < 1.0 - 1e-6:
            cost_std = self.cost_std / np.sqrt(1 - self.cost_phi**2)
        else:
            cost_std = self.cost_std
        c_noise = self.np_random.normal(0, cost_std, size=2)
        costs = self.production_costs_mean + c_noise
        costs = np.maximum(costs, 0.0)
        
        self.state = np.array([p_init[0], p_init[1], self.current_demand_shock, costs[0], costs[1]], dtype=np.float32)
        self.current_step = 0
        
        return self.state, {}
        
    def step(self, actions):
        p1, p2 = actions
        p1 = np.clip(p1, 0, self.max_price)
        p2 = np.clip(p2, 0, self.max_price)
        prices = np.array([p1, p2])
        
        # Extract current costs from state (before update)
        # State: [p1_old, p2_old, signal, c1, c2]
        current_costs = self.state[3:5]
        
        # Determine Demand Parameters based on Mode
        current_quality = self.base_quality
        current_beta = self.base_price_sensitivity
        shock_value = 0.0
        
        if self.market_mode == 'static':
            shock_value = 0.0
            
        elif self.market_mode == 'ar_drift':
            # In AR drift, shock is additive to utility
            # Signal in state is the shock value
            shock_value = self.current_demand_shock
            
        elif self.market_mode == 'regime_switch':
            # Use current regime for this step's dynamics
            if self.current_regime == 0: # Boom
                current_quality = self.boom_quality
                current_beta = self.boom_sensitivity
            else: # Recession
                current_quality = self.recession_quality
                current_beta = self.recession_sensitivity
            
            # Signal is regime index
            shock_value = 0.0 # Shock absorbed into alpha/beta changes
            
        # Calculate Utilities
        # V_i = alpha_i - beta * p_i + shock
        utilities = current_quality - current_beta * prices + shock_value
        
        # Logit Demand
        exp_u = np.exp(utilities)
        total_exp = 1.0 + np.sum(exp_u)
        shares = exp_u / total_exp
        
        # Profits
        profits = (prices - current_costs) * shares
        
        # Update State Variables for Next Step
        next_signal = 0.0
        
        if self.market_mode == 'static':
            next_signal = 0.0
            
        elif self.market_mode == 'ar_drift':
            # Update AR(1) Shock
            noise = self.np_random.normal(0, self.demand_shock_std)
            self.current_demand_shock = self.demand_shock_mean + \
                self.demand_shock_phi * (self.current_demand_shock - self.demand_shock_mean) + \
                noise
            next_signal = self.current_demand_shock
            
        elif self.market_mode == 'regime_switch':
            # Update Regime (Markov Transition) for NEXT step
            if self.np_random.random() < self.regime_transition_prob:
                self.current_regime = 1 - self.current_regime
            
            # Signal is the updated regime
            next_signal = float(self.current_regime)
            
        # Update Costs (AR(1))
        cost_noise = self.np_random.normal(0, self.cost_std, size=2)
        next_costs = self.production_costs_mean + \
                     self.cost_phi * (current_costs - self.production_costs_mean) + \
                     cost_noise
        next_costs = np.maximum(next_costs, 0.0)
        
        # Construct New State
        self.state = np.array([p1, p2, next_signal, next_costs[0], next_costs[1]], dtype=np.float32)
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            'shares': shares,
            'profits': profits,
            'demand_shock': next_signal,
            'costs': current_costs,
            'regime': self.current_regime if self.market_mode == 'regime_switch' else None
        }
        
        return self.state, profits, terminated, truncated, info

    def render(self):
        pass
