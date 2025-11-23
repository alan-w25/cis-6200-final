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
        
        # Market Parameters
        self.quality = np.array(self.config.get('quality', [2.0, 2.0]))
        self.max_price = self.config.get('max_price', 10.0)
        self.production_costs = np.array(self.config.get('production_costs', [1.0, 1.0])) # c1, c2
        self.price_sensitivity = self.config.get('price_sensitivity', 1.0) # mu parameter in logit
        
        # Demand Shock Parameters
        self.demand_shock_mean = self.config.get('demand_shock_mean', 0.0)
        self.demand_shock_std = self.config.get('demand_shock_std', 0.1)
        
        # Action Space: Joint prices [p1, p2]
        # We allow prices in [0, max_price]
        self.action_space = spaces.Box(low=0, high=self.max_price, shape=(2,), dtype=np.float32)
        
        # Observation Space: [p1_t-1, p2_t-1, demand_shock_signal, c1, c2]
        # p_t-1 defaults to 0 initially
        self.observation_space = spaces.Box(
            low=np.array([0, 0, -np.inf, 0, 0]),
            high=np.array([self.max_price, self.max_price, np.inf, np.inf, np.inf]),
            dtype=np.float32
        )
        
        self.state = None
        self.current_step = 0
        self.max_steps = self.config.get('max_steps', 100)
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initial prices (can be random or fixed)
        p_init = np.zeros(2)
        
        # Initial demand shock
        demand_shock = self.np_random.normal(self.demand_shock_mean, self.demand_shock_std)
        
        # Costs (can be non-stationary if configured, but fixed for now per episode usually)
        c1, c2 = self.production_costs
        
        self.state = np.array([p_init[0], p_init[1], demand_shock, c1, c2], dtype=np.float32)
        self.current_step = 0
        
        return self.state, {}
        
    def step(self, actions):
        """
        actions: [p1, p2]
        """
        p1, p2 = actions
        p1 = np.clip(p1, 0, self.max_price)
        p2 = np.clip(p2, 0, self.max_price)
        prices = np.array([p1, p2])
        
        # Extract state info
        _, _, demand_shock, c1, c2 = self.state
        costs = np.array([c1, c2])
        
        # Logit Demand Model
        # Utility V_i = quality_i - mu * p_i + demand_shock
        # Note: Demand shock applies to market potential or specific goods? 
        # Usually demand shock affects overall demand or specific preference.
        # Let's assume it affects the outside option or boosts all goods.
        # Simple version: V_i = Q_i - p_i + shock
        
        # Let's use a standard formulation:
        # U_i = V_i + epsilon
        # V_i = alpha_i - beta * p_i + xi (shock)
        # Outside option V_0 = 0
        
        # We'll apply shock to both for general market demand fluctuation
        utilities = self.quality - self.price_sensitivity * prices + demand_shock
        
        # Probabilities (Market Shares)
        # D_i = exp(V_i) / (1 + sum(exp(V_j)))
        exp_u = np.exp(utilities)
        total_exp = 1.0 + np.sum(exp_u) # 1.0 is for outside option
        shares = exp_u / total_exp
        
        # Profits
        # pi_i = (p_i - c_i) * D_i
        profits = (prices - costs) * shares
        
        # Update State
        # New demand shock for next step
        next_demand_shock = self.np_random.normal(self.demand_shock_mean, self.demand_shock_std)
        self.state = np.array([p1, p2, next_demand_shock, c1, c2], dtype=np.float32)
        
        self.current_step += 1
        terminated = self.current_step >= self.max_steps
        truncated = False
        
        info = {
            'shares': shares,
            'profits': profits,
            'demand_shock': demand_shock
        }
        
        # Reward is usually the profit vector for multi-agent
        return self.state, profits, terminated, truncated, info

    def render(self):
        pass
