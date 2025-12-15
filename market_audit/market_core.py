import gymnasium as gym
import numpy as np
from gymnasium import spaces
from .nash_oracle import NashOracle

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
        
        # Regime Switch Parameters (3-State Markov Chain)
        # Regime 0: Recession - Low Quality, High Sensitivity (consumers broke and picky)
        self.recession_quality = np.array(self.config.get('recession_quality', [1.0, 1.0]))
        self.recession_sensitivity = self.config.get('recession_sensitivity', 1.0)

        # Regime 1: Normal/Stagnation - Baseline calibration
        self.normal_quality = np.array(self.config.get('normal_quality', [2.0, 2.0]))
        self.normal_sensitivity = self.config.get('normal_sensitivity', 0.8)

        # Regime 2: Boom - High Quality, Low Sensitivity (consumers flush with cash)
        self.boom_quality = np.array(self.config.get('boom_quality', [3.0, 3.0]))
        self.boom_sensitivity = self.config.get('boom_sensitivity', 0.5)

        # Transition Matrix: T[i,j] = P(s_{t+1}=j | s_t=i)
        # High self-transition (0.98) creates sticky regimes ~50 steps average duration
        self.transition_matrix = np.array(self.config.get('transition_matrix', [
            [0.98, 0.02, 0.00],  # From Recession
            [0.01, 0.98, 0.01],  # From Normal
            [0.00, 0.02, 0.98]   # From Boom
        ]))
        
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
        self.current_regime = 1 # 0: Recession, 1: Normal, 2: Boom

        # Market size (total consumer mass)
        self.market_size = self.config.get('market_size', 1.0)

        # Initialize Nash Oracle for computing theoretical benchmarks
        self.nash_oracle = NashOracle()
        
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
            # Initialize regime with stationary distribution of Markov chain
            # For simplicity, start in Normal regime (can be randomized)
            regime_probs = self.config.get('initial_regime_probs', [0.0, 1.0, 0.0])
            self.current_regime = self.np_random.choice([0, 1, 2], p=regime_probs)
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
            if self.current_regime == 0: # Recession
                current_quality = self.recession_quality
                current_beta = self.recession_sensitivity
            elif self.current_regime == 1: # Normal
                current_quality = self.normal_quality
                current_beta = self.normal_sensitivity
            else: # Boom (regime == 2)
                current_quality = self.boom_quality
                current_beta = self.boom_sensitivity

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
            # Update Regime using Markov Transition Matrix
            # Sample next regime based on current regime's transition probabilities
            transition_probs = self.transition_matrix[self.current_regime]
            self.current_regime = self.np_random.choice([0, 1, 2], p=transition_probs)

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

    def get_current_nash_equilibrium(self):
        """
        Compute Nash equilibrium for the current market state.

        Returns:
            dict with nash_prices, jpm_prices, and market parameters
        """
        # Extract current costs from state
        current_costs = self.state[3:5] if self.state is not None else self.production_costs_mean

        # Get current market parameters based on mode
        if self.market_mode == 'static':
            quality = self.base_quality
            beta = self.base_price_sensitivity

        elif self.market_mode == 'ar_drift':
            quality = self.base_quality
            beta = self.base_price_sensitivity
            # Note: AR drift affects utility additively, not through alpha/beta

        elif self.market_mode == 'regime_switch':
            # Use current regime parameters
            if self.current_regime == 0:  # Recession
                quality = self.recession_quality
                beta = self.recession_sensitivity
            elif self.current_regime == 1:  # Normal
                quality = self.normal_quality
                beta = self.normal_sensitivity
            else:  # Boom
                quality = self.boom_quality
                beta = self.boom_sensitivity
        else:
            quality = self.base_quality
            beta = self.base_price_sensitivity

        # Compute Nash equilibrium
        nash_prices, nash_profits, converged = self.nash_oracle.compute_nash_equilibrium(
            current_costs, quality, beta
        )

        # Compute Joint Profit Maximum
        jpm_prices, jpm_profits, total_jpm = self.nash_oracle.compute_joint_profit_maximum(
            current_costs, quality, beta
        )

        return {
            'nash_prices': nash_prices,
            'nash_profits': nash_profits,
            'jpm_prices': jpm_prices,
            'jpm_profits': jpm_profits,
            'costs': current_costs,
            'quality': quality,
            'beta': beta,
            'regime': self.current_regime if self.market_mode == 'regime_switch' else None,
            'converged': converged
        }

    def get_regime_name(self, regime=None):
        """Get human-readable name for regime."""
        if regime is None:
            regime = self.current_regime

        regime_names = {0: 'Recession', 1: 'Normal', 2: 'Boom'}
        return regime_names.get(regime, 'Unknown')

