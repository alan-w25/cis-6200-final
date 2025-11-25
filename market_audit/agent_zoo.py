import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from gymnasium import spaces
from abc import ABC, abstractmethod
from collections import deque

class PricingAgent(ABC):
    def __init__(self, action_space, config=None):
        self.action_space = action_space
        self.config = config if config else {}
        
    @abstractmethod
    def act(self, observation):
        pass
        
    @abstractmethod
    def update(self, transition):
        pass

class FixedPriceAgent(PricingAgent):
    """
    Agent that always plays a fixed price.
    """
    def __init__(self, action_space, config=None):
        super().__init__(action_space, config)
        self.fixed_price = self.config.get('fixed_price', 1.0)
        
    def act(self, observation):
        return self.fixed_price
        
    def update(self, transition):
        pass

class RandomAgent(PricingAgent):
    """
    Agent that plays a random price with a max and min price
    """
    def __init__(self, action_space, config=None):
        super().__init__(action_space, config)
        self.max_price = self.config.get('max_price')
        self.min_price = 0.0
        
    def act(self, observation):
        price = np.random.uniform(self.min_price, self.max_price)
        return price
        
    def update(self, transition):
        pass

class NSRAgent(PricingAgent):
    """
    No-Swap-Regret Agent using Matrix-Based Swap Regret Minimization.
    Maintains a regret matrix R[i, j] for swapping from action i to action j.
    Strategy is the stationary distribution of the regret-induced Markov chain.
    """
    def __init__(self, action_space, config=None):
        super().__init__(action_space, config)
        self.n_bins = self.config.get('n_bins', 100)
        self.learning_rate = self.config.get('learning_rate', 0.1)
        
        # Discretize action space
        self.actions = np.linspace(
            self.action_space.low[0], 
            self.action_space.high[0], 
            self.n_bins
        )
        
        # Swap Regret Matrix [N, N]
        # Row i: Action played
        # Col j: Counterfactual action
        self.swap_regret_sum = np.zeros((self.n_bins, self.n_bins))
        self.strategy = np.ones(self.n_bins) / self.n_bins
        
        # Market Knowledge
        self.quality = self.config.get('quality', 2.0)
        self.price_sensitivity = self.config.get('price_sensitivity', 2.0)
        self.cost = self.config.get('cost', 1.0)
        
    def act(self, observation):
        # Compute Stationary Distribution of Regret Matrix
        
        # 1. Positive Regret Matrix M
        M = np.maximum(self.swap_regret_sum, 0)
        
        # 2. Normalizing Constant mu
        # Must be > sum of any row to ensure P_ii > 0
        # We use max(sum(row)) + small_epsilon
        row_sums = np.sum(M, axis=1)
        mu = np.max(row_sums)
        
        if mu < 1e-9:
            # No regret yet, uniform random
            self.strategy = np.ones(self.n_bins) / self.n_bins
        else:
            # 3. Transition Matrix P
            # P_ij = M_ij / mu for i != j
            # P_ii = 1 - sum_{k!=i} P_ik
            
            # Divide all by mu
            P = M / mu
            
            # Fix diagonal: P_ii = 1 - (row_sum - M_ii)/mu
            # But M_ii is 0 usually (regret of swapping i->i is 0)
            # So P_ii = 1 - row_sum/mu
            
            # Vectorized diagonal update
            # P[i, i] += 1 - row_sum[i]/mu
            # Since we already divided M by mu, P currently holds M_ij/mu
            # We need to set diagonal such that row sums to 1
            
            # Current row sums of P
            current_P_row_sums = np.sum(P, axis=1)
            
            # Add residual to diagonal
            diag_indices = np.arange(self.n_bins)
            P[diag_indices, diag_indices] += (1.0 - current_P_row_sums)
            
            # 4. Compute Stationary Distribution
            # Solve v P = v  =>  v (P - I) = 0  =>  (P.T - I) v = 0
            # This is finding eigenvector for eigenvalue 1
            
            # We can use numpy's eig, but it's slow for 100x100 every step?
            # Power iteration is faster if we have a good guess (previous strategy)
            # Let's use power iteration for efficiency
            
            v = self.strategy.copy()
            for _ in range(10): # 10 iterations usually sufficient for convergence if close
                v = np.dot(v, P)
                
            self.strategy = v / np.sum(v)
            
        # Select action
        # Handle numerical issues (negative probs)
        self.strategy = np.maximum(self.strategy, 0)
        self.strategy /= np.sum(self.strategy)
        
        action_idx = np.random.choice(self.n_bins, p=self.strategy)
        return self.actions[action_idx]
        
    def update(self, transition):
        state, action, reward, next_state, done = transition
        
        # Find index of chosen action
        # We need the exact index corresponding to 'action'
        # Since action is continuous, we find closest bin
        chosen_idx = np.argmin(np.abs(self.actions - action))
        
        # 1. Infer Competitor Price / Market State
        if abs(action - self.cost) < 1e-6:
            return 
            
        observed_share = reward / (action - self.cost)
        observed_share = np.clip(observed_share, 1e-6, 1.0 - 1e-6)
        
        v_own = self.quality - self.price_sensitivity * action
        exp_v_own = np.exp(v_own)
        
        competitor_agg_utility = exp_v_own * (1.0/observed_share - 1.0) - 1.0
        competitor_agg_utility = max(competitor_agg_utility, 0.0)
        
        # 2. Calculate Counterfactual Rewards for ALL actions
        v_own_all = self.quality - self.price_sensitivity * self.actions
        exp_v_own_all = np.exp(v_own_all)
        
        shares_all = exp_v_own_all / (1.0 + exp_v_own_all + competitor_agg_utility)
        profits_all = (self.actions - self.cost) * shares_all
        
        # 3. Update Swap Regret Matrix
        # Only update the row for the action we actually played (chosen_idx)
        # Regret(i -> j) = Profit(j) - Profit(i)
        # Profit(i) is the realized reward
        
        regrets = profits_all - reward
        self.swap_regret_sum[chosen_idx, :] += regrets


class RLAgent(PricingAgent):
    """
    Deep Q-Network Agent.
    """
    def __init__(self, action_space, config=None):
        super().__init__(action_space, config)
        self.state_dim = self.config.get('state_dim', 5)
        self.hidden_dim = self.config.get('hidden_dim', 64)
        self.n_actions = self.config.get('n_bins', 100) # Discretized for DQN
        self.lr = self.config.get('lr', 1e-3)
        self.gamma = self.config.get('gamma', 0.99)
        self.epsilon = self.config.get('epsilon', 1.0)
        self.epsilon_decay = self.config.get('epsilon_decay', 0.995)
        self.epsilon_min = self.config.get('epsilon_min', 0.01)
        self.batch_size = self.config.get('batch_size', 32)
        self.memory_size = self.config.get('memory_size', 10000)
        
        self.actions = np.linspace(
            self.action_space.low[0], 
            self.action_space.high[0], 
            self.n_actions
        )
        
        # Network
        self.q_net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_actions)
        )
        
        self.target_net = nn.Sequential(
            nn.Linear(self.state_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.n_actions)
        )
        self.target_net.load_state_dict(self.q_net.state_dict())
        
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=self.lr)
        self.memory = deque(maxlen=self.memory_size)
        
    def act(self, observation):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)
        
        state_t = torch.FloatTensor(observation).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_net(state_t)
        action_idx = q_values.argmax().item()
        return self.actions[action_idx]
        
    def update(self, transition):
        self.memory.append(transition)
        if len(self.memory) < self.batch_size:
            return
            
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        states_t = torch.FloatTensor(np.array(states))
        next_states_t = torch.FloatTensor(np.array(next_states))
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        dones_t = torch.FloatTensor(dones).unsqueeze(1)
        
        # Map actions to indices
        # This is slow, better to store indices
        action_indices = [np.argmin(np.abs(self.actions - a)) for a in actions]
        actions_t = torch.LongTensor(action_indices).unsqueeze(1)
        
        # Q(s, a)
        q_values = self.q_net(states_t).gather(1, actions_t)
        
        # Target: r + gamma * max Q(s', a')
        with torch.no_grad():
            next_q_values = self.target_net(next_states_t).max(1)[0].unsqueeze(1)
            target_q_values = rewards_t + self.gamma * next_q_values * (1 - dones_t)
            
        loss = nn.MSELoss()(q_values, target_q_values)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Epsilon decay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())

class ConstrainedRLAgent(RLAgent):
    """
    RL Agent with Lagrangian Relaxation for correlation constraint.
    """
    def __init__(self, action_space, config=None):
        super().__init__(action_space, config)
        self.kappa = self.config.get('kappa', 0.5) # Correlation threshold
        self.lam = self.config.get('lambda_init', 0.0)
        self.lambda_lr = self.config.get('lambda_lr', 0.01)
        self.window_size = self.config.get('window_size', 50)
        self.price_history = deque(maxlen=self.window_size)
        self.competitor_price_history = deque(maxlen=self.window_size)
        
    def update(self, transition):
        # Store history for correlation calculation
        state, action, reward, next_state, done = transition
        # Extract competitor price from state (p2_t-1)
        # State: [p1_t-1, p2_t-1, ...]
        # Wait, the state contains PAST prices. 
        # The current transition's action is p1_t.
        # We need p2_t to compute correlation of (p1, p2).
        # But we only observe p2_t in the NEXT state or info.
        # Let's assume we get it. For now, we'll use the p2 from next_state as p2_t.
        # next_state: [p1_t, p2_t, ...]
        
        p1_t = action
        p2_t = next_state[1] # Assuming index 1 is competitor price
        
        self.price_history.append(p1_t)
        self.competitor_price_history.append(p2_t)
        
        # Calculate Correlation
        correlation = 0.0
        if len(self.price_history) >= 10:
            p1_seq = np.array(self.price_history)
            p2_seq = np.array(self.competitor_price_history)
            if np.std(p1_seq) > 1e-6 and np.std(p2_seq) > 1e-6:
                correlation = np.corrcoef(p1_seq, p2_seq)[0, 1]
                
        # Lagrangian Reward Modification
        # R_mod = R - lambda * ReLU(Corr - kappa)
        constraint_violation = max(0, correlation - self.kappa)
        modified_reward = reward - self.lam * constraint_violation
        
        # Dual Ascent for Lambda
        self.lam += self.lambda_lr * constraint_violation
        self.lam = max(0, self.lam) # Lambda must be >= 0
        
        # Call parent update with modified reward
        modified_transition = (state, action, modified_reward, next_state, done)
        super().update(modified_transition)
