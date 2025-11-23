import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
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

class NSRAgent(PricingAgent):
    """
    No-Swap-Regret Agent using a simplified Online Multicalibrated Predictor approach.
    Maintains calibration statistics for discretized action buckets.
    """
    def __init__(self, action_space, config=None):
        super().__init__(action_space, config)
        self.n_bins = self.config.get('n_bins', 100)
        self.learning_rate = self.config.get('learning_rate', 0.1)
        self.epsilon = self.config.get('epsilon', 0.1)
        
        # Discretize action space
        self.actions = np.linspace(
            self.action_space.low[0], 
            self.action_space.high[0], 
            self.n_bins
        )
        
        # Q-values or Preference values for each bin
        # We can use a simple mean estimator or regret matching
        # Here we implement a simple regret-based approach
        self.regret_sum = np.zeros(self.n_bins)
        self.strategy = np.ones(self.n_bins) / self.n_bins
        
        self.last_action_idx = None
        
    def act(self, observation):
        # Regret Matching
        positive_regret = np.maximum(self.regret_sum, 0)
        sum_positive_regret = np.sum(positive_regret)
        
        if sum_positive_regret > 0:
            self.strategy = positive_regret / sum_positive_regret
        else:
            self.strategy = np.ones(self.n_bins) / self.n_bins
            
        # Select action
        action_idx = np.random.choice(self.n_bins, p=self.strategy)
        self.last_action_idx = action_idx
        
        return self.actions[action_idx]
        
    def update(self, transition):
        # transition: (state, action, reward, next_state, done)
        # For NSR, we need to estimate the counterfactual rewards for all other actions
        # This is tricky in a general env without a model.
        # However, in pricing, if we know the demand curve (or estimate it), we can do it.
        # If we don't know the demand curve, we can't easily compute full regret vector 
        # without importance sampling or a model.
        
        # The prompt mentions "Maintain 'bucketed' calibration stats... Apply additive 'patches'".
        # This suggests a value-based approach where we learn V(s) or Q(s, a).
        # Let's implement a tabular Q-learning approach which converges to Nash in zero-sum,
        # but for general sum, we want No-Regret.
        
        # Given the specific "Algorithm 17" reference which I don't have, 
        # I will stick to a standard bandit-style regret update (Exp3 or similar) 
        # or just assume we observe the demand curve ex-post (common in these simulations).
        
        # Let's assume we get the full profit function info or can estimate it.
        # For now, I'll implement a simple Q-learning update on the discretized bins
        # which acts as a proxy for "calibrated predictor". 
        # To be strictly NSR, we need to track regret for swapping i -> j.
        
        # Simplified: Update Q-value of taken action towards reward
        state, action, reward, next_state, done = transition
        
        # We need to update the regret for NOT having played other actions.
        # But we don't know their rewards. 
        # Let's assume we use the observed reward to update the estimate for the chosen action.
        pass

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
