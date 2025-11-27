import numpy as np
import pandas as pd
from market_audit.market_core import DuopolyEnv
from market_audit.agent_zoo import RLAgent, FixedPriceAgent
from market_audit.utils import run_episode

def run_experiment(epsilon_decay):
    # 1. Initialize Environment with NEW CONFIG
    env_config = {
        'market_mode': 'static',
        'max_steps': 100,
        'max_price': 6.0,                # Reduced max price
        'price_sensitivity': 0.8,        # Lower sensitivity
        'production_costs': [1.0, 2.0],  # RL Agent (Index 0) has LOWER cost (1.0 vs 2.0)
        'quality': [2.0, 2.0]            # Equal quality
    }
    env = DuopolyEnv(config=env_config)

    # 2. Initialize Agents
    rl_config = {
        'hidden_dim': 64,
        'lr': 1e-3,
        'epsilon': 1.0,
        'epsilon_decay': epsilon_decay,
        'epsilon_min': 0.05,
        'memory_size': 10000,
        'batch_size': 32,
        'n_bins': 50 # Fewer bins for faster learning in smaller space
    }
    agent_rl = RLAgent(env.action_space, config=rl_config)

    # Agent 2: Fixed Price Agent (Price = 2.0)
    # Competitor has HIGHER quality (2.0 vs 1.0) and same cost.
    fixed_config = {'fixed_price': 2.0}
    agent_fixed = FixedPriceAgent(env.action_space, config=fixed_config)

    n_train_episodes = 200 # Minimal iterations as requested
    
    # 3. Training Loop
    for i in range(n_train_episodes):
        run_episode(env, agent_rl, agent_fixed, label="train", train_mode=True)

    # 4. Evaluation
    eval_df = run_episode(env, agent_rl, agent_fixed, label="eval", train_mode=False)
    
    avg_price = eval_df['p1'].mean()
    total_profit = eval_df['r1'].sum()
    
    return avg_price, total_profit

def main():
    decays_to_test = [0.9, 0.95, 0.99, 0.995]
    results = []
    
    print(f"{'Decay':<10} | {'Avg Price':<10} | {'Total Profit':<12}")
    print("-" * 40)
    
    for decay in decays_to_test:
        avg_price, profit = run_experiment(decay)
        results.append((decay, avg_price, profit))
        print(f"{decay:<10} | {avg_price:<10.2f} | {profit:<12.2f}")
        
    # Find best
    best_decay = max(results, key=lambda x: x[2])
    print("\nBest Performance:")
    print(f"Decay: {best_decay[0]}, Profit: {best_decay[2]:.2f}, Price: {best_decay[1]:.2f}")

if __name__ == "__main__":
    main()
