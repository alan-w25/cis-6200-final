import numpy as np
import pandas as pd
from market_audit.market_core import DuopolyEnv
from market_audit.agent_zoo import RLAgent, FixedPriceAgent, RandomAgent
from market_audit.utils import run_episode

def run_experiment(epsilon_decay, opponent_type, opponent_config):
    # 1. Initialize Environment (User Provided Config)
    env_config = {
        'market_mode': 'static',
        'production_costs': [1.0, 2.0],
        'quality': [2.0, 2.0],
        'price_sensitivity': 0.8,
        'max_price': 6.0,
        'cost_std': 0.0,
        'max_steps': 10000 # Long episodes
    }
    env = DuopolyEnv(config=env_config)

    # 2. Initialize RL Agent (User Provided Config Base)
    rl_config = {
        'state_dim': 5,
        'hidden_dim': 128,
        'n_bins': 100,
        'lr': 0.0005,
        'gamma': 0.99,
        'epsilon': 1.0,
        'epsilon_decay': epsilon_decay, # Variable to optimize
        'epsilon_min': 0.01,
        'batch_size': 64,
        'memory_size': 10000,
    }
    agent_rl = RLAgent(env.action_space, config=rl_config)

    # 3. Initialize Opponent
    if opponent_type == 'fixed':
        agent_opponent = FixedPriceAgent(env.action_space, config=opponent_config)
    elif opponent_type == 'random':
        agent_opponent = RandomAgent(env.action_space, config=opponent_config)
    else:
        raise ValueError("Unknown opponent type")

    # 4. Training Loop
    # Since max_steps is 10000, one episode is very long.
    # We will run fewer episodes but they are long.
    # Let's run 10 training episodes (Total 100k steps)
    n_train_episodes = 10 
    
    for i in range(n_train_episodes):
        run_episode(env, agent_rl, agent_opponent, label="train", train_mode=True)

    # 5. Evaluation
    eval_df = run_episode(env, agent_rl, agent_opponent, label="eval", train_mode=False)
    
    avg_price = eval_df['p1'].mean()
    total_profit = eval_df['r1'].sum()
    
    return avg_price, total_profit

def main():
    decays_to_test = [0.90, 0.95, 0.99, 0.995, 0.999]
    
    fixed_config = {'fixed_price': 2.5}
    random_config = {'max_price': 6.0}
    
    print("Optimization Results")
    print("====================")
    
    # --- Scenario 1: RL vs Fixed ---
    print("\nScenario 1: RL vs Fixed Price (2.5)")
    print(f"{'Decay':<10} | {'Avg Price':<10} | {'Total Profit':<12}")
    print("-" * 40)
    
    results_fixed = []
    for decay in decays_to_test:
        avg_price, profit = run_experiment(decay, 'fixed', fixed_config)
        results_fixed.append((decay, avg_price, profit))
        print(f"{decay:<10} | {avg_price:<10.2f} | {profit:<12.2f}")
        
    best_fixed = max(results_fixed, key=lambda x: x[2])
    print(f"Best Decay vs Fixed: {best_fixed[0]} (Profit: {best_fixed[2]:.2f})")

    # --- Scenario 2: RL vs Random ---
    print("\nScenario 2: RL vs Random Price")
    print(f"{'Decay':<10} | {'Avg Price':<10} | {'Total Profit':<12}")
    print("-" * 40)
    
    results_random = []
    for decay in decays_to_test:
        avg_price, profit = run_experiment(decay, 'random', random_config)
        results_random.append((decay, avg_price, profit))
        print(f"{decay:<10} | {avg_price:<10.2f} | {profit:<12.2f}")
        
    best_random = max(results_random, key=lambda x: x[2])
    print(f"Best Decay vs Random: {best_random[0]} (Profit: {best_random[2]:.2f})")

if __name__ == "__main__":
    main()
