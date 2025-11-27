import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from market_audit.market_core import DuopolyEnv
from market_audit.agent_zoo import RLAgent, FixedPriceAgent
from market_audit.utils import run_episode

def main():
    # 1. Initialize Environment
    env_config = {
        'market_mode': 'static', # Simple static market for clear learning signal
        'max_steps': 100,
        'production_costs': [1.0, 1.0],
        'quality': [2.0, 2.0]
    }
    env = DuopolyEnv(config=env_config)

    # 2. Initialize Agents
    # Agent 1: RL Agent
    rl_config = {
        'hidden_dim': 64,
        'lr': 1e-3,
        'epsilon': 1.0,
        'epsilon_decay': 0.995, # Slower decay for better exploration
        'epsilon_min': 0.05,
        'memory_size': 10000,
        'batch_size': 32
    }
    agent_rl = RLAgent(env.action_space, config=rl_config)

    # Agent 2: Fixed Price Agent (Price = 2.0)
    fixed_config = {'fixed_price': 2.0}
    agent_fixed = FixedPriceAgent(env.action_space, config=fixed_config)

    print("Starting Training (RL vs Fixed)...")
    n_train_episodes = 1000
    
    # 3. Training Loop
    for i in range(n_train_episodes):
        # Run episode in train mode
        df = run_episode(env, agent_rl, agent_fixed, label="train", train_mode=True)
        
        total_reward = df['r1'].sum()
        if (i+1) % 100 == 0:
            print(f"Episode {i+1}/{n_train_episodes} - RL Total Reward: {total_reward:.2f} - Epsilon: {agent_rl.epsilon:.2f}")

    print("\nTraining Complete. Running Evaluation Episode...")

    # 4. Evaluation Episode
    eval_df = run_episode(env, agent_rl, agent_fixed, label="eval", train_mode=False)

    # 5. Output Results
    print("\nEvaluation Episode Results (First 20 steps):")
    print(eval_df[['step', 'p1', 'p2', 'r1', 'r2']].head(20))
    
    avg_price_rl = eval_df['p1'].mean()
    avg_price_fixed = eval_df['p2'].mean()
    total_profit_rl = eval_df['r1'].sum()
    
    print(f"\nSummary:")
    print(f"RL Agent Avg Price: {avg_price_rl:.2f}")
    print(f"Fixed Agent Price: {avg_price_fixed:.2f}")
    print(f"RL Agent Total Profit: {total_profit_rl:.2f}")

    # Save to CSV for user inspection
    output_file = "rl_vs_fixed_results.csv"
    eval_df.to_csv(output_file, index=False)
    print(f"\nFull episode data saved to {output_file}")

if __name__ == "__main__":
    main()
