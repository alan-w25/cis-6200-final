import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_episode(env, agent1, agent2, label, train_mode=True):
    """
    Runs a single episode of the market simulation.
    
    Args:
        env: The market environment.
        agent1: The first agent.
        agent2: The second agent.
        label: Label for the episode data.
        train_mode: If True, agents explore and update. If False, agents are in eval mode (no exploration, no updates).
    """
    state, _ = env.reset()
    done = False
    history = []
    
    # Set agents to appropriate mode if they support it
    if hasattr(agent1, 'train') and hasattr(agent1, 'eval'):
        if train_mode: agent1.train()
        else: agent1.eval()
        
    if hasattr(agent2, 'train') and hasattr(agent2, 'eval'):
        if train_mode: agent2.train()
        else: agent2.eval()
    
    step_count = 0
    while not done:
        a1 = agent1.act(state)
        a2 = agent2.act(state)
        
        next_state, rewards, done, _, info = env.step([a1, a2])
        
        # Only update agents if in training mode
        if train_mode:
            agent1.update((state, a1, rewards[0], next_state, done))
            agent2.update((state, a2, rewards[1], next_state, done))
        
        history.append({
            'step': step_count,
            'label': label,
            'p1': a1,
            'p2': a2,
            'r1': rewards[0],
            'r2': rewards[1],
            'demand_shock': info.get('demand_shock', 0),
            'c1': info.get('costs', [0, 0])[0],
            'c2': info.get('costs', [0, 0])[1],
            'train_mode': train_mode
        })
        
        state = next_state
        step_count += 1
        
    return pd.DataFrame(history)
    
def get_best_response(p2_values, env_config):
    """
    Calculates the best response p1 for a given set of p2 values (scalar or array).
    """
    # 1. Setup Grid
    p1_range = np.linspace(0, env_config['max_price'], 1000)
    c1 = env_config['production_costs'][0]
    q1, q2 = env_config['quality']
    mu = env_config['price_sensitivity']
    
    # 2. Vectorized Calculation
    # We need to broadcast p1_range against p2_values if p2 is an array
    # p1_range: [1000], p2_values: [Steps]
    # Result: [1000, Steps]
    
    p2_values = np.atleast_1d(p2_values)
    
    # Utilities
    # U1 [1000, 1]
    u1 = q1 - mu * p1_range[:, np.newaxis]
    # U2 [1, Steps]
    u2 = q2 - mu * p2_values[np.newaxis, :]
    
    # Shares
    exp_u1 = np.exp(u1)
    exp_u2 = np.exp(u2)
    # Denom [1000, Steps]
    denom = 1.0 + exp_u1 + exp_u2
    shares1 = exp_u1 / denom
    
    # Profits [1000, Steps]
    profits1 = (p1_range[:, np.newaxis] - c1) * shares1
    
    # Find max index for each step
    best_indices = np.argmax(profits1, axis=0)
    best_p1_array = p1_range[best_indices]
    
    return best_p1_array

def plot_generalized_convergence(df_history, env_config, title_suffix=""):
    """
    Plots the agent's price vs the instantaneous best response to the opponent.
    """
    # 1. Calculate Instantaneous Best Response (Target)
    target_p1 = get_best_response(df_history['p2'].values, env_config)
    print("Average best price:", np.mean(target_p1))
    
    # 2. Plot
    plt.figure(figsize=(14, 7))
    
    # Agent Price
    plt.plot(df_history['step'], df_history['p1'], 
             label='Agent Price', color='blue', linewidth=1.5, alpha=0.8)
    
    # Target Price (Best Response)
    plt.plot(df_history['step'], target_p1, 
             label='Instantaneous Best Response (Target)', 
             color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    
    # Opponent Price (for context)
    plt.plot(df_history['step'], df_history['p2'], 
             label='Opponent Price', color='gray', linestyle=':', alpha=0.3)
    
    plt.title(f'Agent Convergence Analysis {title_suffix}')
    plt.xlabel('Step')
    plt.ylabel('Price ($p_1$)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Add Moving Average for clearer trend if noisy
    if len(df_history) > 50:
        ma_p1 = df_history['p1'].rolling(window=20).mean()
        plt.plot(df_history['step'], ma_p1, color='blue', linewidth=2, label='NSR Price (MA-20)')
    
    plt.show()
