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
    
def _calculate_single_agent_br(opponent_prices, my_cost, my_quality, opponent_quality, mu, max_price):
    p_range = np.linspace(0, max_price, 1000)
    opponent_prices = np.atleast_1d(opponent_prices)
    
    # Utilities
    # My Utility [1000, 1]
    u_own = my_quality - mu * p_range[:, np.newaxis]
    # Opponent Utility [1, Steps]
    u_opp = opponent_quality - mu * opponent_prices[np.newaxis, :]
    
    exp_u_own = np.exp(u_own)
    exp_u_opp = np.exp(u_opp)
    
    denom = 1.0 + exp_u_own + exp_u_opp
    shares = exp_u_own / denom
    
    profits = (p_range[:, np.newaxis] - my_cost) * shares
    
    best_indices = np.argmax(profits, axis=0)
    return p_range[best_indices]

def get_best_response(history, env_config):
    """
    Calculates the best response prices for both agents given the history.
    Returns (br_p1, br_p2).
    """
    c1, c2 = env_config['production_costs']
    q1, q2 = env_config['quality']
    mu = env_config['price_sensitivity']
    max_price = env_config['max_price']
    
    br_p1 = _calculate_single_agent_br(history['p2'].values, c1, q1, q2, mu, max_price)
    br_p2 = _calculate_single_agent_br(history['p1'].values, c2, q2, q1, mu, max_price)
    
    return br_p1, br_p2

def plot_generalized_convergence(df_history, env_config, title_suffix="", plot_br1=True, plot_br2=True):
    """
    Plots the agent's price vs the instantaneous best response to the opponent.
    """
    # 1. Calculate Instantaneous Best Response (Target)
    br_p1, br_p2 = get_best_response(df_history, env_config)
    
    if plot_br1:
        print("Average best price P1:", np.mean(br_p1))
    if plot_br2:
        print("Average best price P2:", np.mean(br_p2))
    
    # 2. Plot
    plt.figure(figsize=(14, 7))
    
    # Agent 1
    plt.plot(df_history['step'], df_history['p1'], 
             label='Agent 1 Price', color='blue', linewidth=1.5, alpha=0.8)
    if plot_br1:
        plt.plot(df_history['step'], br_p1, 
                 label='Agent 1 BR (Target)', color='blue', linestyle='--', linewidth=1.5, alpha=0.4)
    
    # Agent 2
    plt.plot(df_history['step'], df_history['p2'], 
             label='Agent 2 Price', color='red', linewidth=1.5, alpha=0.8)
    if plot_br2:
        plt.plot(df_history['step'], br_p2, 
                 label='Agent 2 BR (Target)', color='red', linestyle='--', linewidth=1.5, alpha=0.4)
    
    plt.title(f'Agent Convergence Analysis {title_suffix}')
    plt.xlabel('Step')
    plt.ylabel('Price')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.show()

def plot_market_shares(df_history, env_config, title_suffix=""):
    """
    Calculates and plots the market shares of Agent 1 and Agent 2 over the episode.
    """
    # 1. Extract Parameters
    q1, q2 = env_config['quality']
    mu = env_config['price_sensitivity']
    
    # 2. Calculate Utilities
    # U = Q - mu * P
    u1 = q1 - mu * df_history['p1'].values
    u2 = q2 - mu * df_history['p2'].values
    
    # 3. Calculate Shares (Logit)
    exp_u1 = np.exp(u1)
    exp_u2 = np.exp(u2)
    denom = 1.0 + exp_u1 + exp_u2
    
    s1 = exp_u1 / denom
    s2 = exp_u2 / denom
    s0 = 1.0 / denom # Outside option share
    
    # 4. Plot
    plt.figure(figsize=(14, 7))
    
    plt.plot(df_history['step'], s1, label='Agent 1 Share', color='blue', linewidth=1.5)
    plt.plot(df_history['step'], s2, label='Agent 2 Share', color='red', linewidth=1.5)
    plt.plot(df_history['step'], s0, label='Outside Option', color='gray', linestyle=':', alpha=0.5)
    
    plt.title(f'Market Share Evolution {title_suffix}')
    plt.xlabel('Step')
    plt.ylabel('Market Share')
    plt.ylim(0, 1.0)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Add Moving Average
    if len(df_history) > 50:
        ma_s1 = pd.Series(s1).rolling(window=20).mean()
        ma_s2 = pd.Series(s2).rolling(window=20).mean()
        plt.plot(df_history['step'], ma_s1, color='blue', linewidth=2, alpha=0.3)
        plt.plot(df_history['step'], ma_s2, color='red', linewidth=2, alpha=0.3)
        
    plt.show()
