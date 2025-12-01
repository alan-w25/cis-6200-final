import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize

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

def plot_generalized_convergence(df_history, env_config, title_suffix="", agent_1_label="Agent 1", agent_2_label="Agent 2", plot_br1=True, plot_br2=True):
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
             label=agent_1_label, color='blue', linewidth=1.5, alpha=0.8)
    #Agent 1 cost 
    plt.plot(df_history['step'], df_history['c1'], 
             label=f'{agent_1_label} Cost', color='blue', linestyle='--', linewidth=1.5, alpha=0.4)
    if plot_br1:
        plt.plot(df_history['step'], br_p1, 
                 label=f'{agent_1_label} BR (Target)', color='blue', linestyle='--', linewidth=1.5, alpha=0.4)
    
    # Agent 2
    plt.plot(df_history['step'], df_history['p2'], 
             label= agent_2_label, color='red', linewidth=1.5, alpha=0.6)
    #Agent 2 cost 
    plt.plot(df_history['step'], df_history['c2'], 
             label=f'{agent_2_label} Cost', color='red', linestyle='--', linewidth=1.5, alpha=0.4)
    if plot_br2:
        plt.plot(df_history['step'], br_p2, 
                 label=f'{agent_2_label} BR (Target)', color='red', linestyle='--', linewidth=1.5, alpha=0.4)
    

    plt.title(f'Agent Convergence Analysis {title_suffix}')
    plt.xlabel('Step')
    plt.ylabel('Price')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    plt.show()

def plot_market_shares(df_history, env_config, title_suffix="", agent_1_label="Agent 1", agent_2_label="Agent 2"):
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
    
    plt.plot(df_history['step'], s1, label=f'{agent_1_label} Share', color='blue', linewidth=1.5)
    plt.plot(df_history['step'], s2, label=f'{agent_2_label} Share', color='red', linewidth=1.5)
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

def calculate_instantaneous_benchmarks(demand_shock, costs, env_config):
    """
    Calculates NE and Collusive prices for a specific state (demand shock + costs).
    """
    q1, q2 = env_config['quality']
    mu = env_config['price_sensitivity']
    
    # Effective Quality = Base Quality + Shock
    # In AR drift mode, shock is additive to utility
    eff_q1 = q1 + demand_shock
    eff_q2 = q2 + demand_shock
    
    c1, c2 = costs
    
    # --- Helper: Profit Function ---
    def get_profits(p):
        p1, p2 = p
        exp_u1 = np.exp(eff_q1 - mu * p1)
        exp_u2 = np.exp(eff_q2 - mu * p2)
        denom = 1.0 + exp_u1 + exp_u2
        
        s1 = exp_u1 / denom
        s2 = exp_u2 / denom
        
        pi1 = (p1 - c1) * s1
        pi2 = (p2 - c2) * s2
        return pi1, pi2

    # --- 1. Nash Equilibrium (FOCs) ---
    def focs(p):
        p1, p2 = p
        exp_u1 = np.exp(eff_q1 - mu * p1)
        exp_u2 = np.exp(eff_q2 - mu * p2)
        denom = 1.0 + exp_u1 + exp_u2
        
        s1 = exp_u1 / denom
        s2 = exp_u2 / denom
        
        # FOC: 1 - mu * (p - c) * (1 - s) = 0
        foc1 = 1 - mu * (p1 - c1) * (1 - s1)
        foc2 = 1 - mu * (p2 - c2) * (1 - s2)
        return [foc1, foc2]

    # Solve NE
    ne_prices = fsolve(focs, x0=[c1+1.0, c2+1.0])
    
    # --- 2. Collusive Optimum (Joint Profit Max) ---
    def neg_joint_profit(p):
        pi1, pi2 = get_profits(p)
        return -(pi1 + pi2)
    
    # Solve Collusive
    # Bounds to keep optimization stable
    bounds = [(c1, 20), (c2, 20)]
    res = minimize(neg_joint_profit, x0=ne_prices, bounds=bounds)
    collusive_prices = res.x
    
    return ne_prices, collusive_prices

def plot_price_gap(df_history, env_config, title_suffix=""):
    """
    Plots the gap between Agents' prices and their respective Nash Equilibrium prices.
    Gap = Price_Agent - Price_NE
    """
    gaps_p1 = []
    gaps_p2 = []
    
    for _, row in df_history.iterrows():
        shock = row['demand_shock']
        costs = [row['c1'], row['c2']]
        
        ne_p, _ = calculate_instantaneous_benchmarks(shock, costs, env_config)
        
        gaps_p1.append(row['p1'] - ne_p[0])
        gaps_p2.append(row['p2'] - ne_p[1])
        
    plt.figure(figsize=(14, 7))
    
    # Agent 1 Gap
    plt.plot(df_history['step'], gaps_p1, label='Agent 1 Gap (P1 - NE1)', color='blue', linewidth=1.5, alpha=0.8)
    
    # Agent 2 Gap
    plt.plot(df_history['step'], gaps_p2, label='Agent 2 Gap (P2 - NE2)', color='red', linewidth=1.5, alpha=0.8)
    
    # Zero Line (Nash Equilibrium)
    plt.axhline(0, color='black', linestyle='--', alpha=0.5, label='Nash Equilibrium')
    
    plt.title(f'Price Gap Analysis (Deviation from NE) {title_suffix}')
    plt.xlabel('Step')
    plt.ylabel('Price Gap ($)')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Add Moving Averages
    if len(df_history) > 50:
        ma_gap1 = pd.Series(gaps_p1).rolling(window=20).mean()
        ma_gap2 = pd.Series(gaps_p2).rolling(window=20).mean()
        plt.plot(df_history['step'], ma_gap1, color='blue', linewidth=2, alpha=0.4, label='Gap 1 (MA-20)')
        plt.plot(df_history['step'], ma_gap2, color='red', linewidth=2, alpha=0.4, label='Gap 2 (MA-20)')
        
    plt.show()
