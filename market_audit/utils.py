import pandas as pd

def run_episode(env, agent1, agent2, label):
    state, _ = env.reset()
    done = False
    history = []
    
    step_count = 0
    while not done:
        a1 = agent1.act(state)
        a2 = agent2.act(state)
        
        next_state, rewards, done, _, info = env.step([a1, a2])
        
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
            'c2': info.get('costs', [0, 0])[1]
        })
        
        state = next_state
        step_count += 1
        
    return pd.DataFrame(history)
