import numpy as np
import pandas as pd
from .market_core import DuopolyEnv
from .agent_zoo import NSRAgent, RLAgent, ConstrainedRLAgent
from .auditing_engine import ConformalAuditor

class MatchupRunner:
    def __init__(self, config=None):
        self.config = config if config else {}
        self.env_config = self.config.get('env_config', {})
        self.n_episodes = self.config.get('n_episodes', 10)
        self.max_steps = self.config.get('max_steps', 100)
        
    def run_matchup(self, agent1_cls, agent2_cls, scenario_name="Matchup"):
        print(f"Starting Scenario: {scenario_name}")
        
        env = DuopolyEnv(config=self.env_config)
        
        # Initialize Agents
        # We need to pass action space and config
        # Assuming agents take same config for now, or we can separate
        agent1 = agent1_cls(env.action_space, config=self.config.get('agent_config', {}))
        agent2 = agent2_cls(env.action_space, config=self.config.get('agent_config', {}))
        
        results = []
        
        for episode in range(self.n_episodes):
            state, _ = env.reset()
            done = False
            truncated = False
            
            episode_data = {
                'episode': episode,
                'p1': [],
                'p2': [],
                'profit1': [],
                'profit2': [],
                'demand_shock': []
            }
            
            while not (done or truncated):
                # Agents act
                # Note: Agents need observation. 
                # Our env returns full state [p1, p2, shock, c1, c2]
                # Agents might need to see only relevant parts or full state.
                # RLAgent handles vector. NSRAgent handles vector?
                # NSRAgent in our impl takes observation but ignores it mostly (Regret Matching).
                # Let's pass the full state.
                
                action1 = agent1.act(state)
                action2 = agent2.act(state)
                
                next_state, profits, done, truncated, info = env.step([action1, action2])
                
                # Store data
                episode_data['p1'].append(action1)
                episode_data['p2'].append(action2)
                episode_data['profit1'].append(profits[0])
                episode_data['profit2'].append(profits[1])
                episode_data['demand_shock'].append(info['demand_shock'])
                
                # Update Agents
                # Transition for agent 1: state, action1, profit1, next_state, done
                agent1.update((state, action1, profits[0], next_state, done))
                agent2.update((state, action2, profits[1], next_state, done))
                
                state = next_state
                
            # Aggregate episode metrics
            avg_p1 = np.mean(episode_data['p1'])
            avg_p2 = np.mean(episode_data['p2'])
            total_profit1 = np.sum(episode_data['profit1'])
            total_profit2 = np.sum(episode_data['profit2'])
            
            results.append({
                'scenario': scenario_name,
                'episode': episode,
                'avg_p1': avg_p1,
                'avg_p2': avg_p2,
                'total_profit1': total_profit1,
                'total_profit2': total_profit2
            })
            
        return pd.DataFrame(results)

    def run_all_scenarios(self):
        all_results = []
        
        # Scenario A: NSR vs NSR
        df_nsr = self.run_matchup(NSRAgent, NSRAgent, "NSR_vs_NSR")
        all_results.append(df_nsr)
        
        # Scenario B: RL vs RL
        df_rl = self.run_matchup(RLAgent, RLAgent, "RL_vs_RL")
        all_results.append(df_rl)
        
        # Scenario C: Constrained vs Constrained
        df_crl = self.run_matchup(ConstrainedRLAgent, ConstrainedRLAgent, "CRL_vs_CRL")
        all_results.append(df_crl)
        
        # Scenario D: NSR vs RL
        df_mix = self.run_matchup(NSRAgent, RLAgent, "NSR_vs_RL")
        all_results.append(df_mix)
        
        return pd.concat(all_results, ignore_index=True)

if __name__ == "__main__":
    # Simple test run
    runner = MatchupRunner(config={'n_episodes': 2, 'max_steps': 50})
    results = runner.run_all_scenarios()
    print(results)
