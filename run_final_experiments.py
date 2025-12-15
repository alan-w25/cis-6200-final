"""
Final Experimental Campaign - Corrected Configuration

Uses env_config.json to ensure consistent parameters across market modes:
- Static market (no dynamics)
- AR(1) dynamic market (demand shocks)
- Regime switching market (3-state Markov)

All markets share: quality=5.0, price_sensitivity=0.8, max_price=8.0, costs=[1.0,2.0]
Only difference: dynamics mechanism

Experiments:
1. Static: NSR vs NSR, RL vs RL, NSR vs RL (both as low-cost)
2. AR(1): NSR vs NSR, RL vs RL, NSR vs RL (both as low-cost)
3. Regime Switching: NSR vs NSR, RL vs RL, NSR vs RL (both as low-cost)

Each experiment: 5 runs × 10,000 steps
"""

import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime
from market_audit.market_core import DuopolyEnv
from market_audit.agent_zoo import NSRAgent, RLAgent


def load_configs():
    """Load environment configurations from env_config.json"""
    config_path = 'market_audit/env_config.json'
    with open(config_path, 'r') as f:
        configs = json.load(f)

    # Add regime switching config with same base parameters
    configs['regime_switch_config'] = {
        "market_mode": "regime_switch",
        "production_costs": [1.0, 2.0],
        "quality": [5.0, 5.0],
        "price_sensitivity": 0.8,
        "max_price": 8.0,
        "cost_std": 0.0,  # Disable AR(1) cost dynamics
        "cost_phi": 0.0,
        "max_steps": 10000,
        # Regime parameters
        "recession_quality": [3.0, 3.0],      # Lower than normal
        "recession_sensitivity": 1.2,          # Higher sensitivity (price conscious)
        "normal_quality": [5.0, 5.0],          # Same as base
        "normal_sensitivity": 0.8,             # Same as base
        "boom_quality": [7.0, 7.0],            # Higher than normal
        "boom_sensitivity": 0.5,               # Lower sensitivity (flush with cash)
        "transition_matrix": [
            [0.98, 0.02, 0.00],
            [0.01, 0.98, 0.01],
            [0.00, 0.02, 0.98]
        ]
    }

    return configs


def run_matchup(env_config, agent1_cls, agent2_cls, agent_config,
                scenario_name, n_runs=5, output_dir='results'):
    """
    Run a single matchup experiment.

    Args:
        env_config: Environment configuration dict
        agent1_cls: Agent 1 class (NSRAgent or RLAgent)
        agent2_cls: Agent 2 class
        agent_config: Dict with 'agent1_config' and 'agent2_config'
        scenario_name: Experiment name
        n_runs: Number of independent runs
        output_dir: Output directory
    """
    print(f"\n{'='*70}")
    print(f"Running: {scenario_name}")
    print(f"{'='*70}")
    print(f"Configuration: {n_runs} runs × {env_config['max_steps']} steps")
    print(f"Market mode: {env_config['market_mode']}")
    print()

    results = []

    for run_idx in range(n_runs):
        print(f"  Run {run_idx + 1}/{n_runs}...", end='')

        # Create environment
        env = DuopolyEnv(config=env_config)

        # Initialize agents
        agent1 = agent1_cls(env.action_space, config=agent_config['agent1_config'])
        agent2 = agent2_cls(env.action_space, config=agent_config['agent2_config'])

        # Data collection
        episode_data = {
            'step': [],
            'p1': [],
            'p2': [],
            'profit1': [],
            'profit2': [],
            'nash_p1': [],
            'nash_p2': [],
            'jpm_p1': [],
            'jpm_p2': [],
            'demand_shock': [],
            'regime': []
        }

        # Run episode
        state, _ = env.reset(seed=42 + run_idx)

        for step in range(env_config['max_steps']):
            # Agents act
            action1 = agent1.act(state)
            action2 = agent2.act(state)

            # Get benchmarks
            benchmarks = env.get_current_nash_equilibrium()
            nash_prices = benchmarks['nash_prices']
            jpm_prices = benchmarks['jpm_prices']
            regime = benchmarks.get('regime', -1)

            # Execute step
            next_state, profits, done, truncated, info = env.step([action1, action2])

            # Store data
            episode_data['step'].append(step)
            episode_data['p1'].append(action1)
            episode_data['p2'].append(action2)
            episode_data['profit1'].append(profits[0])
            episode_data['profit2'].append(profits[1])
            episode_data['nash_p1'].append(nash_prices[0])
            episode_data['nash_p2'].append(nash_prices[1])
            episode_data['jpm_p1'].append(jpm_prices[0])
            episode_data['jpm_p2'].append(jpm_prices[1])
            episode_data['demand_shock'].append(info.get('demand_shock', 0.0))
            episode_data['regime'].append(regime)

            # Update agents
            agent1.update((state, action1, profits[0], next_state, done))
            agent2.update((state, action2, profits[1], next_state, done))

            if done or truncated:
                state, _ = env.reset()
            else:
                state = next_state

        # Calculate metrics
        p1_array = np.array(episode_data['p1'])
        p2_array = np.array(episode_data['p2'])
        nash_p1_array = np.array(episode_data['nash_p1'])
        nash_p2_array = np.array(episode_data['nash_p2'])

        nash_gap_1 = p1_array - nash_p1_array
        nash_gap_2 = p2_array - nash_p2_array

        run_results = {
            'run_id': run_idx,
            'scenario': scenario_name,
            'market_mode': env_config['market_mode'],
            'mean_price_1': np.mean(p1_array),
            'mean_price_2': np.mean(p2_array),
            'mean_nash_gap_1': np.mean(nash_gap_1),
            'mean_nash_gap_2': np.mean(nash_gap_2),
            'std_nash_gap_1': np.std(nash_gap_1),
            'std_nash_gap_2': np.std(nash_gap_2),
            'total_profit_1': np.sum(episode_data['profit1']),
            'total_profit_2': np.sum(episode_data['profit2']),
            'episode_data': episode_data
        }

        results.append(run_results)

        print(f" Nash gap 1: ${np.mean(nash_gap_1):+.3f}, Nash gap 2: ${np.mean(nash_gap_2):+.3f}")

    return results


def run_all_experiments():
    """
    Run complete experimental campaign across all three market modes.
    """
    print("\n" + "="*70)
    print(" "*20 + "FINAL EXPERIMENTAL CAMPAIGN")
    print(" "*15 + "Using env_config.json Parameters")
    print("="*70 + "\n")

    # Load configurations
    configs = load_configs()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'results/final_experiments_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)

    print(f"Output directory: {output_dir}\n")

    # Print config summary
    print("Configuration Summary:")
    print(f"  Base parameters: quality=5.0, price_sensitivity=0.8, max_price=8.0")
    print(f"  Costs: [1.0, 2.0] (low-cost, high-cost)")
    print(f"  Steps per run: 10,000")
    print(f"  Runs per experiment: 5")
    print()

    all_results = {}

    # Agent configurations
    nsr_low_cost_config = {
        'n_bins': 100,
        'quality': 5.0,
        'price_sensitivity': 0.8,
        'cost': 1.0
    }

    nsr_high_cost_config = {
        'n_bins': 100,
        'quality': 5.0,
        'price_sensitivity': 0.8,
        'cost': 2.0
    }

    rl_low_cost_config = {
        'state_dim': 5,
        'n_bins': 100,
        'lr': 1e-3,
        'gamma': 0.99,
        'epsilon': 0.1
    }

    rl_high_cost_config = {
        'state_dim': 5,
        'n_bins': 100,
        'lr': 1e-3,
        'gamma': 0.99,
        'epsilon': 0.1
    }

    # ===================================================================
    # STATIC MARKET EXPERIMENTS
    # ===================================================================
    print("\n" + "="*70)
    print("STATIC MARKET EXPERIMENTS")
    print("="*70)

    # Static: NSR vs NSR
    all_results['static_nsr_vs_nsr'] = run_matchup(
        configs['static_config'],
        NSRAgent, NSRAgent,
        {'agent1_config': nsr_low_cost_config, 'agent2_config': nsr_high_cost_config},
        'Static_NSR_vs_NSR',
        n_runs=5,
        output_dir=output_dir
    )

    # Static: RL vs RL
    all_results['static_rl_vs_rl'] = run_matchup(
        configs['static_config'],
        RLAgent, RLAgent,
        {'agent1_config': rl_low_cost_config, 'agent2_config': rl_high_cost_config},
        'Static_RL_vs_RL',
        n_runs=5,
        output_dir=output_dir
    )

    # Static: NSR(low-cost) vs RL(high-cost)
    all_results['static_nsr_low_vs_rl_high'] = run_matchup(
        configs['static_config'],
        NSRAgent, RLAgent,
        {'agent1_config': nsr_low_cost_config, 'agent2_config': rl_high_cost_config},
        'Static_NSR(low)_vs_RL(high)',
        n_runs=5,
        output_dir=output_dir
    )

    # Static: RL(low-cost) vs NSR(high-cost)
    all_results['static_rl_low_vs_nsr_high'] = run_matchup(
        configs['static_config'],
        RLAgent, NSRAgent,
        {'agent1_config': rl_low_cost_config, 'agent2_config': nsr_high_cost_config},
        'Static_RL(low)_vs_NSR(high)',
        n_runs=5,
        output_dir=output_dir
    )

    # ===================================================================
    # AR(1) DYNAMIC MARKET EXPERIMENTS
    # ===================================================================
    print("\n" + "="*70)
    print("AR(1) DYNAMIC MARKET EXPERIMENTS")
    print("="*70)

    # AR(1): NSR vs NSR
    all_results['ar1_nsr_vs_nsr'] = run_matchup(
        configs['ar_1_config'],
        NSRAgent, NSRAgent,
        {'agent1_config': nsr_low_cost_config, 'agent2_config': nsr_high_cost_config},
        'AR1_NSR_vs_NSR',
        n_runs=5,
        output_dir=output_dir
    )

    # AR(1): RL vs RL
    all_results['ar1_rl_vs_rl'] = run_matchup(
        configs['ar_1_config'],
        RLAgent, RLAgent,
        {'agent1_config': rl_low_cost_config, 'agent2_config': rl_high_cost_config},
        'AR1_RL_vs_RL',
        n_runs=5,
        output_dir=output_dir
    )

    # AR(1): NSR(low-cost) vs RL(high-cost)
    all_results['ar1_nsr_low_vs_rl_high'] = run_matchup(
        configs['ar_1_config'],
        NSRAgent, RLAgent,
        {'agent1_config': nsr_low_cost_config, 'agent2_config': rl_high_cost_config},
        'AR1_NSR(low)_vs_RL(high)',
        n_runs=5,
        output_dir=output_dir
    )

    # AR(1): RL(low-cost) vs NSR(high-cost)
    all_results['ar1_rl_low_vs_nsr_high'] = run_matchup(
        configs['ar_1_config'],
        RLAgent, NSRAgent,
        {'agent1_config': rl_low_cost_config, 'agent2_config': nsr_high_cost_config},
        'AR1_RL(low)_vs_NSR(high)',
        n_runs=5,
        output_dir=output_dir
    )

    # ===================================================================
    # REGIME SWITCHING MARKET EXPERIMENTS
    # ===================================================================
    print("\n" + "="*70)
    print("REGIME SWITCHING MARKET EXPERIMENTS")
    print("="*70)

    # Regime: NSR vs NSR
    all_results['regime_nsr_vs_nsr'] = run_matchup(
        configs['regime_switch_config'],
        NSRAgent, NSRAgent,
        {'agent1_config': nsr_low_cost_config, 'agent2_config': nsr_high_cost_config},
        'Regime_NSR_vs_NSR',
        n_runs=5,
        output_dir=output_dir
    )

    # Regime: RL vs RL
    all_results['regime_rl_vs_rl'] = run_matchup(
        configs['regime_switch_config'],
        RLAgent, RLAgent,
        {'agent1_config': rl_low_cost_config, 'agent2_config': rl_high_cost_config},
        'Regime_RL_vs_RL',
        n_runs=5,
        output_dir=output_dir
    )

    # Regime: NSR(low-cost) vs RL(high-cost)
    all_results['regime_nsr_low_vs_rl_high'] = run_matchup(
        configs['regime_switch_config'],
        NSRAgent, RLAgent,
        {'agent1_config': nsr_low_cost_config, 'agent2_config': rl_high_cost_config},
        'Regime_NSR(low)_vs_RL(high)',
        n_runs=5,
        output_dir=output_dir
    )

    # Regime: RL(low-cost) vs NSR(high-cost)
    all_results['regime_rl_low_vs_nsr_high'] = run_matchup(
        configs['regime_switch_config'],
        RLAgent, NSRAgent,
        {'agent1_config': rl_low_cost_config, 'agent2_config': nsr_high_cost_config},
        'Regime_RL(low)_vs_NSR(high)',
        n_runs=5,
        output_dir=output_dir
    )

    # ===================================================================
    # SAVE RESULTS
    # ===================================================================
    print("\n" + "="*70)
    print("SAVING RESULTS")
    print("="*70)

    for exp_name, exp_results in all_results.items():
        output_file = os.path.join(output_dir, f'{exp_name}.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(exp_results, f)
        print(f"  Saved: {exp_name}.pkl")

    # Save summary
    summary = {
        'timestamp': timestamp,
        'total_experiments': len(all_results),
        'total_runs': sum(len(r) for r in all_results.values()),
        'total_steps': sum(len(r) * 10000 for r in all_results.values()),
        'configs_used': configs,
        'experiment_names': list(all_results.keys())
    }

    summary_file = os.path.join(output_dir, 'experiment_summary.pkl')
    with open(summary_file, 'wb') as f:
        pickle.dump(summary, f)

    print(f"\n  Saved: experiment_summary.pkl")
    print(f"\nAll results saved to: {output_dir}")

    # Print summary statistics
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    print(f"Total experiments: {len(all_results)}")
    print(f"Total runs: {summary['total_runs']}")
    print(f"Total steps: {summary['total_steps']:,}")
    print()

    return all_results, output_dir


if __name__ == "__main__":
    all_results, output_dir = run_all_experiments()

    print("\n" + "="*70)
    print("FINAL EXPERIMENTAL CAMPAIGN COMPLETE")
    print("="*70)
    print(f"\nResults directory: {output_dir}")
    print("\nNext steps:")
    print("  1. Run analysis: python analyze_final_experiments.py <output_dir>")
    print("  2. Generate visualizations")
    print("  3. Compare across market modes")
    print()
