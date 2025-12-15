"""
Sprint 3 Experimental Campaign

Executes comprehensive experiments to test core hypotheses:
- Experiment A: NSR vs NSR (Rational Baseline)
- Experiment B: RL vs RL (Collusive Stress Test)
- Experiment C: NSR vs RL (Intervention Study)

Each experiment runs 5 independent trials with different random seeds.
Results are saved for downstream analysis and visualization.
"""

import numpy as np
import pandas as pd
import pickle
import os
from datetime import datetime
from market_audit.market_core import DuopolyEnv
from market_audit.agent_zoo import NSRAgent, RLAgent
from market_audit.auditing_engine import ConformalAuditor
from market_audit.simulation_controller import MatchupRunner


def run_experiment_a_nsr_baseline(n_runs=5, n_steps=20000, output_dir='results/sprint3'):
    """
    Experiment A: Rational Baseline (NSR vs NSR)

    Hypothesis: NSR agents maintain competitive Nash prices and adapt quickly
    to regime changes without losing calibration.

    Metrics:
    - Average price gap from Nash
    - Calibration error
    - Convergence lag after regime switches
    - Conformal safe zone establishment
    """
    print("=" * 70)
    print("EXPERIMENT A: Rational Baseline (NSR vs NSR)")
    print("=" * 70)
    print(f"Configuration: {n_runs} runs × {n_steps} steps")
    print()

    os.makedirs(output_dir, exist_ok=True)

    results = []

    for run_idx in range(n_runs):
        print(f"Run {run_idx + 1}/{n_runs}...")

        # Environment configuration with regime switching
        env_config = {
            'market_mode': 'regime_switch',
            'max_steps': n_steps,
            'production_costs': [1.0, 2.0],
            'seed': 42 + run_idx  # Different seed for each run
        }

        env = DuopolyEnv(config=env_config)

        # NSR agent configuration
        agent_config = {
            'n_bins': 100,
            'quality': 2.0,
            'price_sensitivity': 0.8
        }

        agent1 = NSRAgent(env.action_space, config={**agent_config, 'cost': 1.0})
        agent2 = NSRAgent(env.action_space, config={**agent_config, 'cost': 2.0})

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
            'regime': [],
            'demand_shock': []
        }

        state, _ = env.reset(seed=42 + run_idx)

        for step in range(n_steps):
            # Agents act
            action1 = agent1.act(state)
            action2 = agent2.act(state)

            # Get benchmarks
            benchmarks = env.get_current_nash_equilibrium()
            nash_prices = benchmarks['nash_prices']
            jpm_prices = benchmarks['jpm_prices']
            regime = benchmarks['regime']

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
            episode_data['regime'].append(regime)
            episode_data['demand_shock'].append(info['demand_shock'])

            # Update agents
            agent1.update((state, action1, profits[0], next_state, done))
            agent2.update((state, action2, profits[1], next_state, done))

            if done or truncated:
                state, _ = env.reset()
            else:
                state = next_state

        # Calculate metrics
        p1_array = np.array(episode_data['p1'])
        nash_p1_array = np.array(episode_data['nash_p1'])
        jpm_p1_array = np.array(episode_data['jpm_p1'])

        nash_gap_1 = p1_array - nash_p1_array
        jpm_gap_1 = p1_array - jpm_p1_array

        run_results = {
            'run_id': run_idx,
            'mean_price_1': np.mean(p1_array),
            'mean_nash_gap_1': np.mean(nash_gap_1),
            'std_nash_gap_1': np.std(nash_gap_1),
            'mean_jpm_gap_1': np.mean(jpm_gap_1),
            'total_profit_1': np.sum(episode_data['profit1']),
            'total_profit_2': np.sum(episode_data['profit2']),
            'episode_data': episode_data
        }

        results.append(run_results)

        print(f"  Mean price: ${np.mean(p1_array):.2f}")
        print(f"  Mean Nash gap: ${np.mean(nash_gap_1):.3f}")
        print(f"  Total profit (Firm 1): ${np.sum(episode_data['profit1']):.2f}")
        print()

    # Save results
    output_file = os.path.join(output_dir, 'experiment_a_nsr_baseline.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to: {output_file}")
    print()

    return results


def run_experiment_b_rl_collusion(n_runs=5, n_steps=20000, output_dir='results/sprint3'):
    """
    Experiment B: Collusive Stress Test (RL vs RL)

    Hypothesis: RL agents exhibit "Downward Rigidity" - when market shifts from
    Boom to Recession, they maintain supra-competitive prices (Rocket and Feather).

    Metrics:
    - Price premium over Nash
    - Joint profit levels
    - Brier Score spikes
    - Violation rate of conformal bounds
    - Hysteresis asymmetry (upward vs downward adjustment speed)
    """
    print("=" * 70)
    print("EXPERIMENT B: Collusive Stress Test (RL vs RL)")
    print("=" * 70)
    print(f"Configuration: {n_runs} runs × {n_steps} steps")
    print()

    os.makedirs(output_dir, exist_ok=True)

    results = []

    for run_idx in range(n_runs):
        print(f"Run {run_idx + 1}/{n_runs}...")

        # Environment configuration
        env_config = {
            'market_mode': 'regime_switch',
            'max_steps': n_steps,
            'production_costs': [1.0, 2.0],
            'seed': 100 + run_idx
        }

        env = DuopolyEnv(config=env_config)

        # RL agent configuration with Brier logging
        agent_config = {
            'state_dim': 5,
            'n_bins': 100,
            'lr': 1e-3,
            'gamma': 0.99,
            'epsilon': 0.1,
            'enable_brier_logging': True,
            'brier_window': 500
        }

        agent1 = RLAgent(env.action_space, config=agent_config)
        agent2 = RLAgent(env.action_space, config=agent_config)

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
            'regime': [],
            'regime_transition': [],  # Track regime transitions
            'brier_1': [],
            'brier_2': [],
            'demand_shock': []
        }

        state, _ = env.reset(seed=100 + run_idx)
        prev_regime = env.current_regime

        for step in range(n_steps):
            # Agents act
            action1 = agent1.act(state)
            action2 = agent2.act(state)

            # Get benchmarks
            benchmarks = env.get_current_nash_equilibrium()
            nash_prices = benchmarks['nash_prices']
            jpm_prices = benchmarks['jpm_prices']
            regime = benchmarks['regime']

            # Detect regime transition
            regime_transition = (regime != prev_regime)
            prev_regime = regime

            # Execute step
            next_state, profits, done, truncated, info = env.step([action1, action2])

            # Get Brier scores
            brier_1 = agent1.get_brier_scores()[-1] if agent1.get_brier_scores() else np.nan
            brier_2 = agent2.get_brier_scores()[-1] if agent2.get_brier_scores() else np.nan

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
            episode_data['regime'].append(regime)
            episode_data['regime_transition'].append(regime_transition)
            episode_data['brier_1'].append(brier_1)
            episode_data['brier_2'].append(brier_2)
            episode_data['demand_shock'].append(info['demand_shock'])

            # Update agents
            agent1.update((state, action1, profits[0], next_state, done))
            agent2.update((state, action2, profits[1], next_state, done))

            if done or truncated:
                state, _ = env.reset()
            else:
                state = next_state

        # Calculate metrics
        p1_array = np.array(episode_data['p1'])
        nash_p1_array = np.array(episode_data['nash_p1'])
        jpm_p1_array = np.array(episode_data['jpm_p1'])

        nash_gap_1 = p1_array - nash_p1_array
        jpm_gap_1 = jpm_p1_array - p1_array  # Distance from JPM (negative if above Nash)

        run_results = {
            'run_id': run_idx,
            'mean_price_1': np.mean(p1_array),
            'mean_nash_gap_1': np.mean(nash_gap_1),
            'std_nash_gap_1': np.std(nash_gap_1),
            'mean_jpm_gap_1': np.mean(jpm_gap_1),
            'total_profit_1': np.sum(episode_data['profit1']),
            'total_profit_2': np.sum(episode_data['profit2']),
            'joint_profit': np.sum(episode_data['profit1']) + np.sum(episode_data['profit2']),
            'episode_data': episode_data
        }

        results.append(run_results)

        print(f"  Mean price: ${np.mean(p1_array):.2f}")
        print(f"  Mean Nash gap: ${np.mean(nash_gap_1):.3f}")
        print(f"  Joint profit: ${run_results['joint_profit']:.2f}")
        print()

    # Save results
    output_file = os.path.join(output_dir, 'experiment_b_rl_collusion.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to: {output_file}")
    print()

    return results


def run_experiment_c_intervention(n_runs=5, n_steps=20000, output_dir='results/sprint3'):
    """
    Experiment C: Intervention Study (NSR vs RL)

    Hypothesis: NSR agent forces RL agent to adapt faster to regime changes
    and reduces collusive drift ("competitive contagion").

    Metrics:
    - RL price trajectories when facing NSR vs another RL
    - Relative price gaps
    - Convergence speeds
    - Conformal violation rates
    - Market share dynamics
    """
    print("=" * 70)
    print("EXPERIMENT C: Intervention Study (NSR vs RL)")
    print("=" * 70)
    print(f"Configuration: {n_runs} runs × {n_steps} steps")
    print()

    os.makedirs(output_dir, exist_ok=True)

    results = []

    for run_idx in range(n_runs):
        print(f"Run {run_idx + 1}/{n_runs}...")

        # Environment configuration
        env_config = {
            'market_mode': 'regime_switch',
            'max_steps': n_steps,
            'production_costs': [1.0, 2.0],
            'seed': 200 + run_idx
        }

        env = DuopolyEnv(config=env_config)

        # NSR agent (Firm 1)
        nsr_config = {
            'n_bins': 100,
            'quality': 2.0,
            'price_sensitivity': 0.8,
            'cost': 1.0
        }

        # RL agent (Firm 2) with Brier logging
        rl_config = {
            'state_dim': 5,
            'n_bins': 100,
            'lr': 1e-3,
            'gamma': 0.99,
            'epsilon': 0.1,
            'enable_brier_logging': True,
            'brier_window': 500
        }

        agent1 = NSRAgent(env.action_space, config=nsr_config)
        agent2 = RLAgent(env.action_space, config=rl_config)

        # Data collection
        episode_data = {
            'step': [],
            'p1_nsr': [],
            'p2_rl': [],
            'profit1': [],
            'profit2': [],
            'nash_p1': [],
            'nash_p2': [],
            'jpm_p1': [],
            'jpm_p2': [],
            'regime': [],
            'brier_2': [],
            'demand_shock': []
        }

        state, _ = env.reset(seed=200 + run_idx)

        for step in range(n_steps):
            # Agents act
            action1 = agent1.act(state)
            action2 = agent2.act(state)

            # Get benchmarks
            benchmarks = env.get_current_nash_equilibrium()
            nash_prices = benchmarks['nash_prices']
            jpm_prices = benchmarks['jpm_prices']
            regime = benchmarks['regime']

            # Execute step
            next_state, profits, done, truncated, info = env.step([action1, action2])

            # Get Brier score for RL agent
            brier_2 = agent2.get_brier_scores()[-1] if agent2.get_brier_scores() else np.nan

            # Store data
            episode_data['step'].append(step)
            episode_data['p1_nsr'].append(action1)
            episode_data['p2_rl'].append(action2)
            episode_data['profit1'].append(profits[0])
            episode_data['profit2'].append(profits[1])
            episode_data['nash_p1'].append(nash_prices[0])
            episode_data['nash_p2'].append(nash_prices[1])
            episode_data['jpm_p1'].append(jpm_prices[0])
            episode_data['jpm_p2'].append(jpm_prices[1])
            episode_data['regime'].append(regime)
            episode_data['brier_2'].append(brier_2)
            episode_data['demand_shock'].append(info['demand_shock'])

            # Update agents
            agent1.update((state, action1, profits[0], next_state, done))
            agent2.update((state, action2, profits[1], next_state, done))

            if done or truncated:
                state, _ = env.reset()
            else:
                state = next_state

        # Calculate metrics
        p1_nsr = np.array(episode_data['p1_nsr'])
        p2_rl = np.array(episode_data['p2_rl'])
        nash_p2_array = np.array(episode_data['nash_p2'])

        nash_gap_nsr = p1_nsr - np.array(episode_data['nash_p1'])
        nash_gap_rl = p2_rl - nash_p2_array

        run_results = {
            'run_id': run_idx,
            'mean_price_nsr': np.mean(p1_nsr),
            'mean_price_rl': np.mean(p2_rl),
            'mean_nash_gap_nsr': np.mean(nash_gap_nsr),
            'mean_nash_gap_rl': np.mean(nash_gap_rl),
            'std_nash_gap_rl': np.std(nash_gap_rl),
            'total_profit_nsr': np.sum(episode_data['profit1']),
            'total_profit_rl': np.sum(episode_data['profit2']),
            'episode_data': episode_data
        }

        results.append(run_results)

        print(f"  NSR mean price: ${np.mean(p1_nsr):.2f}, Nash gap: ${np.mean(nash_gap_nsr):.3f}")
        print(f"  RL mean price: ${np.mean(p2_rl):.2f}, Nash gap: ${np.mean(nash_gap_rl):.3f}")
        print(f"  Total profit (NSR): ${np.sum(episode_data['profit1']):.2f}")
        print(f"  Total profit (RL): ${np.sum(episode_data['profit2']):.2f}")
        print()

    # Save results
    output_file = os.path.join(output_dir, 'experiment_c_intervention.pkl')
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)

    print(f"Results saved to: {output_file}")
    print()

    return results


def run_all_experiments():
    """
    Execute all Sprint 3 experiments in sequence.
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "SPRINT 3 EXPERIMENTAL CAMPAIGN")
    print("=" * 70 + "\n")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'results/sprint3_{timestamp}'

    print(f"Output directory: {output_dir}\n")

    # Experiment A: Rational Baseline
    results_a = run_experiment_a_nsr_baseline(n_runs=5, n_steps=20000, output_dir=output_dir)

    # Experiment B: Collusive Stress Test
    results_b = run_experiment_b_rl_collusion(n_runs=5, n_steps=20000, output_dir=output_dir)

    # Experiment C: Intervention Study
    results_c = run_experiment_c_intervention(n_runs=5, n_steps=20000, output_dir=output_dir)

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {output_dir}")
    print("\nNext steps:")
    print("  1. Run analysis script to generate summary statistics")
    print("  2. Create visualizations (regime overlay plots)")
    print("  3. Update manuscript with experimental findings")
    print()

    return {
        'experiment_a': results_a,
        'experiment_b': results_b,
        'experiment_c': results_c,
        'output_dir': output_dir
    }


if __name__ == "__main__":
    all_results = run_all_experiments()

    print("Sprint 3 Experimental Campaign Complete!")
    print("\nSummary:")
    print(f"  - Experiment A (NSR vs NSR): {len(all_results['experiment_a'])} runs")
    print(f"  - Experiment B (RL vs RL): {len(all_results['experiment_b'])} runs")
    print(f"  - Experiment C (NSR vs RL): {len(all_results['experiment_c'])} runs")
    print(f"  - Total simulation steps: {5 * 20000 * 3:,}")
