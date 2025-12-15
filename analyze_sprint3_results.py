"""
Sprint 3 Results Analysis

Analyzes experimental results from Sprint 3 to test core hypotheses:
- NSR competitive behavior and Nash convergence
- RL collusive tendency and downward rigidity
- Competitive contagion effect (NSR intervention)

Generates summary statistics, hypothesis tests, and regime-specific analyses.
"""

import numpy as np
import pandas as pd
import pickle
import os
from scipy import stats


def analyze_experiment_a(results_file):
    """
    Analyze Experiment A: NSR vs NSR Rational Baseline

    Tests:
    1. Nash convergence: Mean Nash gap close to 0
    2. Calibration stability across regimes
    3. Regime adaptation speed
    """
    print("=" * 70)
    print("EXPERIMENT A ANALYSIS: Rational Baseline (NSR vs NSR)")
    print("=" * 70)

    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    # Aggregate metrics across runs
    mean_nash_gaps = [r['mean_nash_gap_1'] for r in results]
    std_nash_gaps = [r['std_nash_gap_1'] for r in results]
    total_profits_1 = [r['total_profit_1'] for r in results]
    total_profits_2 = [r['total_profit_2'] for r in results]

    print(f"\nNumber of runs: {len(results)}")
    print(f"\nNash Gap Statistics (across all runs):")
    print(f"  Mean Nash gap: ${np.mean(mean_nash_gaps):.4f} ± ${np.std(mean_nash_gaps):.4f}")
    print(f"  Median Nash gap: ${np.median(mean_nash_gaps):.4f}")
    print(f"  Within-run std: ${np.mean(std_nash_gaps):.4f}")

    # Test if mean Nash gap is significantly different from 0
    t_stat, p_value = stats.ttest_1samp(mean_nash_gaps, 0)
    print(f"\n  Hypothesis Test (H0: Nash gap = 0):")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value: {p_value:.4f}")
    if p_value > 0.05:
        print(f"    ✓ Cannot reject H0: NSR agents converge to Nash equilibrium")
    else:
        print(f"    ✗ Reject H0: Significant deviation from Nash")

    print(f"\nProfit Statistics:")
    print(f"  Mean total profit (Firm 1): ${np.mean(total_profits_1):.2f} ± ${np.std(total_profits_1):.2f}")
    print(f"  Mean total profit (Firm 2): ${np.mean(total_profits_2):.2f} ± ${np.std(total_profits_2):.2f}")

    # Regime-specific analysis (use first run as example)
    first_run = results[0]['episode_data']
    regimes = np.array(first_run['regime'])
    nash_gaps = np.array(first_run['p1']) - np.array(first_run['nash_p1'])

    print(f"\nRegime-Specific Nash Gaps (Run 0):")
    for regime_idx in [0, 1, 2]:
        regime_name = ['Recession', 'Normal', 'Boom'][regime_idx]
        mask = regimes == regime_idx
        if np.sum(mask) > 0:
            regime_gaps = nash_gaps[mask]
            print(f"  {regime_name}: ${np.mean(regime_gaps):.4f} ± ${np.std(regime_gaps):.4f} ({np.sum(mask)} steps)")

    # Regime transition analysis
    print(f"\nRegime Transition Dynamics:")
    transitions = []
    for i in range(1, len(regimes)):
        if regimes[i] != regimes[i-1]:
            transitions.append(i)

    if len(transitions) > 0:
        print(f"  Number of transitions: {len(transitions)}")

        # Calculate convergence lag (steps until Nash gap returns to baseline)
        baseline_threshold = np.std(nash_gaps)
        convergence_lags = []

        for trans_idx in transitions[:10]:  # Analyze first 10 transitions
            # Look at Nash gap in 50-step window after transition
            window = nash_gaps[trans_idx:min(trans_idx+50, len(nash_gaps))]
            abs_window = np.abs(window)

            # Find when gap returns below threshold
            below_threshold = np.where(abs_window < baseline_threshold)[0]
            if len(below_threshold) > 0:
                lag = below_threshold[0]
                convergence_lags.append(lag)

        if len(convergence_lags) > 0:
            print(f"  Mean convergence lag: {np.mean(convergence_lags):.1f} steps")
            print(f"  Median convergence lag: {np.median(convergence_lags):.1f} steps")

    # Calibration for conformal prediction
    print(f"\nConformal Calibration (for External Audit):")
    all_nash_gaps = []
    for r in results:
        p1 = np.array(r['episode_data']['p1'])
        nash_p1 = np.array(r['episode_data']['nash_p1'])
        all_nash_gaps.extend(np.abs(p1 - nash_p1))

    q95 = np.quantile(all_nash_gaps, 0.95)
    q99 = np.quantile(all_nash_gaps, 0.99)

    print(f"  95th percentile non-conformity score: ${q95:.4f}")
    print(f"  99th percentile non-conformity score: ${q99:.4f}")
    print(f"  Safe zone radius (α=0.05): ±${q95:.4f}")

    print()
    return {
        'mean_nash_gap': np.mean(mean_nash_gaps),
        'std_nash_gap': np.std(mean_nash_gaps),
        'p_value_nash_convergence': p_value,
        'conformal_quantile_95': q95,
        'conformal_quantile_99': q99
    }


def analyze_experiment_b(results_file, conformal_quantile=None):
    """
    Analyze Experiment B: RL vs RL Collusive Stress Test

    Tests:
    1. Price premium over Nash
    2. Downward rigidity (Rocket and Feather)
    3. Violation rate relative to conformal safe zone
    4. Hysteresis at regime transitions
    """
    print("=" * 70)
    print("EXPERIMENT B ANALYSIS: Collusive Stress Test (RL vs RL)")
    print("=" * 70)

    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    # Aggregate metrics
    mean_nash_gaps = [r['mean_nash_gap_1'] for r in results]
    std_nash_gaps = [r['std_nash_gap_1'] for r in results]
    joint_profits = [r['joint_profit'] for r in results]

    print(f"\nNumber of runs: {len(results)}")
    print(f"\nNash Gap Statistics (Price Premium):")
    print(f"  Mean Nash gap: ${np.mean(mean_nash_gaps):.4f} ± ${np.std(mean_nash_gaps):.4f}")
    print(f"  Median Nash gap: ${np.median(mean_nash_gaps):.4f}")
    print(f"  Within-run std: ${np.mean(std_nash_gaps):.4f}")

    # Test if RL prices are significantly above Nash
    t_stat, p_value = stats.ttest_1samp(mean_nash_gaps, 0)
    print(f"\n  Hypothesis Test (H0: Nash gap = 0, H1: gap > 0):")
    print(f"    t-statistic: {t_stat:.4f}")
    print(f"    p-value (one-sided): {p_value/2:.4f}")
    if t_stat > 0 and p_value/2 < 0.05:
        print(f"    ✓ Reject H0: RL agents price significantly above Nash (collusive)")
    else:
        print(f"    ✗ Cannot reject H0: No significant collusion detected")

    print(f"\nJoint Profit Statistics:")
    print(f"  Mean joint profit: ${np.mean(joint_profits):.2f} ± ${np.std(joint_profits):.2f}")
    print(f"  Median joint profit: ${np.median(joint_profits):.2f}")

    # Violation rate analysis (if conformal quantile provided)
    if conformal_quantile is not None:
        print(f"\nConformal Violation Rate (Safe Zone: ±${conformal_quantile:.4f}):")
        all_violation_rates = []

        for r in results:
            p1 = np.array(r['episode_data']['p1'])
            nash_p1 = np.array(r['episode_data']['nash_p1'])
            violations = np.abs(p1 - nash_p1) > conformal_quantile
            violation_rate = np.mean(violations)
            all_violation_rates.append(violation_rate)

        print(f"  Mean violation rate: {np.mean(all_violation_rates):.4f} ({np.mean(all_violation_rates)*100:.1f}%)")
        print(f"  Std violation rate: {np.std(all_violation_rates):.4f}")
        print(f"  Expected for competitive agents: ≤ 0.05 (5%)")

        if np.mean(all_violation_rates) > 0.10:
            print(f"  ✓ High violation rate indicates collusive behavior")
        else:
            print(f"  ✗ Low violation rate - competitive behavior")

    # Downward Rigidity Analysis (Rocket and Feather)
    print(f"\nDownward Rigidity Analysis (Rocket and Feather):")

    first_run = results[0]['episode_data']
    regimes = np.array(first_run['regime'])
    nash_gaps = np.array(first_run['p1']) - np.array(first_run['nash_p1'])

    # Identify regime transitions
    upward_transitions = []  # Recession→Normal, Normal→Boom
    downward_transitions = []  # Boom→Normal, Normal→Recession

    for i in range(1, len(regimes)):
        if regimes[i] > regimes[i-1]:
            upward_transitions.append(i)
        elif regimes[i] < regimes[i-1]:
            downward_transitions.append(i)

    # Calculate adjustment speeds
    def calculate_adjustment_speed(transitions, nash_gaps, direction='up'):
        """Calculate how quickly Nash gap changes after transition"""
        adjustment_speeds = []

        for trans_idx in transitions[:5]:  # First 5 transitions
            if trans_idx + 20 < len(nash_gaps):
                gap_before = nash_gaps[trans_idx - 1]
                gap_after_window = nash_gaps[trans_idx:trans_idx+20]

                # Calculate rate of change
                changes = np.diff(gap_after_window)
                mean_change = np.mean(changes)
                adjustment_speeds.append(mean_change)

        return adjustment_speeds

    if len(upward_transitions) > 0 and len(downward_transitions) > 0:
        upward_speeds = calculate_adjustment_speed(upward_transitions, nash_gaps, 'up')
        downward_speeds = calculate_adjustment_speed(downward_transitions, nash_gaps, 'down')

        print(f"  Upward transitions (Recession→Boom): {len(upward_transitions)}")
        if len(upward_speeds) > 0:
            print(f"    Mean adjustment rate: ${np.mean(upward_speeds):.4f}/step")

        print(f"  Downward transitions (Boom→Recession): {len(downward_transitions)}")
        if len(downward_speeds) > 0:
            print(f"    Mean adjustment rate: ${np.mean(downward_speeds):.4f}/step")

        # Test for asymmetry (Rocket and Feather pattern)
        if len(upward_speeds) > 0 and len(downward_speeds) > 0:
            t_stat, p_value = stats.ttest_ind(upward_speeds, downward_speeds)
            print(f"\n  Hysteresis Asymmetry Test:")
            print(f"    t-statistic: {t_stat:.4f}, p-value: {p_value:.4f}")
            if p_value < 0.05:
                print(f"    ✓ Significant asymmetry detected (Rocket and Feather)")
            else:
                print(f"    No significant asymmetry")

    # Brier Score analysis
    print(f"\nBrier Score Statistics (Calibration):")
    first_run_brier = np.array(first_run['brier_1'])
    brier_clean = first_run_brier[~np.isnan(first_run_brier)]

    if len(brier_clean) > 0:
        print(f"  Mean Brier Score: {np.mean(brier_clean):.4f}")
        print(f"  Std Brier Score: {np.std(brier_clean):.4f}")
        print(f"  Brier scores logged: {len(brier_clean)}/{len(first_run_brier)}")

    print()
    return {
        'mean_nash_gap': np.mean(mean_nash_gaps),
        'std_nash_gap': np.std(mean_nash_gaps),
        'p_value_collusion': p_value/2,
        'mean_joint_profit': np.mean(joint_profits),
        'mean_violation_rate': np.mean(all_violation_rates) if conformal_quantile else None
    }


def analyze_experiment_c(results_file, rl_vs_rl_results):
    """
    Analyze Experiment C: NSR vs RL Intervention Study

    Tests:
    1. Competitive contagion: RL Nash gap lower when facing NSR
    2. Convergence speed comparison
    3. Market share dynamics
    """
    print("=" * 70)
    print("EXPERIMENT C ANALYSIS: Intervention Study (NSR vs RL)")
    print("=" * 70)

    with open(results_file, 'rb') as f:
        results = pickle.load(f)

    # NSR vs RL results
    mean_nash_gap_rl = [r['mean_nash_gap_rl'] for r in results]
    mean_nash_gap_nsr = [r['mean_nash_gap_nsr'] for r in results]
    total_profit_rl = [r['total_profit_rl'] for r in results]
    total_profit_nsr = [r['total_profit_nsr'] for r in results]

    print(f"\nNumber of runs: {len(results)}")

    print(f"\nNSR Agent (Firm 1):")
    print(f"  Mean Nash gap: ${np.mean(mean_nash_gap_nsr):.4f} ± ${np.std(mean_nash_gap_nsr):.4f}")
    print(f"  Mean total profit: ${np.mean(total_profit_nsr):.2f}")

    print(f"\nRL Agent (Firm 2) when facing NSR:")
    print(f"  Mean Nash gap: ${np.mean(mean_nash_gap_rl):.4f} ± ${np.std(mean_nash_gap_rl):.4f}")
    print(f"  Mean total profit: ${np.mean(total_profit_rl):.2f}")

    # Compare with RL vs RL results
    if rl_vs_rl_results:
        rl_vs_rl_gaps = [r['mean_nash_gap_1'] for r in rl_vs_rl_results]

        print(f"\nRL Agent (Firm 1) when facing another RL:")
        print(f"  Mean Nash gap: ${np.mean(rl_vs_rl_gaps):.4f} ± ${np.std(rl_vs_rl_gaps):.4f}")

        # Competitive Contagion Test
        print(f"\nCompetitive Contagion Effect:")
        reduction = np.mean(rl_vs_rl_gaps) - np.mean(mean_nash_gap_rl)
        reduction_pct = (reduction / np.mean(rl_vs_rl_gaps)) * 100

        print(f"  RL Nash gap reduction when facing NSR: ${reduction:.4f} ({reduction_pct:.1f}%)")

        # Statistical test
        t_stat, p_value = stats.ttest_ind(rl_vs_rl_gaps, mean_nash_gap_rl)
        print(f"\n  Hypothesis Test (H0: No contagion effect):")
        print(f"    t-statistic: {t_stat:.4f}")
        print(f"    p-value: {p_value:.4f}")

        if p_value < 0.05 and reduction > 0:
            print(f"    ✓ Significant competitive contagion detected")
            print(f"    NSR agent forces RL to price closer to Nash")
        else:
            print(f"    ✗ No significant contagion effect")

    # Profit distribution
    print(f"\nProfit Distribution:")
    profit_ratio = np.array(total_profit_nsr) / np.array(total_profit_rl)
    print(f"  NSR/RL profit ratio: {np.mean(profit_ratio):.3f} ± {np.std(profit_ratio):.3f}")

    if np.mean(profit_ratio) > 1.0:
        print(f"  ✓ NSR agent outperforms RL (competitive discipline advantage)")
    else:
        print(f"  RL agent outperforms NSR (potential exploitation)")

    print()
    return {
        'mean_nash_gap_rl_vs_nsr': np.mean(mean_nash_gap_rl),
        'mean_nash_gap_nsr': np.mean(mean_nash_gap_nsr),
        'contagion_reduction': reduction if rl_vs_rl_results else None,
        'contagion_p_value': p_value if rl_vs_rl_results else None,
        'profit_ratio_nsr_rl': np.mean(profit_ratio)
    }


def generate_summary_report(output_dir):
    """
    Generate comprehensive summary report for all experiments.
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "SPRINT 3 SUMMARY REPORT")
    print("=" * 70 + "\n")

    # Load all results
    exp_a_file = os.path.join(output_dir, 'experiment_a_nsr_baseline.pkl')
    exp_b_file = os.path.join(output_dir, 'experiment_b_rl_collusion.pkl')
    exp_c_file = os.path.join(output_dir, 'experiment_c_intervention.pkl')

    # Analyze Experiment A
    results_a = analyze_experiment_a(exp_a_file)

    # Analyze Experiment B (with conformal quantile from A)
    results_b = analyze_experiment_b(exp_b_file, conformal_quantile=results_a['conformal_quantile_95'])

    # Analyze Experiment C (with RL vs RL comparison)
    with open(exp_b_file, 'rb') as f:
        rl_vs_rl_data = pickle.load(f)
    results_c = analyze_experiment_c(exp_c_file, rl_vs_rl_data)

    # Summary table
    print("=" * 70)
    print("HYPOTHESIS TESTING SUMMARY")
    print("=" * 70)
    print()

    summary = {
        'Experiment A: NSR Nash Convergence': {
            'Hypothesis': 'NSR agents converge to Nash equilibrium',
            'Test': f"Mean gap = ${results_a['mean_nash_gap']:.4f}",
            'p-value': f"{results_a['p_value_nash_convergence']:.4f}",
            'Result': '✓ Supported' if results_a['p_value_nash_convergence'] > 0.05 else '✗ Rejected'
        },
        'Experiment B: RL Collusion': {
            'Hypothesis': 'RL agents price above Nash (collusive)',
            'Test': f"Mean gap = ${results_b['mean_nash_gap']:.4f}",
            'p-value': f"{results_b['p_value_collusion']:.4f}",
            'Result': '✓ Supported' if results_b['p_value_collusion'] < 0.05 and results_b['mean_nash_gap'] > 0 else '✗ Rejected'
        },
        'Experiment C: Competitive Contagion': {
            'Hypothesis': 'NSR forces RL closer to Nash',
            'Test': f"Reduction = ${results_c['contagion_reduction']:.4f}" if results_c['contagion_reduction'] else 'N/A',
            'p-value': f"{results_c['contagion_p_value']:.4f}" if results_c['contagion_p_value'] else 'N/A',
            'Result': '✓ Supported' if results_c['contagion_p_value'] and results_c['contagion_p_value'] < 0.05 else 'Pending'
        }
    }

    for exp_name, results in summary.items():
        print(f"{exp_name}:")
        print(f"  Hypothesis: {results['Hypothesis']}")
        print(f"  Test statistic: {results['Test']}")
        print(f"  p-value: {results['p-value']}")
        print(f"  Result: {results['Result']}")
        print()

    # Save summary
    summary_file = os.path.join(output_dir, 'summary_statistics.pkl')
    with open(summary_file, 'wb') as f:
        pickle.dump({
            'experiment_a': results_a,
            'experiment_b': results_b,
            'experiment_c': results_c,
            'summary_table': summary
        }, f)

    print(f"Summary statistics saved to: {summary_file}")
    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python analyze_sprint3_results.py <output_dir>")
        print("Example: python analyze_sprint3_results.py results/sprint3_20241213_120000")
        sys.exit(1)

    output_dir = sys.argv[1]

    if not os.path.exists(output_dir):
        print(f"Error: Directory '{output_dir}' not found")
        sys.exit(1)

    generate_summary_report(output_dir)

    print("=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print("\nNext steps:")
    print("  1. Review summary statistics")
    print("  2. Create visualizations (regime overlay plots)")
    print("  3. Update manuscript with experimental findings")
