"""
Sprint 4 Visualization Generation

Creates publication-quality visualizations for the manuscript:
1. Regime overlay plots (price trajectories with regime shading)
2. Brier Score time series
3. Violation rate comparisons
4. Nash gap distributions
5. Convergence lag analysis

All figures saved as high-resolution PNG and PDF for manuscript inclusion.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import os
from scipy import stats


def set_publication_style():
    """Configure matplotlib for publication-quality figures."""
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 11
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 14
    plt.rcParams['lines.linewidth'] = 1.5


def plot_regime_overlay(episode_data, title, output_file, max_steps=2000):
    """
    Create regime overlay plot: price trajectory with regime-dependent background shading.

    Args:
        episode_data: Dict with 'p1', 'nash_p1', 'jpm_p1', 'regime', 'step'
        title: Plot title
        output_file: Output filename (without extension)
        max_steps: Maximum steps to plot (for readability)
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Extract data
    steps = episode_data['step'][:max_steps]
    p1 = np.array(episode_data['p1'][:max_steps])
    nash_p1 = np.array(episode_data['nash_p1'][:max_steps])
    jpm_p1 = np.array(episode_data['jpm_p1'][:max_steps])
    regimes = np.array(episode_data['regime'][:max_steps])

    # Regime background shading
    regime_colors = {
        0: '#FFCCCC',  # Recession - light red
        1: '#FFFFFF',  # Normal - white
        2: '#CCFFCC'   # Boom - light green
    }

    regime_names = {0: 'Recession', 1: 'Normal', 2: 'Boom'}

    # Plot regime shading
    current_regime = regimes[0]
    regime_start = 0

    for i in range(1, len(regimes)):
        if regimes[i] != current_regime or i == len(regimes) - 1:
            ax.axvspan(regime_start, i, alpha=0.3, color=regime_colors[current_regime])
            regime_start = i
            current_regime = regimes[i]

    # Plot price trajectories
    ax.plot(steps, p1, 'b-', linewidth=2, label='Observed Price', alpha=0.8)
    ax.plot(steps, nash_p1, 'k--', linewidth=1.5, label='Nash Equilibrium', alpha=0.7)
    ax.plot(steps, jpm_p1, 'r:', linewidth=1.5, label='Joint Profit Max', alpha=0.7)

    # Labels and formatting
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Price ($)')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add regime legend
    recession_patch = mpatches.Patch(color=regime_colors[0], alpha=0.3, label='Recession')
    normal_patch = mpatches.Patch(color=regime_colors[1], alpha=0.3, label='Normal')
    boom_patch = mpatches.Patch(color=regime_colors[2], alpha=0.3, label='Boom')

    # Second legend for regimes
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.legend(handles=[recession_patch, normal_patch, boom_patch],
               loc='lower right', title='Regime')

    plt.tight_layout()

    # Save in multiple formats
    plt.savefig(f'{output_file}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_file}.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}.png, {output_file}.pdf")


def plot_brier_score_timeseries(episode_data, title, output_file, max_steps=2000):
    """
    Create Brier Score time series plot with regime overlay.
    """
    fig, ax = plt.subplots(figsize=(12, 5))

    # Extract data
    steps = episode_data['step'][:max_steps]
    brier_scores = episode_data.get('brier_1', [])[:max_steps]
    regimes = np.array(episode_data['regime'][:max_steps])

    # Clean Brier scores (remove NaN)
    brier_array = np.array(brier_scores)
    valid_mask = ~np.isnan(brier_array)

    if np.sum(valid_mask) == 0:
        print(f"Warning: No valid Brier scores for {title}")
        return

    valid_steps = np.array(steps)[valid_mask]
    valid_brier = brier_array[valid_mask]
    valid_regimes = regimes[valid_mask]

    # Regime background shading
    regime_colors = {0: '#FFCCCC', 1: '#FFFFFF', 2: '#CCFFCC'}

    current_regime = valid_regimes[0]
    regime_start = valid_steps[0]

    for i in range(1, len(valid_regimes)):
        if valid_regimes[i] != current_regime or i == len(valid_regimes) - 1:
            ax.axvspan(regime_start, valid_steps[i], alpha=0.3, color=regime_colors[current_regime])
            regime_start = valid_steps[i]
            current_regime = valid_regimes[i]

    # Plot Brier scores
    ax.plot(valid_steps, valid_brier, 'purple', linewidth=1.5, alpha=0.8)

    # Add smoothed trend line
    if len(valid_brier) > 100:
        window = 50
        smoothed = np.convolve(valid_brier, np.ones(window)/window, mode='valid')
        smoothed_steps = valid_steps[window-1:]
        ax.plot(smoothed_steps, smoothed, 'darkviolet', linewidth=2.5,
                label=f'Smoothed (window={window})', alpha=0.9)

    # Labels
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Brier Score')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_file}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_file}.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}.png, {output_file}.pdf")


def plot_violation_rate_comparison(results_a, results_b, conformal_quantile, output_file):
    """
    Create bar chart comparing violation rates between NSR and RL agents.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Calculate violation rates for NSR (Experiment A)
    nsr_violation_rates = []
    for run in results_a:
        p1 = np.array(run['episode_data']['p1'])
        nash_p1 = np.array(run['episode_data']['nash_p1'])
        violations = np.abs(p1 - nash_p1) > conformal_quantile
        nsr_violation_rates.append(np.mean(violations))

    # Calculate violation rates for RL (Experiment B)
    rl_violation_rates = []
    for run in results_b:
        p1 = np.array(run['episode_data']['p1'])
        nash_p1 = np.array(run['episode_data']['nash_p1'])
        violations = np.abs(p1 - nash_p1) > conformal_quantile
        rl_violation_rates.append(np.mean(violations))

    # Bar chart
    x = np.arange(2)
    means = [np.mean(nsr_violation_rates), np.mean(rl_violation_rates)]
    stds = [np.std(nsr_violation_rates), np.std(rl_violation_rates)]

    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7,
                   color=['steelblue', 'coral'], edgecolor='black', linewidth=1.5)

    # Add horizontal line for α=0.05 threshold
    ax.axhline(y=0.05, color='red', linestyle='--', linewidth=2,
               label='Competitive Threshold (α=0.05)', alpha=0.8)

    # Labels
    ax.set_ylabel('Violation Rate')
    ax.set_title('Conformal Violation Rates: NSR vs RL')
    ax.set_xticks(x)
    ax.set_xticklabels(['NSR\n(Competitive)', 'RL\n(Collusive)'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.01,
                f'{mean:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f'{output_file}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_file}.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}.png, {output_file}.pdf")


def plot_nash_gap_distributions(results_a, results_b, results_c, output_file):
    """
    Create violin plots comparing Nash gap distributions across experiments.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Collect Nash gaps from all runs
    nsr_gaps = []
    for run in results_a:
        p1 = np.array(run['episode_data']['p1'])
        nash_p1 = np.array(run['episode_data']['nash_p1'])
        nsr_gaps.extend(p1 - nash_p1)

    rl_gaps = []
    for run in results_b:
        p1 = np.array(run['episode_data']['p1'])
        nash_p1 = np.array(run['episode_data']['nash_p1'])
        rl_gaps.extend(p1 - nash_p1)

    rl_vs_nsr_gaps = []
    for run in results_c:
        p2_rl = np.array(run['episode_data']['p2_rl'])
        nash_p2 = np.array(run['episode_data']['nash_p2'])
        rl_vs_nsr_gaps.extend(p2_rl - nash_p2)

    # Sample for visualization (use every 10th point to reduce density)
    nsr_sample = np.array(nsr_gaps)[::10]
    rl_sample = np.array(rl_gaps)[::10]
    rl_vs_nsr_sample = np.array(rl_vs_nsr_gaps)[::10]

    # Violin plot
    data = [nsr_sample, rl_sample, rl_vs_nsr_sample]
    positions = [1, 2, 3]

    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)

    # Color the violins
    colors = ['steelblue', 'coral', 'gold']
    for pc, color in zip(parts['bodies'], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.7)

    # Add horizontal line at y=0 (Nash equilibrium)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5,
               label='Nash Equilibrium', alpha=0.7)

    # Labels
    ax.set_ylabel('Nash Gap ($)')
    ax.set_title('Nash Gap Distributions Across Experiments')
    ax.set_xticks(positions)
    ax.set_xticklabels(['NSR vs NSR\n(Competitive)',
                        'RL vs RL\n(Collusive)',
                        'RL vs NSR\n(Intervention)'])
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Add mean values as text
    means = [np.mean(d) for d in data]
    for pos, mean in zip(positions, means):
        ax.text(pos, ax.get_ylim()[1] * 0.9, f'μ={mean:.2f}',
                ha='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    plt.savefig(f'{output_file}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_file}.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}.png, {output_file}.pdf")


def plot_convergence_lag_analysis(results_a, output_file):
    """
    Analyze and plot convergence lags after regime transitions for NSR agents.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    all_lags = []

    for run in results_a:
        regimes = np.array(run['episode_data']['regime'])
        nash_gaps = np.array(run['episode_data']['p1']) - np.array(run['episode_data']['nash_p1'])

        # Find regime transitions
        transitions = []
        for i in range(1, len(regimes)):
            if regimes[i] != regimes[i-1]:
                transitions.append(i)

        # Calculate convergence lag for each transition
        baseline_threshold = np.std(nash_gaps)

        for trans_idx in transitions[:20]:  # First 20 transitions
            if trans_idx + 50 < len(nash_gaps):
                window = nash_gaps[trans_idx:trans_idx+50]
                abs_window = np.abs(window)

                below_threshold = np.where(abs_window < baseline_threshold)[0]
                if len(below_threshold) > 0:
                    lag = below_threshold[0]
                    all_lags.append(lag)

    if len(all_lags) == 0:
        print("Warning: No convergence lags calculated")
        return

    # Histogram of lags
    ax1.hist(all_lags, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(x=np.mean(all_lags), color='red', linestyle='--',
                linewidth=2, label=f'Mean = {np.mean(all_lags):.1f}')
    ax1.axvline(x=np.median(all_lags), color='orange', linestyle='--',
                linewidth=2, label=f'Median = {np.median(all_lags):.1f}')
    ax1.set_xlabel('Convergence Lag (steps)')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Distribution of Convergence Lags')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_lags = np.sort(all_lags)
    cumulative = np.arange(1, len(sorted_lags) + 1) / len(sorted_lags)

    ax2.plot(sorted_lags, cumulative, 'steelblue', linewidth=2)
    ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='50th percentile')
    ax2.axhline(y=0.9, color='orange', linestyle='--', alpha=0.7, label='90th percentile')
    ax2.set_xlabel('Convergence Lag (steps)')
    ax2.set_ylabel('Cumulative Probability')
    ax2.set_title('Cumulative Distribution of Convergence Lags')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_file}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_file}.pdf', bbox_inches='tight')
    plt.close()

    print(f"Saved: {output_file}.png, {output_file}.pdf")
    print(f"  Convergence lag statistics: mean={np.mean(all_lags):.1f}, median={np.median(all_lags):.1f}, std={np.std(all_lags):.1f}")


def generate_all_visualizations(data_dir, output_dir):
    """
    Generate all Sprint 4 visualizations.
    """
    print("=" * 70)
    print("SPRINT 4 VISUALIZATION GENERATION")
    print("=" * 70)
    print()

    # Set publication style
    set_publication_style()

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Load experimental results
    print("Loading experimental data...")
    with open(f'{data_dir}/experiment_a_nsr_baseline.pkl', 'rb') as f:
        results_a = pickle.load(f)

    with open(f'{data_dir}/experiment_b_rl_collusion.pkl', 'rb') as f:
        results_b = pickle.load(f)

    with open(f'{data_dir}/experiment_c_intervention.pkl', 'rb') as f:
        results_c = pickle.load(f)

    print(f"Loaded: {len(results_a)} NSR runs, {len(results_b)} RL runs, {len(results_c)} Intervention runs")
    print()

    # 1. Regime overlay plots (one for each experiment type)
    print("Generating regime overlay plots...")

    # NSR vs NSR
    plot_regime_overlay(
        results_a[0]['episode_data'],
        'NSR vs NSR: Competitive Pricing with Regime Dynamics',
        f'{output_dir}/fig1_nsr_regime_overlay',
        max_steps=2000
    )

    # RL vs RL
    plot_regime_overlay(
        results_b[0]['episode_data'],
        'RL vs RL: Collusive Pricing with Regime Dynamics',
        f'{output_dir}/fig2_rl_regime_overlay',
        max_steps=2000
    )

    # NSR vs RL
    # Need to adjust for NSR vs RL (different data structure)
    intervention_data = {
        'step': results_c[0]['episode_data']['step'],
        'p1': results_c[0]['episode_data']['p2_rl'],  # RL prices
        'nash_p1': results_c[0]['episode_data']['nash_p2'],
        'jpm_p1': results_c[0]['episode_data']['jpm_p2'],
        'regime': results_c[0]['episode_data']['regime']
    }
    plot_regime_overlay(
        intervention_data,
        'NSR vs RL: Strategic Brittleness in Intervention',
        f'{output_dir}/fig3_intervention_regime_overlay',
        max_steps=2000
    )

    # 2. Brier Score time series (RL experiments only)
    print("\nGenerating Brier Score time series...")
    plot_brier_score_timeseries(
        results_b[0]['episode_data'],
        'RL Agent Brier Score Evolution (Internal Audit)',
        f'{output_dir}/fig4_brier_score_timeseries',
        max_steps=2000
    )

    # 3. Violation rate comparison
    print("\nGenerating violation rate comparison...")
    conformal_quantile = 1.85  # From Experiment A
    plot_violation_rate_comparison(
        results_a, results_b, conformal_quantile,
        f'{output_dir}/fig5_violation_rate_comparison'
    )

    # 4. Nash gap distributions
    print("\nGenerating Nash gap distributions...")
    plot_nash_gap_distributions(
        results_a, results_b, results_c,
        f'{output_dir}/fig6_nash_gap_distributions'
    )

    # 5. Convergence lag analysis
    print("\nGenerating convergence lag analysis...")
    plot_convergence_lag_analysis(
        results_a,
        f'{output_dir}/fig7_convergence_lag_analysis'
    )

    print()
    print("=" * 70)
    print("VISUALIZATION GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nAll figures saved to: {output_dir}/")
    print("\nGenerated figures:")
    print("  1. fig1_nsr_regime_overlay.png/pdf - NSR competitive pricing")
    print("  2. fig2_rl_regime_overlay.png/pdf - RL collusive pricing")
    print("  3. fig3_intervention_regime_overlay.png/pdf - NSR vs RL intervention")
    print("  4. fig4_brier_score_timeseries.png/pdf - Brier Score evolution")
    print("  5. fig5_violation_rate_comparison.png/pdf - Conformal violation rates")
    print("  6. fig6_nash_gap_distributions.png/pdf - Nash gap distributions")
    print("  7. fig7_convergence_lag_analysis.png/pdf - Convergence lag analysis")
    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python generate_sprint4_visualizations.py <data_dir>")
        print("Example: python generate_sprint4_visualizations.py results/sprint3_20251213_190554")
        sys.exit(1)

    data_dir = sys.argv[1]
    output_dir = f"{data_dir}/figures"

    if not os.path.exists(data_dir):
        print(f"Error: Data directory '{data_dir}' not found")
        sys.exit(1)

    generate_all_visualizations(data_dir, output_dir)

    print("Sprint 4 Visualization Generation Complete!")
    print("\nNext steps:")
    print("  1. Review generated figures")
    print("  2. Add figures to manuscript")
    print("  3. Complete Discussion section")
