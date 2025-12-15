"""
Quick Analysis of Corrected Final Experiments
Compares behavior across 3 market modes with proper env_config.json parameters
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

results_dir = Path('results/final_experiments_20251214_112028')

def load_experiment(exp_name):
    """Load experiment results"""
    with open(results_dir / f'{exp_name}.pkl', 'rb') as f:
        return pickle.load(f)

def compute_stats(exp_name):
    """Compute statistics for an experiment"""
    results = load_experiment(exp_name)

    nash_gaps_1 = [r['mean_nash_gap_1'] for r in results]
    nash_gaps_2 = [r['mean_nash_gap_2'] for r in results]

    return {
        'experiment': exp_name,
        'n_runs': len(results),
        'nash_gap_1_mean': np.mean(nash_gaps_1),
        'nash_gap_1_std': np.std(nash_gaps_1),
        'nash_gap_2_mean': np.mean(nash_gaps_2),
        'nash_gap_2_std': np.std(nash_gaps_2),
        'total_profit_1_mean': np.mean([r['total_profit_1'] for r in results]),
        'total_profit_2_mean': np.mean([r['total_profit_2'] for r in results])
    }

# All 12 experiments
experiments = [
    # Static
    'static_nsr_vs_nsr',
    'static_rl_vs_rl',
    'static_nsr_low_vs_rl_high',
    'static_rl_low_vs_nsr_high',
    # AR(1)
    'ar1_nsr_vs_nsr',
    'ar1_rl_vs_rl',
    'ar1_nsr_low_vs_rl_high',
    'ar1_rl_low_vs_nsr_high',
    # Regime Switching
    'regime_nsr_vs_nsr',
    'regime_rl_vs_rl',
    'regime_nsr_low_vs_rl_high',
    'regime_rl_low_vs_nsr_high'
]

print("\n" + "="*80)
print(" "*25 + "CORRECTED EXPERIMENTAL RESULTS")
print(" "*20 + "Using env_config.json Parameters")
print("="*80 + "\n")

print("Configuration:")
print("  Base: quality=5.0, price_sensitivity=0.8, max_price=8.0, costs=[1.0, 2.0]")
print("  Static: No dynamics (market_mode='static')")
print("  AR(1): Demand shocks (φ=0.8, σ=0.2)")
print("  Regime: 3-state Markov switching\n")

# Collect all stats
all_stats = []
for exp_name in experiments:
    all_stats.append(compute_stats(exp_name))

df = pd.DataFrame(all_stats)

print("\n" + "="*80)
print("STATIC MARKET RESULTS")
print("="*80)
print("\nNSR vs NSR (Competitive Baseline):")
row = df[df['experiment'] == 'static_nsr_vs_nsr'].iloc[0]
print(f"  Nash Gap 1: ${row['nash_gap_1_mean']:+.3f} ± ${row['nash_gap_1_std']:.3f}")
print(f"  Nash Gap 2: ${row['nash_gap_2_mean']:+.3f} ± ${row['nash_gap_2_std']:.3f}")
print(f"  Total Profit 1: ${row['total_profit_1_mean']:,.0f}")

print("\nRL vs RL (Collusion Test):")
row = df[df['experiment'] == 'static_rl_vs_rl'].iloc[0]
print(f"  Nash Gap 1: ${row['nash_gap_1_mean']:+.3f} ± ${row['nash_gap_1_std']:.3f}")
print(f"  Nash Gap 2: ${row['nash_gap_2_mean']:+.3f} ± ${row['nash_gap_2_std']:.3f}")
print(f"  Total Profit 1: ${row['total_profit_1_mean']:,.0f}")

print("\nNSR(low-cost) vs RL(high-cost):")
row = df[df['experiment'] == 'static_nsr_low_vs_rl_high'].iloc[0]
print(f"  Nash Gap 1 (NSR): ${row['nash_gap_1_mean']:+.3f} ± ${row['nash_gap_1_std']:.3f}")
print(f"  Nash Gap 2 (RL):  ${row['nash_gap_2_mean']:+.3f} ± ${row['nash_gap_2_std']:.3f}")

print("\nRL(low-cost) vs NSR(high-cost):")
row = df[df['experiment'] == 'static_rl_low_vs_nsr_high'].iloc[0]
print(f"  Nash Gap 1 (RL):  ${row['nash_gap_1_mean']:+.3f} ± ${row['nash_gap_1_std']:.3f}")
print(f"  Nash Gap 2 (NSR): ${row['nash_gap_2_mean']:+.3f} ± ${row['nash_gap_2_std']:.3f}")

print("\n" + "="*80)
print("AR(1) DYNAMIC MARKET RESULTS")
print("="*80)
print("\nNSR vs NSR (Competitive Baseline):")
row = df[df['experiment'] == 'ar1_nsr_vs_nsr'].iloc[0]
print(f"  Nash Gap 1: ${row['nash_gap_1_mean']:+.3f} ± ${row['nash_gap_1_std']:.3f}")
print(f"  Nash Gap 2: ${row['nash_gap_2_mean']:+.3f} ± ${row['nash_gap_2_std']:.3f}")
print(f"  Total Profit 1: ${row['total_profit_1_mean']:,.0f}")

print("\nRL vs RL (Collusion Test):")
row = df[df['experiment'] == 'ar1_rl_vs_rl'].iloc[0]
print(f"  Nash Gap 1: ${row['nash_gap_1_mean']:+.3f} ± ${row['nash_gap_1_std']:.3f}")
print(f"  Nash Gap 2: ${row['nash_gap_2_mean']:+.3f} ± ${row['nash_gap_2_std']:.3f}")
print(f"  Total Profit 1: ${row['total_profit_1_mean']:,.0f}")

print("\nNSR(low-cost) vs RL(high-cost):")
row = df[df['experiment'] == 'ar1_nsr_low_vs_rl_high'].iloc[0]
print(f"  Nash Gap 1 (NSR): ${row['nash_gap_1_mean']:+.3f} ± ${row['nash_gap_1_std']:.3f}")
print(f"  Nash Gap 2 (RL):  ${row['nash_gap_2_mean']:+.3f} ± ${row['nash_gap_2_std']:.3f}")

print("\nRL(low-cost) vs NSR(high-cost):")
row = df[df['experiment'] == 'ar1_rl_low_vs_nsr_high'].iloc[0]
print(f"  Nash Gap 1 (RL):  ${row['nash_gap_1_mean']:+.3f} ± ${row['nash_gap_1_std']:.3f}")
print(f"  Nash Gap 2 (NSR): ${row['nash_gap_2_mean']:+.3f} ± ${row['nash_gap_2_std']:.3f}")

print("\n" + "="*80)
print("REGIME SWITCHING MARKET RESULTS")
print("="*80)
print("\nNSR vs NSR (Competitive Baseline):")
row = df[df['experiment'] == 'regime_nsr_vs_nsr'].iloc[0]
print(f"  Nash Gap 1: ${row['nash_gap_1_mean']:+.3f} ± ${row['nash_gap_1_std']:.3f}")
print(f"  Nash Gap 2: ${row['nash_gap_2_mean']:+.3f} ± ${row['nash_gap_2_std']:.3f}")
print(f"  Total Profit 1: ${row['total_profit_1_mean']:,.0f}")
print("  ⚠️  NEGATIVE gaps: NSR pricing BELOW Nash in regime switching!")

print("\nRL vs RL (Collusion Test):")
row = df[df['experiment'] == 'regime_rl_vs_rl'].iloc[0]
print(f"  Nash Gap 1: ${row['nash_gap_1_mean']:+.3f} ± ${row['nash_gap_1_std']:.3f}")
print(f"  Nash Gap 2: ${row['nash_gap_2_mean']:+.3f} ± ${row['nash_gap_2_std']:.3f}")
print(f"  Total Profit 1: ${row['total_profit_1_mean']:,.0f}")
print("  ⚠️  MUCH HIGHER than static/AR(1)!")

print("\nNSR(low-cost) vs RL(high-cost):")
row = df[df['experiment'] == 'regime_nsr_low_vs_rl_high'].iloc[0]
print(f"  Nash Gap 1 (NSR): ${row['nash_gap_1_mean']:+.3f} ± ${row['nash_gap_1_std']:.3f}")
print(f"  Nash Gap 2 (RL):  ${row['nash_gap_2_mean']:+.3f} ± ${row['nash_gap_2_std']:.3f}")

print("\nRL(low-cost) vs NSR(high-cost):")
row = df[df['experiment'] == 'regime_rl_low_vs_nsr_high'].iloc[0]
print(f"  Nash Gap 1 (RL):  ${row['nash_gap_1_mean']:+.3f} ± ${row['nash_gap_1_std']:.3f}")
print(f"  Nash Gap 2 (NSR): ${row['nash_gap_2_mean']:+.3f} ± ${row['nash_gap_2_std']:.3f}")
print("  ⚠️  RL much more competitive when low-cost facing NSR!")

print("\n" + "="*80)
print("CROSS-MARKET COMPARISON: RL vs RL COLLUSION")
print("="*80)

# Compare RL collusion across markets
static_rl = df[df['experiment'] == 'static_rl_vs_rl'].iloc[0]
ar1_rl = df[df['experiment'] == 'ar1_rl_vs_rl'].iloc[0]
regime_rl = df[df['experiment'] == 'regime_rl_vs_rl'].iloc[0]

print("\nAverage Nash Gap (RL Agent 1):")
print(f"  Static:          ${static_rl['nash_gap_1_mean']:+.3f}")
print(f"  AR(1):           ${ar1_rl['nash_gap_1_mean']:+.3f}")
print(f"  Regime Switch:   ${regime_rl['nash_gap_1_mean']:+.3f}")
print(f"\n  Regime/Static Ratio: {regime_rl['nash_gap_1_mean']/static_rl['nash_gap_1_mean']:.2f}x")

print("\n" + "="*80)
print("KEY FINDINGS")
print("="*80)

print("\n1. REGIME SWITCHING AMPLIFIES RL COLLUSION:")
print(f"   - Static: ${static_rl['nash_gap_1_mean']:.3f} above Nash")
print(f"   - Regime: ${regime_rl['nash_gap_1_mean']:.3f} above Nash ({regime_rl['nash_gap_1_mean']/static_rl['nash_gap_1_mean']:.1f}x higher)")

print("\n2. NSR OVERLY AGGRESSIVE IN REGIME SWITCHING:")
nsr_regime = df[df['experiment'] == 'regime_nsr_vs_nsr'].iloc[0]
print(f"   - Negative Nash gaps: ${nsr_regime['nash_gap_1_mean']:.3f}")
print("   - Pricing below Nash equilibrium (over-competitive)")

print("\n3. COST POSITION MATTERS FOR RL:")
rl_low = df[df['experiment'] == 'regime_rl_low_vs_nsr_high'].iloc[0]
rl_high = df[df['experiment'] == 'regime_nsr_low_vs_rl_high'].iloc[0]
print(f"   - RL as low-cost: ${rl_low['nash_gap_1_mean']:+.3f} gap")
print(f"   - RL as high-cost: ${rl_high['nash_gap_2_mean']:+.3f} gap")
print("   - RL more competitive when low-cost facing NSR")

print("\n4. AR(1) SHOWS INTERMEDIATE BEHAVIOR:")
print(f"   - Collusion gap between static and regime switching")
print(f"   - Static: ${static_rl['nash_gap_1_mean']:.3f}")
print(f"   - AR(1):  ${ar1_rl['nash_gap_1_mean']:.3f}")
print(f"   - Regime: ${regime_rl['nash_gap_1_mean']:.3f}")

print("\n" + "="*80)
print("NEXT STEPS")
print("="*80)
print("\n1. Generate visualizations comparing market modes")
print("2. Statistical testing (t-tests, ANOVA)")
print("3. Update manuscript with corrected results")
print("4. Investigate why NSR prices below Nash in regime switching")
print("5. Analyze time-series for regime transition behavior\n")
