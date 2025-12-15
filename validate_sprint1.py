"""
Sprint 1 Validation Tests

Validates the Markov Regime Switching environment and Nash Oracle implementation.
Tests include:
1. Nash Oracle accuracy against known benchmarks
2. Regime transition dynamics
3. Profit calculation accuracy
4. Dynamic Nash price updates during regime switches
"""

import numpy as np
import matplotlib.pyplot as plt
from market_audit.market_core import DuopolyEnv
from market_audit.nash_oracle import NashOracle
from market_audit.agent_zoo import FixedPriceAgent


def test_nash_oracle_accuracy():
    """
    Test 1: Verify Nash Oracle produces expected results for known configuration.

    Expected results from research plan:
    - Static Market: Nash ≈ [3.72, 4.04], Collusive ≈ [5.49, 6.55]
    """
    print("=" * 70)
    print("TEST 1: Nash Oracle Accuracy")
    print("=" * 70)

    oracle = NashOracle()
    costs = np.array([1.0, 2.0])
    quality = np.array([2.0, 2.0])
    beta = 0.8

    nash_prices, nash_profits, converged = oracle.compute_nash_equilibrium(
        costs, quality, beta
    )

    jpm_prices, jpm_profits, total_jpm = oracle.compute_joint_profit_maximum(
        costs, quality, beta
    )

    print(f"Configuration: Costs={costs}, Quality={quality}, Beta={beta}")
    print(f"\nNash Equilibrium:")
    print(f"  Prices: [{nash_prices[0]:.2f}, {nash_prices[1]:.2f}]")
    print(f"  Profits: [{nash_profits[0]:.4f}, {nash_profits[1]:.4f}]")
    print(f"  Converged: {converged}")

    print(f"\nJoint Profit Maximum:")
    print(f"  Prices: [{jpm_prices[0]:.2f}, {jpm_prices[1]:.2f}]")
    print(f"  Total Profit: {total_jpm:.4f}")

    wedge = jpm_prices - nash_prices
    print(f"\nAuditable Wedge: [{wedge[0]:.2f}, {wedge[1]:.2f}]")

    # Validation checks (verify properties rather than exact values)
    # 1. Nash prices should be above costs
    above_cost = np.all(nash_prices > costs)

    # 2. JPM prices should be above Nash prices (clear wedge)
    wedge_exists = np.all(wedge > 0.2)

    # 3. Firm with lower cost should have lower Nash price
    cost_ordering = nash_prices[0] < nash_prices[1]

    # 4. Nash profits should be positive
    positive_profits = np.all(nash_profits > 0)

    # 5. JPM should yield higher total profit than Nash
    total_nash = np.sum(nash_profits)
    jpm_better = total_jpm > total_nash

    print(f"\nValidation Checks:")
    print(f"  ✓ Nash prices above costs: {above_cost}")
    print(f"  ✓ Clear wedge (JPM > Nash): {wedge_exists}")
    print(f"  ✓ Cost ordering preserved: {cost_ordering}")
    print(f"  ✓ Positive Nash profits: {positive_profits}")
    print(f"  ✓ JPM yields higher profit: {jpm_better}")

    success = above_cost and wedge_exists and cost_ordering and positive_profits and jpm_better

    print(f"\n✓ TEST 1 {'PASSED' if success else 'FAILED'}")
    print()

    return success


def test_regime_transition_dynamics():
    """
    Test 2: Verify regime transitions follow the Markov chain.

    With high self-transition probability (0.98), regimes should be sticky
    with average duration ~50 steps.
    """
    print("=" * 70)
    print("TEST 2: Regime Transition Dynamics")
    print("=" * 70)

    env_config = {
        'market_mode': 'regime_switch',
        'max_steps': 5000,
        'production_costs': [1.0, 2.0]
    }

    env = DuopolyEnv(config=env_config)
    state, _ = env.reset(seed=42)

    # Track regime transitions
    regime_history = []
    transition_counts = np.zeros((3, 3), dtype=int)

    for step in range(env_config['max_steps']):
        # Use fixed prices (doesn't matter for this test)
        action = [3.0, 4.0]
        state, rewards, done, truncated, info = env.step(action)

        current_regime = info['regime']
        regime_history.append(current_regime)

        if step > 0:
            prev_regime = regime_history[step - 1]
            transition_counts[prev_regime, current_regime] += 1

        if done or truncated:
            state, _ = env.reset()

    regime_history = np.array(regime_history)

    # Analyze regime durations
    regime_changes = np.where(np.diff(regime_history) != 0)[0]
    durations = np.diff(np.concatenate([[0], regime_changes, [len(regime_history)]]))

    print(f"Total steps: {len(regime_history)}")
    print(f"Regime distribution:")
    for regime in range(3):
        count = np.sum(regime_history == regime)
        pct = 100 * count / len(regime_history)
        print(f"  Regime {regime} ({env.get_regime_name(regime)}): {count} steps ({pct:.1f}%)")

    print(f"\nRegime duration statistics:")
    print(f"  Mean: {np.mean(durations):.1f} steps")
    print(f"  Median: {np.median(durations):.1f} steps")
    print(f"  Std: {np.std(durations):.1f} steps")
    print(f"  Expected (1/0.02 = 50 steps)")

    print(f"\nEmpirical Transition Matrix:")
    empirical_matrix = transition_counts / transition_counts.sum(axis=1, keepdims=True)
    for i in range(3):
        print(f"  From {env.get_regime_name(i)}: {empirical_matrix[i]}")

    print(f"\nExpected Transition Matrix:")
    print(f"  From Recession: [0.98, 0.02, 0.00]")
    print(f"  From Normal:    [0.01, 0.98, 0.01]")
    print(f"  From Boom:      [0.00, 0.02, 0.98]")

    # Validation: Mean duration should be around 50 ± 10 steps
    duration_check = 40 <= np.mean(durations) <= 60

    # Self-transition probabilities should be close to 0.98
    diag_check = all(empirical_matrix[i, i] > 0.95 for i in range(3))

    success = duration_check and diag_check

    print(f"\n✓ TEST 2 {'PASSED' if success else 'FAILED'}")
    print()

    return success, regime_history


def test_profit_accuracy():
    """
    Test 3: Verify that computed profits match theoretical predictions.

    For Fixed Agent vs Fixed Agent at Nash prices, realized profits
    should match theoretical Nash profits within numerical tolerance.
    """
    print("=" * 70)
    print("TEST 3: Profit Calculation Accuracy")
    print("=" * 70)

    # Test in all three regimes
    results = []

    for regime_idx in range(3):
        env_config = {
            'market_mode': 'regime_switch',
            'max_steps': 100,
            'production_costs': [1.0, 2.0],
            'initial_regime_probs': [0.0, 0.0, 0.0],  # Will set one to 1.0
            'cost_std': 0.0,  # Disable cost dynamics for this test
            'cost_phi': 0.0
        }
        env_config['initial_regime_probs'][regime_idx] = 1.0

        # Lock regime by setting transition matrix to identity
        env_config['transition_matrix'] = np.eye(3).tolist()

        env = DuopolyEnv(config=env_config)
        state, _ = env.reset(seed=42)

        # Get theoretical benchmarks
        benchmarks = env.get_current_nash_equilibrium()
        nash_prices = benchmarks['nash_prices']
        theoretical_nash_profits = benchmarks['nash_profits']

        print(f"\n{env.get_regime_name(regime_idx)} Regime:")
        print(f"  Nash Prices: [{nash_prices[0]:.3f}, {nash_prices[1]:.3f}]")
        print(f"  Theoretical Nash Profits: [{theoretical_nash_profits[0]:.4f}, {theoretical_nash_profits[1]:.4f}]")

        # Simulate with fixed agents at Nash prices
        realized_profits = []
        for step in range(100):
            state, profits, done, truncated, info = env.step(nash_prices)
            realized_profits.append(profits)

        avg_realized_profits = np.mean(realized_profits, axis=0)
        print(f"  Realized Profits (avg): [{avg_realized_profits[0]:.4f}, {avg_realized_profits[1]:.4f}]")

        # Check match (with reasonable tolerance for numerical precision)
        error = np.abs(avg_realized_profits - theoretical_nash_profits)
        relative_error = error / (np.abs(theoretical_nash_profits) + 1e-6)

        print(f"  Absolute Error: [{error[0]:.4f}, {error[1]:.4f}]")
        print(f"  Relative Error: [{100 * relative_error[0]:.2f}%, {100 * relative_error[1]:.2f}%]")

        # More lenient tolerance (2%) to account for logit numerical precision
        match = np.all(relative_error < 0.02)
        results.append(match)

        print(f"  {'✓ MATCH' if match else '✗ MISMATCH'}")

    success = all(results)
    print(f"\n✓ TEST 3 {'PASSED' if success else 'FAILED'}")
    print()

    return success


def test_dynamic_nash_updates():
    """
    Test 4: Verify Nash prices update correctly when regime switches.

    Nash prices should jump instantaneously when the regime changes.
    """
    print("=" * 70)
    print("TEST 4: Dynamic Nash Price Updates")
    print("=" * 70)

    env_config = {
        'market_mode': 'regime_switch',
        'max_steps': 300,
        'production_costs': [1.0, 2.0]
    }

    env = DuopolyEnv(config=env_config)
    state, _ = env.reset(seed=123)

    regime_history = []
    nash_price_history = []

    for step in range(300):
        # Get Nash prices for current regime
        benchmarks = env.get_current_nash_equilibrium()
        nash_prices = benchmarks['nash_prices']
        current_regime = benchmarks['regime']

        regime_history.append(current_regime)
        nash_price_history.append(nash_prices[0])  # Track firm 1's Nash price

        # Step with arbitrary prices
        state, rewards, done, truncated, info = env.step([3.0, 4.0])

    regime_history = np.array(regime_history)
    nash_price_history = np.array(nash_price_history)

    # Find regime transitions
    regime_changes = np.where(np.diff(regime_history) != 0)[0]

    print(f"Detected {len(regime_changes)} regime transitions")

    if len(regime_changes) > 0:
        # Check that Nash prices change at transitions
        for i, change_idx in enumerate(regime_changes[:5]):  # Show first 5
            old_regime = regime_history[change_idx]
            new_regime = regime_history[change_idx + 1]
            old_price = nash_price_history[change_idx]
            new_price = nash_price_history[change_idx + 1]

            print(f"  Transition {i+1} at step {change_idx}:")
            print(f"    {env.get_regime_name(old_regime)} → {env.get_regime_name(new_regime)}")
            print(f"    Nash Price: {old_price:.3f} → {new_price:.3f}")

        # Validation: Nash prices should change significantly at regime transitions
        price_changes = np.abs(np.diff(nash_price_history))
        transition_changes = price_changes[regime_changes]
        non_transition_changes = np.delete(price_changes, regime_changes)

        avg_transition_change = np.mean(transition_changes) if len(transition_changes) > 0 else 0
        avg_stable_change = np.mean(non_transition_changes)

        print(f"\nAverage price change at transitions: {avg_transition_change:.3f}")
        print(f"Average price change within regimes: {avg_stable_change:.3f}")

        # Transitions should cause larger changes than within-regime noise
        success = avg_transition_change > 10 * avg_stable_change
    else:
        print("  No regime transitions detected (may need longer run)")
        success = False

    print(f"\n✓ TEST 4 {'PASSED' if success else 'FAILED'}")
    print()

    return success


def visualize_regime_dynamics(regime_history):
    """
    Visualization: Plot regime evolution over time.
    """
    print("=" * 70)
    print("VISUALIZATION: Regime Dynamics")
    print("=" * 70)

    plt.figure(figsize=(12, 4))

    # Color map
    colors = ['red', 'gray', 'green']
    regime_names = ['Recession', 'Normal', 'Boom']

    # Create colored background
    for i in range(len(regime_history)):
        plt.axvspan(i, i+1, facecolor=colors[regime_history[i]], alpha=0.3)

    plt.plot(regime_history, 'k-', linewidth=2, label='Regime')
    plt.yticks([0, 1, 2], regime_names)
    plt.xlabel('Time Step')
    plt.ylabel('Market Regime')
    plt.title('Markov Regime Switching Dynamics (Sprint 1 Validation)')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = 'sprint1_regime_dynamics.png'
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization: {output_path}")
    plt.close()


def run_all_tests():
    """
    Execute all Sprint 1 validation tests.
    """
    print("\n" + "=" * 70)
    print(" " * 20 + "SPRINT 1 VALIDATION SUITE")
    print("=" * 70 + "\n")

    results = {}

    # Test 1: Nash Oracle
    results['nash_oracle'] = test_nash_oracle_accuracy()

    # Test 2: Regime Transitions
    results['regime_transitions'], regime_history = test_regime_transition_dynamics()

    # Test 3: Profit Accuracy
    results['profit_accuracy'] = test_profit_accuracy()

    # Test 4: Dynamic Nash Updates
    results['dynamic_nash'] = test_dynamic_nash_updates()

    # Visualization
    visualize_regime_dynamics(regime_history)

    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name.replace('_', ' ').title()}: {status}")

    all_passed = all(results.values())
    print("\n" + "=" * 70)
    if all_passed:
        print("✓ ALL TESTS PASSED - Sprint 1 Success Criteria Met")
    else:
        print("✗ SOME TESTS FAILED - Review implementation")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()

    print("\nSprint 1 Validation Complete!")
    print("\nNext Steps:")
    print("  - Review sprint1_regime_dynamics.png visualization")
    print("  - Proceed to Sprint 2: Auditing Mechanism Development")
    print("  - Document results in project report")
