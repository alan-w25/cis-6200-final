"""
Sprint 2 Validation Tests

Validates the dual-layer auditing framework:
1. Internal Audit (Brier Score) for DQN agent calibration
2. External Audit (Conformal Prediction) for collusion detection
3. Integration with simulation pipeline

Tests include:
1. Brier Score logging and calculation correctness
2. Conformal calibration and coverage guarantees
3. Violation rate accuracy
4. End-to-end auditing pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
from market_audit.market_core import DuopolyEnv
from market_audit.agent_zoo import RLAgent, NSRAgent
from market_audit.auditing_engine import ConformalAuditor
from market_audit.simulation_controller import MatchupRunner


def test_brier_score_logging():
    """
    Test 1: Verify Brier Score logging works correctly for RL agents.

    Tests that:
    - Predictions are logged during training
    - Brier Score is calculated correctly
    - Rolling window updates properly
    """
    print("=" * 70)
    print("TEST 1: Brier Score Logging and Calculation")
    print("=" * 70)

    env_config = {
        'market_mode': 'static',
        'max_steps': 1000,
        'production_costs': [1.0, 2.0]
    }

    env = DuopolyEnv(config=env_config)

    agent_config = {
        'state_dim': 5,
        'n_bins': 100,
        'enable_brier_logging': True,
        'brier_window': 500
    }

    agent = RLAgent(env.action_space, config=agent_config)

    # Run simulation to generate predictions
    state, _ = env.reset(seed=42)

    for step in range(1000):
        action = agent.act(state)
        next_state, profits, done, truncated, info = env.step([action, 3.5])  # Agent 1 vs fixed agent 2

        agent.update((state, action, profits[0], next_state, done))

        if done or truncated:
            state, _ = env.reset()
        else:
            state = next_state

    # Check that predictions were logged
    n_predictions = len(agent.prediction_log)
    n_brier_scores = len(agent.get_brier_scores())

    print(f"Total predictions logged: {n_predictions}")
    print(f"Brier scores calculated: {n_brier_scores}")

    # Verify that Brier scores were calculated after window size reached
    expected_brier_scores = max(0, n_predictions - agent_config['brier_window'] + 1)

    print(f"Expected Brier scores (after window fills): ~{expected_brier_scores}")

    # Manual Brier Score calculation verification
    if n_predictions >= agent_config['brier_window']:
        recent_preds = list(agent.prediction_log)[-agent_config['brier_window']:]
        predicted = np.array([p[0] for p in recent_preds])
        actual = np.array([p[1] for p in recent_preds])

        manual_brier = np.mean((predicted - actual) ** 2)
        agent_brier = agent.get_brier_scores()[-1]

        print(f"\nManual Brier Score: {manual_brier:.6f}")
        print(f"Agent Brier Score: {agent_brier:.6f}")
        print(f"Difference: {abs(manual_brier - agent_brier):.10f}")

        match = abs(manual_brier - agent_brier) < 1e-6

        print(f"\n{'✓ MATCH' if match else '✗ MISMATCH'}")
        success = match and n_brier_scores > 0
    else:
        print("\nNot enough data for Brier Score calculation")
        success = False

    print(f"\n✓ TEST 1 {'PASSED' if success else 'FAILED'}")
    print()

    return success


def test_conformal_calibration():
    """
    Test 2: Verify conformal prediction calibration and coverage guarantees.

    Uses NSR vs NSR as calibration set to establish safe zone,
    then tests coverage properties.
    """
    print("=" * 70)
    print("TEST 2: Conformal Prediction Calibration")
    print("=" * 70)

    # Run NSR vs NSR to generate calibration data
    env_config = {
        'market_mode': 'regime_switch',
        'max_steps': 2000,
        'production_costs': [1.0, 2.0]
    }

    env = DuopolyEnv(config=env_config)

    agent1 = NSRAgent(env.action_space, config={
        'n_bins': 100,
        'quality': 2.0,
        'price_sensitivity': 0.8,
        'cost': 1.0
    })

    agent2 = NSRAgent(env.action_space, config={
        'n_bins': 100,
        'quality': 2.0,
        'price_sensitivity': 0.8,
        'cost': 2.0
    })

    # Collect calibration data
    calibration_prices_1 = []
    calibration_nash_1 = []

    state, _ = env.reset(seed=42)

    for step in range(2000):
        action1 = agent1.act(state)
        action2 = agent2.act(state)

        benchmarks = env.get_current_nash_equilibrium()
        nash_prices = benchmarks['nash_prices']

        calibration_prices_1.append(action1)
        calibration_nash_1.append(nash_prices[0])

        next_state, profits, done, truncated, info = env.step([action1, action2])

        agent1.update((state, action1, profits[0], next_state, done))
        agent2.update((state, action2, profits[1], next_state, done))

        if done or truncated:
            state, _ = env.reset()
        else:
            state = next_state

    # Create and calibrate conformal auditor
    auditor = ConformalAuditor(significance_level=0.05)
    auditor.update_calibration(calibration_prices_1, calibration_nash_1)
    quantile = auditor.finalize_calibration()

    print(f"Calibration data points: {len(calibration_prices_1)}")
    print(f"Computed quantile (95th percentile): {quantile:.4f}")

    summary = auditor.get_calibration_summary()
    print(f"\nCalibration Summary:")
    print(f"  Mean non-conformity score: {summary['mean_score']:.4f}")
    print(f"  Std non-conformity score: {summary['std_score']:.4f}")
    print(f"  Expected coverage: {summary['expected_coverage']:.2%}")

    # Test coverage on the calibration set itself (should be ~95%)
    violation_rate, violations = auditor.calculate_violation_rate(
        calibration_prices_1,
        calibration_nash_1
    )

    print(f"\nEmpirical Coverage on Calibration Set:")
    print(f"  Violation rate: {violation_rate:.4f} (expected ≤ {auditor.alpha:.4f})")
    print(f"  Coverage rate: {1 - violation_rate:.4f} (expected ≥ {1 - auditor.alpha:.4f})")

    # For conformal prediction, violation rate should be ≤ alpha
    # With some tolerance for finite sample effects
    coverage_valid = violation_rate <= auditor.alpha + 0.05  # 5% tolerance

    print(f"\n{'✓ Coverage Guarantee Satisfied' if coverage_valid else '✗ Coverage Violated'}")

    success = coverage_valid and quantile > 0

    print(f"\n✓ TEST 2 {'PASSED' if success else 'FAILED'}")
    print()

    return success, auditor


def test_violation_rate_discrimination():
    """
    Test 3: Verify that conformal auditor can discriminate between
    competitive (NSR) and collusive (RL) behavior.

    NSR agents should have low violation rates (~alpha),
    while collusive RL agents should have higher violation rates.
    """
    print("=" * 70)
    print("TEST 3: Violation Rate Discrimination")
    print("=" * 70)

    # Use calibration from previous test or create new one
    env_config = {
        'market_mode': 'static',
        'max_steps': 1000,
        'production_costs': [1.0, 2.0]
    }

    # Calibrate on NSR vs NSR
    print("Calibrating on NSR vs NSR...")
    runner_config = {
        'env_config': env_config,
        'n_episodes': 1,
        'max_steps': 1000,
        'enable_auditing': True,
        'agent_config': {
            'n_bins': 100,
            'quality': 2.0,
            'price_sensitivity': 0.8
        }
    }

    runner = MatchupRunner(config=runner_config)
    nsr_results = runner.run_matchup(NSRAgent, NSRAgent, "NSR_vs_NSR_Calibration")

    # Extract calibration data
    episode_data = nsr_results.iloc[0]['episode_data']
    cal_prices = np.array(episode_data['p1'][500:])  # Skip warmup
    cal_nash = np.array(episode_data['nash_p1'][500:])

    auditor = ConformalAuditor(significance_level=0.05)
    auditor.update_calibration(cal_prices, cal_nash)
    auditor.finalize_calibration()

    print(f"Calibration quantile: {auditor.quantile:.4f}")

    # Test on NSR (should have low violation rate)
    print("\nTesting NSR vs NSR (competitive)...")
    nsr_test_results = runner.run_matchup(NSRAgent, NSRAgent, "NSR_vs_NSR_Test")
    test_data = nsr_test_results.iloc[0]['episode_data']
    nsr_prices = np.array(test_data['p1'][500:])
    nsr_nash = np.array(test_data['nash_p1'][500:])

    nsr_violation_rate, _ = auditor.calculate_violation_rate(nsr_prices, nsr_nash)

    print(f"  NSR Violation Rate: {nsr_violation_rate:.4f} (expected ≤ {auditor.alpha + 0.05:.4f})")

    # Test on RL vs RL (should have higher violation rate if collusive)
    print("\nTesting RL vs RL (potentially collusive)...")
    rl_runner_config = runner_config.copy()
    rl_runner_config['agent_config'] = {
        'state_dim': 5,
        'n_bins': 100,
        'lr': 1e-3,
        'gamma': 0.99,
        'epsilon': 0.1  # Lower epsilon for more exploitation
    }
    rl_runner = MatchupRunner(config=rl_runner_config)

    # Pre-train RL agents
    print("  Pre-training RL agents...")
    for _ in range(5):  # Multiple episodes for training
        _ = rl_runner.run_matchup(RLAgent, RLAgent, "RL_Training")

    # Test RL agents
    rl_results = rl_runner.run_matchup(RLAgent, RLAgent, "RL_vs_RL_Test")
    rl_test_data = rl_results.iloc[0]['episode_data']
    rl_prices = np.array(rl_test_data['p1'][500:])
    rl_nash = np.array(rl_test_data['nash_p1'][500:])

    rl_violation_rate, _ = auditor.calculate_violation_rate(rl_prices, rl_nash)

    print(f"  RL Violation Rate: {rl_violation_rate:.4f}")

    print(f"\nDiscrimination:")
    print(f"  NSR Violation Rate: {nsr_violation_rate:.4f}")
    print(f"  RL Violation Rate: {rl_violation_rate:.4f}")
    print(f"  Difference: {rl_violation_rate - nsr_violation_rate:.4f}")

    # Success if:
    # 1. NSR violation rate is within tolerance of alpha
    # 2. RL violation rate is higher (indicating collusion detection capability)
    nsr_valid = nsr_violation_rate <= auditor.alpha + 0.10
    rl_higher = rl_violation_rate >= nsr_violation_rate  # RL should violate more

    success = nsr_valid and rl_higher

    print(f"\n{'✓ Discrimination Successful' if success else '✗ Discrimination Failed'}")
    print(f"\n✓ TEST 3 {'PASSED' if success else 'FAILED'}")
    print()

    return success


def test_end_to_end_auditing():
    """
    Test 4: Verify end-to-end auditing pipeline integration.

    Tests that the MatchupRunner correctly collects:
    - Nash price benchmarks
    - Brier scores from RL agents
    - Price gaps for analysis
    """
    print("=" * 70)
    print("TEST 4: End-to-End Auditing Pipeline")
    print("=" * 70)

    runner_config = {
        'env_config': {
            'market_mode': 'regime_switch',
            'max_steps': 500
        },
        'n_episodes': 2,
        'max_steps': 500,
        'enable_auditing': True,
        'agent_config': {
            'state_dim': 5,
            'n_bins': 100,
            'enable_brier_logging': True
        }
    }

    runner = MatchupRunner(config=runner_config)

    print("Running RL vs RL matchup with auditing enabled...")
    results = runner.run_matchup(RLAgent, RLAgent, "RL_vs_RL_Audited")

    print(f"\nCollected {len(results)} episodes")

    # Check that auditing data was collected
    for idx, row in results.iterrows():
        episode_data = row['episode_data']

        print(f"\nEpisode {idx}:")
        print(f"  Steps: {len(episode_data['p1'])}")
        print(f"  Nash prices collected: {len(episode_data.get('nash_p1', []))}")
        print(f"  Regimes tracked: {len(episode_data.get('regime', []))}")

        if 'brier_score_1' in episode_data:
            brier_1 = np.array(episode_data['brier_score_1'])
            brier_1_clean = brier_1[~np.isnan(brier_1)]
            if len(brier_1_clean) > 0:
                print(f"  Brier scores (Agent 1): {len(brier_1_clean)} points, mean={np.mean(brier_1_clean):.4f}")

        # Check that Nash gaps were calculated
        if 'avg_nash_gap_1' in row:
            print(f"  Avg Nash gap (Agent 1): {row['avg_nash_gap_1']:.4f}")

    # Verify data completeness
    first_episode = results.iloc[0]['episode_data']

    has_nash_prices = len(first_episode.get('nash_p1', [])) > 0
    has_regimes = len(first_episode.get('regime', [])) > 0
    has_brier = 'brier_score_1' in first_episode

    print(f"\nData Completeness:")
    print(f"  ✓ Nash prices collected: {has_nash_prices}")
    print(f"  ✓ Regimes tracked: {has_regimes}")
    print(f"  ✓ Brier scores logged: {has_brier}")

    success = has_nash_prices and has_regimes and has_brier

    print(f"\n✓ TEST 4 {'PASSED' if success else 'FAILED'}")
    print()

    return success


def run_all_tests():
    """
    Execute all Sprint 2 validation tests.
    """
    print("\n" + "=" * 70)
    print(" " * 15 + "SPRINT 2 VALIDATION SUITE")
    print(" " * 10 + "Dual-Layer Auditing Framework")
    print("=" * 70 + "\n")

    results = {}

    # Test 1: Brier Score Logging
    results['brier_logging'] = test_brier_score_logging()

    # Test 2: Conformal Calibration
    results['conformal_calibration'], auditor = test_conformal_calibration()

    # Test 3: Violation Rate Discrimination
    results['violation_discrimination'] = test_violation_rate_discrimination()

    # Test 4: End-to-End Pipeline
    results['end_to_end'] = test_end_to_end_auditing()

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
        print("✓ ALL TESTS PASSED - Sprint 2 Success Criteria Met")
    else:
        print("✗ SOME TESTS FAILED - Review implementation")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()

    print("\nSprint 2 Validation Complete!")
    print("\nNext Steps:")
    print("  - Review test results and auditing metrics")
    print("  - Update manuscript with Sprint 2 implementation details")
    print("  - Proceed to Sprint 3: Experimental Campaign")
