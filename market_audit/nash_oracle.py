"""
Nash Equilibrium Oracle for Duopoly Market

Computes theoretical Nash equilibrium and Joint Profit Maximizing prices
for the multinomial logit duopoly model across different market regimes.
"""

import numpy as np
from scipy.optimize import fsolve, minimize


class NashOracle:
    """
    Computes Nash equilibrium prices and joint profit maximizing prices
    for a differentiated Bertrand duopoly with logit demand.
    """

    def __init__(self, config=None):
        self.config = config if config else {}
        self.max_iterations = self.config.get('max_iterations', 1000)
        self.tolerance = self.config.get('tolerance', 1e-6)

    def compute_market_shares(self, prices, quality, price_sensitivity):
        """
        Compute market shares using multinomial logit demand.

        Args:
            prices: array [p1, p2]
            quality: array [alpha1, alpha2]
            price_sensitivity: scalar beta

        Returns:
            shares: array [q1, q2]
        """
        utilities = quality - price_sensitivity * prices
        exp_utilities = np.exp(utilities)
        denominator = 1.0 + np.sum(exp_utilities)
        shares = exp_utilities / denominator
        return shares

    def compute_profits(self, prices, costs, quality, price_sensitivity):
        """
        Compute profits for both firms.

        Args:
            prices: array [p1, p2]
            costs: array [c1, c2]
            quality: array [alpha1, alpha2]
            price_sensitivity: scalar beta

        Returns:
            profits: array [pi1, pi2]
        """
        shares = self.compute_market_shares(prices, quality, price_sensitivity)
        margins = prices - costs
        profits = margins * shares
        return profits

    def nash_first_order_conditions(self, prices, costs, quality, price_sensitivity):
        """
        Compute the first-order conditions for Nash equilibrium.

        For firm i, FOC is:
        dπ_i/dp_i = q_i + (p_i - c_i) * dq_i/dp_i = 0

        For logit demand:
        dq_i/dp_i = -beta * q_i * (1 - q_i)

        So FOC becomes:
        q_i - beta * (p_i - c_i) * q_i * (1 - q_i) = 0

        Rearranging:
        p_i = c_i + 1 / (beta * (1 - q_i))

        Returns residual that should be zero at equilibrium.
        """
        shares = self.compute_market_shares(prices, quality, price_sensitivity)

        # FOC for each firm
        residuals = np.zeros(2)
        for i in range(2):
            # Best response: p_i = c_i + 1 / (beta * (1 - q_i))
            # Residual: p_i - c_i - 1/(beta * (1 - q_i))
            if shares[i] < 0.999:  # Avoid division by zero
                best_response_price = costs[i] + 1.0 / (price_sensitivity * (1.0 - shares[i]))
                residuals[i] = prices[i] - best_response_price
            else:
                # If share approaches 1, firm has monopoly power
                residuals[i] = 0.0

        return residuals

    def compute_nash_equilibrium(self, costs, quality, price_sensitivity, initial_guess=None):
        """
        Solve for Nash equilibrium prices using fixed-point iteration.

        Args:
            costs: array [c1, c2] - marginal costs
            quality: array [alpha1, alpha2] - quality parameters
            price_sensitivity: scalar beta - price sensitivity
            initial_guess: optional starting point

        Returns:
            nash_prices: array [p1*, p2*]
            nash_profits: array [pi1*, pi2*]
            converged: bool
        """
        if initial_guess is None:
            # Heuristic: Start at monopoly markup over cost
            initial_guess = costs + 2.0 / price_sensitivity

        # Use scipy.optimize.fsolve to find root of FOCs
        try:
            solution = fsolve(
                lambda p: self.nash_first_order_conditions(p, costs, quality, price_sensitivity),
                initial_guess,
                full_output=True
            )

            nash_prices = solution[0]
            info = solution[1]
            converged = (info['fvec']**2).sum() < self.tolerance

            # Ensure prices are non-negative and above cost
            nash_prices = np.maximum(nash_prices, costs + 0.01)

            # Compute equilibrium profits
            nash_profits = self.compute_profits(nash_prices, costs, quality, price_sensitivity)

            return nash_prices, nash_profits, converged

        except Exception as e:
            print(f"Nash equilibrium computation failed: {e}")
            # Return fallback: cost + markup
            fallback_prices = costs + 1.0 / price_sensitivity
            fallback_profits = self.compute_profits(fallback_prices, costs, quality, price_sensitivity)
            return fallback_prices, fallback_profits, False

    def compute_joint_profit_maximum(self, costs, quality, price_sensitivity, initial_guess=None):
        """
        Solve for joint profit maximizing (collusive) prices.

        Args:
            costs: array [c1, c2]
            quality: array [alpha1, alpha2]
            price_sensitivity: scalar beta
            initial_guess: optional starting point

        Returns:
            jpm_prices: array [p1^M, p2^M]
            jpm_profits: array [pi1^M, pi2^M]
            total_profit: scalar
        """
        if initial_guess is None:
            # Start higher than Nash
            initial_guess = costs + 3.0 / price_sensitivity

        def negative_joint_profit(prices):
            """Objective to minimize (negative of joint profit)."""
            profits = self.compute_profits(prices, costs, quality, price_sensitivity)
            return -np.sum(profits)

        # Constraint: prices must be above cost
        bounds = [(costs[i] + 0.01, costs[i] + 10.0) for i in range(2)]

        try:
            result = minimize(
                negative_joint_profit,
                initial_guess,
                method='L-BFGS-B',
                bounds=bounds
            )

            jpm_prices = result.x
            jpm_profits = self.compute_profits(jpm_prices, costs, quality, price_sensitivity)
            total_profit = np.sum(jpm_profits)

            return jpm_prices, jpm_profits, total_profit

        except Exception as e:
            print(f"JPM computation failed: {e}")
            # Return Nash as fallback
            nash_prices, nash_profits, _ = self.compute_nash_equilibrium(costs, quality, price_sensitivity)
            return nash_prices, nash_profits, np.sum(nash_profits)

    def get_regime_benchmarks(self, regime, costs):
        """
        Compute both Nash and JPM benchmarks for a specific regime.

        Args:
            regime: int (0=Recession, 1=Normal, 2=Boom)
            costs: array [c1, c2]

        Returns:
            dict with nash_prices, nash_profits, jpm_prices, jpm_profits
        """
        # Define regime parameters (matching DuopolyEnv defaults)
        regime_params = {
            0: {'quality': np.array([1.0, 1.0]), 'beta': 1.0},      # Recession
            1: {'quality': np.array([2.0, 2.0]), 'beta': 0.8},      # Normal
            2: {'quality': np.array([3.0, 3.0]), 'beta': 0.5}       # Boom
        }

        params = regime_params[regime]
        quality = params['quality']
        beta = params['beta']

        nash_prices, nash_profits, nash_converged = self.compute_nash_equilibrium(
            costs, quality, beta
        )

        jpm_prices, jpm_profits, total_jpm = self.compute_joint_profit_maximum(
            costs, quality, beta
        )

        return {
            'nash_prices': nash_prices,
            'nash_profits': nash_profits,
            'nash_converged': nash_converged,
            'jpm_prices': jpm_prices,
            'jpm_profits': jpm_profits,
            'total_jpm': total_jpm,
            'regime': regime,
            'quality': quality,
            'beta': beta
        }


def test_nash_oracle():
    """Test the Nash oracle with known market configurations."""
    oracle = NashOracle()

    # Test case from research plan: Static market
    # Costs: [1.0, 2.0], Quality: [2.0, 2.0], Beta: 0.8
    # Expected Nash: approximately [3.72, 4.04]
    # Expected Collusive: approximately [5.49, 6.55]

    costs = np.array([1.0, 2.0])
    quality = np.array([2.0, 2.0])
    beta = 0.8

    print("=" * 60)
    print("Nash Oracle Test: Static Market Configuration")
    print("=" * 60)
    print(f"Costs: {costs}")
    print(f"Quality: {quality}")
    print(f"Price Sensitivity (beta): {beta}")
    print()

    # Compute Nash equilibrium
    nash_prices, nash_profits, converged = oracle.compute_nash_equilibrium(
        costs, quality, beta
    )

    print("Nash Equilibrium:")
    print(f"  Prices: {nash_prices}")
    print(f"  Profits: {nash_profits}")
    print(f"  Total Profit: {np.sum(nash_profits):.4f}")
    print(f"  Converged: {converged}")
    print()

    # Compute Joint Profit Maximum
    jpm_prices, jpm_profits, total_jpm = oracle.compute_joint_profit_maximum(
        costs, quality, beta
    )

    print("Joint Profit Maximum (Collusive):")
    print(f"  Prices: {jpm_prices}")
    print(f"  Profits: {jpm_profits}")
    print(f"  Total Profit: {total_jpm:.4f}")
    print()

    # Compute wedge
    price_wedge = jpm_prices - nash_prices
    print(f"Auditable Wedge (JPM - Nash): {price_wedge}")
    print()

    # Test all three regimes
    print("=" * 60)
    print("Regime-Specific Benchmarks")
    print("=" * 60)

    for regime in [0, 1, 2]:
        regime_name = ['Recession', 'Normal', 'Boom'][regime]
        benchmarks = oracle.get_regime_benchmarks(regime, costs)

        print(f"\n{regime_name} Regime (regime={regime}):")
        print(f"  Quality: {benchmarks['quality']}, Beta: {benchmarks['beta']}")
        print(f"  Nash Prices: {benchmarks['nash_prices']}")
        print(f"  JPM Prices: {benchmarks['jpm_prices']}")
        print(f"  Wedge: {benchmarks['jpm_prices'] - benchmarks['nash_prices']}")


if __name__ == "__main__":
    test_nash_oracle()
