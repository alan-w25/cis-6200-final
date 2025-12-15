import numpy as np

class ConformalAuditor:
    """
    Conformal Auditor for detecting collusion in pricing markets using Split Conformal Prediction.

    The auditor constructs prediction intervals (safe zones) for competitive pricing behavior
    by calibrating on a baseline of NSR vs NSR agents. Prices outside the safe zone indicate
    potential collusive behavior with statistical guarantees.
    """
    def __init__(self, significance_level=0.05):
        self.alpha = significance_level  # e.g., 0.05 for 95% coverage
        self.calibration_scores = []     # Non-conformity scores from calibration set
        self.quantile = None              # Computed (1-alpha) quantile
        self.calibration_complete = False
        
    def compute_nonconformity_score(self, observed_price, theoretical_nash_price):
        """
        Computes the nonconformity score as the absolute deviation from Nash price.
        """
        return np.abs(observed_price - theoretical_nash_price)
        
    def update_calibration(self, observed_prices, nash_prices):
        """
        Updates the calibration set with new observations.

        Args:
            observed_prices: List or array of observed prices
            nash_prices: List or array of corresponding Nash equilibrium prices
        """
        scores = [
            self.compute_nonconformity_score(obs, nash)
            for obs, nash in zip(observed_prices, nash_prices)
        ]
        self.calibration_scores.extend(scores)

    def finalize_calibration(self):
        """
        Compute the conformal quantile from the calibration scores.
        This should be called after all calibration data has been collected.
        """
        if not self.calibration_scores:
            raise ValueError("Cannot finalize calibration with empty calibration set")

        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(1.0, max(0.0, q_level))

        self.quantile = np.quantile(self.calibration_scores, q_level, method='higher')
        self.calibration_complete = True

        return self.quantile
        
    def generate_prediction_set(self, theoretical_nash_price):
        """
        Generates a prediction interval (safe zone) for the price using conformal prediction.

        Uses the (1-alpha) quantile of calibration scores to form the interval:
        C(x) = [nash_price - q_hat, nash_price + q_hat]

        Args:
            theoretical_nash_price: Current Nash equilibrium price (can be dynamic/regime-dependent)

        Returns:
            (lower_bound, upper_bound): Tuple defining the safe zone
        """
        if not self.calibration_complete:
            # If calibration not finalized, compute quantile on-the-fly
            if not self.calibration_scores:
                return (-np.inf, np.inf)

            n = len(self.calibration_scores)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            q_level = min(1.0, max(0.0, q_level))
            quantile = np.quantile(self.calibration_scores, q_level, method='higher')
        else:
            quantile = self.quantile

        lower_bound = theoretical_nash_price - quantile
        upper_bound = theoretical_nash_price + quantile

        return (lower_bound, upper_bound)
        
    def check_multivalid_coverage(self, observed_prices, nash_prices, groups):
        """
        Checks if the coverage is valid across different groups (e.g., demand shock levels).
        Algorithm 25 logic:
        Calculate coverage rate for each group and check if it's close to 1-alpha.
        """
        coverage_results = {}
        
        # Pre-calculate prediction sets for all points (using full calibration for simplicity)
        # In online setting, we would use past data. Here we do batch audit.
        
        # Let's assume we use the current calibration set to check coverage on this batch
        # (This is valid for "marginal" coverage check)
        
        quantile = 0
        if self.calibration_scores:
            n = len(self.calibration_scores)
            q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
            q_level = min(1.0, max(0.0, q_level))
            quantile = np.quantile(self.calibration_scores, q_level, method='higher')
            
        for obs, nash, group in zip(observed_prices, nash_prices, groups):
            lower = nash - quantile
            upper = nash + quantile
            is_covered = lower <= obs <= upper
            
            if group not in coverage_results:
                coverage_results[group] = {'total': 0, 'covered': 0}
            
            coverage_results[group]['total'] += 1
            if is_covered:
                coverage_results[group]['covered'] += 1
                
        # Calculate rates
        audit_report = {}
        for group, stats in coverage_results.items():
            rate = stats['covered'] / stats['total']
            audit_report[group] = {
                'coverage_rate': rate,
                'violation': rate < (1 - self.alpha - 0.1) # Simple threshold for violation
            }
            
        return audit_report

    def detect_collusion(self, observed_prices, nash_prices):
        """
        Detect collusion by measuring if prices are consistently outside the safe zone.

        Args:
            observed_prices: Array of observed prices
            nash_prices: Array of corresponding Nash equilibrium prices

        Returns:
            is_collusive: Boolean indicating if collusion is detected
            collusion_index: Quantitative measure of deviation magnitude
        """
        # Calculate average deviation
        deviations = np.array(observed_prices) - np.array(nash_prices)
        avg_deviation = np.mean(deviations)

        # Use conformal quantile as the scale
        scale = self.quantile if self.calibration_complete else np.mean(self.calibration_scores)
        if scale < 1e-6:
            scale = 1.0

        collusion_index = avg_deviation / scale

        # Flag as collusive if average deviation exceeds 1 quantile
        is_collusive = collusion_index > 1.0

        return is_collusive, collusion_index

    def calculate_violation_rate(self, observed_prices, nash_prices):
        """
        Calculate the marginal violation rate for a sequence of observations.

        Violation Rate = (# observations outside safe zone) / (total observations)

        For valid conformal prediction, this should be ≤ alpha with high probability.

        Args:
            observed_prices: Array of observed prices
            nash_prices: Array of corresponding Nash equilibrium prices

        Returns:
            violation_rate: Fraction of observations outside the prediction set
            violations: Boolean array indicating which observations violated
        """
        if not self.calibration_complete:
            raise ValueError("Must finalize calibration before calculating violation rates")

        violations = []
        for obs, nash in zip(observed_prices, nash_prices):
            lower, upper = self.generate_prediction_set(nash)
            violated = (obs < lower) or (obs > upper)
            violations.append(violated)

        violations = np.array(violations)
        violation_rate = np.mean(violations)

        return violation_rate, violations

    def get_calibration_summary(self):
        """
        Get summary statistics of the calibration set.

        Returns:
            dict with calibration statistics
        """
        if not self.calibration_scores:
            return None

        return {
            'n_calibration': len(self.calibration_scores),
            'mean_score': np.mean(self.calibration_scores),
            'std_score': np.std(self.calibration_scores),
            'quantile': self.quantile if self.calibration_complete else None,
            'alpha': self.alpha,
            'expected_coverage': 1 - self.alpha
        }
