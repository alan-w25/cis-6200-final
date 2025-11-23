import numpy as np

class ConformalAuditor:
    """
    Conformal Auditor for detecting collusion in pricing markets.
    """
    def __init__(self, significance_level=0.05):
        self.alpha = significance_level
        self.calibration_scores = []
        
    def compute_nonconformity_score(self, observed_price, theoretical_nash_price):
        """
        Computes the nonconformity score as the absolute deviation from Nash price.
        """
        return np.abs(observed_price - theoretical_nash_price)
        
    def update_calibration(self, observed_prices, nash_prices):
        """
        Updates the calibration set with new observations.
        """
        scores = [
            self.compute_nonconformity_score(obs, nash) 
            for obs, nash in zip(observed_prices, nash_prices)
        ]
        self.calibration_scores.extend(scores)
        
    def generate_prediction_set(self, theoretical_nash_price):
        """
        Generates a prediction set for the price using the calibration scores.
        Algorithm 24 (Adversarial-Marginal) logic:
        Use the (1-alpha) quantile of the calibration scores to form the interval.
        """
        if not self.calibration_scores:
            return (-np.inf, np.inf)
            
        n = len(self.calibration_scores)
        q_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        q_level = min(1.0, max(0.0, q_level))
        
        quantile = np.quantile(self.calibration_scores, q_level, method='higher')
        
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
        Simple collusion detection: if prices are consistently higher than Nash + tolerance.
        """
        # Calculate average deviation
        deviations = np.array(observed_prices) - np.array(nash_prices)
        avg_deviation = np.mean(deviations)
        
        # If average deviation is significantly positive, flag as collusion
        # We can use the conformal interval width as a scale
        scale = np.mean(self.calibration_scores) if self.calibration_scores else 1.0
        collusion_index = avg_deviation / (scale + 1e-6)
        
        is_collusive = collusion_index > 1.0 # Threshold
        
        return is_collusive, collusion_index
