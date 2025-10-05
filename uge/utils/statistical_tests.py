"""
Statistical Testing Utilities for GE-Lab

This module provides statistical tests and effect size calculations
for comparing evolutionary algorithm results following best practices
from the EC/ML literature.

Author: GE-Lab Team
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon, kruskal, friedmanchisquare, shapiro
import warnings


class StatisticalTests:
    """
    Statistical testing utilities for comparing EA/GE results.
    
    Implements best practices from:
    - Demšar (JMLR 2006)
    - Derrac et al. (SWARM & EC 2011)
    - García & Herrera (2008)
    """
    
    @staticmethod
    def check_normality(data: np.ndarray, alpha: float = 0.05) -> Dict[str, Any]:
        """
        Test normality using Shapiro-Wilk test.
        
        Args:
            data: 1D array of samples
            alpha: Significance level (default 0.05)
            
        Returns:
            Dict with test results
        """
        if len(data) < 3:
            return {
                'test': 'Shapiro-Wilk',
                'statistic': None,
                'p_value': None,
                'is_normal': None,
                'message': 'Insufficient data (n<3)'
            }
        
        statistic, p_value = shapiro(data)
        is_normal = p_value > alpha
        
        return {
            'test': 'Shapiro-Wilk',
            'statistic': statistic,
            'p_value': p_value,
            'is_normal': is_normal,
            'alpha': alpha,
            'interpretation': f"{'Normal' if is_normal else 'Non-normal'} distribution (p={'>' if is_normal else '<'}{alpha})"
        }
    
    @staticmethod
    def mann_whitney_u(group1: np.ndarray, group2: np.ndarray, 
                       alternative: str = 'two-sided') -> Dict[str, Any]:
        """
        Mann-Whitney U test for two independent samples (nonparametric).
        
        Args:
            group1: First group samples
            group2: Second group samples
            alternative: 'two-sided', 'less', or 'greater'
            
        Returns:
            Dict with test results including effect size
        """
        statistic, p_value = mannwhitneyu(group1, group2, alternative=alternative)
        
        # Calculate Vargha-Delaney A12 effect size
        a12 = StatisticalTests.vargha_delaney_a12(group1, group2)
        
        # Calculate Cliff's Delta
        delta = StatisticalTests.cliffs_delta(group1, group2)
        
        return {
            'test': 'Mann-Whitney U',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size_a12': a12,
            'effect_size_delta': delta,
            'a12_interpretation': StatisticalTests._interpret_a12(a12),
            'delta_interpretation': StatisticalTests._interpret_delta(delta),
            'alternative': alternative
        }
    
    @staticmethod
    def welch_t_test(group1: np.ndarray, group2: np.ndarray) -> Dict[str, Any]:
        """
        Welch's t-test for two independent samples (parametric, unequal variances).
        
        Args:
            group1: First group samples
            group2: Second group samples
            
        Returns:
            Dict with test results including effect size
        """
        statistic, p_value = stats.ttest_ind(group1, group2, equal_var=False)
        
        # Calculate Cohen's d
        cohens_d = StatisticalTests.cohens_d(group1, group2)
        
        return {
            'test': "Welch's t-test",
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size_cohens_d': cohens_d,
            'd_interpretation': StatisticalTests._interpret_cohens_d(cohens_d)
        }
    
    @staticmethod
    def wilcoxon_signed_rank(group1: np.ndarray, group2: np.ndarray) -> Dict[str, Any]:
        """
        Wilcoxon signed-rank test for paired samples (nonparametric).
        
        Args:
            group1: First group samples (paired)
            group2: Second group samples (paired)
            
        Returns:
            Dict with test results
        """
        if len(group1) != len(group2):
            raise ValueError("Groups must have equal length for paired test")
        
        statistic, p_value = wilcoxon(group1, group2)
        
        # Effect size for paired samples
        differences = group1 - group2
        r = statistic / (len(group1) * (len(group1) + 1) / 2)
        
        return {
            'test': 'Wilcoxon Signed-Rank',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size_r': r,
            'median_difference': np.median(differences)
        }
    
    @staticmethod
    def kruskal_wallis(groups: List[np.ndarray]) -> Dict[str, Any]:
        """
        Kruskal-Wallis H-test for k>2 independent samples (nonparametric).
        
        Args:
            groups: List of sample arrays
            
        Returns:
            Dict with test results
        """
        if len(groups) < 3:
            raise ValueError("Need at least 3 groups for Kruskal-Wallis")
        
        statistic, p_value = kruskal(*groups)
        
        # Calculate eta-squared (effect size)
        n = sum(len(g) for g in groups)
        k = len(groups)
        eta_squared = (statistic - k + 1) / (n - k)
        
        return {
            'test': 'Kruskal-Wallis H',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size_eta_squared': eta_squared,
            'num_groups': k
        }
    
    @staticmethod
    def friedman_test(groups: List[np.ndarray]) -> Dict[str, Any]:
        """
        Friedman test for k>2 related samples (nonparametric, paired across datasets).
        
        Args:
            groups: List of sample arrays (must be same length)
            
        Returns:
            Dict with test results
        """
        if len(groups) < 3:
            raise ValueError("Need at least 3 groups for Friedman test")
        
        if len(set(len(g) for g in groups)) > 1:
            raise ValueError("All groups must have equal length for Friedman test")
        
        statistic, p_value = friedmanchisquare(*groups)
        
        # Calculate Kendall's W (effect size)
        k = len(groups)
        n = len(groups[0])
        w = statistic / (n * (k - 1))
        
        return {
            'test': 'Friedman',
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'effect_size_kendalls_w': w,
            'num_groups': k,
            'num_datasets': n
        }
    
    @staticmethod
    def vargha_delaney_a12(group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Vargha-Delaney A12 effect size.
        
        A12 represents the probability that a random value from group1
        is greater than a random value from group2.
        
        Interpretation:
        - 0.50: No effect
        - 0.56: Small effect
        - 0.64: Medium effect
        - 0.71: Large effect
        
        Args:
            group1: First group samples
            group2: Second group samples
            
        Returns:
            A12 value (0 to 1)
        """
        m = len(group1)
        n = len(group2)
        
        # Count wins
        r = 0
        for a in group1:
            for b in group2:
                if a > b:
                    r += 1
                elif a == b:
                    r += 0.5
        
        a12 = r / (m * n)
        return a12
    
    @staticmethod
    def cliffs_delta(group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Cliff's Delta effect size.
        
        Delta ranges from -1 to 1:
        - Positive: group1 tends to be larger
        - Negative: group2 tends to be larger
        
        Interpretation (|delta|):
        - < 0.147: Negligible
        - < 0.33: Small
        - < 0.474: Medium
        - >= 0.474: Large
        
        Args:
            group1: First group samples
            group2: Second group samples
            
        Returns:
            Delta value (-1 to 1)
        """
        m = len(group1)
        n = len(group2)
        
        # Count wins and losses
        wins = 0
        losses = 0
        for a in group1:
            for b in group2:
                if a > b:
                    wins += 1
                elif a < b:
                    losses += 1
        
        delta = (wins - losses) / (m * n)
        return delta
    
    @staticmethod
    def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
        """
        Cohen's d effect size for parametric tests.
        
        Interpretation (|d|):
        - 0.2: Small effect
        - 0.5: Medium effect
        - 0.8: Large effect
        
        Args:
            group1: First group samples
            group2: Second group samples
            
        Returns:
            Cohen's d value
        """
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        d = (np.mean(group1) - np.mean(group2)) / pooled_std
        return d
    
    @staticmethod
    def bootstrap_ci(data: np.ndarray, statistic=np.mean, 
                     n_bootstrap: int = 10000, ci: float = 0.95) -> Tuple[float, float, float]:
        """
        Bootstrap confidence interval for a statistic.
        
        Args:
            data: Sample data
            statistic: Function to compute (default: mean)
            n_bootstrap: Number of bootstrap samples
            ci: Confidence level (default 0.95)
            
        Returns:
            Tuple of (point_estimate, lower_ci, upper_ci)
        """
        bootstrap_samples = []
        n = len(data)
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_samples.append(statistic(sample))
        
        bootstrap_samples = np.array(bootstrap_samples)
        alpha = 1 - ci
        lower = np.percentile(bootstrap_samples, alpha / 2 * 100)
        upper = np.percentile(bootstrap_samples, (1 - alpha / 2) * 100)
        point = statistic(data)
        
        return point, lower, upper
    
    @staticmethod
    def bootstrap_difference_ci(group1: np.ndarray, group2: np.ndarray,
                                statistic=np.mean, n_bootstrap: int = 10000,
                                ci: float = 0.95) -> Tuple[float, float, float]:
        """
        Bootstrap confidence interval for difference between two groups.
        
        Args:
            group1: First group samples
            group2: Second group samples
            statistic: Function to compute (default: mean)
            n_bootstrap: Number of bootstrap samples
            ci: Confidence level (default 0.95)
            
        Returns:
            Tuple of (difference, lower_ci, upper_ci)
        """
        differences = []
        n1, n2 = len(group1), len(group2)
        
        for _ in range(n_bootstrap):
            sample1 = np.random.choice(group1, size=n1, replace=True)
            sample2 = np.random.choice(group2, size=n2, replace=True)
            differences.append(statistic(sample1) - statistic(sample2))
        
        differences = np.array(differences)
        alpha = 1 - ci
        lower = np.percentile(differences, alpha / 2 * 100)
        upper = np.percentile(differences, (1 - alpha / 2) * 100)
        point = statistic(group1) - statistic(group2)
        
        return point, lower, upper
    
    @staticmethod
    def _interpret_a12(a12: float) -> str:
        """Interpret Vargha-Delaney A12 effect size."""
        abs_diff = abs(a12 - 0.5)
        if abs_diff < 0.06:
            return "Negligible"
        elif abs_diff < 0.14:
            return "Small"
        elif abs_diff < 0.21:
            return "Medium"
        else:
            return "Large"
    
    @staticmethod
    def _interpret_delta(delta: float) -> str:
        """Interpret Cliff's Delta effect size."""
        abs_delta = abs(delta)
        if abs_delta < 0.147:
            return "Negligible"
        elif abs_delta < 0.33:
            return "Small"
        elif abs_delta < 0.474:
            return "Medium"
        else:
            return "Large"
    
    @staticmethod
    def _interpret_cohens_d(d: float) -> str:
        """Interpret Cohen's d effect size."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Negligible"
        elif abs_d < 0.5:
            return "Small"
        elif abs_d < 0.8:
            return "Medium"
        else:
            return "Large"
    
    @staticmethod
    def compare_two_setups(setup1_runs: np.ndarray, setup2_runs: np.ndarray,
                          setup1_name: str = "Setup 1", 
                          setup2_name: str = "Setup 2",
                          metric_name: str = "Fitness",
                          check_assumptions: bool = True) -> Dict[str, Any]:
        """
        Complete statistical comparison for two setups (Scenario A).
        
        Implements the recommended pipeline:
        1. Check normality (optional)
        2. Choose appropriate test (parametric vs nonparametric)
        3. Calculate effect sizes
        4. Bootstrap CIs
        
        Args:
            setup1_runs: Array of results from setup 1
            setup2_runs: Array of results from setup 2
            setup1_name: Name of first setup
            setup2_name: Name of second setup
            metric_name: Name of metric being compared
            check_assumptions: Whether to check normality
            
        Returns:
            Dict with complete comparison results
        """
        results = {
            'setup1_name': setup1_name,
            'setup2_name': setup2_name,
            'metric_name': metric_name,
            'n1': len(setup1_runs),
            'n2': len(setup2_runs)
        }
        
        # Descriptive statistics
        results['descriptive'] = {
            'setup1': {
                'mean': np.mean(setup1_runs),
                'median': np.median(setup1_runs),
                'std': np.std(setup1_runs, ddof=1),
                'min': np.min(setup1_runs),
                'max': np.max(setup1_runs)
            },
            'setup2': {
                'mean': np.mean(setup2_runs),
                'median': np.median(setup2_runs),
                'std': np.std(setup2_runs, ddof=1),
                'min': np.min(setup2_runs),
                'max': np.max(setup2_runs)
            }
        }
        
        # Check normality
        use_nonparametric = False
        if check_assumptions and len(setup1_runs) >= 3 and len(setup2_runs) >= 3:
            norm1 = StatisticalTests.check_normality(setup1_runs)
            norm2 = StatisticalTests.check_normality(setup2_runs)
            results['normality'] = {
                'setup1': norm1,
                'setup2': norm2
            }
            # Use nonparametric if either is non-normal
            use_nonparametric = not (norm1['is_normal'] and norm2['is_normal'])
            results['recommendation'] = 'nonparametric' if use_nonparametric else 'parametric'
        else:
            results['recommendation'] = 'nonparametric (default for EA/GE)'
            use_nonparametric = True
        
        # Perform appropriate test
        if use_nonparametric:
            test_result = StatisticalTests.mann_whitney_u(setup1_runs, setup2_runs)
            results['primary_test'] = test_result
        else:
            test_result = StatisticalTests.welch_t_test(setup1_runs, setup2_runs)
            results['primary_test'] = test_result
        
        # Bootstrap CIs for difference
        diff, lower, upper = StatisticalTests.bootstrap_difference_ci(
            setup1_runs, setup2_runs, statistic=np.mean
        )
        results['bootstrap_ci'] = {
            'difference': diff,
            'ci_lower': lower,
            'ci_upper': upper,
            'ci_level': 0.95
        }
        
        # Summary interpretation
        if test_result['significant']:
            direction = "better" if diff > 0 else "worse"
            results['interpretation'] = (
                f"{setup1_name} performs significantly {direction} than {setup2_name} "
                f"(p={test_result['p_value']:.4f}). "
            )
            if 'effect_size_a12' in test_result:
                results['interpretation'] += (
                    f"Effect size: A₁₂={test_result['effect_size_a12']:.3f} "
                    f"({test_result['a12_interpretation']}), "
                    f"δ={test_result['effect_size_delta']:.3f} "
                    f"({test_result['delta_interpretation']})."
                )
            elif 'effect_size_cohens_d' in test_result:
                results['interpretation'] += (
                    f"Effect size: d={test_result['effect_size_cohens_d']:.3f} "
                    f"({test_result['d_interpretation']})."
                )
        else:
            results['interpretation'] = (
                f"No statistically significant difference between {setup1_name} "
                f"and {setup2_name} (p={test_result['p_value']:.4f})."
            )
        
        return results

