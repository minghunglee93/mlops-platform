"""
A/B Testing Framework for Model Comparison

Supports:
- Champion/Challenger model testing
- Multi-armed bandit algorithms
- Traffic splitting strategies
- Statistical significance testing
- Automated winner selection
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
from enum import Enum
from scipy import stats

from config import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class TrafficSplitStrategy(Enum):
    """Traffic splitting strategies."""
    FIXED = "fixed"  # Fixed percentage split
    EPSILON_GREEDY = "epsilon_greedy"  # Explore/exploit
    THOMPSON_SAMPLING = "thompson_sampling"  # Bayesian optimization
    UCB = "ucb"  # Upper Confidence Bound


@dataclass
class ModelVariant:
    """Represents a model variant in the experiment."""
    name: str
    model_version: str
    traffic_weight: float = 0.5
    total_requests: int = 0
    successful_predictions: int = 0
    total_reward: float = 0.0
    metrics: Dict[str, float] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_predictions / self.total_requests if self.total_requests > 0 else 0.0
    
    @property
    def average_reward(self) -> float:
        """Calculate average reward."""
        return self.total_reward / self.total_requests if self.total_requests > 0 else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class ABExperiment:
    """
    A/B Testing experiment manager.
    
    Manages multiple model variants, traffic splitting, and performance tracking.
    """
    
    def __init__(
        self,
        experiment_name: str,
        variants: List[ModelVariant],
        strategy: TrafficSplitStrategy = TrafficSplitStrategy.FIXED,
        epsilon: float = 0.1,
        min_samples_per_variant: int = 100,
        confidence_level: float = 0.95
    ):
        """
        Initialize A/B experiment.
        
        Args:
            experiment_name: Name of the experiment
            variants: List of model variants to test
            strategy: Traffic splitting strategy
            epsilon: Exploration rate for epsilon-greedy
            min_samples_per_variant: Minimum samples before statistical tests
            confidence_level: Confidence level for significance tests
        """
        self.experiment_name = experiment_name
        self.variants = {v.name: v for v in variants}
        self.strategy = strategy
        self.epsilon = epsilon
        self.min_samples_per_variant = min_samples_per_variant
        self.confidence_level = confidence_level
        
        self.start_time = datetime.now()
        self.experiment_history = []
        self.experiment_dir = Path("ab_experiments")
        self.experiment_dir.mkdir(exist_ok=True)
        
        logger.info(f"A/B Experiment '{experiment_name}' initialized with {len(variants)} variants")
        logger.info(f"  Strategy: {strategy.value}")
        logger.info(f"  Variants: {[v.name for v in variants]}")
    
    def select_variant(self, context: Optional[Dict] = None) -> str:
        """
        Select a variant based on the strategy.
        
        Args:
            context: Optional context for contextual bandits
            
        Returns:
            Selected variant name
        """
        if self.strategy == TrafficSplitStrategy.FIXED:
            return self._fixed_split()
        elif self.strategy == TrafficSplitStrategy.EPSILON_GREEDY:
            return self._epsilon_greedy()
        elif self.strategy == TrafficSplitStrategy.THOMPSON_SAMPLING:
            return self._thompson_sampling()
        elif self.strategy == TrafficSplitStrategy.UCB:
            return self._ucb()
        else:
            return self._fixed_split()
    
    def _fixed_split(self) -> str:
        """Fixed traffic split based on weights."""
        variant_names = list(self.variants.keys())
        weights = [v.traffic_weight for v in self.variants.values()]
        
        # Normalize weights
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        return np.random.choice(variant_names, p=normalized_weights)
    
    def _epsilon_greedy(self) -> str:
        """Epsilon-greedy exploration/exploitation."""
        if np.random.random() < self.epsilon:
            # Explore: random selection
            return np.random.choice(list(self.variants.keys()))
        else:
            # Exploit: select best performing
            best_variant = max(
                self.variants.values(),
                key=lambda v: v.average_reward
            )
            return best_variant.name
    
    def _thompson_sampling(self) -> str:
        """Thompson Sampling (Bayesian optimization)."""
        samples = {}
        
        for name, variant in self.variants.items():
            # Beta distribution for success rate
            alpha = variant.successful_predictions + 1
            beta = (variant.total_requests - variant.successful_predictions) + 1
            samples[name] = np.random.beta(alpha, beta)
        
        return max(samples.items(), key=lambda x: x[1])[0]
    
    def _ucb(self) -> str:
        """Upper Confidence Bound algorithm."""
        total_requests = sum(v.total_requests for v in self.variants.values())
        
        if total_requests == 0:
            # Random selection if no data
            return np.random.choice(list(self.variants.keys()))
        
        ucb_scores = {}
        for name, variant in self.variants.items():
            if variant.total_requests == 0:
                # Infinite UCB for untested variants
                ucb_scores[name] = float('inf')
            else:
                # UCB formula: mean + sqrt(2 * ln(total) / n)
                exploration_bonus = np.sqrt(2 * np.log(total_requests) / variant.total_requests)
                ucb_scores[name] = variant.average_reward + exploration_bonus
        
        return max(ucb_scores.items(), key=lambda x: x[1])[0]
    
    def record_result(
        self,
        variant_name: str,
        success: bool = True,
        reward: float = 1.0,
        metrics: Optional[Dict[str, float]] = None
    ):
        """
        Record the result of a prediction.
        
        Args:
            variant_name: Name of the variant used
            success: Whether prediction was successful
            reward: Reward value (e.g., user engagement, revenue)
            metrics: Additional metrics to track
        """
        if variant_name not in self.variants:
            logger.warning(f"Unknown variant: {variant_name}")
            return
        
        variant = self.variants[variant_name]
        variant.total_requests += 1
        
        if success:
            variant.successful_predictions += 1
        
        variant.total_reward += reward
        
        if metrics:
            # Update running metrics
            for metric_name, metric_value in metrics.items():
                if metric_name not in variant.metrics:
                    variant.metrics[metric_name] = metric_value
                else:
                    # Running average
                    n = variant.total_requests
                    variant.metrics[metric_name] = (
                        variant.metrics[metric_name] * (n - 1) + metric_value
                    ) / n
        
        # Store in history
        self.experiment_history.append({
            'timestamp': datetime.now().isoformat(),
            'variant': variant_name,
            'success': success,
            'reward': reward,
            'metrics': metrics or {}
        })
    
    def get_results(self) -> Dict[str, Any]:
        """
        Get current experiment results.
        
        Returns:
            Dictionary with experiment results
        """
        results = {
            'experiment_name': self.experiment_name,
            'start_time': self.start_time.isoformat(),
            'duration_hours': (datetime.now() - self.start_time).total_seconds() / 3600,
            'strategy': self.strategy.value,
            'total_requests': sum(v.total_requests for v in self.variants.values()),
            'variants': {}
        }
        
        for name, variant in self.variants.items():
            results['variants'][name] = {
                'model_version': variant.model_version,
                'total_requests': variant.total_requests,
                'success_rate': variant.success_rate,
                'average_reward': variant.average_reward,
                'traffic_share': variant.total_requests / results['total_requests'] if results['total_requests'] > 0 else 0,
                'metrics': variant.metrics
            }
        
        return results
    
    def run_statistical_test(
        self,
        variant_a: str,
        variant_b: str,
        metric: str = 'success_rate'
    ) -> Dict[str, Any]:
        """
        Run statistical significance test between two variants.
        
        Args:
            variant_a: First variant name
            variant_b: Second variant name
            metric: Metric to compare ('success_rate' or 'average_reward')
            
        Returns:
            Dictionary with test results
        """
        if variant_a not in self.variants or variant_b not in self.variants:
            raise ValueError(f"Unknown variant: {variant_a} or {variant_b}")
        
        var_a = self.variants[variant_a]
        var_b = self.variants[variant_b]
        
        # Check minimum sample size
        if var_a.total_requests < self.min_samples_per_variant or \
           var_b.total_requests < self.min_samples_per_variant:
            return {
                'test': 'insufficient_data',
                'significant': False,
                'message': f"Need at least {self.min_samples_per_variant} samples per variant"
            }
        
        if metric == 'success_rate':
            # Two-proportion z-test
            p1 = var_a.success_rate
            p2 = var_b.success_rate
            n1 = var_a.total_requests
            n2 = var_b.total_requests
            
            # Pooled proportion
            p_pool = (var_a.successful_predictions + var_b.successful_predictions) / (n1 + n2)
            
            # Standard error
            se = np.sqrt(p_pool * (1 - p_pool) * (1/n1 + 1/n2))
            
            # Z-score
            z_score = (p1 - p2) / se if se > 0 else 0
            
            # P-value (two-tailed)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            
            result = {
                'test': 'two_proportion_z_test',
                'variant_a': {
                    'name': variant_a,
                    'success_rate': p1,
                    'n': n1
                },
                'variant_b': {
                    'name': variant_b,
                    'success_rate': p2,
                    'n': n2
                },
                'z_score': z_score,
                'p_value': p_value,
                'significant': p_value < (1 - self.confidence_level),
                'confidence_level': self.confidence_level,
                'winner': variant_a if p1 > p2 else variant_b
            }
            
        elif metric == 'average_reward':
            # Two-sample t-test (approximation)
            # In practice, you'd need raw data; this is simplified
            mean_a = var_a.average_reward
            mean_b = var_b.average_reward
            n1 = var_a.total_requests
            n2 = var_b.total_requests
            
            # Simplified t-test (assumes equal variance)
            # For production, use actual sample data
            result = {
                'test': 'two_sample_t_test_approximation',
                'variant_a': {
                    'name': variant_a,
                    'average_reward': mean_a,
                    'n': n1
                },
                'variant_b': {
                    'name': variant_b,
                    'average_reward': mean_b,
                    'n': n2
                },
                'significant': abs(mean_a - mean_b) > 0.05,  # Simplified
                'confidence_level': self.confidence_level,
                'winner': variant_a if mean_a > mean_b else variant_b
            }
        
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        logger.info(f"Statistical test: {variant_a} vs {variant_b}")
        logger.info(f"  P-value: {result.get('p_value', 'N/A')}")
        logger.info(f"  Significant: {result['significant']}")
        logger.info(f"  Winner: {result['winner']}")
        
        return result
    
    def should_promote_challenger(
        self,
        champion: str,
        challenger: str,
        min_improvement: float = 0.01
    ) -> Tuple[bool, str]:
        """
        Determine if challenger should replace champion.
        
        Args:
            champion: Current champion variant
            challenger: Challenger variant
            min_improvement: Minimum improvement required (e.g., 1%)
            
        Returns:
            Tuple of (should_promote, reason)
        """
        # Run statistical test
        test_result = self.run_statistical_test(champion, challenger)
        
        if not test_result['significant']:
            return False, "No statistically significant difference"
        
        # Check if challenger is winner
        if test_result['winner'] != challenger:
            return False, f"Champion ({champion}) still performs better"
        
        # Check minimum improvement
        champ_rate = self.variants[champion].success_rate
        chall_rate = self.variants[challenger].success_rate
        improvement = (chall_rate - champ_rate) / champ_rate if champ_rate > 0 else 0
        
        if improvement < min_improvement:
            return False, f"Improvement ({improvement:.2%}) below threshold ({min_improvement:.2%})"
        
        return True, f"Challenger outperforms by {improvement:.2%}"
    
    def save_results(self):
        """Save experiment results to file."""
        results = self.get_results()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.experiment_dir / f"{self.experiment_name}_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Experiment results saved: {results_file}")
        
        return str(results_file)
    
    def generate_report(self) -> str:
        """
        Generate experiment report.
        
        Returns:
            Path to generated report
        """
        results = self.get_results()
        
        report_lines = [
            f"# A/B Experiment Report: {self.experiment_name}",
            f"\n## Overview",
            f"- **Start Time**: {results['start_time']}",
            f"- **Duration**: {results['duration_hours']:.2f} hours",
            f"- **Strategy**: {results['strategy']}",
            f"- **Total Requests**: {results['total_requests']:,}",
            f"\n## Variants Performance\n"
        ]
        
        for name, variant_results in results['variants'].items():
            report_lines.extend([
                f"### {name}",
                f"- **Model Version**: {variant_results['model_version']}",
                f"- **Requests**: {variant_results['total_requests']:,} ({variant_results['traffic_share']:.1%})",
                f"- **Success Rate**: {variant_results['success_rate']:.2%}",
                f"- **Average Reward**: {variant_results['average_reward']:.4f}",
                ""
            ])
        
        # Statistical comparison
        if len(self.variants) == 2:
            variant_names = list(self.variants.keys())
            test_result = self.run_statistical_test(variant_names[0], variant_names[1])
            
            report_lines.extend([
                f"\n## Statistical Test",
                f"- **Test**: {test_result['test']}",
                f"- **P-value**: {test_result.get('p_value', 'N/A')}",
                f"- **Significant**: {test_result['significant']}",
                f"- **Winner**: {test_result['winner']}",
            ])
        
        report_text = "\n".join(report_lines)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.experiment_dir / f"{self.experiment_name}_report_{timestamp}.md"
        
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Experiment report generated: {report_file}")
        
        return str(report_file)


if __name__ == "__main__":
    # Example usage
    logger.info("Creating A/B experiment...")
    
    # Create variants
    champion = ModelVariant(
        name="champion",
        model_version="v1.0",
        traffic_weight=0.7
    )
    
    challenger = ModelVariant(
        name="challenger",
        model_version="v2.0",
        traffic_weight=0.3
    )
    
    # Initialize experiment
    experiment = ABExperiment(
        experiment_name="model_comparison_v1_v2",
        variants=[champion, challenger],
        strategy=TrafficSplitStrategy.EPSILON_GREEDY,
        epsilon=0.1
    )
    
    # Simulate traffic
    logger.info("Simulating traffic...")
    for i in range(1000):
        variant = experiment.select_variant()
        
        # Simulate prediction (challenger is slightly better)
        if variant == "challenger":
            success = np.random.random() < 0.85
            reward = np.random.normal(1.2, 0.1)
        else:
            success = np.random.random() < 0.80
            reward = np.random.normal(1.0, 0.1)
        
        experiment.record_result(variant, success, reward)
    
    # Get results
    results = experiment.get_results()
    logger.info(f"\nResults: {json.dumps(results, indent=2)}")
    
    # Check if should promote
    should_promote, reason = experiment.should_promote_challenger("champion", "challenger")
    logger.info(f"\nShould promote challenger: {should_promote}")
    logger.info(f"Reason: {reason}")
    
    # Generate report
    report_file = experiment.generate_report()
    logger.info(f"\nReport generated: {report_file}")
