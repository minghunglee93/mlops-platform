"""
A/B Testing Example

Demonstrates:
1. Champion/Challenger model comparison
2. Multiple traffic splitting strategies
3. Statistical significance testing
4. Automated winner selection
5. Multi-armed bandit algorithms
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from datetime import datetime
import logging
import json

from ab_testing.experiment import (
    ABExperiment,
    ModelVariant,
    TrafficSplitStrategy
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simulate_model_performance(model_name: str, n_samples: int = 1) -> tuple:
    """
    Simulate model performance with different characteristics.

    Args:
        model_name: Name of the model variant
        n_samples: Number of predictions to simulate

    Returns:
        Tuple of (successes, rewards)
    """
    # Different models have different performance profiles
    if model_name == "champion_v1":
        # Baseline model: 80% success, reward ~1.0
        success_rate = 0.80
        reward_mean = 1.0
    elif model_name == "challenger_v2":
        # Better model: 85% success, reward ~1.2
        success_rate = 0.85
        reward_mean = 1.2
    elif model_name == "challenger_v3":
        # Even better: 90% success, reward ~1.5
        success_rate = 0.90
        reward_mean = 1.5
    elif model_name == "bad_model":
        # Worse model: 70% success, reward ~0.8
        success_rate = 0.70
        reward_mean = 0.8
    else:
        success_rate = 0.75
        reward_mean = 1.0

    successes = np.random.binomial(n_samples, success_rate)
    rewards = np.random.normal(reward_mean, 0.1, n_samples)

    return successes, rewards


def demo_fixed_split():
    """Demonstrate fixed traffic split A/B test."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 1: Fixed Traffic Split (70/30)")
    logger.info("=" * 60)

    # Create variants
    champion = ModelVariant(
        name="champion_v1",
        model_version="1.0.0",
        traffic_weight=0.7
    )

    challenger = ModelVariant(
        name="challenger_v2",
        model_version="2.0.0",
        traffic_weight=0.3
    )

    # Create experiment
    experiment = ABExperiment(
        experiment_name="fixed_split_test",
        variants=[champion, challenger],
        strategy=TrafficSplitStrategy.FIXED,
        min_samples_per_variant=100
    )

    # Simulate traffic
    logger.info("Simulating 1000 requests...")
    for i in range(1000):
        variant_name = experiment.select_variant()
        successes, rewards = simulate_model_performance(variant_name, 1)

        experiment.record_result(
            variant_name=variant_name,
            success=successes > 0,
            reward=rewards[0]
        )

    # Get results
    results = experiment.get_results()
    logger.info("\nResults:")
    for name, metrics in results['variants'].items():
        logger.info(f"  {name}:")
        logger.info(f"    Requests: {metrics['total_requests']}")
        logger.info(f"    Success Rate: {metrics['success_rate']:.2%}")
        logger.info(f"    Avg Reward: {metrics['average_reward']:.3f}")

    # Statistical test
    test_result = experiment.run_statistical_test("champion_v1", "challenger_v2")
    logger.info(f"\nStatistical Test:")
    logger.info(f"  P-value: {test_result.get('p_value', 'N/A')}")
    logger.info(f"  Significant: {test_result['significant']}")
    logger.info(f"  Winner: {test_result['winner']}")

    # Save results
    experiment.save_results()
    experiment.generate_report()

    return experiment


def demo_epsilon_greedy():
    """Demonstrate epsilon-greedy strategy."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 2: Epsilon-Greedy Strategy")
    logger.info("=" * 60)

    # Create variants
    variants = [
        ModelVariant(name="champion_v1", model_version="1.0.0"),
        ModelVariant(name="challenger_v2", model_version="2.0.0"),
    ]

    # Create experiment with epsilon-greedy
    experiment = ABExperiment(
        experiment_name="epsilon_greedy_test",
        variants=variants,
        strategy=TrafficSplitStrategy.EPSILON_GREEDY,
        epsilon=0.1  # 10% exploration
    )

    # Simulate traffic
    logger.info("Simulating 1000 requests with exploration...")
    for i in range(1000):
        variant_name = experiment.select_variant()
        successes, rewards = simulate_model_performance(variant_name, 1)

        experiment.record_result(
            variant_name=variant_name,
            success=successes > 0,
            reward=rewards[0]
        )

        # Log progress
        if (i + 1) % 250 == 0:
            results = experiment.get_results()
            logger.info(f"\nAfter {i + 1} requests:")
            for name, metrics in results['variants'].items():
                logger.info(f"  {name}: {metrics['total_requests']} requests, "
                            f"{metrics['success_rate']:.2%} success")

    # Final results
    results = experiment.get_results()
    logger.info("\nFinal Results:")
    for name, metrics in results['variants'].items():
        logger.info(f"  {name}:")
        logger.info(f"    Requests: {metrics['total_requests']}")
        logger.info(f"    Traffic Share: {metrics['traffic_share']:.2%}")
        logger.info(f"    Success Rate: {metrics['success_rate']:.2%}")

    return experiment


def demo_thompson_sampling():
    """Demonstrate Thompson Sampling."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 3: Thompson Sampling (Bayesian Optimization)")
    logger.info("=" * 60)

    # Create three variants
    variants = [
        ModelVariant(name="champion_v1", model_version="1.0.0"),
        ModelVariant(name="challenger_v2", model_version="2.0.0"),
        ModelVariant(name="challenger_v3", model_version="3.0.0"),
    ]

    # Create experiment
    experiment = ABExperiment(
        experiment_name="thompson_sampling_test",
        variants=variants,
        strategy=TrafficSplitStrategy.THOMPSON_SAMPLING
    )

    # Simulate traffic
    logger.info("Simulating 1500 requests with Bayesian optimization...")
    for i in range(1500):
        variant_name = experiment.select_variant()
        successes, rewards = simulate_model_performance(variant_name, 1)

        experiment.record_result(
            variant_name=variant_name,
            success=successes > 0,
            reward=rewards[0]
        )

        if (i + 1) % 500 == 0:
            results = experiment.get_results()
            logger.info(f"\nAfter {i + 1} requests:")
            for name, metrics in results['variants'].items():
                logger.info(f"  {name}: {metrics['traffic_share']:.1%} traffic, "
                            f"{metrics['success_rate']:.2%} success")

    # Final results
    results = experiment.get_results()
    logger.info("\nFinal Traffic Distribution:")
    for name, metrics in results['variants'].items():
        logger.info(f"  {name}: {metrics['traffic_share']:.1%}")

    return experiment


def demo_promotion_decision():
    """Demonstrate automated promotion decision."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 4: Automated Promotion Decision")
    logger.info("=" * 60)

    # Create variants
    variants = [
        ModelVariant(name="champion", model_version="1.0.0"),
        ModelVariant(name="challenger", model_version="2.0.0"),
    ]

    # Create experiment
    experiment = ABExperiment(
        experiment_name="promotion_test",
        variants=variants,
        strategy=TrafficSplitStrategy.FIXED,
        min_samples_per_variant=200
    )

    # Simulate traffic
    logger.info("Collecting data for promotion decision...")
    for i in range(1000):
        variant_name = experiment.select_variant()
        successes, rewards = simulate_model_performance(variant_name, 1)

        experiment.record_result(
            variant_name=variant_name,
            success=successes > 0,
            reward=rewards[0]
        )

    # Check promotion
    should_promote, reason = experiment.should_promote_challenger(
        champion="champion",
        challenger="challenger",
        min_improvement=0.02  # Require 2% improvement
    )

    logger.info(f"\nPromotion Decision:")
    logger.info(f"  Should Promote: {should_promote}")
    logger.info(f"  Reason: {reason}")

    if should_promote:
        logger.info("\n✓ RECOMMENDATION: Promote challenger to production")
    else:
        logger.info("\n✗ RECOMMENDATION: Keep current champion")

    return experiment


def demo_multi_variant():
    """Demonstrate multi-variant testing."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 5: Multi-Variant Testing (4 models)")
    logger.info("=" * 60)

    # Create four variants
    variants = [
        ModelVariant(name="champion_v1", model_version="1.0.0"),
        ModelVariant(name="challenger_v2", model_version="2.0.0"),
        ModelVariant(name="challenger_v3", model_version="3.0.0"),
        ModelVariant(name="bad_model", model_version="0.5.0"),
    ]

    # Use UCB strategy
    experiment = ABExperiment(
        experiment_name="multi_variant_test",
        variants=variants,
        strategy=TrafficSplitStrategy.UCB
    )

    logger.info("Running UCB algorithm to find best model...")
    for i in range(2000):
        variant_name = experiment.select_variant()
        successes, rewards = simulate_model_performance(variant_name, 1)

        experiment.record_result(
            variant_name=variant_name,
            success=successes > 0,
            reward=rewards[0]
        )

        if (i + 1) % 500 == 0:
            results = experiment.get_results()
            logger.info(f"\nAfter {i + 1} requests:")
            sorted_variants = sorted(
                results['variants'].items(),
                key=lambda x: x[1]['average_reward'],
                reverse=True
            )
            for rank, (name, metrics) in enumerate(sorted_variants, 1):
                logger.info(f"  #{rank} {name}: "
                            f"{metrics['traffic_share']:.1%} traffic, "
                            f"{metrics['average_reward']:.3f} reward")

    # Final ranking
    results = experiment.get_results()
    logger.info("\nFinal Model Ranking:")
    sorted_variants = sorted(
        results['variants'].items(),
        key=lambda x: x[1]['average_reward'],
        reverse=True
    )
    for rank, (name, metrics) in enumerate(sorted_variants, 1):
        logger.info(f"  #{rank} {name}:")
        logger.info(f"      Success Rate: {metrics['success_rate']:.2%}")
        logger.info(f"      Avg Reward: {metrics['average_reward']:.3f}")
        logger.info(f"      Traffic Share: {metrics['traffic_share']:.1%}")

    return experiment


def main():
    """Run all A/B testing demonstrations."""
    logger.info("\n" + "=" * 60)
    logger.info("A/B TESTING FRAMEWORK - COMPLETE DEMONSTRATIONS")
    logger.info("=" * 60)

    # Run all demos
    demo_fixed_split()
    demo_epsilon_greedy()
    demo_thompson_sampling()
    demo_promotion_decision()
    demo_multi_variant()

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("ALL DEMONSTRATIONS COMPLETE! 🎉")
    logger.info("=" * 60)
    logger.info("\nGenerated reports in: ab_experiments/")
    logger.info("\nKey Takeaways:")
    logger.info("  1. Fixed split: Simple, predictable traffic distribution")
    logger.info("  2. Epsilon-greedy: Balances exploration and exploitation")
    logger.info("  3. Thompson Sampling: Bayesian optimization, adapts quickly")
    logger.info("  4. Promotion: Automated decision-making with statistical tests")
    logger.info("  5. Multi-variant: Compare many models simultaneously")
    logger.info("\nNext Steps:")
    logger.info("  1. Integrate with production serving API")
    logger.info("  2. Set up automated promotion pipelines")
    logger.info("  3. Configure monitoring dashboards")
    logger.info("  4. Define business metrics as rewards")


if __name__ == "__main__":
    main()