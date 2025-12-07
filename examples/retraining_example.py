"""
Automated Retraining Example

Demonstrates:
1. Setting up automated retraining
2. Performance-based triggers
3. Drift-based triggers
4. Scheduled retraining
5. Automatic model promotion
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
import logging

from retraining.scheduler import (
    RetrainingScheduler,
    RetrainingConfig,
    RetrainingTrigger
)
from training.pipeline import TrainingPipeline
from monitoring.drift_detector import generate_sample_drift_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_training_data(n_samples=2000, drift=0.0):
    """Create synthetic training data with optional drift."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    df['target'] = y

    # Add drift if specified
    if drift > 0:
        df = generate_sample_drift_data(df, drift_magnitude=drift)

    return df


def demo_manual_retraining():
    """Demonstrate manual retraining trigger."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 1: Manual Retraining")
    logger.info("=" * 60)

    # Configuration
    config = RetrainingConfig(
        model_name="demo_model",
        min_training_samples=1000,
        auto_promote=False
    )

    # Data loader
    def data_loader():
        return create_training_data(n_samples=2000)

    # Initialize
    pipeline = TrainingPipeline(
        experiment_name="manual_retraining_demo",
        model_type="sklearn"
    )

    scheduler = RetrainingScheduler(
        config=config,
        data_loader=data_loader,
        training_pipeline=pipeline
    )

    # Initialize monitors
    ref_data = create_training_data(n_samples=1000)
    baseline_metrics = {'accuracy': 0.85, 'f1_score': 0.83, 'precision': 0.84, 'recall': 0.82}
    scheduler.initialize_monitors(ref_data, baseline_metrics)

    # Trigger manual retraining
    logger.info("Triggering manual retraining...")
    job = scheduler.trigger_retraining(
        trigger=RetrainingTrigger.MANUAL,
        reason="User-initiated retraining for testing"
    )

    logger.info(f"\n✓ Retraining completed!")
    logger.info(f"  Job ID: {job.job_id}")
    logger.info(f"  Status: {job.status}")
    logger.info(f"  Duration: {(job.end_time - job.start_time).total_seconds():.1f}s")
    if job.metrics:
        logger.info(f"  New Metrics: {job.metrics}")

    return scheduler


def demo_drift_triggered_retraining():
    """Demonstrate drift-triggered retraining."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 2: Drift-Triggered Retraining")
    logger.info("=" * 60)

    config = RetrainingConfig(
        model_name="drift_sensitive_model",
        drift_threshold=0.1,
        drift_alert_threshold=0.2,
        auto_promote=True,
        promotion_min_improvement=0.01
    )

    def data_loader():
        return create_training_data(n_samples=2000)

    pipeline = TrainingPipeline(
        experiment_name="drift_retraining_demo",
        model_type="sklearn"
    )

    scheduler = RetrainingScheduler(
        config=config,
        data_loader=data_loader,
        training_pipeline=pipeline
    )

    # Initialize with baseline data
    ref_data = create_training_data(n_samples=1000, drift=0.0)
    baseline_metrics = {'accuracy': 0.85, 'f1_score': 0.83}
    scheduler.initialize_monitors(ref_data, baseline_metrics)

    # Simulate production data with drift
    logger.info("Simulating production data with drift...")
    drifted_data = create_training_data(n_samples=500, drift=0.5)

    # Check if retraining needed
    needs_retraining, trigger, reason = scheduler.check_retraining_needed(drifted_data)

    if needs_retraining:
        logger.info(f"\n⚠️  DRIFT DETECTED!")
        logger.info(f"  Trigger: {trigger.value}")
        logger.info(f"  Reason: {reason}")
        logger.info("\nTriggering automatic retraining...")

        job = scheduler.trigger_retraining(
            trigger=trigger,
            reason=reason
        )

        logger.info(f"\n✓ Retraining completed!")
        logger.info(f"  Job ID: {job.job_id}")
        logger.info(f"  Status: {job.status}")
    else:
        logger.info("✓ No drift detected, retraining not needed")

    return scheduler


def demo_performance_triggered_retraining():
    """Demonstrate performance degradation retraining."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 3: Performance-Triggered Retraining")
    logger.info("=" * 60)

    config = RetrainingConfig(
        model_name="performance_monitored_model",
        performance_threshold=0.05,  # 5% degradation
        auto_promote=True
    )

    def data_loader():
        return create_training_data(n_samples=2000)

    pipeline = TrainingPipeline(
        experiment_name="performance_retraining_demo",
        model_type="sklearn"
    )

    scheduler = RetrainingScheduler(
        config=config,
        data_loader=data_loader,
        training_pipeline=pipeline
    )

    # Initialize monitors
    ref_data = create_training_data(n_samples=1000)
    baseline_metrics = {
        'accuracy': 0.85,
        'f1_score': 0.83,
        'precision': 0.84,
        'recall': 0.82
    }
    scheduler.initialize_monitors(ref_data, baseline_metrics)

    # Simulate degraded performance
    logger.info("Simulating degraded model performance...")
    degraded_metrics = {
        'accuracy': 0.78,  # 8.2% degradation
        'f1_score': 0.76,  # 8.4% degradation
        'precision': 0.79,
        'recall': 0.75
    }

    # Check if retraining needed
    current_data = create_training_data(n_samples=200)
    needs_retraining, trigger, reason = scheduler.check_retraining_needed(
        current_data,
        current_metrics=degraded_metrics
    )

    if needs_retraining:
        logger.info(f"\n⚠️  PERFORMANCE DEGRADATION DETECTED!")
        logger.info(f"  Baseline: {baseline_metrics}")
        logger.info(f"  Current: {degraded_metrics}")
        logger.info(f"  Reason: {reason}")
        logger.info("\nTriggering automatic retraining...")

        job = scheduler.trigger_retraining(
            trigger=trigger,
            reason=reason
        )

        logger.info(f"\n✓ Retraining completed!")
        logger.info(f"  Status: {job.status}")
        logger.info(f"  New metrics: {job.metrics}")
    else:
        logger.info("✓ Performance within acceptable range")

    return scheduler


def demo_scheduled_retraining():
    """Demonstrate scheduled retraining."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 4: Scheduled Retraining")
    logger.info("=" * 60)

    config = RetrainingConfig(
        model_name="scheduled_model",
        schedule_enabled=True,
        schedule_interval_days=7,
        schedule_time="02:00",
        auto_promote=True
    )

    def data_loader():
        return create_training_data(n_samples=2000)

    pipeline = TrainingPipeline(
        experiment_name="scheduled_retraining_demo",
        model_type="sklearn"
    )

    scheduler = RetrainingScheduler(
        config=config,
        data_loader=data_loader,
        training_pipeline=pipeline
    )

    # Initialize monitors
    ref_data = create_training_data(n_samples=1000)
    baseline_metrics = {'accuracy': 0.85, 'f1_score': 0.83}
    scheduler.initialize_monitors(ref_data, baseline_metrics)

    # Setup schedule
    logger.info("Setting up scheduled retraining...")
    scheduler.schedule_periodic_retraining()

    logger.info(f"✓ Scheduled retraining configured:")
    logger.info(f"  Interval: Every {config.schedule_interval_days} days")
    logger.info(f"  Time: {config.schedule_time}")
    logger.info(f"  Auto-promote: {config.auto_promote}")

    # Note: In production, call scheduler.start_monitoring() to run in background
    logger.info("\nTo run continuously, call: scheduler.start_monitoring()")

    return scheduler


def demo_end_to_end_workflow():
    """Demonstrate complete end-to-end workflow."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO 5: End-to-End Automated Retraining Workflow")
    logger.info("=" * 60)

    # Step 1: Initial training
    logger.info("\n📊 Step 1: Initial model training...")
    initial_data = create_training_data(n_samples=3000)

    pipeline = TrainingPipeline(
        experiment_name="e2e_workflow",
        model_type="sklearn"
    )

    X_train, X_test, y_train, y_test = pipeline.prepare_data(
        data=initial_data,
        target_column='target'
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    trained_model, initial_metrics = pipeline.train(
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        hyperparameters={'n_estimators': 100},
        tags={'phase': 'initial_training'}
    )

    logger.info(f"✓ Initial model trained: {initial_metrics}")

    # Step 2: Setup automated retraining
    logger.info("\n🔧 Step 2: Setting up automated retraining...")
    config = RetrainingConfig(
        model_name="production_model",
        performance_threshold=0.05,
        drift_threshold=0.1,
        schedule_enabled=True,
        schedule_interval_days=7,
        auto_promote=True,
        promotion_min_improvement=0.01
    )

    def data_loader():
        return create_training_data(n_samples=2000)

    scheduler = RetrainingScheduler(
        config=config,
        data_loader=data_loader,
        training_pipeline=pipeline
    )

    # Initialize monitors with baseline
    ref_data = initial_data[:1000]
    scheduler.initialize_monitors(ref_data, initial_metrics)
    scheduler.schedule_periodic_retraining()

    logger.info("✓ Automated retraining configured")

    # Step 3: Simulate production monitoring
    logger.info("\n👁️  Step 3: Simulating production monitoring...")

    # Scenario 1: Normal data (no retraining)
    logger.info("\nScenario 1: Normal production data...")
    normal_data = create_training_data(n_samples=200, drift=0.0)
    needs_retraining, trigger, reason = scheduler.check_retraining_needed(normal_data)
    logger.info(f"  Retraining needed: {needs_retraining}")

    # Scenario 2: Drifted data (trigger retraining)
    logger.info("\nScenario 2: Data with significant drift...")
    drifted_data = create_training_data(n_samples=200, drift=0.6)
    needs_retraining, trigger, reason = scheduler.check_retraining_needed(drifted_data)
    logger.info(f"  Retraining needed: {needs_retraining}")

    if needs_retraining:
        logger.info(f"  Trigger: {trigger.value}")
        logger.info(f"  Reason: {reason}")

        # Step 4: Execute retraining
        logger.info("\n🚀 Step 4: Executing automated retraining...")
        job = scheduler.trigger_retraining(trigger=trigger, reason=reason)

        logger.info(f"✓ Retraining job completed:")
        logger.info(f"  Job ID: {job.job_id}")
        logger.info(f"  Status: {job.status}")
        logger.info(f"  Duration: {(job.end_time - job.start_time).total_seconds():.1f}s")
        if job.metrics:
            logger.info(f"  New metrics: {job.metrics}")

    # Step 5: Generate report
    logger.info("\n📄 Step 5: Generating summary report...")
    report_path = scheduler.generate_report()
    logger.info(f"✓ Report generated: {report_path}")

    # Summary
    job_history = scheduler.get_job_history()
    logger.info(f"\n📈 Summary:")
    logger.info(f"  Total retraining jobs: {len(job_history)}")
    logger.info(f"  Successful: {sum(1 for j in job_history if j['status'] == 'completed')}")
    logger.info(f"  Failed: {sum(1 for j in job_history if j['status'] == 'failed')}")

    return scheduler


def main():
    """Run all automated retraining demonstrations."""
    logger.info("\n" + "=" * 60)
    logger.info("AUTOMATED RETRAINING - COMPLETE DEMONSTRATIONS")
    logger.info("=" * 60)

    # Run demos
    demo_manual_retraining()
    demo_drift_triggered_retraining()
    demo_performance_triggered_retraining()
    demo_scheduled_retraining()
    demo_end_to_end_workflow()

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("ALL DEMONSTRATIONS COMPLETE! 🎉")
    logger.info("=" * 60)
    logger.info("\nKey Features Demonstrated:")
    logger.info("  ✓ Manual retraining triggers")
    logger.info("  ✓ Drift-based automatic retraining")
    logger.info("  ✓ Performance-based retraining")
    logger.info("  ✓ Scheduled periodic retraining")
    logger.info("  ✓ Automatic model promotion")
    logger.info("  ✓ Job tracking and reporting")
    logger.info("\nNext Steps:")
    logger.info("  1. Integrate with production serving API")
    logger.info("  2. Configure notification channels")
    logger.info("  3. Set up monitoring dashboards")
    logger.info("  4. Define custom retraining policies")
    logger.info("\nGenerated files:")
    logger.info("  - retraining_jobs/*.json (job records)")
    logger.info("  - retraining_jobs/*.md (reports)")


if __name__ == "__main__":
    main()