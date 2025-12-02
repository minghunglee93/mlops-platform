"""
Drift Detection Example

Demonstrates:
1. Data drift detection
2. Target drift detection
3. Feature-level drift analysis
4. Automated testing
5. Performance monitoring
6. Alert mechanisms
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import logging

from monitoring.drift_detector import (
    DriftDetector,
    ModelPerformanceMonitor,
    generate_sample_drift_data
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_baseline_data(n_samples=2000):
    """Create baseline training data."""
    logger.info("Creating baseline data...")

    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        random_state=42
    )

    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
    df['target'] = y

    # Add timestamps
    base_time = datetime.now() - timedelta(days=30)
    df['timestamp'] = [base_time + timedelta(hours=i) for i in range(n_samples)]

    logger.info(f"✓ Created {len(df)} baseline samples")
    return df


def simulate_production_data(baseline_df, scenario='no_drift'):
    """Simulate different production scenarios."""
    logger.info(f"Simulating production data: {scenario}")

    if scenario == 'no_drift':
        # Sample from same distribution
        production_df = baseline_df.sample(n=500, replace=True, random_state=123)

    elif scenario == 'feature_drift':
        # Shift feature distributions
        production_df = generate_sample_drift_data(
            baseline_df.sample(n=500, random_state=123),
            drift_magnitude=0.5
        )

    elif scenario == 'target_drift':
        # Change target distribution
        production_df = baseline_df.sample(n=500, random_state=123).copy()
        # Flip 30% of targets
        flip_indices = np.random.choice(
            production_df.index,
            size=int(len(production_df) * 0.3),
            replace=False
        )
        production_df.loc[flip_indices, 'target'] = 1 - production_df.loc[flip_indices, 'target']

    elif scenario == 'severe_drift':
        # Severe feature drift
        production_df = generate_sample_drift_data(
            baseline_df.sample(n=500, random_state=123),
            drift_magnitude=1.5
        )

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    # Update timestamps to recent
    base_time = datetime.now()
    production_df['timestamp'] = [
        base_time + timedelta(hours=i) for i in range(len(production_df))
    ]

    logger.info(f"✓ Generated {len(production_df)} production samples")
    return production_df.reset_index(drop=True)


def train_baseline_model(baseline_df):
    """Train a baseline model."""
    logger.info("Training baseline model...")

    feature_cols = [col for col in baseline_df.columns if col.startswith('feature_')]
    X = baseline_df[feature_cols]
    y = baseline_df['target']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Get baseline metrics
    y_pred = model.predict(X_test)

    baseline_metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred)
    }

    logger.info(f"✓ Model trained. Baseline metrics: {baseline_metrics}")

    return model, baseline_metrics, feature_cols


def demo_data_drift_detection(detector, production_df):
    """Demonstrate data drift detection."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Data Drift Detection")
    logger.info("=" * 60)

    # Detect overall data drift
    drift_summary = detector.detect_data_drift(production_df)

    return drift_summary


def demo_drift_tests(detector, production_df):
    """Demonstrate automated drift testing."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Automated Drift Tests")
    logger.info("=" * 60)

    test_results = detector.run_drift_tests(production_df, max_drift_share=0.3)

    if test_results['all_tests_passed']:
        logger.info("✓ All drift tests PASSED")
    else:
        logger.warning(f"⚠️  {test_results['failed_tests']} tests FAILED")

    return test_results


def demo_target_drift(detector, production_df):
    """Demonstrate target drift detection."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Target Drift Detection")
    logger.info("=" * 60)

    if 'target' in production_df.columns:
        target_drift = detector.detect_target_drift(production_df, 'target')
        logger.info(f"✓ Target drift analysis complete")
        return target_drift
    else:
        logger.warning("No target column in production data")
        return None


def demo_performance_monitoring(
        monitor,
        model,
        production_df,
        feature_cols
):
    """Demonstrate performance monitoring."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Performance Monitoring")
    logger.info("=" * 60)

    # Get predictions
    X_prod = production_df[feature_cols]
    y_pred = model.predict(X_prod)

    # Calculate current metrics (if we have true labels)
    if 'target' in production_df.columns:
        y_true = production_df['target']

        current_metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred)
        }

        logger.info(f"Current metrics: {current_metrics}")

        # Check for degradation
        perf_check = monitor.check_performance(current_metrics)

        if perf_check['alert']:
            logger.warning("⚠️  PERFORMANCE ALERT: Model degradation detected!")
            for metric_name, info in perf_check['degradations'].items():
                if info['alert']:
                    logger.warning(f"  {metric_name}: {info['baseline']:.4f} → "
                                   f"{info['current']:.4f} "
                                   f"({info['degradation_pct']:.2%})")
        else:
            logger.info("✓ Model performance within acceptable range")

        return perf_check
    else:
        logger.warning("Cannot monitor performance without true labels")
        return None


def demo_data_quality(detector, production_df):
    """Demonstrate data quality reporting."""
    logger.info("\n" + "=" * 60)
    logger.info("DEMO: Data Quality Report")
    logger.info("=" * 60)

    report_path = detector.generate_data_quality_report(production_df)
    logger.info(f"✓ Data quality report generated: {report_path}")

    return report_path


def main():
    """Run complete drift detection demonstration."""
    logger.info("\n" + "=" * 60)
    logger.info("DRIFT DETECTION EXAMPLE - COMPLETE WORKFLOW")
    logger.info("=" * 60)

    # Step 1: Create baseline data and train model
    logger.info("\n📊 Step 1: Creating baseline data and training model...")
    baseline_df = create_baseline_data(n_samples=2000)
    model, baseline_metrics, feature_cols = train_baseline_model(baseline_df)

    # Prepare reference data (exclude timestamp for drift detection)
    reference_data = baseline_df.drop(columns=['timestamp'])

    # Step 2: Initialize monitoring
    logger.info("\n🔧 Step 2: Initializing drift detector and performance monitor...")
    detector = DriftDetector(
        reference_data=reference_data,
        drift_threshold=0.1,
        alert_threshold=0.3
    )

    performance_monitor = ModelPerformanceMonitor(baseline_metrics)

    # Step 3: Test different scenarios
    scenarios = [
        ('no_drift', 'No Drift (Baseline)'),
        ('feature_drift', 'Feature Drift'),
        ('target_drift', 'Target Drift'),
        ('severe_drift', 'Severe Drift')
    ]

    for scenario_name, scenario_desc in scenarios:
        logger.info("\n" + "=" * 60)
        logger.info(f"SCENARIO: {scenario_desc}")
        logger.info("=" * 60)

        # Generate production data
        production_df = simulate_production_data(baseline_df, scenario=scenario_name)
        production_data = production_df.drop(columns=['timestamp'])

        # Run all detection methods
        drift_summary = demo_data_drift_detection(detector, production_data)
        test_results = demo_drift_tests(detector, production_data)
        target_drift = demo_target_drift(detector, production_data)
        perf_check = demo_performance_monitoring(
            performance_monitor, model, production_df, feature_cols
        )

        # Generate quality report
        quality_report = demo_data_quality(detector, production_data)

        # Check if alert needed
        if detector.should_alert(drift_summary):
            logger.warning(f"\n🚨 ALERT TRIGGERED for scenario: {scenario_desc}")
        else:
            logger.info(f"\n✓ No alerts for scenario: {scenario_desc}")

        # Log to MLflow
        detector.log_drift_metrics(drift_summary, model_name=f"scenario_{scenario_name}")

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("DRIFT DETECTION DEMO COMPLETE! 🎉")
    logger.info("=" * 60)
    logger.info("\nGenerated reports:")
    logger.info(f"  Location: {detector.drift_reports_dir}")
    logger.info(f"  Reports: {len(list(detector.drift_reports_dir.glob('*.html')))}")
    logger.info("\nNext steps:")
    logger.info("  1. Open HTML reports to visualize drift")
    logger.info("  2. Check MLflow for logged metrics: mlflow ui")
    logger.info("  3. Set up automated monitoring in production")
    logger.info("  4. Configure alerts for drift thresholds")


if __name__ == "__main__":
    main()