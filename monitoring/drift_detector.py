"""
Model and Data Drift Detection using Evidently

Monitors:
- Data drift (distribution changes)
- Concept drift (target distribution changes)
- Model performance degradation
- Feature drift
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import json

from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset, DataQualityPreset
from evidently.metrics import (
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
    ColumnDriftMetric,
    ColumnSummaryMetric
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfDriftedColumns,
    TestShareOfDriftedColumns,
    TestColumnsType,
    TestNumberOfMissingValues
)

from config import settings

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class DriftDetector:
    """
    Detects drift in features and model predictions.

    Supports:
    - Data drift detection
    - Target drift detection
    - Feature-level drift analysis
    - Automated alerting
    """

    def __init__(
            self,
            reference_data: pd.DataFrame,
            drift_threshold: float = 0.1,
            alert_threshold: float = 0.3
    ):
        """
        Initialize drift detector.

        Args:
            reference_data: Baseline/training data for comparison
            drift_threshold: Threshold for individual feature drift (0-1)
            alert_threshold: Threshold for overall drift alert (0-1)
        """
        self.reference_data = reference_data
        self.drift_threshold = drift_threshold
        self.alert_threshold = alert_threshold
        self.drift_reports_dir = Path("drift_reports")
        self.drift_reports_dir.mkdir(exist_ok=True)

        logger.info(f"DriftDetector initialized with {len(reference_data)} reference samples")

    def detect_data_drift(
            self,
            current_data: pd.DataFrame,
            column_mapping: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Detect data drift between reference and current data.

        Args:
            current_data: New production data
            column_mapping: Column type mapping (numerical, categorical, target)

        Returns:
            Dictionary with drift metrics and report path
        """
        logger.info(f"Detecting data drift on {len(current_data)} samples...")

        # Create drift report
        data_drift_report = Report(metrics=[
            DataDriftPreset(drift_share=self.drift_threshold),
        ])

        data_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.drift_reports_dir / f"data_drift_{timestamp}.html"
        data_drift_report.save_html(str(report_path))

        # Extract metrics
        report_dict = data_drift_report.as_dict()
        metrics = report_dict['metrics'][0]['result']

        drift_summary = {
            'timestamp': timestamp,
            'dataset_drift': metrics.get('dataset_drift', False),
            'drift_share': metrics.get('drift_share', 0.0),
            'number_of_drifted_columns': metrics.get('number_of_drifted_columns', 0),
            'report_path': str(report_path),
            'drift_by_columns': metrics.get('drift_by_columns', {})
        }

        logger.info(f"Data drift analysis complete:")
        logger.info(f"  Dataset drift detected: {drift_summary['dataset_drift']}")
        logger.info(f"  Drift share: {drift_summary['drift_share']:.2%}")
        logger.info(f"  Drifted columns: {drift_summary['number_of_drifted_columns']}")
        logger.info(f"  Report saved: {report_path}")

        return drift_summary

    def detect_target_drift(
            self,
            current_data: pd.DataFrame,
            target_column: str
    ) -> Dict[str, Any]:
        """
        Detect drift in target variable distribution.

        Args:
            current_data: New production data with target
            target_column: Name of target column

        Returns:
            Dictionary with target drift metrics
        """
        logger.info(f"Detecting target drift for column: {target_column}")

        # Create target drift report
        target_drift_report = Report(metrics=[
            TargetDriftPreset(),
        ])

        column_mapping = {'target': target_column}

        target_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data,
            column_mapping=column_mapping
        )

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.drift_reports_dir / f"target_drift_{timestamp}.html"
        target_drift_report.save_html(str(report_path))

        # Extract metrics
        report_dict = target_drift_report.as_dict()

        drift_summary = {
            'timestamp': timestamp,
            'report_path': str(report_path)
        }

        logger.info(f"Target drift analysis complete")
        logger.info(f"  Report saved: {report_path}")

        return drift_summary

    def detect_column_drift(
            self,
            current_data: pd.DataFrame,
            column_name: str
    ) -> Dict[str, Any]:
        """
        Detect drift for a specific column.

        Args:
            current_data: New production data
            column_name: Column to analyze

        Returns:
            Dictionary with column drift metrics
        """
        logger.info(f"Detecting drift for column: {column_name}")

        # Create column drift report
        column_drift_report = Report(metrics=[
            ColumnDriftMetric(column_name=column_name),
            ColumnSummaryMetric(column_name=column_name)
        ])

        column_drift_report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )

        # Extract metrics
        report_dict = column_drift_report.as_dict()
        drift_metric = report_dict['metrics'][0]['result']

        drift_summary = {
            'column': column_name,
            'drift_detected': drift_metric.get('drift_detected', False),
            'drift_score': drift_metric.get('drift_score', 0.0),
            'stattest_name': drift_metric.get('stattest_name', 'unknown')
        }

        logger.info(f"Column '{column_name}' drift: {drift_summary['drift_detected']} "
                    f"(score: {drift_summary['drift_score']:.4f})")

        return drift_summary

    def run_drift_tests(
            self,
            current_data: pd.DataFrame,
            max_drift_share: float = 0.3
    ) -> Dict[str, Any]:
        """
        Run automated drift tests with pass/fail criteria.

        Args:
            current_data: New production data
            max_drift_share: Maximum allowed share of drifted columns

        Returns:
            Dictionary with test results
        """
        logger.info("Running drift test suite...")

        # Create test suite
        drift_tests = TestSuite(tests=[
            TestNumberOfDriftedColumns(),
            TestShareOfDriftedColumns(lt=max_drift_share),
            TestColumnsType(),
            TestNumberOfMissingValues()
        ])

        drift_tests.run(
            reference_data=self.reference_data,
            current_data=current_data
        )

        # Save test results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.drift_reports_dir / f"drift_tests_{timestamp}.html"
        drift_tests.save_html(str(report_path))

        # Extract results
        test_results = drift_tests.as_dict()

        summary = {
            'timestamp': timestamp,
            'all_tests_passed': test_results['summary']['all_passed'],
            'total_tests': test_results['summary']['total_tests'],
            'success_tests': test_results['summary']['success_tests'],
            'failed_tests': test_results['summary']['failed_tests'],
            'report_path': str(report_path)
        }

        logger.info(f"Drift tests complete:")
        logger.info(f"  All tests passed: {summary['all_tests_passed']}")
        logger.info(f"  Success: {summary['success_tests']}/{summary['total_tests']}")
        logger.info(f"  Report saved: {report_path}")

        return summary

    def generate_data_quality_report(
            self,
            current_data: pd.DataFrame
    ) -> str:
        """
        Generate comprehensive data quality report.

        Args:
            current_data: New production data

        Returns:
            Path to generated report
        """
        logger.info("Generating data quality report...")

        # Create quality report
        quality_report = Report(metrics=[
            DataQualityPreset(),
        ])

        quality_report.run(
            reference_data=self.reference_data,
            current_data=current_data
        )

        # Save report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.drift_reports_dir / f"data_quality_{timestamp}.html"
        quality_report.save_html(str(report_path))

        logger.info(f"Data quality report saved: {report_path}")

        return str(report_path)

    def should_alert(self, drift_summary: Dict[str, Any]) -> bool:
        """
        Determine if drift warrants an alert.

        Args:
            drift_summary: Drift detection results

        Returns:
            True if alert should be triggered
        """
        drift_share = drift_summary.get('drift_share', 0.0)
        dataset_drift = drift_summary.get('dataset_drift', False)

        # Alert if drift exceeds threshold or dataset drift detected
        should_alert = drift_share >= self.alert_threshold or dataset_drift

        if should_alert:
            logger.warning(f"⚠️  DRIFT ALERT: drift_share={drift_share:.2%}, "
                           f"dataset_drift={dataset_drift}")

        return should_alert

    def log_drift_metrics(
            self,
            drift_summary: Dict[str, Any],
            model_name: str = "default"
    ):
        """
        Log drift metrics to MLflow.

        Args:
            drift_summary: Drift detection results
            model_name: Name of the model being monitored
        """
        try:
            import mlflow

            with mlflow.start_run(run_name=f"drift_monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.set_tag("model_name", model_name)
                mlflow.set_tag("monitoring_type", "drift_detection")

                mlflow.log_metric("drift_share", drift_summary.get('drift_share', 0.0))
                mlflow.log_metric("number_of_drifted_columns",
                                  drift_summary.get('number_of_drifted_columns', 0))
                mlflow.log_param("dataset_drift", drift_summary.get('dataset_drift', False))

                # Log report as artifact
                if 'report_path' in drift_summary:
                    mlflow.log_artifact(drift_summary['report_path'])

                logger.info("Drift metrics logged to MLflow")

        except Exception as e:
            logger.warning(f"Could not log to MLflow: {e}")


class ModelPerformanceMonitor:
    """
    Monitors model performance over time to detect degradation.
    """

    def __init__(self, baseline_metrics: Dict[str, float]):
        """
        Initialize performance monitor.

        Args:
            baseline_metrics: Reference metrics from validation/test set
        """
        self.baseline_metrics = baseline_metrics
        self.performance_history = []

        logger.info(f"Performance monitor initialized with baseline: {baseline_metrics}")

    def check_performance(
            self,
            current_metrics: Dict[str, float],
            degradation_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Check if current performance has degraded significantly.

        Args:
            current_metrics: Current model metrics
            degradation_threshold: Threshold for acceptable degradation

        Returns:
            Dictionary with performance analysis
        """
        logger.info("Checking model performance...")

        degradations = {}
        alert = False

        for metric_name, baseline_value in self.baseline_metrics.items():
            if metric_name in current_metrics:
                current_value = current_metrics[metric_name]
                degradation = baseline_value - current_value
                degradation_pct = degradation / baseline_value if baseline_value != 0 else 0

                degradations[metric_name] = {
                    'baseline': baseline_value,
                    'current': current_value,
                    'degradation': degradation,
                    'degradation_pct': degradation_pct,
                    'alert': abs(degradation_pct) > degradation_threshold
                }

                if abs(degradation_pct) > degradation_threshold:
                    alert = True
                    logger.warning(f"⚠️  Performance degradation in {metric_name}: "
                                   f"{baseline_value:.4f} → {current_value:.4f} "
                                   f"({degradation_pct:.2%})")

        # Store in history
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': current_metrics,
            'degradations': degradations,
            'alert': alert
        })

        return {
            'alert': alert,
            'degradations': degradations,
            'timestamp': datetime.now().isoformat()
        }

    def get_performance_trend(self, metric_name: str) -> List[float]:
        """
        Get historical trend for a specific metric.

        Args:
            metric_name: Name of the metric

        Returns:
            List of metric values over time
        """
        return [
            entry['metrics'].get(metric_name, 0.0)
            for entry in self.performance_history
        ]


def generate_sample_drift_data(
        reference_data: pd.DataFrame,
        drift_magnitude: float = 0.5
) -> pd.DataFrame:
    """
    Generate synthetic drifted data for testing.

    Args:
        reference_data: Original data
        drift_magnitude: How much drift to introduce (0-1)

    Returns:
        Drifted dataframe
    """
    drifted_data = reference_data.copy()

    # Add drift to numerical columns
    numeric_cols = drifted_data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if col != 'target':  # Don't drift the target
            mean_shift = drifted_data[col].std() * drift_magnitude
            drifted_data[col] = drifted_data[col] + mean_shift

    logger.info(f"Generated drifted data with magnitude {drift_magnitude}")

    return drifted_data


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification

    # Generate sample data
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    reference_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
    reference_df['target'] = y

    # Initialize detector
    detector = DriftDetector(reference_df)

    # Generate drifted data
    current_df = generate_sample_drift_data(reference_df, drift_magnitude=0.3)

    # Detect drift
    drift_summary = detector.detect_data_drift(current_df)

    # Run tests
    test_results = detector.run_drift_tests(current_df)

    # Check if alert needed
    if detector.should_alert(drift_summary):
        print("🚨 DRIFT ALERT: Significant drift detected!")

    print(f"\nReports generated in: {detector.drift_reports_dir}")