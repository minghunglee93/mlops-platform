"""
Automated Model Retraining Scheduler

Monitors model performance and triggers retraining based on:
- Performance degradation
- Data drift detection
- Scheduled intervals
- Manual triggers
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Callable
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import json
from enum import Enum
import threading
import time
import schedule

from config import settings
from monitoring.drift_detector import DriftDetector, ModelPerformanceMonitor
from training.pipeline import TrainingPipeline
from registry.model_registry import ModelRegistry

logging.basicConfig(level=settings.LOG_LEVEL)
logger = logging.getLogger(__name__)


class RetrainingTrigger(Enum):
    """Reasons for triggering retraining."""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    DATA_DRIFT = "data_drift"
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    CONCEPT_DRIFT = "concept_drift"


@dataclass
class RetrainingConfig:
    """Configuration for automated retraining."""
    model_name: str
    
    # Performance thresholds
    performance_threshold: float = 0.05  # 5% degradation
    min_samples_for_check: int = 100
    
    # Drift thresholds
    drift_threshold: float = 0.1
    drift_alert_threshold: float = 0.3
    
    # Scheduling
    schedule_enabled: bool = False
    schedule_interval_days: int = 7
    schedule_day_of_week: Optional[str] = None  # "monday", "tuesday", etc.
    schedule_time: str = "02:00"  # 2 AM default
    
    # Data requirements
    min_training_samples: int = 1000
    training_data_window_days: int = 30
    
    # Model promotion
    auto_promote: bool = False
    promotion_min_improvement: float = 0.01  # 1%
    promotion_stage: str = "Staging"
    
    # Notifications
    enable_notifications: bool = True
    notification_email: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class RetrainingJob:
    """Represents a retraining job."""
    job_id: str
    model_name: str
    trigger: RetrainingTrigger
    trigger_reason: str
    start_time: datetime
    status: str = "pending"  # pending, running, completed, failed
    end_time: Optional[datetime] = None
    new_model_version: Optional[str] = None
    metrics: Optional[Dict[str, float]] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'trigger': self.trigger.value
        }


class RetrainingScheduler:
    """
    Manages automated model retraining.
    
    Features:
    - Performance monitoring
    - Drift detection
    - Scheduled retraining
    - Automatic model promotion
    """
    
    def __init__(
        self,
        config: RetrainingConfig,
        data_loader: Callable[[], pd.DataFrame],
        training_pipeline: Optional[TrainingPipeline] = None
    ):
        """
        Initialize retraining scheduler.
        
        Args:
            config: Retraining configuration
            data_loader: Function that returns training data
            training_pipeline: Optional custom training pipeline
        """
        self.config = config
        self.data_loader = data_loader
        self.training_pipeline = training_pipeline
        
        self.registry = ModelRegistry()
        self.drift_detector: Optional[DriftDetector] = None
        self.performance_monitor: Optional[ModelPerformanceMonitor] = None
        
        self.jobs_dir = Path("retraining_jobs")
        self.jobs_dir.mkdir(exist_ok=True)
        
        self.job_history: List[RetrainingJob] = []
        self.monitoring_thread: Optional[threading.Thread] = None
        self.is_monitoring = False
        
        logger.info(f"RetrainingScheduler initialized for model: {config.model_name}")
    
    def initialize_monitors(
        self,
        reference_data: pd.DataFrame,
        baseline_metrics: Dict[str, float]
    ):
        """
        Initialize drift detector and performance monitor.
        
        Args:
            reference_data: Baseline data for drift detection
            baseline_metrics: Baseline performance metrics
        """
        self.drift_detector = DriftDetector(
            reference_data=reference_data,
            drift_threshold=self.config.drift_threshold,
            alert_threshold=self.config.drift_alert_threshold
        )
        
        self.performance_monitor = ModelPerformanceMonitor(
            baseline_metrics=baseline_metrics
        )
        
        logger.info("Monitors initialized successfully")
    
    def check_retraining_needed(
        self,
        current_data: pd.DataFrame,
        current_metrics: Optional[Dict[str, float]] = None
    ) -> tuple[bool, RetrainingTrigger, str]:
        """
        Check if retraining is needed.
        
        Args:
            current_data: Recent production data
            current_metrics: Current model performance metrics
            
        Returns:
            Tuple of (needs_retraining, trigger, reason)
        """
        # Check data drift
        if self.drift_detector and len(current_data) >= self.config.min_samples_for_check:
            drift_summary = self.drift_detector.detect_data_drift(current_data)
            
            if self.drift_detector.should_alert(drift_summary):
                reason = (f"Data drift detected: {drift_summary['drift_share']:.2%} features drifted, "
                         f"{drift_summary['number_of_drifted_columns']} columns affected")
                return True, RetrainingTrigger.DATA_DRIFT, reason
        
        # Check performance degradation
        if self.performance_monitor and current_metrics:
            perf_check = self.performance_monitor.check_performance(
                current_metrics,
                degradation_threshold=self.config.performance_threshold
            )
            
            if perf_check['alert']:
                degraded_metrics = [
                    f"{name}: {info['degradation_pct']:.2%}"
                    for name, info in perf_check['degradations'].items()
                    if info['alert']
                ]
                reason = f"Performance degradation detected: {', '.join(degraded_metrics)}"
                return True, RetrainingTrigger.PERFORMANCE_DEGRADATION, reason
        
        return False, None, ""
    
    def trigger_retraining(
        self,
        trigger: RetrainingTrigger,
        reason: str,
        custom_data: Optional[pd.DataFrame] = None
    ) -> RetrainingJob:
        """
        Trigger a retraining job.
        
        Args:
            trigger: Reason for retraining
            reason: Detailed explanation
            custom_data: Optional custom training data
            
        Returns:
            RetrainingJob instance
        """
        job_id = f"{self.config.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        job = RetrainingJob(
            job_id=job_id,
            model_name=self.config.model_name,
            trigger=trigger,
            trigger_reason=reason,
            start_time=datetime.now(),
            status="running"
        )
        
        self.job_history.append(job)
        
        logger.info(f"Retraining triggered: {trigger.value}")
        logger.info(f"  Reason: {reason}")
        logger.info(f"  Job ID: {job_id}")
        
        try:
            # Load training data
            if custom_data is not None:
                training_data = custom_data
            else:
                training_data = self.data_loader()
            
            if len(training_data) < self.config.min_training_samples:
                raise ValueError(
                    f"Insufficient training data: {len(training_data)} < {self.config.min_training_samples}"
                )
            
            logger.info(f"Loaded {len(training_data)} training samples")
            
            # Execute retraining
            new_model, metrics = self._execute_retraining(training_data)
            
            # Update job status
            job.status = "completed"
            job.end_time = datetime.now()
            job.metrics = metrics
            job.new_model_version = getattr(new_model, 'version', 'unknown')
            
            logger.info(f"Retraining completed successfully")
            logger.info(f"  New metrics: {metrics}")
            
            # Auto-promote if configured
            if self.config.auto_promote:
                self._attempt_promotion(job)
            
            # Save job record
            self._save_job(job)
            
            # Send notification
            if self.config.enable_notifications:
                self._send_notification(job)
            
            return job
            
        except Exception as e:
            job.status = "failed"
            job.end_time = datetime.now()
            job.error = str(e)
            
            logger.error(f"Retraining failed: {e}")
            self._save_job(job)
            
            if self.config.enable_notifications:
                self._send_notification(job)
            
            raise
    
    def _execute_retraining(
        self,
        training_data: pd.DataFrame
    ) -> tuple[Any, Dict[str, float]]:
        """
        Execute the actual retraining process.
        
        Args:
            training_data: Training dataset
            
        Returns:
            Tuple of (trained_model, metrics)
        """
        if self.training_pipeline is None:
            raise ValueError("No training pipeline configured")
        
        # Prepare data
        target_col = 'target'  # Should be configurable
        X_train, X_test, y_train, y_test = self.training_pipeline.prepare_data(
            data=training_data,
            target_column=target_col
        )
        
        # Load model architecture (use same as previous version)
        # In practice, you'd load the model class/config from registry
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Train
        trained_model, metrics = self.training_pipeline.train(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            hyperparameters={"n_estimators": 100},
            tags={
                "retraining": "automated",
                "trigger": "scheduler",
                "original_model": self.config.model_name
            }
        )
        
        return trained_model, metrics
    
    def _attempt_promotion(self, job: RetrainingJob):
        """
        Attempt to promote new model if it meets criteria.
        
        Args:
            job: Completed retraining job
        """
        if not job.metrics or job.status != "completed":
            logger.warning("Cannot promote: job not completed or no metrics")
            return
        
        # Get baseline metrics
        baseline_metrics = self.performance_monitor.baseline_metrics
        
        # Compare metrics
        improvements = {}
        for metric_name, baseline_value in baseline_metrics.items():
            if metric_name in job.metrics:
                new_value = job.metrics[metric_name]
                improvement = (new_value - baseline_value) / baseline_value if baseline_value != 0 else 0
                improvements[metric_name] = improvement
        
        # Check if improvement threshold met
        avg_improvement = np.mean(list(improvements.values()))
        
        if avg_improvement >= self.config.promotion_min_improvement:
            logger.info(f"Promoting new model (avg improvement: {avg_improvement:.2%})")
            
            # Register and promote
            # In practice, get run_id from training pipeline
            run_id = "placeholder_run_id"
            
            metadata = self.registry.register_model(
                model_name=f"{self.config.model_name}_retrained",
                run_id=run_id,
                description=f"Automated retraining ({job.trigger.value})",
                tags={"automated": "true", "trigger": job.trigger.value}
            )
            
            self.registry.promote_model(
                model_name=metadata.name,
                version=metadata.version,
                stage=self.config.promotion_stage
            )
            
            logger.info(f"Model promoted to {self.config.promotion_stage}")
        else:
            logger.info(f"New model not promoted (improvement {avg_improvement:.2%} < threshold)")
    
    def _save_job(self, job: RetrainingJob):
        """Save job record to file."""
        job_file = self.jobs_dir / f"{job.job_id}.json"
        with open(job_file, 'w') as f:
            json.dump(job.to_dict(), f, indent=2)
    
    def _send_notification(self, job: RetrainingJob):
        """Send notification about job status."""
        # Placeholder for notification logic
        # In production, integrate with email/Slack/PagerDuty
        logger.info(f"NOTIFICATION: Retraining job {job.job_id} - Status: {job.status}")
    
    def schedule_periodic_retraining(self):
        """Setup scheduled periodic retraining."""
        if not self.config.schedule_enabled:
            logger.info("Scheduled retraining is disabled")
            return
        
        if self.config.schedule_day_of_week:
            # Weekly schedule
            schedule_func = getattr(schedule.every(), self.config.schedule_day_of_week.lower())
            schedule_func.at(self.config.schedule_time).do(
                self._scheduled_retraining_task
            )
            logger.info(f"Scheduled weekly retraining: {self.config.schedule_day_of_week} at {self.config.schedule_time}")
        else:
            # Daily/interval schedule
            schedule.every(self.config.schedule_interval_days).days.at(
                self.config.schedule_time
            ).do(self._scheduled_retraining_task)
            logger.info(f"Scheduled retraining every {self.config.schedule_interval_days} days at {self.config.schedule_time}")
    
    def _scheduled_retraining_task(self):
        """Task executed on schedule."""
        logger.info("Executing scheduled retraining")
        self.trigger_retraining(
            trigger=RetrainingTrigger.SCHEDULED,
            reason="Scheduled periodic retraining"
        )
    
    def start_monitoring(self, check_interval_seconds: int = 3600):
        """
        Start continuous monitoring in background thread.
        
        Args:
            check_interval_seconds: How often to check (default: 1 hour)
        """
        if self.is_monitoring:
            logger.warning("Monitoring already active")
            return
        
        self.is_monitoring = True
        
        def monitor_loop():
            while self.is_monitoring:
                try:
                    # Run scheduled jobs
                    schedule.run_pending()
                    
                    # Check if retraining needed (on-demand check)
                    # This would need production data input
                    # For now, just sleep
                    time.sleep(check_interval_seconds)
                    
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    time.sleep(60)  # Wait before retry
        
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info(f"Monitoring started (check interval: {check_interval_seconds}s)")
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_monitoring = False
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        logger.info("Monitoring stopped")
    
    def get_job_history(self, limit: int = 10) -> List[Dict]:
        """
        Get recent job history.
        
        Args:
            limit: Number of recent jobs to return
            
        Returns:
            List of job dictionaries
        """
        return [job.to_dict() for job in self.job_history[-limit:]]
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """
        Get status of a specific job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job dictionary or None
        """
        for job in self.job_history:
            if job.job_id == job_id:
                return job.to_dict()
        return None
    
    def generate_report(self) -> str:
        """
        Generate retraining summary report.
        
        Returns:
            Path to generated report
        """
        report_lines = [
            f"# Automated Retraining Report: {self.config.model_name}",
            f"\nGenerated: {datetime.now().isoformat()}",
            f"\n## Configuration",
            f"- Performance Threshold: {self.config.performance_threshold:.1%}",
            f"- Drift Threshold: {self.config.drift_threshold:.1%}",
            f"- Scheduled: {self.config.schedule_enabled}",
            f"- Auto-promote: {self.config.auto_promote}",
            f"\n## Job History (Last 10)\n"
        ]
        
        for job in self.job_history[-10:]:
            duration = (job.end_time - job.start_time).total_seconds() if job.end_time else 0
            report_lines.extend([
                f"### {job.job_id}",
                f"- Status: {job.status}",
                f"- Trigger: {job.trigger.value}",
                f"- Reason: {job.trigger_reason}",
                f"- Duration: {duration:.1f}s",
                f"- Metrics: {job.metrics}" if job.metrics else "",
                ""
            ])
        
        report_text = "\n".join(report_lines)
        report_file = self.jobs_dir / f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        with open(report_file, 'w') as f:
            f.write(report_text)
        
        logger.info(f"Report generated: {report_file}")
        return str(report_file)


if __name__ == "__main__":
    # Example usage
    from sklearn.datasets import make_classification
    
    def sample_data_loader():
        """Example data loader."""
        X, y = make_classification(n_samples=2000, n_features=20, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        df['target'] = y
        return df
    
    # Configuration
    config = RetrainingConfig(
        model_name="example_model",
        performance_threshold=0.05,
        drift_threshold=0.1,
        schedule_enabled=True,
        schedule_interval_days=7,
        auto_promote=True
    )
    
    # Initialize pipeline
    pipeline = TrainingPipeline(
        experiment_name="automated_retraining",
        model_type="sklearn"
    )
    
    # Create scheduler
    scheduler = RetrainingScheduler(
        config=config,
        data_loader=sample_data_loader,
        training_pipeline=pipeline
    )
    
    # Initialize monitors
    ref_data = sample_data_loader()
    baseline_metrics = {'accuracy': 0.85, 'f1_score': 0.83}
    scheduler.initialize_monitors(ref_data, baseline_metrics)
    
    # Trigger manual retraining
    job = scheduler.trigger_retraining(
        trigger=RetrainingTrigger.MANUAL,
        reason="Testing automated retraining"
    )
    
    logger.info(f"Job completed: {job.job_id}")
    logger.info(f"Status: {job.status}")
