# Automated Model Retraining

## Overview

Automated retraining ensures your models stay accurate and relevant by detecting when retraining is needed and executing it automatically.

## Features

- **Performance Monitoring**: Detects model degradation
- **Drift Detection**: Identifies data and concept drift
- **Scheduled Retraining**: Periodic model updates
- **Manual Triggers**: On-demand retraining
- **Auto-Promotion**: Automatic model deployment
- **Job Tracking**: Complete audit trail

## Quick Start

```bash
# Run the complete example
python run_retraining.py

# Or run specific examples
python examples/retraining_example.py
```

## Configuration

### Basic Setup

```python
from retraining.scheduler import RetrainingScheduler, RetrainingConfig
from training.pipeline import TrainingPipeline

# Configure retraining
config = RetrainingConfig(
    model_name="my_model",
    performance_threshold=0.05,  # 5% degradation triggers retraining
    drift_threshold=0.1,         # Drift detection sensitivity
    schedule_enabled=True,
    schedule_interval_days=7,    # Weekly retraining
    auto_promote=True            # Auto-deploy if better
)

# Data loader function
def load_training_data():
    # Your data loading logic
    return pd.DataFrame(...)

# Initialize
pipeline = TrainingPipeline("automated_retraining", "sklearn")
scheduler = RetrainingScheduler(config, load_training_data, pipeline)
```

### Initialize Monitors

```python
# Set baseline
reference_data = load_reference_data()
baseline_metrics = {
    'accuracy': 0.85,
    'f1_score': 0.83
}

scheduler.initialize_monitors(reference_data, baseline_metrics)
```

## Retraining Triggers

### 1. Performance Degradation

Triggers when model performance drops below threshold:

```python
# Automatic check
current_metrics = {'accuracy': 0.78, 'f1_score': 0.75}
needs_retraining, trigger, reason = scheduler.check_retraining_needed(
    current_data,
    current_metrics=current_metrics
)
```

### 2. Data Drift

Triggers when input data distribution changes:

```python
# Check drift on production data
production_data = get_recent_predictions()
needs_retraining, trigger, reason = scheduler.check_retraining_needed(
    production_data
)
```

### 3. Scheduled

Periodic retraining on fixed schedule:

```python
config = RetrainingConfig(
    schedule_enabled=True,
    schedule_interval_days=7,
    schedule_time="02:00"
)

scheduler.schedule_periodic_retraining()
scheduler.start_monitoring()  # Runs in background
```

### 4. Manual

On-demand retraining:

```python
job = scheduler.trigger_retraining(
    trigger=RetrainingTrigger.MANUAL,
    reason="New feature added"
)
```

## API Endpoints

### Configure Retraining

```bash
curl -X POST "http://localhost:8000/retraining/configure" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "my_model",
    "performance_threshold": 0.05,
    "drift_threshold": 0.1,
    "schedule_enabled": true,
    "auto_promote": true
  }'
```

### Initialize Monitors

```bash
curl -X POST "http://localhost:8000/retraining/my_model/initialize"
```

### Trigger Retraining

```bash
curl -X POST "http://localhost:8000/retraining/my_model/trigger" \
  -H "Content-Type: application/json" \
  -d '{
    "trigger_type": "manual",
    "reason": "Testing automated retraining"
  }'
```

### Check Status

```bash
# Check if retraining needed
curl "http://localhost:8000/retraining/my_model/check"

# Get job history
curl "http://localhost:8000/retraining/my_model/jobs"

# Get specific job
curl "http://localhost:8000/retraining/my_model/jobs/{job_id}"

# System status
curl "http://localhost:8000/retraining/status"
```

### Monitoring Control

```bash
# Start continuous monitoring
curl -X POST "http://localhost:8000/retraining/my_model/start-monitoring"

# Stop monitoring
curl -X POST "http://localhost:8000/retraining/my_model/stop-monitoring"
```

### Generate Report

```bash
curl "http://localhost:8000/retraining/my_model/report"
```

## Configuration Options

```python
RetrainingConfig(
    # Model identification
    model_name: str,
    
    # Performance thresholds
    performance_threshold: float = 0.05,     # 5% degradation
    min_samples_for_check: int = 100,
    
    # Drift thresholds
    drift_threshold: float = 0.1,
    drift_alert_threshold: float = 0.3,
    
    # Scheduling
    schedule_enabled: bool = False,
    schedule_interval_days: int = 7,
    schedule_day_of_week: str = None,        # "monday", "tuesday", etc.
    schedule_time: str = "02:00",
    
    # Data requirements
    min_training_samples: int = 1000,
    training_data_window_days: int = 30,
    
    # Auto-promotion
    auto_promote: bool = False,
    promotion_min_improvement: float = 0.01,  # 1%
    promotion_stage: str = "Staging",
    
    # Notifications
    enable_notifications: bool = True,
    notification_email: str = None
)
```

## Job Tracking

Each retraining creates a tracked job:

```python
job = scheduler.trigger_retraining(...)

print(f"Job ID: {job.job_id}")
print(f"Status: {job.status}")
print(f"Metrics: {job.metrics}")
print(f"Duration: {job.end_time - job.start_time}")
```

Job statuses:
- `pending`: Queued
- `running`: In progress
- `completed`: Successful
- `failed`: Error occurred

## Automatic Promotion

When enabled, new models are automatically promoted if they meet criteria:

```python
config = RetrainingConfig(
    auto_promote=True,
    promotion_min_improvement=0.02,  # Require 2% improvement
    promotion_stage="Production"
)

# Promotion happens automatically after retraining
# if new model outperforms by 2%+
```

## Monitoring & Alerts

### Continuous Monitoring

```python
# Start background monitoring
scheduler.start_monitoring(check_interval_seconds=3600)  # Check hourly

# Stop when done
scheduler.stop_monitoring()
```

### Custom Notifications

```python
# Implement custom notification handler
class MyNotifier:
    def notify(self, job):
        if job.status == "failed":
            send_pagerduty_alert(job)
        else:
            send_slack_message(job)

# Use in scheduler
scheduler.notification_handler = MyNotifier()
```

## Best Practices

1. **Set Appropriate Thresholds**
   - Performance: 5-10% degradation
   - Drift: 0.1-0.3 depending on criticality

2. **Schedule During Low Traffic**
   - Default: 2:00 AM
   - Avoid peak hours

3. **Start with Staging**
   - Test auto-promotion in staging first
   - Require manual production promotion initially

4. **Monitor Data Quality**
   - Ensure sufficient training samples
   - Validate data before retraining

5. **Track Job History**
   - Review failed jobs
   - Monitor retraining frequency
   - Analyze improvement trends

## Examples

### Performance-Based Retraining

```python
scheduler.initialize_monitors(ref_data, baseline_metrics)

# Check performance regularly
current_metrics = evaluate_model(production_data)
needs_retraining, trigger, reason = scheduler.check_retraining_needed(
    production_data,
    current_metrics=current_metrics
)

if needs_retraining:
    job = scheduler.trigger_retraining(trigger, reason)
```

### Drift-Based Retraining

```python
# Monitor production data
production_data = get_recent_data()
needs_retraining, trigger, reason = scheduler.check_retraining_needed(
    production_data
)

if trigger == RetrainingTrigger.DATA_DRIFT:
    logger.warning(f"Data drift detected: {reason}")
    job = scheduler.trigger_retraining(trigger, reason)
```

### Scheduled with Promotion

```python
config = RetrainingConfig(
    schedule_enabled=True,
    schedule_day_of_week="sunday",
    schedule_time="02:00",
    auto_promote=True,
    promotion_min_improvement=0.02
)

scheduler = RetrainingScheduler(config, data_loader, pipeline)
scheduler.schedule_periodic_retraining()
scheduler.start_monitoring()
```

## Troubleshooting

### Insufficient Training Data

```python
config = RetrainingConfig(
    min_training_samples=1000,
    training_data_window_days=30
)

# Validate before retraining
if len(training_data) < config.min_training_samples:
    logger.error("Not enough training data")
```

### Retraining Failures

```python
# Check job status
job = scheduler.get_job_status(job_id)
if job['status'] == 'failed':
    logger.error(f"Retraining failed: {job['error']}")
```

### Performance Not Improving

- Check data quality
- Review feature engineering
- Consider hyperparameter tuning
- Analyze drift reports

## Integration

### With Serving API

```python
# In serving/api.py
@app.post("/predict")
async def predict(request):
    # Make prediction
    prediction = model.predict(features)
    
    # Check if retraining needed
    if scheduler.check_retraining_needed(...):
        background_tasks.add_task(trigger_retraining)
    
    return prediction
```

### With CI/CD

```yaml
# .github/workflows/retrain.yml
name: Scheduled Retraining
on:
  schedule:
    - cron: '0 2 * * 0'  # Weekly Sunday 2 AM

jobs:
  retrain:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Trigger Retraining
        run: |
          curl -X POST "$API_URL/retraining/trigger" \
            -H "Authorization: Bearer $TOKEN"
```

## Metrics

Track retraining effectiveness:

```python
# Generate report
report_path = scheduler.generate_report()

# View job history
jobs = scheduler.get_job_history(limit=50)
successful = [j for j in jobs if j['status'] == 'completed']
avg_improvement = np.mean([j['metrics']['accuracy'] for j in successful])
```

## Next Steps

1. Configure for your models
2. Set up monitoring dashboards
3. Integrate with notification systems
4. Define custom retraining policies
5. Review and tune thresholds

For more examples, see `examples/retraining_example.py`.