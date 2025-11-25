# Feature Store Implementation

## 🎯 Overview

The feature store provides centralized feature management for your ML platform with:

- **Consistency**: Same features in training and serving
- **Reusability**: Define features once, use everywhere
- **Performance**: Low-latency online serving
- **Correctness**: Point-in-time correctness for training
- **Monitoring**: Track feature drift and quality
- **Governance**: Centralized versioning and management

## 🚀 Quick Start

```bash
# Install dependencies (includes Feast)
pip install -r requirements.txt

# Run the complete example
python run_feature_store.py
```

This will:
1. ✅ Setup the feature store
2. ✅ Engineer features from raw data
3. ✅ Register features in Feast
4. ✅ Materialize to online store
5. ✅ Train a model using features

## 📁 Architecture

```
┌─────────────────────────────────────────┐
│         Raw Data Sources                │
│  (CSV, Parquet, Database, Streams)      │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│      Feature Engineering                │
│  (feature_store/engineering.py)         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│         Feature Store (Feast)           │
│  ┌─────────────┐    ┌────────────────┐ │
│  │   Offline   │    │     Online     │ │
│  │   Storage   │    │    Storage     │ │
│  │  (Parquet)  │    │   (SQLite)     │ │
│  └─────────────┘    └────────────────┘ │
└──────────────┬──────────────────────────┘
               │
        ┌──────┴──────┐
        ▼             ▼
   ┌─────────┐   ┌──────────┐
   │Training │   │ Serving  │
   │Pipeline │   │   API    │
   └─────────┘   └──────────┘
```

## 🏗️ Project Structure

```
feature_store/
├── __init__.py           # Package initialization
├── store.py              # Core feature store wrapper
├── features.py           # Feature definitions (entities, views)
├── engineering.py        # Feature engineering utilities
└── README.md

feature_repo/             # Feast repository (auto-created)
├── feature_store.yaml    # Feast configuration
├── features_def.py       # Feature definitions
└── data/                 # Feature data storage
    ├── user_features.parquet
    └── transaction_features.parquet

examples/
└── feature_store_example.py  # Complete workflow demo
```

## 📚 Key Components

### 1. Feature Store Wrapper (`store.py`)

High-level interface for Feast operations:

```python
from feature_store import MLOpsFeatureStore

# Initialize
store = MLOpsFeatureStore()

# Get online features (serving)
features = store.get_online_features(
    entity_rows=[{"user_id": "user_123"}],
    features=["user_features:age", "user_features:income"]
)

# Get historical features (training)
training_data = store.get_historical_features(
    entity_df=entity_df,
    features=["user_features:age", "user_features:income"]
)
```

### 2. Feature Engineering (`engineering.py`)

Create ML-ready features:

```python
from feature_store.engineering import FeatureEngineer

engineer = FeatureEngineer()

# Time-based features
df = engineer.create_time_features(df, timestamp_col="timestamp")

# Aggregation features
df = engineer.create_aggregation_features(
    df, 
    group_by="user_id",
    value_col="amount",
    windows=[7, 30, 90]
)

# Statistical features
df = engineer.create_statistical_features(
    df,
    numeric_cols=["amount", "age"]
)
```

### 3. Feature Definitions (`features.py`)

Define features in Feast:

```python
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

# Define entity
user = Entity(
    name="user",
    join_keys=["user_id"],
)

# Define feature view
user_features = FeatureView(
    name="user_features",
    entities=[user],
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="income", dtype=Float32),
    ],
    source=FileSource(path="data/users.parquet"),
)
```

## 🔄 Complete Workflow

### Step 1: Engineer Features

```python
from feature_store.engineering import generate_sample_data_with_features

# Generate or load your data
df = generate_sample_data_with_features(n_samples=1000)

# Apply transformations
engineer = FeatureEngineer()
df = engineer.create_time_features(df)
df = engineer.create_statistical_features(df, numeric_cols=["amount"])
```

### Step 2: Define Features

Create `feature_repo/features_def.py`:

```python
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

user = Entity(name="user", join_keys=["user_id"])

user_features = FeatureView(
    name="user_features",
    entities=[user],
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="income", dtype=Float32),
    ],
    source=FileSource(
        path="data/user_features.parquet",
        timestamp_field="event_timestamp"
    ),
    ttl=timedelta(days=365),
    online=True
)
```

### Step 3: Apply Features

```bash
cd feature_repo
feast apply
```

### Step 4: Materialize Features

```bash
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

Or programmatically:

```python
store.materialize_features(
    start_date=datetime(2024, 1, 1),
    end_date=datetime.now()
)
```

### Step 5: Use in Training

```python
from feature_store import MLOpsFeatureStore

store = MLOpsFeatureStore()

# Get historical features
entity_df = pd.DataFrame({
    'user_id': ['user_1', 'user_2'],
    'event_timestamp': [datetime.now(), datetime.now()]
})

training_df = store.get_historical_features(
    entity_df=entity_df,
    features=['user_features:age', 'user_features:income']
)

# Train model
X = training_df[['age', 'income']]
y = training_df['target']
model.fit(X, y)
```

### Step 6: Use in Serving

```python
# In your API
@app.post("/predict")
def predict(user_id: str):
    # Get online features
    features = store.get_online_features(
        entity_rows=[{"user_id": user_id}],
        features=['user_features:age', 'user_features:income']
    )
    
    # Make prediction
    prediction = model.predict(features)
    return {"prediction": prediction}
```

## 🎓 Advanced Features

### Feature Services

Group related features for specific use cases:

```python
from feast import FeatureService

fraud_detection_service = FeatureService(
    name="fraud_detection_v1",
    features=[
        user_demographics,
        user_activity,
        transaction_features
    ]
)
```

### Feature Versioning

```python
# Version 1
user_features_v1 = FeatureView(
    name="user_features_v1",
    # ... features
)

# Version 2 (new features added)
user_features_v2 = FeatureView(
    name="user_features_v2",
    # ... updated features
)
```

### Custom Data Sources

```python
# PostgreSQL source
from feast.infra.offline_stores.contrib.postgres_offline_store.postgres_source import PostgreSQLSource

postgres_source = PostgreSQLSource(
    query="SELECT * FROM user_features",
    timestamp_field="event_timestamp"
)

# Snowflake source
from feast.infra.offline_stores.snowflake_source import SnowflakeSource

snowflake_source = SnowflakeSource(
    database="ML_DB",
    schema="FEATURES",
    table="user_features",
    timestamp_field="event_timestamp"
)
```

## 📊 Monitoring Features

### Feature Drift Detection

```python
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

# Compare reference vs current features
report = Report(metrics=[DataDriftPreset()])
report.run(reference_data=ref_df, current_data=curr_df)
```

### Feature Quality Checks

```python
def validate_features(df: pd.DataFrame) -> bool:
    """Validate feature quality."""
    checks = {
        'no_nulls': df.isnull().sum().sum() == 0,
        'age_range': (df['age'] >= 0).all() and (df['age'] <= 120).all(),
        'income_positive': (df['income'] > 0).all()
    }
    return all(checks.values())
```

## 🔧 Configuration

### Feast Configuration (`feature_store.yaml`)

```yaml
project: mlops_features
registry: data/registry.db
provider: local

online_store:
  type: sqlite
  path: data/online_store.db

offline_store:
  type: file
```

### For Production (PostgreSQL + Redis)

```yaml
project: mlops_features
registry: postgresql://user:pass@localhost/feast
provider: local

online_store:
  type: redis
  connection_string: redis://localhost:6379

offline_store:
  type: postgres
  host: localhost
  database: feast
  user: feast_user
  password: feast_pass
```

## 🐛 Troubleshooting

### "Feature view not found"

```bash
cd feature_repo
feast apply
```

### "No online features available"

```bash
cd feature_repo
feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

### "Database locked" errors

```bash
rm feature_repo/data/online_store.db
cd feature_repo && feast materialize-incremental $(date -u +"%Y-%m-%dT%H:%M:%S")
```

## 📈 Performance Optimization

### Batch Materialization

```python
# Materialize in parallel
from concurrent.futures import ThreadPoolExecutor

def materialize_view(view_name):
    store.materialize_features(feature_view_name=view_name)

with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(materialize_view, feature_view_names)
```

### Caching

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def get_user_features(user_id: str):
    return store.get_online_features(
        entity_rows=[{"user_id": user_id}],
        features=["user_features:age", "user_features:income"]
    )
```

## 🎯 Best Practices

1. **Version your features**: Use descriptive names like `user_features_v1`
2. **Set appropriate TTLs**: Fresh data for activity, longer for demographics
3. **Monitor feature drift**: Track distribution changes over time
4. **Document features**: Add descriptions to all feature views
5. **Test feature pipelines**: Unit test feature engineering logic
6. **Use feature services**: Group related features for models

## 📚 Resources

- [Feast Documentation](https://docs.feast.dev/)
- [Feature Store Architecture](https://www.featurestore.org/)
- [MLOps Best Practices](https://ml-ops.org/)

## 🚀 Next Steps

1. ✅ Run the example: `python run_feature_store.py`
2. ✅ Define your features in `feature_repo/features_def.py`
3. ✅ Integrate with training pipeline
4. ✅ Add feature monitoring
5. ✅ Deploy to production with Redis/PostgreSQL