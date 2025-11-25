
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

# Define entity
user = Entity(
    name="user",
    join_keys=["user_id"],
    description="User entity"
)

# Define data source
user_source = FileSource(
    path="data/user_features.parquet",
    timestamp_field="event_timestamp",
)

# Define feature view
user_features = FeatureView(
    name="user_features",
    entities=[user],
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="income", dtype=Float32),
        Field(name="transaction_count", dtype=Int64),
    ],
    source=user_source,
    ttl=timedelta(days=365),
    online=True,
)
