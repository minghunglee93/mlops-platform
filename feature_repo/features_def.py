
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64
from datetime import timedelta

user_entity = Entity(
    name="user",
    join_keys=["user_id"],
    description="User entity"
)

user_source = FileSource(
    path="feature_repo/data/user_features.parquet",
    timestamp_field="event_timestamp",
)

user_features = FeatureView(
    name="user_features",
    entities=[user_entity],
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="income", dtype=Float32),
        Field(name="amount", dtype=Float32),
        Field(name="hour", dtype=Int64),
        Field(name="day_of_week", dtype=Int64),
        Field(name="is_weekend", dtype=Int64),
    ],
    source=user_source,
    ttl=timedelta(days=365),
    online=True,
)
