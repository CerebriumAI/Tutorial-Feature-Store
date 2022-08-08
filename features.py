from datetime import timedelta
from feast import Entity, Feature, FeatureView, RedshiftSource, ValueType

transaction = Entity(name="transaction", value_type=ValueType.INT64)

transaction_source = RedshiftSource(
    query="SELECT * FROM spectrum.transaction_features",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

transaction_features = FeatureView(
    name="transaction_features",
    entities=["transaction"],
    ttl=timedelta(days=365),
    features=[
        Feature(name="ProductCD", dtype=ValueType.STRING),
        Feature(name="TransactionAmt", dtype=ValueType.DOUBLE),
        Feature(name="P_emaildomain", dtype=ValueType.STRING),
        Feature(name="R_emaildomain", dtype=ValueType.STRING),
        Feature(name="card4", dtype=ValueType.STRING),
        Feature(name="M1", dtype=ValueType.STRING),
        Feature(name="M2", dtype=ValueType.STRING),
        Feature(name="M3", dtype=ValueType.STRING),
    ],
    batch_source=transaction_source,
)
