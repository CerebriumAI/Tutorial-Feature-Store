from datetime import timedelta
from feast import Entity, Feature, FeatureView, RedshiftSource, ValueType

transaction = Entity(name="transactionid")

transaction_source = RedshiftSource(
    query="SELECT * FROM spectrum.transaction_features",
    event_timestamp_column="event_timestamp",
    created_timestamp_column="created_timestamp",
)

transaction_features = FeatureView(
    name="transaction_features",
    entities=["TransactionID"],
    ttl=timedelta(days=365),
    features=[
        Feature(name="productcd", dtype=ValueType.STRING),
        Feature(name="transactionamt", dtype=ValueType.DOUBLE),
        Feature(name="p_emaildomain", dtype=ValueType.STRING),
        Feature(name="r_emaildomain", dtype=ValueType.STRING),
        Feature(name="card4", dtype=ValueType.STRING),
        Feature(name="m1", dtype=ValueType.STRING),
        Feature(name="m2", dtype=ValueType.STRING),
        Feature(name="m3", dtype=ValueType.STRING),
    ],
    batch_source=transaction_source,
)
