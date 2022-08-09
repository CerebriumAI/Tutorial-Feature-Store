import boto3
import pandas as pd

from fraud_detection_model import FraudClassifierModel

# Get historic transactions from parquet
transactions = pd.read_parquet("data/train_transaction.parquet")

# Create model
model = FraudClassifierModel()

# Train model (using Redshift for transaction history features)
if not model.is_model_trained():
    model.train(transactions)

# Make online prediction (using DynamoDB for retrieving online features)
loan_request = {
    "transactionid": [3577537],
    "transactionamt": [30.95],
    "productcd": ["W"],
    "card4": ["mastercard"],
    "p_emaildomain": ["gmail.com"],
    "r_emaildomain": [None],
    "m1": ["T"],
    "m2": ["F"],
    "m3": ["F"],
}

result = model.predict(loan_request)

if result == 0:
    print("Transaction OK!")
elif result == 1:
    print("Transaction FRAUDULENT!")
