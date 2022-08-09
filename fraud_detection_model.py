from pathlib import Path

import feast
import joblib
import pandas as pd
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


class FraudClassifierModel:
    categorical_features = [
        "productcd",
        "p_emaildomain",
        "r_emaildomain",
        "card4",
        "m1",
        "m2",
        "m3",
    ]

    feast_features = [
        "transaction_features:productcd",
        "transaction_features:transactionamt",
        "transaction_features:p_emaildomain",
        "transaction_features:r_emaildomain",
        "transaction_features:card4",
        "transaction_features:m1",
        "transaction_features:m2",
        "transaction_features:m3",
    ]

    target = "isfraud"
    model_filename = "fraud_model.joblib"
    encoder_filename = "encoder.joblib"

    def __init__(self):
        # Load model
        if Path(self.model_filename).exists():
            self.classifier = joblib.load(self.model_filename)
        else:
            self.classifier = XGBClassifier()

        # Load ohe encoder
        if Path(self.encoder_filename).exists():
            self.encoder = joblib.load(self.encoder_filename)
        else:
            self.encoder = OneHotEncoder()

        # Set up feature store
        self.fs = feast.FeatureStore(repo_path="feature_repo")

    def train(self, transactions):
        train_X, train_Y = self._get_training_features(transactions)

        self.classifier.fit(train_X[sorted(train_X)], train_Y)
        joblib.dump(self.classifier, self.model_filename)

    def _get_training_features(self, transactions):
        training_df = self.fs.get_historical_features(
            entity_df=transactions[["transactionid", "event_timestamp", "isfraud"]],
            features=self.feast_features,
        ).to_df()

        self._fit_ohe_encoder(training_df)
        train_X = self._apply_ohe_encoding(training_df)

        train_X = train_X.reindex(sorted(train_X.columns), axis=1)
        train_Y = training_df.loc[:, self.target]

        return train_X, train_Y

    def _fit_ohe_encoder(self, requests):
        self.encoder.fit(requests[self.categorical_features])
        joblib.dump(self.encoder, self.encoder_filename)

    def _apply_ohe_encoding(self, requests):
        X = requests[self.categorical_features]
        X = pd.DataFrame(
            self.encoder.transform(X).toarray(), columns=self.encoder.get_feature_names_out().reshape(-1)
        )
        X["transactionamt"] = requests[["transactionamt"]].to_numpy()
        return X

    def predict(self, request):
        # Get online features from Feast
        feature_vector = self._get_online_features_from_feast(request)

        # Join features to request features
        features = request.copy()
        features.update(feature_vector)
        features_df = pd.DataFrame.from_dict(features)

        # Apply ohe encoding to categorical features
        features_df = self._apply_ohe_encoding(features_df)

        # Sort columns
        features_df = features_df.reindex(sorted(features_df.columns), axis=1)

        # Make prediction
        features_df["prediction"] = self.classifier.predict(features_df)

        # return result of credit scoring
        return features_df["prediction"].iloc[0]

    def _get_online_features_from_feast(self, request):
        transaction = request["transactionid"][0]

        return self.fs.get_online_features(
            entity_rows=[{"transactionid": transaction}],
            features=self.feast_features,
        ).to_dict()

    def is_model_trained(self):
        try:
            check_is_fitted(self.classifier, "xgboost_")
        except NotFittedError:
            return False
        return True
