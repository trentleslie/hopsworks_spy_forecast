import unittest
import pandas as pd
from spy_forecast_training import prepare_data, train_and_evaluate_model

def test_train_model():
    _, _, X_train, y_train = load_and_split_data()
    preprocessor = preprocess_data(X_train)
    clf = train_model(X_train, y_train, preprocessor)
    assert isinstance(clf, Pipeline), "train_model should return a Pipeline"

test_train_model()
