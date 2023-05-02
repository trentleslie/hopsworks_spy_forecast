import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import time
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectPercentile, chi2

import hopsworks
from hsml.schema import Schema
from hsml.model_schema import ModelSchema
import joblib

def process_data(file_path):
    data = pd.read_csv(file_path, index_col=0)
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    y_train = train_data[['close_20']]
    y_test = test_data[['close_20']]

    columns_to_drop = ['close_20', 'open_20', 'high_20', 'low_20']
    X_train = train_data.drop(columns=columns_to_drop)
    X_test = test_data.drop(columns=columns_to_drop)
    
    # Hopsworks train_test_split
    #proj = hopsworks.login()
    #fs = proj.get_feature_store()
    
    #fg_spy = fs.get_feature_group(name="historical_data_21", version=1)

    #query = fg_spy.select_except(["id","high_20","low_20","open_20"])
    
    #try:
    #    fv = fs.get_feature_view(name="historical_data_21", version=1)
    #except:
    #    fv = fs.create_feature_view(name="historical_data_21", 
    #                    version=1,
    #                        description="Historical Data_21",
    #                        labels=["close_20"],
    #                        query=query
    #                    )
    
    #start_time = time.time()
    #X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)
    #print("Get Random Split Training Data in %s seconds ---" % (time.time() - start_time))
    
    # Map values > 0 to 1 and values <= 0 to 0
    y_train['close_20'] = y_train['close_20'].apply(lambda x: 1 if x > 0 else 0)
    y_test['close_20'] = y_test['close_20'].apply(lambda x: 1 if x > 0 else 0)

    return X_train, X_test, y_train, y_test


def preprocess_data(X_train, X_test):
    categorical_features = []
    numeric_features = []
    for col in X_train.columns:
        if X_train[col].dtype == object:
            categorical_features.append(col)
        else:
            numeric_features.append(col)

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("selector", SelectPercentile(chi2, percentile=50))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ])

    return preprocessor


def train_model(X_train, y_train, preprocessor):
    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression())
    ])

    clf.fit(X_train, y_train['close_20'].ravel())
    return clf


def main():
    file_path = "/home/trent/github/hopsworks_spy_forecast/data/processed/all_processed_21.csv"
    X_train, X_test, y_train, y_test = process_data(file_path)
    preprocessor = preprocess_data(X_train, X_test)
    clf = train_model(X_train, y_train, preprocessor)

    # Save model
    os.makedirs("spy_forecasting_model_21/features", exist_ok=True)
    joblib.dump(clf, 'spy_forecasting_model_21/spy_forecasting_model.pkl')

    # Register the model with Model Registry
    proj = hopsworks.login()
    mr = proj.get_model_registry()
    input_schema = Schema(X_test)
    output_schema = Schema(y_test)

    spy_forecasting_model = mr.sklearn.create_model(
        "spy_forecasting_model_21",
        metrics={'accuracy': roc_auc_score(y_test, clf.predict(X_test))},
        input_example=X_test.sample().to_numpy(),
        model_schema=ModelSchema(input_schema=input_schema, output_schema=output_schema)
    )
    spy_forecasting_model.save('spy_forecasting_model_21')


if __name__ == "__main__":
    main()
