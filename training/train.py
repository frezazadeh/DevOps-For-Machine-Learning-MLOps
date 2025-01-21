import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import train_test_split
import lightgbm

# Split the dataframe into train and validation data
def split_data(data_df):
    """Split a dataframe into training and validation datasets."""
    # If this column exists, convert to 'category' dtype so LightGBM recognizes it
    if 'ps_ind_04_cat' in data_df.columns:
        data_df['ps_ind_04_cat'] = data_df['ps_ind_04_cat'].astype('category')

    # Drop label and ID columns from features
    features = data_df.drop(['target', 'id'], axis=1)
    labels = np.array(data_df['target'])

    # Standard train/validation split
    features_train, features_valid, labels_train, labels_valid = \
        train_test_split(features, labels, test_size=0.2, random_state=0)

    # Automatically detect categorical columns (anything typed as 'category')
    categorical_cols = [
        col for col in features_train.columns
        if str(features_train[col].dtype) == 'category'
    ]

    # Create LightGBM Dataset objects
    train_data = lightgbm.Dataset(
        features_train,
        label=labels_train,
        categorical_feature=categorical_cols
    )
    valid_data = lightgbm.Dataset(
        features_valid,
        label=labels_valid,
        free_raw_data=False,
        categorical_feature=categorical_cols
    )

    return (train_data, valid_data)

# Train the model, return the model
def train_model(data, parameters, early_stopping_rounds=10):
    """
    Train a model with the given datasets and parameters,
    including early stopping.
    """
    train_data, valid_data = data

    # Use callbacks for early stopping
    early_stopping_cb = lightgbm.early_stopping(2)

    model = lightgbm.train(
        params=parameters,
        train_set=train_data,
        valid_sets=[valid_data],
        num_boost_round=10,
        callbacks=[early_stopping_cb]
    )

    return model

# Evaluate the metrics for the model
def get_model_metrics(model, data):
    """
    Construct and return a dictionary of metrics for the model.
    """
    # data[1] is valid_data in the tuple (train_data, valid_data)
    predictions = model.predict(data[1].data)
    fpr, tpr, thresholds = metrics.roc_curve(data[1].label, predictions)
    model_metrics = {
        "auc": metrics.auc(fpr, tpr)
    }
    print(f"Model metrics: {model_metrics}")
    return model_metrics
