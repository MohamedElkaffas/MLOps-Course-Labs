<<<<<<< Updated upstream
"""
This module contains functions to preprocess and train the model
for bank consumer churn prediction.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder,  StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

### Import MLflow

def rebalance(data):
    """
    Resample data to keep balance between target classes.

    The function uses the resample function to downsample the majority class to match the minority class.

    Args:
        data (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame): balanced DataFrame
    """
    churn_0 = data[data["Exited"] == 0]
    churn_1 = data[data["Exited"] == 1]
    if len(churn_0) > len(churn_1):
        churn_maj = churn_0
        churn_min = churn_1
    else:
        churn_maj = churn_1
        churn_min = churn_0
    churn_maj_downsample = resample(
        churn_maj, n_samples=len(churn_min), replace=False, random_state=1234
    )

    return pd.concat([churn_maj_downsample, churn_min])


def preprocess(df):
    """
    Preprocess and split data into training and test sets.

    Args:
        df (pd.DataFrame): DataFrame with features and target variables

    Returns:
        ColumnTransformer: ColumnTransformer with scalers and encoders
        pd.DataFrame: training set with transformed features
        pd.DataFrame: test set with transformed features
        pd.Series: training set target
        pd.Series: test set target
    """
    filter_feat = [
        "CreditScore",
        "Geography",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Exited",
    ]
    cat_cols = ["Geography", "Gender"]
    num_cols = [
        "CreditScore",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
    ]
    data = df.loc[:, filter_feat]
    data_bal = rebalance(data=data)
    X = data_bal.drop("Exited", axis=1)
    y = data_bal["Exited"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1912
    )
    col_transf = make_column_transformer(
        (StandardScaler(), num_cols), 
        (OneHotEncoder(handle_unknown="ignore", drop="first"), cat_cols),
        remainder="passthrough",
    )

    X_train = col_transf.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=col_transf.get_feature_names_out())

    X_test = col_transf.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=col_transf.get_feature_names_out())

    # Log the transformer as an artifact

    return col_transf, X_train, X_test, y_train, y_test


def train(X_train, y_train):
    """
    Train a logistic regression model.

    Args:
        X_train (pd.DataFrame): DataFrame with features
        y_train (pd.Series): Series with target

    Returns:
        LogisticRegression: trained logistic regression model
    """
    log_reg = LogisticRegression(max_iter=1000)
    log_reg.fit(X_train, y_train)

    ### Log the model with the input and output schema
    # Infer signature (input and output schema)

    # Log model

    ### Log the data

    return log_reg


def main():
    ### Set the tracking URI for MLflow

    ### Set the experiment name


    ### Start a new run and leave all the main function code as part of the experiment

    df = pd.read_csv("data/Churn_Modelling.csv")
    col_transf, X_train, X_test, y_train, y_test = preprocess(df)

    ### Log the max_iter parameter

    model = train(X_train, y_train)

    
    y_pred = model.predict(X_test)

    ### Log metrics after calculating them


    ### Log tag


    
    conf_mat = confusion_matrix(y_test, y_pred, labels=model.classes_)
    conf_mat_disp = ConfusionMatrixDisplay(
        confusion_matrix=conf_mat, display_labels=model.classes_
    )
    conf_mat_disp.plot()
    
    # Log the image as an artifact in MLflow
    
    plt.show()


if __name__ == "__main__":
    main()
=======
# Train
# PRIVATE: Kaggle dataset ==> https://www.kaggle.com/datasets/shantanudhakadd/bank-customer-churn-prediction/data
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def load_data(path="data/Churn_Modelling.csv"):
    df = pd.read_csv(path)

    # Drop IDs and target
    y = df["Exited"]
    df = df.drop(columns=["RowNumber", "CustomerId", "Surname", "Exited"])

    # One-hot encode the categorical features
    df = pd.get_dummies(df,columns=["Geography", "Gender"],drop_first=True)

    # Train/test split
    return train_test_split(df, y, test_size=0.2, random_state=42)


def run_experiment(model, params: dict, run_name: str):
    mlflow.set_experiment("Bank_Customer_Churn")
    with mlflow.start_run(run_name=run_name):
        # Log which model and its hyperparameters
        mlflow.log_param("model_type", model.__class__.__name__)
        mlflow.log_params(params)

        # Data split & training
        X_train, X_test, y_train, y_test = load_data()
        model.set_params(**params)
        model.fit(X_train, y_train)

        # Evaluation
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        mlflow.log_metric("accuracy", acc)

        # Model logging
        mlflow.sklearn.log_model(model, "model")

        print(f"[{run_name}] accuracy = {acc:.4f}")
        return mlflow.active_run().info.run_id

if __name__ == "__main__":
    lr_params = {"C": 1.0, "solver": "liblinear"}
    run_id_lr = run_experiment(LogisticRegression(), lr_params, run_name="LR_baseline")

    rf_params = {"n_estimators": 100, "max_depth": 6, "random_state": 42}
    run_id_rf = run_experiment(RandomForestClassifier(), rf_params, run_name="RF_100_6")

    from xgboost import XGBClassifier
    xgb_params = {
        "n_estimators": 100,
        "learning_rate": 0.1,
        "use_label_encoder": False,
        "eval_metric": "logloss"
    }
    run_experiment(XGBClassifier(), xgb_params, run_name="XGB_100_0.1")

    from mlflow.tracking import MlflowClient
    
    client = MlflowClient()
    client.create_registered_model("ChurnModel")
    
    # Stage LR to Staging
    uri_lr = f"runs:/{run_id_lr}/model"
    mv1 = client.create_model_version("ChurnModel", uri_lr, run_id_lr)
    client.transition_model_version_stage("ChurnModel", mv1.version, "Staging")
    
    # Promote RF to Production
    uri_rf = f"runs:/{run_id_rf}/model"
    mv2 = client.create_model_version("ChurnModel", uri_rf, run_id_rf)
    client.transition_model_version_stage("ChurnModel", mv2.version, "Production")
>>>>>>> Stashed changes
