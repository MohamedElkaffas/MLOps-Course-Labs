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
