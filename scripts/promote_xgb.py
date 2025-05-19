from mlflow.tracking import MlflowClient

client     = MlflowClient()
MODEL_NAME = "ChurnModel"
RUN_ID_XGB = "d38114219d4242dea71933b91e03e66b"

# See if the model already exists by searching for its name
existing = client.search_registered_models(f"name = '{MODEL_NAME}'")
if not existing:
    client.create_registered_model(MODEL_NAME)

# Register the XGBoost run as a new version
model_uri = f"runs:/{RUN_ID_XGB}/model"
mv        = client.create_model_version(
    name   = MODEL_NAME,
    source = model_uri,
    run_id = RUN_ID_XGB
)

# Promote that version to Production
client.transition_model_version_stage(
    name    = MODEL_NAME,
    version = mv.version,
    stage   = "Production"
)