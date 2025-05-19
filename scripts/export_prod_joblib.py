from mlflow.sklearn import load_model
import joblib, os

# Which run to export
RUN_ID = "d38114219d4242dea71933b91e03e66b"

# Path to the model inside mlruns/
MODEL_URI = f"runs:/{RUN_ID}/model"

# Where to save the pickle
dest_folder = "models"
os.makedirs(dest_folder, exist_ok=True)
DEST_PATH = os.path.join(dest_folder, "xgb_best.pkl")

# Load & dump
model = load_model(MODEL_URI)
joblib.dump(model, DEST_PATH)

