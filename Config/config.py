# config.py
import os
from pathlib import Path

# Répertoire du script
BASE_DIR = Path(__file__).resolve().parent

# Chemins vers les ressources
PIPELINE_PATH = BASE_DIR.parent / "Output" / "Pipelines" / "pipeline_to_deployed.joblib"
CLIENT_PATH = BASE_DIR.parent / "Output" / "Data_clients" / "App_test_final.csv"
LOGO_PATH = BASE_DIR.parent  / "Images" / "Logo.png"

# Hyperparamètres / seuils
DEFAULT_THRESHOLD = float(os.getenv("THRESHOLD", 0.53))
DEFAULT_PORT = int(os.getenv("PORT", 8001))
