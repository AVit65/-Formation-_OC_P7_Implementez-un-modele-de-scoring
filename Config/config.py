# config.py
import os
from pathlib import Path

# Répertoire du script
BASE_DIR = Path(__file__).resolve().parent

# Chemins vers les ressources
PIPELINE_PATH = BASE_DIR.parent / "Output" / "Pipelines" / "pipeline_to_deployed.joblib"
CLIENT_PATH = BASE_DIR.parent / "Output" / "Data_clients" / "App_test_final.csv"
VAR_DESC_ENG_PATH = BASE_DIR.parent / "Output" / "Variables" / "Variable_description_eng.csv"
VAR_DESC_FR_PATH = BASE_DIR.parent / "Output" / "Variables" / "Variable_description_fr.csv"
SHAP_VALUE_TEST_PATH = BASE_DIR.parent / "Output" / "Shap_value" / "shap_value_test_sample.joblib"
SHAP_VALUE_TRAIN_PATH = BASE_DIR.parent / "Output" / "Shap_value" / "shap_value_train_sample.joblib"
RAW_DATA_TEST_ALIGNED_PATH = BASE_DIR.parent / "Output" / "Shap_value" / "raw_data_test_with_colname_aligned_sample.csv"
EXPLAINER_PATH = BASE_DIR.parent / "Output" / "Shap_value" / "exp_sample.joblib"
VARIABLE_TYPE_PATH = BASE_DIR.parent / "Output" / "Variables" / "Variable_type.csv"
COL_TYPE_PATH = BASE_DIR.parent / "Output" / "Variables" / "Col_type.csv"
LOGO_PATH = BASE_DIR.parent  / "Images" / "Logo.png"
MAPPING_DICT = BASE_DIR.parent / "Output" / "Shap_value" / "Mapping_Dictionnaire.joblib"

# Hyperparamètres / seuils
THRESHOLD = 0.27
DEFAULT_PORT = int(os.getenv("PORT", 8001))
