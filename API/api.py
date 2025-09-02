"""
API de scoring de clients avec FastAPI.

API compatible avec deux modes de lancement :
  1. python api.py (utilise le bloc `if __name__ == "__main__"`)
  2. uvicorn api:app --reload --port 8001

Fonctionnalités :
- Chargement du pipeline ML et des données clients au démarrage
- Routes pour prédiction, SHAP global/local, informations sur les variables
- Gestion des erreurs et logging
- Lazy loading pour fichiers lourds (SHAP, raw_data, variable_type)

Auteur : [Aline Vitrac]
Date : [17/08/2025]
"""

# ----------------------------------------------------------------------------
# Imports
# ----------------------------------------------------------------------------

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from typing import List
import joblib
import pandas as pd
import numpy as np
import logging
import uvicorn

# Imports depuis config
from Config import (
    PIPELINE_PATH,
    CLIENT_PATH,
    VAR_DESC_PATH,
    SHAP_VALUE_TEST_PATH,
    SHAP_VALUE_TRAIN_PATH,
    RAW_DATA_TEST_ALIGNED_PATH,
    VARIABLE_TYPE_PATH,
    DEFAULT_THRESHOLD,
    DEFAULT_PORT
)

# ----------------------------------------------------------------------------
# Initialisation de FastAPI et logging
# ----------------------------------------------------------------------------

app = FastAPI()
logger = logging.getLogger("uvicorn.error")

# ----------------------------------------------------------------------------
# Gestion des erreurs
# ----------------------------------------------------------------------------

# Erreur non spécifique
@app.exception_handler(Exception)
async def all_exception_handler(request: Request, exc: Exception):
    logger.error(f"Erreur non gérée : {exc}", exc_info=True)
    return JSONResponse(status_code=500, content={"detail": str(exc)})

# Erreur de validation
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc}", exc_info=True)
    return JSONResponse(status_code=422, content={"detail": exc.errors()})

# ----------------------------------------------------------------------------
# Chargement des ressources au démarrage
# ----------------------------------------------------------------------------

@app.on_event("startup")
async def load_resources():
    """
    Chargement initial du pipeline et des données clients.
    Stockage dans app.state pour accès global.
    """
    logger.info("Chargement des ressources principales...")
    app.state.pipeline = joblib.load(PIPELINE_PATH)
    app.state.clients_df = pd.read_csv(CLIENT_PATH).set_index("SK_ID_CURR")
    app.state.variable_desc = pd.read_csv(VAR_DESC_PATH)

# ----------------------------------------------------------------------------
# Lazy loading pour fichiers lourds
# ----------------------------------------------------------------------------

def get_shap_value_train():
    if not hasattr(app.state, "shap_value_train"):
        app.state.shap_value_train = joblib.load(SHAP_VALUE_TRAIN_PATH)
    return app.state.shap_value_train

def get_shap_value_test():
    if not hasattr(app.state, "shap_value_test"):
        app.state.shap_value_test = joblib.load(SHAP_VALUE_TEST_PATH)
    return app.state.shap_value_test

def get_raw_data_test():
    if not hasattr(app.state, "raw_data_test_aligned"):
        app.state.raw_data_test_aligned = pd.read_csv(RAW_DATA_TEST_ALIGNED_PATH)
    return app.state.raw_data_test_aligned

def get_variable_type():
    if not hasattr(app.state, "variable_type"):
        app.state.variable_type = pd.read_csv(VARIABLE_TYPE_PATH)
    return app.state.variable_type

# ----------------------------------------------------------------------------
# Définition des Pydantic Models
# ----------------------------------------------------------------------------

class ClientID(BaseModel):
    client_id: int

class SHAPRequest(BaseModel):
    client_id: int
    top_n: int = 10

class G_SHAPRequest(BaseModel):
    top_n: int = 10

# ----------------------------------------------------------------------------
# Définition des routes
# ----------------------------------------------------------------------------

@app.get("/")
def read_root():
    """Route d'accueil pour tester l'API"""
    return {"message": "Bienvenue sur mon API de scoring !"}

# Prédiction du risque de défaut de paiment
# ----------------------------------------------------------------------------

@app.post("/predict")
async def predict(data: ClientID):
    """
    Prédiction du risque pour un client donné
    """
    
    # Accès aux données stocké dans app.state lors du startup
    pipeline = app.state.pipeline
    clients_df = app.state.clients_df

    # Vérrification de la présence de l'ID en index
    if data.client_id not in clients_df.index:
        raise HTTPException(status_code=404, detail=f"Client {data.client_id} non trouvé")

    # Suppression de la colonne TARGET avant prédiction
    X = clients_df.drop(columns="TARGET").loc[[data.client_id]]
    
    # Prédiction de la probabilité de défaut de paiment
    prob_pos = pipeline.predict_proba(X)[0][1]

    # Prédiction de la classe 
    prediction_seuil = 1 if prob_pos >= DEFAULT_THRESHOLD else 0

    return {"prediction": prediction_seuil, "proba": prob_pos}

# Liste de toutes les variables
# ----------------------------------------------------------------------------

@app.get("/liste_variables")
async def get_all_variables():
    """Retourne toutes les variables disponibles"""
    try:
        # Envoie de la liste de toutes les variables disponibles
        return app.state.variable_desc["Row"].tolist()
    except Exception as e:
        
        # Renvoie d'une erreur 500 si la récupération échoue
        raise HTTPException(status_code=500, detail=f"Impossible de récupérer les variables : {str(e)}")


# Infos sur des variables spécifiques
# ----------------------------------------------------------------------------

@app.get("/variables_info")
async def get_variable_info(variable_names: str):
    """
    Retourne description et source pour une ou plusieurs variables.
    variable_names : string, noms séparés par des virgules
    """
    try:
        # Séparation des noms de variables
        var_list = variable_names.split(",")

        # Filte de la Dataframe
        df_filtered = app.state.variable_desc[app.state.variable_desc["Row"].isin(var_list)]
        result = df_filtered[["Row", "Description", "Source"]].to_dict(orient="records")

        # Si aucune variable trouvée, renvoie une erreur 404
        if not result:
            raise HTTPException(status_code=404, detail=f"Variable(s) non trouvée(s) : {', '.join(var_list)}")

        return result
    except Exception as e:
        # Si problème inattendu lors du filtrage, renvoie une erreur 500
        raise HTTPException(status_code=500, detail=f"Erreur lors de la récupération des variables : {str(e)}")


# Explicabilité globale
# ----------------------------------------------------------------------------

@app.post("/global_explicativity")
async def global_explain(data: G_SHAPRequest):
    """
    Retourne les top N features globales selon SHAP (importance moyenne absolue)
    """
    try:
        # Chargement lazy des shap values globales
        shap_train = get_shap_value_train()
        if not shap_train:
            raise HTTPException(status_code=500, detail="Les SHAP values globales ne sont pas disponibles")

        # Extraction des valeurs et des noms de features
        shap_values_all = np.array([sv.values for sv in shap_train])
        data_all = np.array([sv.data for sv in shap_train])
        feature_names = shap_train[0].feature_names

        # Calcul de l'importance moyenne absolue des features
        mean_abs_shap = np.mean(np.abs(shap_values_all), axis=0)

        # Sélection des indices des top N features
        top_features_idx = np.argsort(mean_abs_shap)[::-1][:data.top_n]

        return {
            "shap_values": shap_values_all[:, top_features_idx].tolist(),
            "values": data_all[:, top_features_idx].tolist(),
            "feature_names": [feature_names[i] for i in top_features_idx]
        }

    except HTTPException as he:
        # Propagation des erreurs HTTP
        raise he
    except Exception as e:
        # Erreur serveur inattendue
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul des SHAP globales : {str(e)}")


# Explicabilité locale
# ----------------------------------------------------------------------------

@app.post("/local_explicativity")
async def local_explain(data: SHAPRequest):
    """
    Retourne les valeurs SHAP pour un client donné.
    """
    try:
        # Chargement lazy des ressources
        shap_test = get_shap_value_test()
        raw_data = get_raw_data_test()
        client_id = data.client_id

        # Vérification que le client existe
        if client_id not in raw_data['SK_ID_CURR'].values:
            raise HTTPException(status_code=404, detail=f"Client {client_id} non trouvé")

        # Position du client dans le DataFrame
        pos = raw_data.index[raw_data['SK_ID_CURR'] == client_id][0]
        sw = shap_test[pos]

        return {
            "client_id": client_id,
            "values": sw.data.tolist(),
            "shap_values": sw.values.tolist(),
            "feature_names": sw.feature_names,
            "base_value": sw.base_values.tolist() if hasattr(sw, 'base_values') else 0,
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de l'extraction des SHAP locales : {str(e)}")


# Données pour plots
# ----------------------------------------------------------------------------
@app.post("/Data_plot_dist")
async def plot_dist(data: SHAPRequest):
    """
    Retourne top N features + distributions pour visualisation.
    """
    
    try:
        # Chargement lazy des ressources
        shap_test = get_shap_value_test()
        raw_data = get_raw_data_test()
        variable_type_df = get_variable_type()
        client_id = data.client_id
        top_n = data.top_n

        # Vérification que le client existe
        if client_id not in raw_data['SK_ID_CURR'].values:
            raise HTTPException(status_code=404, detail=f"Client {client_id} non trouvé")

        # Position du client dans le DataFrame
        pos = raw_data.index[raw_data['SK_ID_CURR'] == client_id][0]
        sw = shap_test[pos]

        # Construction du DataFrame client avec SHAP et valeurs
        df_client = pd.DataFrame({
            "feature": sw.feature_names,
            "shap_value": sw.values,
            "value_transformed": sw.data,
            "value_raw": [raw_data.loc[pos, f] if f in raw_data.columns else None for f in sw.feature_names]
        })

        # Top N features par importance SHAP
        df_client = df_client.iloc[df_client.shap_value.abs().sort_values(ascending=False).index].head(top_n)

        # Préparation des données pour plots (boxplot / countplot)
        plots_data = []
        for f in df_client['feature']:
            val_client = df_client.loc[df_client['feature'] == f, 'value_raw'].values[0]
            plot_type = variable_type_df.loc[variable_type_df['variable'] == f, 'plot_type'].iloc[0]
            col_data = raw_data[f]

            if plot_type == 'countplot':
                values_population = col_data.value_counts().to_dict()
            elif plot_type == 'boxplot':
                values_population = col_data.dropna().tolist()

            plots_data.append({
                "feature": f,
                "type": plot_type,
                "value_client": val_client,
                "values_population": values_population
            })

        return {
            "client_id": client_id,
            "top_features": df_client.to_dict(orient="records"),
            "plots_data": plots_data
        }

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la préparation des données pour plots : {str(e)}")

# ----------------------------------------------------------------------------
# Lancement du serveur
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("API.api:app", host="0.0.0.0", port=DEFAULT_PORT, reload=True, log_level="debug")
