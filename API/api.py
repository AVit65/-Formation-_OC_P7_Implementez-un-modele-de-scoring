"""
API de scoring de clients avec FastAPI.

API compatible avec deux modes de lancement :
  1. python api.py (utilise le bloc `if __name__ == "__main__"`)
  2. uvicorn api:app --reload --port 8001

Fonctionnalités :
- Chargement du pipeline ML et des données clients au démarrage
- Routes pour prédiction
- Gestion des erreurs et logging

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

# ----------------------------------------------------------------------------
# Définition des Pydantic Models
# ----------------------------------------------------------------------------

class ClientID(BaseModel):
    client_id: int

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



# ----------------------------------------------------------------------------
# Lancement du serveur
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("API.api:app", host="0.0.0.0", port=DEFAULT_PORT, reload=True, log_level="debug")
