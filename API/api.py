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

from fastapi import FastAPI, Request, HTTPException, Query
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
import logging
import uvicorn
import shap
import unicodedata
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier


# Imports depuis config
from Config import (
    PIPELINE_PATH,
    CLIENT_PATH,
    VAR_DESC_FR_PATH,
    SHAP_VALUE_TEST_PATH,
    SHAP_VALUE_TRAIN_PATH,
    RAW_DATA_TEST_ALIGNED_PATH,
    EXPLAINER_PATH,
    VARIABLE_TYPE_PATH,
    COL_TYPE_PATH,
    THRESHOLD,
    DEFAULT_PORT, 
    ClientData,
    MAPPING_DICT
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
    app.state.variable_desc_fr = pd.read_csv(VAR_DESC_FR_PATH)


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

        # Création du mapping client_id -> position shap
        if not hasattr(app.state, "id_to_pos"):
            app.state.id_to_pos = {}

        raw_data = get_raw_data_test()
        for idx, client_id in enumerate(raw_data["SK_ID_CURR"].tolist()):
            app.state.id_to_pos[client_id] = idx

    return app.state.shap_value_test

def get_explainer():
    if not hasattr(app.state, "explainer"):
        app.state.explainer = joblib.load(EXPLAINER_PATH)
    return app.state.explainer

def get_raw_data_test():
    if not hasattr(app.state, "raw_data_test_aligned"):
        app.state.raw_data_test_aligned = pd.read_csv(RAW_DATA_TEST_ALIGNED_PATH)
    return app.state.raw_data_test_aligned

def get_variable_type():
    if not hasattr(app.state, "variable_type"):
        app.state.variable_type = pd.read_csv(VARIABLE_TYPE_PATH)
    return app.state.variable_type

def get_col_type():
    if not hasattr(app.state, "col_type"):
        app.state.col_type = pd.read_csv(COL_TYPE_PATH)
    return app.state.col_type

def get_mapping_dict():
    if not hasattr(app.state, "mapping_dict"):
        app.state.mapping_dict = joblib.load(MAPPING_DICT)
    return app.state.mapping_dict


# ----------------------------------------------------------------------------
# Définition des Pydantic Models
# ----------------------------------------------------------------------------

class PredictRequest(BaseModel):
    client_id: int
    modified_data: Optional[Dict[str, Any]] = None

class SHAPRequest(BaseModel):
    client_id: int
    top_n: Optional[int] = None
    variables: Optional[List[str]] = None
    modified_data: Optional[Dict[str, Any]] = None

class G_SHAPRequest(BaseModel):
    top_n: int = 10

class VariableInfoRequest(BaseModel):
    variable_names: list[str]
    goal: str

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
async def predict(data: PredictRequest):
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
    
    # Mise à jour avec les données modifiées si elles existent
    if hasattr(data, "modified_data") and data.modified_data:
        for k, v in data.modified_data.items():
            if k in X.columns:
                
                # Conversion des valeurs invalides en np.nan pour les colonnes numériques
                if pd.api.types.is_numeric_dtype(X[k]):
                    try:
                        X.loc[data.client_id, k] = float(v)
                    except (ValueError, TypeError):
                        X.loc[data.client_id, k] = np.nan
                else:
                    X.loc[data.client_id, k] = v


    # Prédiction de la probabilité de défaut de paiment
    prob_pos = pipeline.predict_proba(X)[0][1]

    # Prédiction de la classe 
    prediction_seuil = 1 if prob_pos >= THRESHOLD else 0

    return {"prediction": prediction_seuil, "proba": prob_pos}

# Liste de toutes les variables
# ----------------------------------------------------------------------------

@app.get("/liste_variables")
async def get_all_variables():
    """Retourne toutes les variables disponibles utilisées par le modèle"""
    try:
        # Filtrage pour ne garder que les variables utilisées
        df_used = app.state.variable_desc_fr[app.state.variable_desc_fr["Utilisé"] == True]

        # Renvoi de la liste des noms de variables
        return df_used["Row"].tolist()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Impossible de récupérer les variables : {str(e)}")

@app.get("/liste_variables_raw_data")
async def get_raw_data_columns():
    """
    Retourne la liste des noms de colonnes du jeu de données raw_data_test_aligned,
    à l'exception de TARGET et SK_ID_CURR.
    """
    try:
        # Récupération du DataFrame
        raw_data = get_raw_data_test()

        # Liste des colonnes à exclure
        exclude_cols = ["TARGET", "SK_ID_CURR"]

        # Renvoi de la liste des colonnes filtrée
        return [col for col in raw_data.columns if col not in exclude_cols]
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Impossible de récupérer les colonnes de raw data : {str(e)}"
        )

# Ajout d'un nouveau client 
# ----------------------------------------------------------------------------

@app.post("/add_client")
async def add_client(modified_data: dict):

    """ Ajoute un nouveau client à la base de données interne de l'application, met à jour
    la table de données de clients, la table contenant les valeurs shap et
     la table de données brutes alignées avec les noms de sortie de la pipeline."""
    
    try:
        
        # Chargement et/ou extraction des données nécessaires
        feature_mapping = get_mapping_dict() 
        clients_df = app.state.clients_df.copy()
        shap_val_test_list = get_shap_value_test()

        # --------------------------------------------------------------------
        # Update de la table de client
        # --------------------------------------------------------------------
        
        # Génération d'un nouvel ID unique
        new_id = int(clients_df.index.max()) + 1

        # Création d'une ligne avec les variables attendues
        new_row = pd.DataFrame([modified_data], index=[new_id])

        # Harmonisation des colonnes (ajout des colonnes manquantes)
        for col in clients_df.columns:
            if col not in new_row.columns:
                new_row[col] = None
        
        # Les variables sont replacées dans le même ordre que dans la table d'origine
        new_row = new_row[clients_df.columns]  

        # Ajout de la ligne dans la table d'origine
        app.state.clients_df = pd.concat([clients_df, new_row])

        # Mise à jour des identifiants des demandes de crédits
        all_ids_update = app.state.clients_df.index.tolist()

        # --------------------------------------------------------------------
        # Update de la table contenant les valeurs shap
        # --------------------------------------------------------------------

        # Obtention des types des colonnes 
        num_cols = app.state.variable_desc_fr.loc[
            app.state.variable_desc_fr['Type_Enc'] == 'Num', 'Row'
        ].tolist()
        label_cols = app.state.variable_desc_fr.loc[
            app.state.variable_desc_fr['Type_Enc'] == 'Label', 'Row'
        ].tolist()
        onehot_cols = app.state.variable_desc_fr.loc[
            app.state.variable_desc_fr['Type_Enc'] == 'OneHot', 'Row'
        ].tolist()

        # Préparation des données pour le pipeline
        expected_columns = app.state.pipeline.named_steps['preprocessing'].feature_names_in_

        # Préparer le pipeline pour le nouveau client
        new_client_pipeline = new_row.copy()

        # Conversion types
        new_client_pipeline = new_client_pipeline.astype(object)
        new_client_pipeline = new_client_pipeline.where(pd.notnull(new_client_pipeline), np.nan)
        for col in num_cols:
            if col in new_client_pipeline.columns:
                new_client_pipeline[col] = pd.to_numeric(new_client_pipeline[col], errors='coerce')

        # Selection des variables dans l'ordre attendu 
        new_client_pipeline = new_client_pipeline[expected_columns]

        # Prétraitement des données
        preprocessing = app.state.pipeline.named_steps['preprocessing']
        X_pre = preprocessing.transform(new_client_pipeline)

        # Feature selection si présente
        if 'feature_selection' in app.state.pipeline.named_steps:
            selector = app.state.pipeline.named_steps['feature_selection']
            X_transformed = selector.transform(X_pre)
            mask = selector.get_support()
            feature_names_after_preprocessing = preprocessing.get_feature_names_out(new_client_pipeline.columns)
            selected_feature_names = feature_names_after_preprocessing[mask]
        else:
            X_transformed = X_pre
            selected_feature_names = preprocessing.get_feature_names_out(new_client_pipeline.columns)

        # Conversion en dataframe
        feature_names_clean = [f.split("__")[-1] for f in selected_feature_names]
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names_clean)

        # Calcul SHAP pour le nouveau client
        explainer = get_explainer()
        new_client_shap = explainer(X_transformed_df)

        # Ajouter à shap_value_test
        if isinstance(shap_val_test_list, list):
            shap_val_test_list.append(new_client_shap)
        else:
            shap_val_test_list = list(shap_val_test_list)
            shap_val_test_list.append(new_client_shap)

        # Update de app.state.shap_value_test
        app.state.shap_value_test = shap_val_test_list
        app.state.id_to_pos[new_id] = len(app.state.shap_value_test) - 1

        #--------------------------------------------------------------------
        # Update de la table raw data
        # --------------------------------------------------------------------

        # mapping final des features pour raw_data_test_aligned
        raw_data = new_row.copy()

        X_raw_preprocessed = preprocessing.transform(raw_data)
        if 'feature_selection' in app.state.pipeline.named_steps:
            X_raw_transformed = selector.transform(X_raw_preprocessed)
        else:
            X_raw_transformed = X_raw_preprocessed

        # Extraction des noms de variables à mettre dans raw data
        final_columns = [v['final'] for v in feature_mapping.values()]

        # Conversion en DataFrame et ajout des colonnes SK_ID_CURR et TARGET
        new_raw_aligned = pd.DataFrame(X_raw_transformed, columns=final_columns, index=[new_id])

        # Ajout de l'identificant et de la target
        new_raw_aligned['SK_ID_CURR'] = new_id
        new_raw_aligned['TARGET'] = np.nan

        # Remplacement des 0.0/1.0 par 'non'/'oui'
        for mapping in feature_mapping.values():
            f_orig, f_final = mapping['original'], mapping['final']
            if f_orig in onehot_cols:
                new_raw_aligned[f_final] = new_raw_aligned[f_final].replace({0.0: 'non', 1.0: 'oui'})
            elif f_orig in label_cols:
                new_raw_aligned[f_final] = new_raw_aligned[f_final].replace({0.0: 'non', 1.0: 'oui', "0.0": "non", "1.0": "oui"})
            elif f_orig in num_cols:
                new_raw_aligned[f_final] = new_row[f_orig].values

        # Gestion du cas particulier de la variable CODE_GENDER
        if 'CODE_GENDER' in new_raw_aligned.columns:
            new_raw_aligned['CODE_GENDER'] = new_raw_aligned['CODE_GENDER'].replace({0.0: 'F', 1.0: 'M'})

        # Concat de la nouvelle ligne dans la table existante
        if hasattr(app.state, "raw_data_test_aligned") and app.state.raw_data_test_aligned is not None:
            app.state.raw_data_test_aligned = pd.concat([app.state.raw_data_test_aligned, new_raw_aligned])
        else:
            app.state.raw_data_test_aligned = new_raw_aligned  
        
        return {"new_id": new_id}

    except Exception as e:
        print("Erreur lors de l'ajout du client :", str(e))
        raise HTTPException(status_code=500, detail=str(e))

# Infos sur des variables spécifiques
# ----------------------------------------------------------------------------

@app.post("/variables_info")
async def get_variable_info(request: VariableInfoRequest):
    """
    Retourne description et source pour une ou plusieurs variables.
    request: VariableInfoRequest contenant variable_names (str, séparés par ,), lang et goal
    """
    try:

        # Récupération des données depuis le request
        variable_names = request.variable_names
        goal = request.goal
        df = app.state.variable_desc_fr
        
        # Filtre de la DataFrame
        var_list = variable_names
        df_filtered = df[df["Row"].isin(var_list)]
        if goal == "Description": 
            result = df_filtered[["Row", "Description", "Source", "Utilisé"]].to_dict(orient="records")
        elif goal == "Formulaire":
            result = df_filtered[["Row", "Description", "Source", "Type", "Values", "Utilisé"]].to_dict(orient="records")
        else:
            raise HTTPException(status_code=400, detail="Goal non supporté. Utilisez 'Description' ou 'Formulaire'.")
        

        for rec in result:
            for k, v in rec.items():
                if pd.isna(v):
                    rec[k] = None

        if not result:
            raise HTTPException(status_code=404, detail=f"Variable(s) non trouvée(s) : {', '.join(var_list)}")

        return result

    except HTTPException:
        raise
    except Exception as e:
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
        pos = app.state.id_to_pos[client_id]
        sw = app.state.shap_value_test[pos]

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
    Gère correctement les variables one-hot encodées via mapping_dict.
    """
    try:
        # Chargement des données nécessaire
        shap_test = get_shap_value_test()      
        raw_data = get_raw_data_test()         
        col_type_df = get_col_type() 
        mapping_dict = get_mapping_dict()   

        # Extraction des paramètres envoyés à l'API
        client_id = data.client_id

        # Vérification client
        if client_id not in raw_data['SK_ID_CURR'].values:
            raise HTTPException(status_code=404, detail=f"Client {client_id} non trouvé")

        # Extraction de la position du clients dans la table de donnée
        pos = app.state.id_to_pos[client_id]
        sw = app.state.shap_value_test[pos]
        
        # Construction dataframe client
        #------------------------------------------

        # Initilisation de la liste
        value_raw = []

        # Pour chaque variable 
        for f in sw.feature_names:

            # Extraction du type de la variable
            col_type_row = col_type_df.loc[col_type_df['variable'] == f, 'col_type']
            col_type = col_type_row.iloc[0] if not col_type_row.empty else "Other"

            # Extraction des valeurs associées à la variable
            mapping = mapping_dict.get(f, {"final": f})
            
            # Extraction du nom cleané
            final_col = mapping["final"]

            if final_col in raw_data.columns:
                
                # Extraction de la valeur du client
                val = raw_data.iloc[pos][final_col]

                # Transformation des Y, 1, "1.0" en oui
                if val in ["Y", 1, "1.0"] and col_type in ["OneHot", "Label"] and final_col!="CODE_GENDER":
                    val = "oui"
                # Conversion des N, 0, "0.0" en non
                elif val in ["N", 0, "0.0"] and col_type in ["OneHot", "Label"] and final_col!="CODE_GENDER":
                    val = "non"
                # Conversion des N, 0, "0.0" en non
                elif val in [ "0", "0.0", 0, 0.0, "non"] and final_col=="CODE_GENDER":
                    val = "F"
                elif val in [ "1", "1.0", 1, 1.0, "oui"] and final_col=="CODE_GENDER":
                    val = "M"
            else:
                val = None
                logger.warning(f"Colonne finale non trouvée dans raw_data : {final_col}")
            
            # Ajout de la valeur la liste
            value_raw.append(val)
        
        # Consutruction de la table de donnée du client
        df_client = pd.DataFrame({
            "feature": [mapping_dict.get(f, {"final": f})["final"] for f in sw.feature_names],
            "shap_value": sw.values.flatten(),
            "value_transformed": sw.data.flatten(),
            "value_raw": np.array(value_raw).flatten()
        })

        # Mise à jour avec données modifiées
        if data.modified_data:
            for original_name, modified_val in data.modified_data.items():
                
                # Récupération du nom final correspondant à au nom de la table modifié
                final_name = None
                for key, v in mapping_dict.items():
                    if v["original"] == original_name:
                        final_name = v["final"]
                        break
                if final_name is None or final_name not in df_client.feature.values:
                    continue  

                # Extraction du type de la variable
                col_type_row = col_type_df.loc[col_type_df['variable'] == final_name, 'col_type']
                col_type = col_type_row.iloc[0] if not col_type_row.empty else "Other"

                # Pour les variable OneHotEncodé, comparaison de la valeur modifiée avec le final_name
                if col_type == "OneHot":
                    val_to_set = "oui" if modified_val == final_name else "non"
                else:
                    val_to_set = modified_val

                # Application de la mise à jour
                df_client.loc[df_client['feature'] == final_name, 'value_raw'] = val_to_set
        
        # Sélection des features à afficher
        if data.variables:
            features_to_plot = [f for f in data.variables if f in df_client.feature.values]
            if not features_to_plot:
                raise HTTPException(status_code=404, detail="Aucune des variables sélectionnées n'existe pour ce client")
            df_client = df_client[df_client.feature.isin(features_to_plot)]
        else:
            raise HTTPException(status_code=400, detail="Veuillez variables")

        # Construction des données a utiliser pour construire les plots
        plots_data = []
        for f in df_client['feature']:
            
            # Extraction de la valeur du client
            val_client = df_client.loc[df_client['feature'] == f, 'value_raw'].values[0]

            # Extraction des valeurs de la variable de l'ensembles des clients
            col_data = raw_data[f]

            # Extraction des valeurs de la variable cible de l'ensemble des clients
            target = raw_data['TARGET']

            # Liste des valeurs des bons payeurs
            values_0 = col_data[target == 0].dropna().tolist()

            # Liste de valeur des mauvais payeurs
            values_1 = col_data[target == 1].dropna().tolist()

            # Ajout des données dans le dictionnaire
            plots_data.append({
                    "feature": f,
                    "value_client": val_client,
                    "values_0": values_0,
                    "values_1": values_1
            })
        
        return {
            "client_id": client_id,
            "top_features": df_client.to_dict(orient="records"),
            "plots_data": plots_data
        }

    except Exception as e:
        logger.error(f"Erreur dans /Data_plot_dist : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/Top_features")
async def top_features(data: SHAPRequest):
    """
    Retourne uniquement les noms des top N features les plus importantes pour un client donné.
    - Si modified_data est fourni : recalcul des SHAP via /recalculate_shap
    - Sinon : SHAP pré-calculés (shap_test)
    - Les noms renvoyés sont ceux “final” après mapping
    """
    try:
        # Chargement des données
        shap_test = get_shap_value_test()      
        raw_data = get_raw_data_test()         
        mapping_dict = get_mapping_dict()

        # Extraction des paramètres envoyé à l'API
        client_id = data.client_id
        top_n = data.top_n

        # Vérification que le client existe
        if client_id not in raw_data['SK_ID_CURR'].values:
            raise HTTPException(status_code=404, detail=f"Client {client_id} non trouvé")

        # Obtention de la position du client
        pos = app.state.id_to_pos[client_id]

        # Cas 1 : données modifiées -> recalcul SHAP
        if data.modified_data:

            client_row = raw_data.iloc[pos].to_dict()
            client_row.update(data.modified_data)
            client_data = ClientData(**client_row)
            recalculated = recalculate_shap(client_data, top_n=top_n)
            feature_names = recalculated["feature_names"]

        # Cas 2 : données originales -> shap_test
        else:
            sw = shap_test[pos]
            importance = np.abs(sw.values)
            idx_sorted = np.argsort(importance)[::-1][:top_n]
            feature_names = [sw.feature_names[i] for i in idx_sorted]

        # Appliquer le mapping pour obtenir les noms “finaux”
        feature_names_mapped = [mapping_dict.get(f, {"final": f})["final"] for f in feature_names]

        return {
            "client_id": client_id,
            "top_features": feature_names_mapped
        }

    except Exception as e:
        logger.error(f"Erreur dans /Top_features : {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/client_data")
async def get_client_data(client_id: int):
    
    # Chargement de la table de données
    clients_df = app.state.clients_df

    # Gestion du cas où l'ID de la demande n'est pas dans la table de données
    if client_id not in clients_df.index:
        raise HTTPException(status_code=404, detail=f"Client {client_id} non trouvé")

    # Obtention des données correspondant à l'ID de la demande souhaitée
    client_dict = clients_df.drop(columns="TARGET").loc[client_id].replace({np.nan: None}).to_dict()
    return client_dict

@app.post("/recalculate_shap")
def recalculate_shap(client_data: ClientData, top_n: int = Query(10, ge=1)):
    """
    Retourne les valeurs SHAP pour un client donné (calcul à la volée).
    Le format de sortie est identique à /local_explicativity.
    """
    try:
        # Conversion des données en dataframe
        df = pd.DataFrame([client_data.dict()])
        df = df.replace({"None": np.nan})


        # Changement des noms de certaines colonnes pour les adapter à ceux passé dans le modèle
        df = df.rename(columns={
            "Another_type_of_loan": "Another type of loan",
            "Cash_loans": "Cash loans",
            "Consumer_credit": "Consumer credit",
            "Consumer_loans": "Consumer loans",
            "Microloan": "Microloan",
            "Revolving_loans": "Revolving loans",
            "business_credit": "business credit"
        })

        # Chargement de la pipeline entrainé et de l'explainer SHAP
        pipeline = app.state.pipeline
        explainer = get_explainer()

        # Prétraitement des données
        X_pre = pipeline.named_steps['preprocessing'].transform(df)
        feature_names_after_preprocessing = pipeline.named_steps['preprocessing'].get_feature_names_out(df.columns)

        # Selection des variables
        selector = pipeline.named_steps['feature_selection']
        X_transformed = selector.transform(X_pre)
        mask = selector.get_support()
        selected_feature_names = feature_names_after_preprocessing[mask]
        
        # Nettoyage des noms des variables
        feature_names_clean = [f.split("__")[-1] for f in selected_feature_names]

        # Conversion en dataframe
        X_transformed_df = pd.DataFrame(X_transformed, columns=feature_names_clean)

        # Extraction du modèle
        model_final = pipeline.named_steps['clf']

        # Calcul des valeur shap 
        shap_values = explainer(X_transformed_df)

        # Extractions des valeurs selon le type de modèle
        if isinstance(model_final, (lgb.LGBMClassifier, RandomForestClassifier)):
            shap_arr = shap_values.values
            base_value = shap_values.base_values[0]
        else:
            pos_idx = list(model_final.classes_).index(1)
            shap_arr = shap_values.values[..., pos_idx]
            base_value = shap_values.base_values[0][pos_idx]

        # Selection des contribution du client
        shap_row = shap_arr[0]
        values_row = X_transformed_df.iloc[0].values
        
        importance = np.abs(shap_row)
        top_idx = np.argsort(importance)[::-1][:top_n]

        shap_row = shap_row[top_idx]
        values_row = values_row[top_idx]
        feature_names_out = [feature_names_clean[i] for i in top_idx]

        # Création du dictionnaire de résultat
        result = {
            "client_id": None, 
            "values": values_row.tolist(),
            "shap_values": shap_row.tolist(),
            "feature_names": feature_names_out,
            "base_value": float(base_value)
        }

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du calcul des SHAP : {str(e)}")

# ----------------------------------------------------------------------------
# Lancement du serveur
# ----------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("API.api:app", host="0.0.0.0", port=DEFAULT_PORT, reload=True, log_level="debug")
