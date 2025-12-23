# Import des librairies
#---------------------------------------------------------------------
import sys
from fastapi.testclient import TestClient
from pathlib import Path

# Définition du chemin absolu du dossier du script
script_dir = Path(__file__).resolve().parent

# Définition du chemin vers le dossier API 
api_path = script_dir.parent / "API"

# Ajout du chemin au path Python pour pouvoir importer app.py
sys.path.append(str(api_path))

# Importation de l'application
from api import app

# Test de la racine 
#--------------------------------------------------------------------------------------
def test_read_root():
    """Test de la route racine"""
    with TestClient(app) as client:  
        # Envoie d'une requête get
        response = client.get("/")

        # Vérification que le code HTTP envoyé est 200
        assert response.status_code == 200

        # Vérification que la réponse JSON correspond au message envoyé
        assert response.json() == {"message": "Bienvenue sur mon API de scoring !"}


# Test de la route predict avec un client existant
#--------------------------------------------------------------------------------------
def test_predict_success():
    """Test prédiction pour un client existant"""
    clientid = {"client_id": 351445}  

    with TestClient(app) as client:
        response = client.post("/predict", json=clientid)

        # Vérification que le code HTTP envoyé est 200
        assert response.status_code == 200

        # Conversion de la réponse en dictionnaire
        data = response.json()

        # Vérification que la réponse contient les clés prédiction et proba
        assert "prediction" in data
        assert "proba" in data
        assert isinstance(data["proba"], (float, int))
        assert 0.0 <= float(data["proba"]) <= 1.0


# Test de la route predict avec un client inexistant
#--------------------------------------------------------------------------------------
def test_predict_client_not_found():
    """Test prédiction pour un client inexistant"""
    clientid = {"client_id": 999999999}

    with TestClient(app) as client:
        response = client.post("/predict", json=clientid)

        # Vérification que le code HTTP envoyé est 404
        assert response.status_code == 404

        # Conversion de la réponse en dictionnaire 
        data = response.json()

        # Vérification que la réponse contient la clé error avec le message non trouvé
        assert "detail" in data
        assert "non trouvé" in data["detail"]


# Test de la validation des entrées pour la route predict 
#--------------------------------------------------------------------------------------
def test_predict_validation_error():
    """Test prédiction avec un id client non valide"""
    clientid = {}  # ID non valide
    
    with TestClient(app) as client:
        response = client.post("/predict", json=clientid)

        # Vérification que l'API renvoie bien 422
        assert response.status_code == 422

        # Vérification que la réponse contient 'detail'
        assert "detail" in response.json()
