# Import des librairies
#---------------------------------------------------------------------

import streamlit as st
import requests
import plotly.graph_objects as go
import os
from pathlib import Path
import sys
import time

# Ajout du répertoire racine au path
project_root = Path.cwd()
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

# Import du fichier Config
from Config import LOGO_PATH, THRESHOLD

#---------------------------------------------------------------------
# Constantes
#---------------------------------------------------------------------

API_URL = os.getenv("API_URL", "http://localhost:8001")

#---------------------------------------------------------------------
# Vérification que l'API est disponible au démarrage
#---------------------------------------------------------------------

# Vérification de l'API
try:
    api_ready = requests.get(API_URL, timeout=5).status_code == 200
except requests.exceptions.RequestException:
    api_ready = False

# Affichage du statut
st.sidebar.info("API opérationnelle ✅" if api_ready else "Impossible de joindre l'API ❌")

# bouton pour réessayer
if not api_ready and st.sidebar.button("Réessayer"):
    try:
        api_ready = requests.get(API_URL, timeout=5).status_code == 200
        st.sidebar.info("API opérationnelle ✅" if api_ready else "Impossible de joindre l'API ❌")
    except requests.exceptions.RequestException:
        st.sidebar.warning("Impossible de joindre l'API ❌")

#----------------------------------------------------------------------------------------------------------------------
# Fonctions
#----------------------------------------------------------------------------------------------------------------------

def plot_gauge(proba, colors, threshold=THRESHOLD):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={'suffix': "%", 'font': {'size': 55}},
        title={'text': "Risque de défaut de paiment(%)", 'font': {'size': 28}},
        gauge={
            'axis': {'range': [0, 100],
                     'tickfont' : {'size': 20, 'color':'black'}},
            'bar': {'color': "black", 'thickness':0.3 },
            'steps': [
                {'range': [0, threshold*100-5], 'color': colors['low']},
                {'range': [threshold*100-5, threshold*100+5], 'color': colors['mid']},
                {'range': [threshold*100+5, 100], 'color': colors['high']}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.8, 'value': threshold*100}
        }
    ))
    fig.update_layout(width=350, height=300, margin=dict(l=20, r=20, t=80, b=20))
    st.plotly_chart(fig, use_container_width=True)


#----------------------------------------------------------------------------------------------------------------------
# Menu
#----------------------------------------------------------------------------------------------------------------------

# Sidebar pour la navigation
st.sidebar.title("Menu")
section = st.sidebar.radio("Choisir une section :", ["Présentation du Dashboard", "Présentation de l'outil (à venir)", 
                                                     "Outil d'aide à la décision", "Description des variables (à venir)"])

#----------------------------------------------------------------------------------------------------------------------
# Section Présentation
#----------------------------------------------------------------------------------------------------------------------


if section == "Présentation du Dashboard":

    # Ajout d'un logo
    st.image(str(LOGO_PATH), use_column_width=True)

    # Ajout d'un titre
    st.header("Présentation du Dashboard")

    # Ajout d'une description
    st.markdown(f"""
    Ce dashboard permet de visualiser les résultats de prédictions de la proabilité de défaut de paiment d'un client demandant un crédit. 
    Le modèle a été entrainé pour classer les demandes de crédits en deux catégories :
                
    - les demandes jugées peu risquées, qui pourront être acceptées,
    - les demandes jugées risquées, qui seront refusées.
    
    Une présentation de l'outil est disponible dans la section <u><i> Présentation de l'outil </i></u>
                
    Pour obtenir la classe prédite par le modèle, seul l'identifiant de la demande est nécessaire. Les résultats peuvent être
    visualisés sous la forme d'une jauge qui indique le niveau de risque de la demande dans la section 
    <u><i>Outil d'aide à la décision</i></u>.
    A noter que le seuil utilisé pour décider si une demande est risquée ou non a été ajusté en fonction de critères définis par 
    les équipes métier afin de mieux refléter la réalité du risque. Ici le seuil utilisé est de **{THRESHOLD:.2f}**. 
    
    Ainsi les demandes dont la probabilité est inférieure à **{THRESHOLD:.2f}**
    seront jugées comment peu risquées et celles dont la probabilité est supérieur à ce seuil seront jugée comme risquées.
                
            
    Le modèle a été construit à partir de différentes sources de données. Une description des variables utilisées par le 
    modèle est fournie dans la section <u><i>Description des variables</i></u>. 
  
    
    """, unsafe_allow_html=True)                                            



#----------------------------------------------------------------------------------------------------------------------
# Section Outil d'aide à la décision
#----------------------------------------------------------------------------------------------------------------------

elif section == "Outil d'aide à la décision":

    # Ajout d'un logo
    st.image(str(LOGO_PATH), use_column_width=True)

    # Ajout d'un titre
    st.header("Prédiction du risque de défaut de paiment d'un client")

    # Saisie manuelle de l'ID client
    client_id = st.number_input("Entrez l'ID de la demande", min_value=0, step=1, value=351445)
    st.caption("Exemple : 351445")

    # Prédiction
    # --------------------------------------------------------------------------------
    
    # Si l'utilisateur clique sur prédire
    if st.button("Lancer la prédiction"):
        
        # Gestion du cas où l'ID est invalide
        if client_id < 0:
            st.error("Veuillez entrer un ID de demande valide.")
        
        # Gestion du cas où l'ID est valide
        else:
            
            # Ajout d'une roue de progression
            with st.spinner("Prédiction en cours, merci de patienter..."):
                try:
                    # Appel de l'API pour obtenir les résultat de la prédiction
                    url = f"{API_URL}/predict"
                    payload = {"client_id": int(client_id)}
                    response = requests.post(url, json=payload)

                    # Si l'API répond avec un code 200
                    if response.status_code == 200:
                        result = response.json()

                        # Gestion du cas où l'API renverrait une erreur
                        if "error" in result:
                            st.error(result["error"])
                        else:
                            # Stockage des résultats dans session_state
                            st.session_state.proba = result["proba"]
                            st.session_state.prediction = result["prediction"]
                    
                    # Sinon affichage de l'erreur API
                    else:
                        st.error(f"Erreur API : {response.status_code} - {response.text}")
                        st.session_state.pop("proba", None)
                        st.session_state.pop("prediction", None)        

                # Gestion des erreurs techniques python
                except Exception as e:
                    st.error(f"Erreur lors de l'appel à l'API : {e}")
            
        
    # Si proba a été stocké dans la session
    if "proba" in st.session_state:

        # Extraction de à la probas
        proba = st.session_state.proba

        # Affichage des résultats
        st.markdown(f"""
                - Probabilité de défaut de paiement : **{round(st.session_state.proba, 2)}**
                - {"La demande a été classée comme risquée  ❌" if st.session_state.prediction else "La demande a été classée comme fiable ✅"}
                """)
            
        st.text("")
        
        # Checkbox pour activer le mode inclusif
        daltonien_mode = st.checkbox("Afficher la jauge avec des couleurs inclusives")

        st.text("")
        
        # Affichage d'une jauge 
        # --------------------------------------------------------------------------------
        

        # Définition des palettes
        if daltonien_mode:
            # Palette optimisée daltonisme
            colors = {
                    "low": "#2ca02c" ,    
                    "mid": "#F0E442",    
                    "high": "#56B4E9"    
                }

            # Ajout de la jauge
            plot_gauge(st.session_state.proba, colors)
        else:
            # Palette standard
            colors = {
                    "low": "#2ca02c",    
                    "mid": "#E69F00",    
                    "high": "#ff6666"     
                }

            # Ajout de la jauge
            plot_gauge(st.session_state.proba, colors)

        # Séparateur après la jauge
        st.markdown("---")