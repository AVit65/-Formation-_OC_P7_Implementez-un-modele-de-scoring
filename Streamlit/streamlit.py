# Import des librairies
#---------------------------------------------------------------------

import streamlit as st
import requests
import plotly.graph_objects as go
import os
from pathlib import Path
import time


# chemin absolu basé sur le script 
script_dir = Path(__file__).resolve().parent
logo_path = script_dir.parent / "Images" / "Logo.png"

#---------------------------------------------------------------------
# Constantes
#---------------------------------------------------------------------

API_URL = os.getenv("API_URL", "http://localhost:8001")

#---------------------------------------------------------------------
# Vérification que l'API est disponible au démarrage
#---------------------------------------------------------------------

if "api_status" not in st.session_state:
    st.session_state.api_status = "en cours"

    # Création d'un placeholder qui affiche un message 
    status_placeholder = st.empty()
    status_placeholder.info("En attente que l'API démarre...")

    # Nombre d'essai
    max_retries = 10

    # Pour chaque essai
    for i in range(max_retries):
        try:
            # Tentative de connextion à l'API via une requête sur la route d'entrée
            response = requests.get(f"{API_URL}/", timeout=5)
            
            # Si l'API a répondu correctement
            if response.status_code == 200:
                
                # Affichage d'un nouveau message
                status_placeholder.success("API opérationnelle ✅")
                
                # Mise à jour de api_status
                st.session_state.api_status = "ok"
                
                # Suppression du message après 3 secondes
                time.sleep(3)  
                status_placeholder.empty()
                break
            else:

                # Affichage d'un message d'avertissement
                status_placeholder.warning(
                    f"L'API répond mais avec un code inattendu : {response.status_code}"
                )
        except requests.exceptions.RequestException:
            # Le message reste affiché tant que l'API n'est pas prête
            pass

        # Attente de 1 sec avant l'essai suivant
        time.sleep(1)  

    else:

        # Affichage d'un message d'erreur
        status_placeholder.error(
            "Impossible de joindre l'API après plusieurs tentatives."
        )
        st.session_state.api_status = "error"

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
    st.image(str(logo_path), use_column_width=True)

    st.header("Présentation du Dashboard")
    st.markdown("""
    Ce dashboard permet de visualiser les résultats de prédictions de la proabilité de défaut de paiment d'un client demandant un crédit. 
    Le modèle a été entrainé pour classer les demandes de crédits en deux catégories :
                
    - les demandes jugées peu risquées, qui pourront être acceptées,
    - les demandes jugées risquées, qui seront refusées.
    
    Une présentation de l'outil est disponible dans la section <u><i> Presentation de l'outil </i></u>
                
    Pour obtenir la classe prédite par le modèle, seul l'identitiant de la demande est nécessaire. Les résultats peuvent être
    visualisés sous la forme d'une jauge qui indique le niveau de risque de la demande dans la section 
    <u><i>Outil d'aide à la décesion</i></u>.
    A noter que le seuil utilisé pour décider si une demande est risquée ou non a été ajusté en fonction de critères définis par 
    les équipes métier afin de mieux refléter la réalité du risque. Ici le seuil utilisé est de . 
    
    Ainsi les demandes dont la probabilité est inférieure à
    seront jugées comment peu risquées et les demandes supérieur à ce seuil seront jugée comme risquées.
                
            
    Le modèle a été construit à partir de différentes sources de données. Une description des variables utilisée par le 
    modèle est fournie dans la section <u><i>Description des variables</i></u>. 
  
    
    """, unsafe_allow_html=True)                                            



#----------------------------------------------------------------------------------------------------------------------
# Section Outil d'aide à la décision
#----------------------------------------------------------------------------------------------------------------------

elif section == "Outil d'aide à la décision":

    # Ajout d'un logo
    st.image(str(logo_path), use_column_width=True)

    # Ajout d'un titre
    st.header("Prédiction du risque de défaut de paiment d'un client")

    # Saisie manuelle de l'ID client
    client_id = st.number_input("Entrez l'ID de la demande", min_value=0, step=1, value=351445)
    st.caption("Exemple : 351445")

    # Prédiction
    # --------------------------------------------------------------------------------
    
    if st.button("Prédire"):
        if client_id < 0:
            st.error("Veuillez entrer un ID de demande valide.")
        else:
            with st.spinner("Prédiction en cours, merci de patienter..."):
                try:
                    # Appel API pour prédiction
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

                # Gestion des erreurs techniques python
                except Exception as e:
                    st.error(f"Erreur lors de l'appel à l'API : {e}")

        # Affichage d'une jauge 
        # --------------------------------------------------------------------------------
        if "proba" in st.session_state:

            # Accès à la probas
            proba = st.session_state.proba

            # Affichage des résultats
            st.write("Probabilité de défaut :", round(proba, 2))
            st.write("La demande a été classée comme risquée"
                    if st.session_state.prediction == 1 else "La demande a été classée comme peu risquée")

            # Ajout de la jauge jauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number={'suffix': "%"},
                title={'text': "Risque (%)"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "black"},
                    'steps': [
                        {'range': [0, 53], 'color': "Gold"},
                        {'range': [53, 100], 'color': "DarkSlateBlue"}
                    ],
                    'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.8, 'value': 53}
                }
            ))

            # Mise en page
            fig.update_layout(width=350, height=250, margin=dict(l=20, r=20, t=40, b=20))
            
            # Affichage dans streamlit
            st.plotly_chart(fig, use_container_width=True)

        # Séparateur après la jauge
        st.markdown("---")