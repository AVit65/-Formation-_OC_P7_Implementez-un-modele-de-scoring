#---------------------------------------------------------------------
# Imports
#---------------------------------------------------------------------
import streamlit as st
import requests
import plotly.graph_objects as go
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import shap
import seaborn as sns
import os
import sys
import time

# Ajout du dossier racine au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

#  import de Config
from Config.config import LOGO_PATH, DEFAULT_THRESHOLD


#---------------------------------------------------------------------
# Constantes
#---------------------------------------------------------------------
API_URL = os.getenv("API_URL", "http://localhost:8001")

#---------------------------------------------------------------------
# Fonctions
#---------------------------------------------------------------------

def call_api(url, payload=None, params=None):
    """Appel API GET ou POST avec gestion d'erreurs"""
    try:
        if payload:
            response = requests.post(url, json=payload)
        else:
            response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if "error" in data:
            st.error(data["error"])
            return None
        return data
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API : {e}")
        return None

def shap_summary_plot(shap_values, values, feature_names):
    
    "Affiche un summary plot à partir des shap value"

    # Conversion des valeurs shap et des valeurs brutes en array
    shap_values = np.array(shap_values)
    values = np.array(values)

    # Construction du summary plot
    shap.summary_plot(
        shap_values,
        values,
        feature_names=feature_names,
        plot_type="dot",
        show=False
    )

    # Affichage de la figure
    fig = plt.gcf()
    st.pyplot(fig, bbox_inches="tight")
    plt.clf()


def shap_waterfall_plot(shap_values, values, feature_names, base_value, top_n=10):
    
    shap_val_client = shap.Explanation(values=np.array(shap_values),
                                      data=np.array(values),
                                      feature_names=feature_names,
                                      base_values=base_value)
    fig, ax = plt.subplots(figsize=(12, 6))
    shap.plots.waterfall(shap_val_client, show=False, max_display=top_n)
    st.pyplot(fig)
    plt.clf()

def plot_gauge(proba, threshold=DEFAULT_THRESHOLD):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=proba * 100,
        number={'suffix': "%"},
        title={'text': "Risque (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "black"},
            'steps': [
                {'range': [0, threshold*100], 'color': "Gold"},
                {'range': [threshold*100, 100], 'color': "DarkSlateBlue"}
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.8, 'value': threshold*100}
        }
    ))
    fig.update_layout(width=350, height=250, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

@st.cache_data
def get_all_variables():
    return call_api(f"{API_URL}/liste_variables") or []

def plot_client_distributions(plots_data):
    """Affiche les distributions des features et la position du client"""
    n_features = len(plots_data)
    n_cols = 4
    n_rows = int(np.ceil(n_features / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), constrained_layout=True)
    axes = axes.flatten()
    
    for i, fdata in enumerate(plots_data):
        feature = fdata["feature"]
        val_client = fdata["value_client"]
        values_population = fdata["values_population"]

        if fdata["type"] == "countplot":
            values_clean = {}
            for cat, count in values_population.items():
                try:
                    clean_cat = int(float(cat))
                except ValueError:
                    clean_cat = cat
                values_clean[clean_cat] = count

            df_tmp = pd.DataFrame({feature: sum([[cat]*count for cat, count in values_clean.items()], [])})
            df_tmp['is_client'] = df_tmp[feature] == val_client
            sns.countplot(x=feature, data=df_tmp, hue='is_client',
                          palette={False: 'skyblue', True: 'blue'},
                          ax=axes[i],
                          order=sorted(values_clean.keys(), key=str))
            axes[i].legend_.remove()
            axes[i].set_xlabel(f"Count of {feature}", fontsize=12)
            axes[i].set_ylabel("Count", fontsize=12)

        elif fdata["type"] == "boxplot":
            sns.boxplot(y=values_population, ax=axes[i], color="skyblue")
            axes[i].axhline(val_client, color='blue', linestyle='--', linewidth=2)
            axes[i].set_ylabel(feature, fontsize=12)

        axes[i].set_title(feature, fontsize=14)
        axes[i].tick_params(axis='x', labelsize=10)
        axes[i].tick_params(axis='y', labelsize=10)
    
    # Suppression des axes inutilisés
    for j in range(i+1, n_rows * n_cols):
        fig.delaxes(axes[j])

    plt.subplots_adjust(wspace=0.3, hspace=0.5)
    st.pyplot(fig)
    plt.clf()

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
                
                # Suppression du message après 5 secondes
                time.sleep(5)  
                status_placeholder.empty()

                break
            # Cas codes temporaires (API en train de redémarrer / surcharge)
            elif response.status_code in [429, 503]:
                status_placeholder.info(
                    f"API pas encore prête (code {response.status_code}), nouvel essai dans 5s..."
                )

            # Autres cas inattendus
            else:
                status_placeholder.warning(
                    f"L'API répond mais avec un code inattendu : {response.status_code}"
                )

        except requests.exceptions.RequestException:
            # Le message reste affiché tant que l'API n'est pas prête
            pass

        # Attente de 5 sec avant l'essai suivant
        time.sleep(5)  

    else:

        # Affichage d'un message d'erreur
        status_placeholder.error(
            "Impossible de joindre l'API après plusieurs tentatives."
        )
        st.session_state.api_status = "error"
        
#---------------------------------------------------------------------
# Création du Menu
#---------------------------------------------------------------------


st.sidebar.title("Menu")
section = st.sidebar.radio("Choisir une section :", ["Présentation du Dashboard",
                                                     "Présentation de l'outil",
                                                     "Outil d'aide à la décision",
                                                     "Description des variables"])

#---------------------------------------------------------------------
# Présentation du Dashboard
#---------------------------------------------------------------------
if section == "Présentation du Dashboard":
    
    # Ajout d'un logo
    st.image(str(LOGO_PATH), use_column_width=True)

    st.header("Présentation du Dashboard")
    st.markdown("""

    Ce dashboard permet de visualiser les résultats de prédictions de la proabilité de défaut de paiment d'un client demandant un crédit. 
    Le modèle a été entrainé pour classer les demandes de crédits en deux catégories :
                
    - les demandes jugées peu risquées, qui pourront être acceptées,
    - les demandes jugées risquées, qui seront refusées.
    
    Une présentation de l'outil est disponible dans la section <u><i> Presentation de l'outil </i></u>
                
    Pour obtenir la classe prédite par le modèle, seul l'identifiant de la demande est nécessaire. Les résultats peuvent être
    visualisés sous la forme d'une jauge qui indique le niveau de risque de la demande dans la section 
    <u><i>Outil d'aide à la décision</i></u>.
    A noter que le seuil utilisé pour décider si une demande est risquée ou non a été ajusté en fonction de critères définis par 
    les équipes métier afin de mieux refléter la réalité du risque. Ici le seuil utilisé est de . 
    
    Ainsi les demandes dont la probabilité est inférieure à
    seront jugées comment peu risquées et les demandes supérieur à ce seuil seront jugée comme risquées.
                
            
    Le modèle a été construit à partir de différentes sources de données. Une description des variables utilisée par le 
    modèle est fournie dans la section <u><i>Description des variables</i></u>. 
                
    """, unsafe_allow_html=True)

#---------------------------------------------------------------------
# Présentation du modèle / outil
#---------------------------------------------------------------------

elif section == "Présentation de l'outil":
    
    # Ajout d'un logo
    st.image(str(LOGO_PATH), use_column_width=True)

    # Ajout d'un titre 
    st.header("Présentation de l'outil")

    # Ajout d'une description
    st.markdown("""
        L’outil d’aide à la décision a été construit à partir d’un modèle de régression logistique.

        Les données d’entrée du modèle proviennent de la table de demande de crédit, préalablement 
        nettoyée et enrichie de nouvelles variables créées à partir des sources suivantes :

        - Previous Applications
        - Payment Installments
        - Bureau Data
        - Bureau Balance

        Les principales étapes de préparation ont consisté à supprimer les variables 
        présentant plus de 20 % de valeurs manquantes et à corriger certaines valeurs aberrantes.

        Au total, 85 variables ont été retenues pour l’entraînement du modèle.
        Ci dessous, il est possible de visualiser quelles variable ont le plus contribuée au décision du modèle. 
        
    """, unsafe_allow_html=True)

    # Ajout d'un sous titre
    st.header("Contribution globale des variables")
    
    # Choix du nombre de variables à afficher
    top_n = st.slider("Choisissez le nombre de variable à afficher", min_value=0, max_value=20, value=10)
    
    # Requète à l'API pour récupérer les valeur SHAP des top variables
    data_summary = call_api(f"{API_URL}/global_explicativity", payload={"top_n": int(top_n)})

    # Si les données récupérées sont non vide
    if data_summary:

        with st.spinner("Récupération des données..."):

            # Construction du summary plot
            shap_summary_plot(data_summary["shap_values"], data_summary["values"], data_summary["feature_names"])

#---------------------------------------------------------------------
# Outil d'aide à la décision
#---------------------------------------------------------------------

elif section == "Outil d'aide à la décision":

    # Ajout d'un logo
    st.image(str(LOGO_PATH), use_column_width=True)

    # Ajout d'un titre 
    st.header("Prédiction du risque de défaut de paiement d'un client")
    
    # Saisie du client
    client_id = st.number_input("Entrez l'ID de la demande", min_value=0, step=1, value=351445)
    st.caption("Exemple : 351445")

    # Bouton pour prédiction
    if st.button("Prédire"):
        if client_id < 0:
            st.error("Veuillez entrer un ID de demande valide.")
        else:
            with st.spinner("Prédiction en cours, merci de patienter..."):
                result = call_api(f"{API_URL}/predict", payload={"client_id": int(client_id)})
                if result:
                    st.session_state.client_id = client_id
                    st.session_state.proba = result["proba"]
                    st.session_state.prediction = result["prediction"]
                else:
                    st.session_state.pop("proba", None)
                    st.session_state.pop("prediction", None)

    # Affichage de la prédiction si elle existe
    if "proba" in st.session_state:
        st.write("Probabilité de défaut :", round(st.session_state.proba, 2))
        st.write(
            "La demande a été classée comme risquée" if st.session_state.prediction else "La demande a été classée comme peu risquée"
        )
        plot_gauge(st.session_state.proba)

        st.markdown("---")

        # Explicabilité SHAP
        show_explain = st.checkbox("Afficher l'explicabilité locale (SHAP)")
        if show_explain:
            top_n = st.number_input("Nombre de variables à afficher pour SHAP", min_value=1, value=10)
            with st.spinner("Récupération des données..."):
                explain_data = call_api(
                    f"{API_URL}/local_explicativity",
                    payload={"client_id": int(st.session_state.client_id), "top_n": int(top_n)}
                )
                if explain_data:
                    shap_waterfall_plot(
                        explain_data["shap_values"],
                        explain_data["values"],
                        explain_data["feature_names"],
                        explain_data["base_value"],
                        top_n,
                    )

            # Position du client
            show_dist = st.checkbox("Afficher la position du client")
            if show_dist:
                dist_data = call_api(
                    f"{API_URL}/Data_plot_dist",
                    payload={"client_id": int(st.session_state.client_id), "top_n": int(top_n)},
                )
                if dist_data:
                    with st.spinner("Construction des graphiques..."):
                        plot_client_distributions(dist_data["plots_data"])

#---------------------------------------------------------------------
# Description des variables
#---------------------------------------------------------------------
elif section == "Description des variables":

    # Ajout d'un logo
    st.image(str(LOGO_PATH), use_column_width=True)

    # Ajout d'un titre
    st.header("Description des variables")

    st.markdown("""
        Pour obtenir davantage d’informations sur la signification des variables, 
        la section ci-dessous permet de sélectionner celles qui vous intéressent et 
        d’accéder à leur description. Vous pourrez également savoir si chaque variable
        provient directement de la table d’application (Application) ou si elle a été 
        extraite ou bien construite à partir d’autres sources de données (Engineered). 
        
    """, unsafe_allow_html=True)


    all_variables = get_all_variables() or []
    all_variables.sort()

    selected_vars = st.multiselect(
        "Sélectionnez les variables pour voir leur description et type",
        options=all_variables,
        placeholder="Selectionnez une ou plusieurs variables..."
    )

    if selected_vars:
        data = call_api(f"{API_URL}/variables_info", params={"variable_names": ",".join(selected_vars)})
        if data:
            df = pd.DataFrame(data)
            st.dataframe(df, use_container_width=True)
