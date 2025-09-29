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

# Ajout du dossier racine au PYTHONPATH
sys.path.append(str(Path(__file__).resolve().parent.parent))

#  import de Config
from Config.config import LOGO_PATH, THRESHOLD


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

def input_type_champs(var, val, vtype, modalities=None):
    """Champ Streamlit sans boutons +/-, valeur par défaut si existante"""
    
    if vtype in ["int", "integer"]:
        default = "" if val is None else str(int(val))
        user_input = st.text_input(f"{var} (entier)", value=default)
        try:
            return int(user_input)
        except ValueError:
            return None  # fallback si l'utilisateur écrit autre chose

    elif vtype in ["float", "double"]:
        default = "" if val is None else f"{float(val):.2f}"
        user_input = st.text_input(f"{var} (float)", value=default)
        try:
            return float(user_input)
        except ValueError:
            return None

    elif vtype == "cat" and modalities:
        if isinstance(modalities, str):
            modalities = [m.strip() for m in modalities.split(",")]
        options = [""] + modalities
        index_default = options.index(val) if val in options else 0
        return st.selectbox(var, options, index=index_default)

    else:
        default = "" if val is None else str(val)
        return st.text_input(var, value=default)


def shap_waterfall_plot(shap_values, values, feature_names, base_value, top_n=10):
    
    shap_val_client = shap.Explanation(values=np.array(shap_values),
                                      data=np.array(values),
                                      feature_names=feature_names,
                                      base_values=base_value)
    fig, ax = plt.subplots(figsize=(12, 6))
    shap.plots.waterfall(shap_val_client, show=False, max_display=top_n)
    st.pyplot(fig)
    plt.clf()

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

@st.cache_data
def get_all_variables():
    return call_api(f"{API_URL}/liste_variables") or []

def get_all_variables_raw_data():
    return call_api(f"{API_URL}/liste_variables_raw_data") or []

def get_palette(daltonien=False):
    return {
        "low": "#96f59c",
        "mid": "#fdae61",
        "high": "#2448FC" if daltonien else "#f46c6c",
    }

def plot_client_distributions(plots_data, colors):
    
    """
    Affiche les distributions des features avec TARGET 0/1 et client identifié
    
    Paramètres:
        - plots_data (list of dict): liste de dictionnaire contenant les données 
        de chacune des variables
        - colors (dict): Dictionnaire des couleurs à utiliser
    """
    
    # Paramètrage de la grille 
    n_cols = 3
    n_rows = (len(plots_data) + n_cols - 1) // n_cols

    # Section Légende
    st.markdown("**Légende :**")
    
    # Création d'une figure vide
    legend_fig = go.Figure()

    # Ajout de la légende correspondant aux bon payeurs
    legend_fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(color=colors["low"], size=12), name='Bons payeurs', showlegend=True
    ))

    # Ajout de la légende correspondant aux mauvais payeurs
    legend_fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(color=colors["high"], size=12), name='Mauvais payeurs', showlegend=True
    ))

    # Ajout de la légende correspondant au client
    legend_fig.add_trace(go.Scatter(
        x=[None], y=[None], mode='markers',
        marker=dict(color="#04d3fd", size=12, symbol='diamond'), name='Client', showlegend=True
    ))

    # Mise en forme de la légende
    legend_fig.update_layout(
        height=50,
        margin=dict(l=0, r=0, t=0, b=0),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        legend=dict(orientation="h", y=1, x=0.2, font=dict(size=16))
    )

    st.plotly_chart(legend_fig, use_container_width=True)

    #  Section graphique

    # Création de la grille
    for i in range(n_rows):
        cols = st.columns(n_cols)
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx >= len(plots_data):
                break

            # Extraction des données
            fdata = plots_data[idx]
            feature = fdata["feature"]

            # Initialisation du graphique
            fig = go.Figure()

            # Ajout de l'histogramme des bons payeurs
            fig.add_trace(go.Histogram(
                    x=fdata["values_0"], opacity=0.5, nbinsx=20, marker_color=colors["low"], showlegend=False
                ))
            
            # Ajout de l'histogramme des mauvais payeurs
            fig.add_trace(go.Histogram(
                    x=fdata["values_1"], opacity=0.5, nbinsx=20, marker_color=colors["high"], showlegend=False
                ))
            
            # Ajout d'une ligne correspondant à la position du client
            if fdata.get("value_client") is not None and not pd.isna(fdata["value_client"]):
                fig.add_vline(
                    x=fdata["value_client"],
                    line=dict(color='#04d3fd', dash='dash', width=3)
                )
          
            # Mise en forme du graphique
            fig.update_layout(
                title=feature,
                height=250,
                margin=dict(l=10, r=10, t=40, b=20),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=False),
                barmode='stack',
                bargap=0.1
            )

            # Affichage du graphique
            cols[j].plotly_chart(fig, use_container_width=True)


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
    st.markdown(f"""

    Ce dashboard permet de visualiser les résultats de prédictions de la probabilité de défaut de paiement d'un client demandant un crédit. 
    Le modèle a été entraîné pour classer les demandes de crédits en deux catégories :
                
    - les demandes jugées peu risquées, qui pourront être acceptées,
    - les demandes jugées risquées, qui seront refusées.
    
    Une présentation de l'outil est disponible dans la section <u><i> Presentation de l'outil </i></u>. 
    On pourra y trouver les sources des données utilisées pour construire l'outil de scoring ainsi qu'un 
    graphique illustrant quelles caractéristiques influencent le plus les résultats de prédiction de manière globale.
                
    Pour obtenir la classe prédite par le modèle, seul l'identifiant de la demande est nécessaire. Les résultats peuvent être
    visualisés sous la forme d'une jauge qui indique le niveau de risque de la demande dans la section 
    <u><i>Outil d'aide à la décision</i></u>.
    A noter que le seuil utilisé pour décider si une demande est risquée ou non a été ajusté en fonction de critères définis par 
    les équipes métier afin de mieux refléter la réalité du risque. Ici le seuil utilisé est de **{THRESHOLD:.2f}**. 
    
    Ainsi les demandes dont la probabilité est inférieure à **{THRESHOLD:.2f}**
    seront jugées comme peu risquées et les demandes supérieur à ce seuil seront jugée comme risquées.

    Dans cette section, il est également possible de voir la contribution des caractéristiques pour un client en particulier grâce à un graphique
    indiquant quelles caractéristiques ont le plus pesé dans la décision, et dans quel sens. Une autre série de graphique pourra être visualisée 
    dans le but de comparer la position du client par rapport aux autres demandes. Ces différents graphiques permettront d'aider le conseiller
    financier à comprendre et à expliquer la décision renvoyée par le modèle.
    
    Il sera également possible de modifier les informations d’un client ou d'ajouter un nouveau dossier.
    Cette fonctionnalité offre la possibilité de tester l’impact de changements concrets, comme une augmentation des revenus ou une diminution 
    des dettes, et ainsi d’identifier les leviers qui pourraient améliorer le profil de risque d’un client. Cela permet non seulement d’évaluer 
    une demande, mais aussi d’accompagner le client de manière constructive en lui indiquant sur quels aspects il peut agir pour renforcer son dossier.
        
            
    Le modèle a été construit à partir de différentes sources de données. Une description des variables utilisée par le 
    modèle est disponible dans la section <u><i>Description des variables</i></u>. 
                
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
        L’outil d’aide à la décision a été construit à partir d’un modèle de machine learning.

        Les données d’entrée du modèle proviennent de la table de demande de crédit, préalablement 
        nettoyée et enrichie de nouvelles variables créées à partir des sources suivantes :

        - Previous Applications
        - Payment Installments
        - Bureau Data
        - Bureau Balance

        Les principales étapes de préparation ont consisté à supprimer les variables 
        présentant plus de 25 % de valeurs manquantes et à corriger certaines valeurs aberrantes.

        Au total, 86 variables ont été retenues pour l’entraînement du modèle.
        Ci dessous, il est possible de visualiser quelles variable ont le plus contribuée aux décisions du modèle. 
        
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

    # Ajout du logo
    st.image(str(LOGO_PATH), use_column_width=True)
    
    # Ajout d'un titre
    st.header("Prédiction du risque de défaut de paiement d'un client")

    # Création d'un champ pour entrer l'identifiant
    client_id = st.number_input(
        "Entrez l'ID de la demande",
        min_value=0, step=1,
        value=st.session_state.get("client_id_input", 372777)
    )
    
    # Ajout d'un message d'aide
    st.caption("Exemple : 372777")

    # Ajout du client_id dans la session
    st.session_state.client_id_input = client_id

    # États des variables de sessions
    st.session_state.setdefault("page_formulaire", False)
    st.session_state.setdefault("data_modified", False)
    st.session_state.setdefault("modified_data", {})
    st.session_state.setdefault("is_new_client", False)  

    # Section Formulaire
    #----------------------------------------------------------------------

    if st.session_state.page_formulaire:

        # ------------------------------------------------------------------
        # Cas de l'ajout d'un nouveau client
        # ------------------------------------------------------------------

        if st.session_state.is_new_client:
            
            # Ajout d'un titre
            st.subheader("Ajout d'un nouveau client")

            # Obtiention d'un client existant pour avoir les colonnes
            sample_client = call_api(
                f"{API_URL}/client_data",
                params={"client_id": 372777}
            ) or {}

            # Liste des variables
            variable_names = list(sample_client.keys())

            # Obtiention des infos sur les variables
            data_info = call_api(
                f"{API_URL}/variables_info",
                payload={"variable_names": variable_names, "goal": "Formulaire"}
            ) or []
            var_info_dict = {d["Row"]: d for d in data_info} if data_info else {}

            modified_data = {}
            vars_utilisees = [d["Row"] for d in data_info if d.get("Utilisé") is True]

            # Création des champs pour entrer les données
            for var in vars_utilisees:
                info = var_info_dict.get(var, {})
                vtype = info.get("Type", "object")
                modalities = info.get("Values", None)     
                modified_data[var] = input_type_champs(var, None, vtype, modalities)

            # Bouton pour ajouter le nouveau client et lancer les prédiction
            if st.button("Créer le client et lancer la prédiction"):
                
                # Ajout du nouveau client
                with st.spinner("Ajout du client..."):
                    response = call_api(
                        f"{API_URL}/add_client",
                        payload=modified_data
                    )

                    # Extraction du nouvel ID renvoyé
                    new_id = response["new_id"]

                # Prédiction
                with st.spinner("Prédiction en cours..."):
                    result = call_api(
                        f"{API_URL}/predict",
                        payload={"client_id": new_id, "modified_data": {}}
                    )
                # Mise à jour des variables de session
                if result:
                    st.session_state.client_id = new_id
                    st.session_state.proba = result["proba"]
                    st.session_state.prediction = result["prediction"]

                # Mise a jour des variables de session
                st.session_state.page_formulaire = False
                st.session_state.data_modified = True
                st.session_state.modified_data = modified_data
                st.rerun()

        # ------------------------------------------------------------------
        # Cas de la modification des données d'un client
        # ------------------------------------------------------------------
        else: 
            # Ajout d'un titre
            st.subheader(f"Modification des données du client {client_id}")

            # Obtention des données brute du client
            client_data = call_api(f"{API_URL}/client_data", params={"client_id": client_id})

            if client_data:

                # Initialisation de la liste de résultat
                modified_data = {}

                # Obtention des informations des variables de la table client
                data_info = call_api(
                    f"{API_URL}/variables_info",
                    payload={
                        "variable_names": list(client_data.keys()),
                        "goal": "Formulaire"                }
                ) or []

                # Reformatage du dictionnaire renvoyé par l'API
                var_info_dict = {d["Row"]: d for d in data_info} if data_info else {}

                # Extraction des variables utilisées et non utilisées
                vars_utilisees = [d["Row"] for d in data_info if d.get("Utilisé") is True]
                vars_non_utilisees = [d["Row"] for d in data_info if d.get("Utilisé") is False]

                # Création des champs de valeurs à modifier
                for var in vars_utilisees:
                    val = client_data.get(var)
                    info = var_info_dict.get(var, {})
                    vtype = info.get("Type", "object")
                    modalities = info.get("Values", None)
                    modified_data[var] = input_type_champs(var, val, vtype, modalities)

                # Ajout des variables non utilisées sans modification
                for var in vars_non_utilisees:
                    modified_data[var] = client_data.get(var)

                # Bouton pour lancer la prédiction sur les données modifiées
                if st.button("Mettre à jour et lancer la prédiction"):
                    with st.spinner("Prédiction en cours..."):
                        result = call_api(
                            f"{API_URL}/predict",
                            payload={"client_id": client_id, "modified_data": modified_data}
                        )

                    # Mise à jour des variables de session
                    if result:
                        st.session_state.client_id = client_id
                        st.session_state.proba = result["proba"]
                        st.session_state.prediction = result["prediction"]

                    # Retour + mémorisation
                    st.session_state.page_formulaire = False
                    st.session_state.data_modified = True
                    st.session_state.modified_data = modified_data
                    st.rerun()

        st.stop()


    # Retour à la section outil d'aide à la décision
    #-----------------------------------------------------------------------
    
    # Ajustement de la taille des boutons pour qu'ils apparaissent tous sur la même ligne
    st.markdown("""
        <style>
        div.stButton > button {
            padding: 0.25rem 0.5rem;  /* réduit la hauteur et largeur */
            font-size: 0.85rem;       /* réduit le texte */
        }
        </style>
    """, unsafe_allow_html=True)


    if "last_clicked_button" not in st.session_state:
        st.session_state.last_clicked_button = None
    
    # Emplacement des boutons 
    cols = st.columns([1.9, 1, 1.2])

    # Bouton pour lancer les prédictions sur les données brutes 
    with cols[0]:
        if st.button("Lancer la prédiction sur les données brutes"):
            st.session_state.last_clicked_button = "predict_button"
            with st.spinner("Prédiction en cours..."):
                result = call_api(f"{API_URL}/predict", payload={"client_id": client_id})
            if result:
                st.session_state.client_id = client_id
                st.session_state.proba = result["proba"]
                st.session_state.prediction = result["prediction"]
                st.session_state.data_modified = False
                st.rerun()
            else :
                for key in ["client_id", "proba", "prediction"]:
                    if key in st.session_state:
                        del st.session_state[key]


    # Bouton pour modifier les données du client séléctionné
    with cols[1]:
        if st.button("Modifier les données"):
            st.session_state.last_clicked_button = "modify_button"
            st.session_state.page_formulaire = True
            st.session_state.is_new_client = False  
            st.rerun()
    
    # Bouton pour ajouter un nouveau client
    with cols[2]:
        if st.button("Ajouter un nouveau client"):
            st.session_state.last_clicked_button = "add_button"
            st.session_state.page_formulaire = True
            st.session_state.is_new_client = True   
            st.rerun()
    
    # Affichage d'un message pour rappeler à l'utilisateur quel est le dernier bouton utilisé
    if st.session_state.last_clicked_button == "predict_button":
        st.markdown(
            '<div style="padding:8px; background-color:#FFD1DC; border-radius:5px; text-align:center;">'
            'Prédiction sur les données brutes'
            '</div>', 
            unsafe_allow_html=True
        )
    elif st.session_state.last_clicked_button == "modify_button":
        st.markdown(
            '<div style="padding:8px; background-color:#FFD1DC; border-radius:5px; text-align:center;">'
            'Prédiction sur les données modifiées'
            '</div>', 
            unsafe_allow_html=True
        )
    elif st.session_state.last_clicked_button == "add_button":
        st.markdown(
            '<div style="padding:8px; background-color:#FFD1DC; border-radius:5px; text-align:center;">'
            'Prédiction sur les données du nouveau client'
            '</div>', 
            unsafe_allow_html=True
        )
    st.text("")


    # Affichage des résultats sous la forme d'une jauge
    #--------------------------------------------------------------------------------------
    
    # Checkbox pour activer le mode inclusif
    daltonien_mode = st.checkbox("Afficher la jauge avec des couleurs inclusives")

    # Affichage des résultats s'ils existent
    if "proba" in st.session_state and "prediction" in st.session_state:
        
        # Extraction du dernier identifiant de client enregistré
        client_id = st.session_state.client_id
        
        # Affichage des résultats en format texte
        st.markdown(f"""
        - Probabilité de défaut de paiement : **{round(st.session_state.proba, 2)}**
        - {"La demande n° **" + str(client_id) + "** a été classée comme risquée  ❌" if st.session_state.prediction else "La demande n° **" + str(client_id) + "** a été classée comme fiable ✅"}
        """)
        st.text("")
        st.text("")
        
        # Affichage des résultats sous la forme d'une jauge
        if daltonien_mode:
            
            # Palette optimisée daltonisme
            colors =  get_palette(daltonien_mode)

            # Ajout de la jauge
            plot_gauge(st.session_state.proba, colors)

            # Ajout d'une légende 
            st.markdown("""
                <div style="font-size:15px; margin-top:10px; text-align:center;">
                <span style="display:inline-block; width:15px; height:15px; background-color:#96f59c; margin-right:5px;"></span> Risque faible &nbsp;&nbsp;
                <span style="display:inline-block; width:15px; height:15px; background-color:#fdae61; margin-right:5px;"></span> Zone intermédiaire &nbsp;&nbsp;
                <span style="display:inline-block; width:15px; height:15px; background-color:#2448FC; margin-right:5px;"></span> Risque élevé
                </div>""", unsafe_allow_html=True)
        else:
            # Palette standard
            colors = get_palette(daltonien_mode)

            # Ajout de la jauge
            plot_gauge(st.session_state.proba, colors)

            # Ajout d'une légende 
            st.markdown("""
                <div style="font-size:15px; margin-top:10px; text-align:center;">
                <span style="display:inline-block; width:15px; height:15px; background-color:#96f59c; margin-right:5px;"></span> Risque faible &nbsp;&nbsp;
                <span style="display:inline-block; width:15px; height:15px; background-color:#f3aa6a; margin-right:5px;"></span> Zone intermédiaire &nbsp;&nbsp;
                <span style="display:inline-block; width:15px; height:15px; background-color:#f46c6c; margin-right:5px;"></span> Risque élevé
                </div>""", unsafe_allow_html=True)

        # Ajout d'un séparateur après la jauge   
        st.markdown("---")

        # Explicabilité SHAP
        #------------------------------------------------------------------------------------------

        # Checknox pour demander l'affichage de l'analyse SHAP locale
        show_explain = st.checkbox("Afficher l'explicabilité locale (SHAP)")
        
        if show_explain:
            
            # Selection du nombre de top feature à afficher
            top_n = st.slider("Nombre de variables à afficher pour SHAP", 
                                    min_value=1, max_value=20, value=10, key="top_n_shap")  
            with st.spinner("Calcul des valeurs SHAP..."):

                # Si des données ont été modifié
                if st.session_state.data_modified:

                    # Extraction des données modifiées
                    client_json = st.session_state.modified_data
                    
                    # Conversion des chaînes "None" en None
                    cleaned_data = {k: (None if v in ["None", ""] else v) for k, v in client_json.items()}

                    # Appel de l'API pour recalculer les valeurs SHAP
                    explain_data = call_api(
                        f"{API_URL}/recalculate_shap?top_n={int(top_n)}",
                        payload=cleaned_data)
                else:

                    # Appel à l'API pour récupérer les valeur shap de base
                    explain_data = call_api(
                        f"{API_URL}/local_explicativity",
                        payload={"client_id": int(st.session_state.client_id), "top_n": int(top_n)}
                    )
            # Construction et affichage du waterfall plot
            if explain_data:
                shap_waterfall_plot(
                    explain_data["shap_values"],
                    explain_data["values"],
                    explain_data["feature_names"],
                    explain_data["base_value"],
                    top_n,
                )

                # Ajout d'une légende
                st.markdown(
                    """
                    <div style="display:flex; flex-direction:column; gap:16px;">

                    <!-- Flèche rouge style waterfall -->
                    <div style="display:flex; align-items:center; gap:12px;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="120" height="30" viewBox="0 0 120 30">
                        <rect x="0" y="10" width="90" height="10" fill="#e53935"/>
                        <polygon points="90,5 120,15 90,25" fill="#e53935"/>
                        </svg>
                        <span style="font-size:16px; color:#333;">
                        Ces variables ont eu tendance à faire <b style="color:#e53935;">augmenter</b> la probabilité de défaut de paiement du client.
                        </span>
                    </div>

                    <!-- Flèche bleue style waterfall -->
                    <div style="display:flex; align-items:center; gap:12px;">
                        <svg xmlns="http://www.w3.org/2000/svg" width="120" height="30" viewBox="0 0 120 30">
                        <rect x="30" y="10" width="90" height="10" fill="#1e88e5"/>
                        <polygon points="30,5 0,15 30,25" fill="#1e88e5"/>
                        </svg>
                        <span style="font-size:16px; color:#333;">
                        Ces variables ont eu tendance à faire <b style="color:#1e88e5;">diminuer</b> la probabilité de défaut de paiement du client.
                        </span>
                    </div>

                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # Ajout d'un séparateur
            st.markdown("---")

        # Histogramme représentant  la position du client
        #------------------------------------------------------------------------------------------

        # Checkbox pour demander l'affichage des histogrammes
        show_dist = st.checkbox("Afficher la position du client")
        if show_dist:
            
            # Définition des paramètres à envoyer à l'API
            payload_top = {
                "client_id": int(st.session_state.client_id),
                "top_n": 10
            }

            # Ajout des données modifiées si elles existent
            if st.session_state.data_modified:
                payload_top["modified_data"] = st.session_state.modified_data

            # Appel de l'API pour récupérer les données à utiliser
            top_features_resp = call_api(f"{API_URL}/Top_features", payload=payload_top)
            top_vars = top_features_resp["top_features"] if top_features_resp else []

            # Liste de toutes les variables disponibles
            all_vars = get_all_variables_raw_data()
            all_vars_clean = [str(x).strip() for x in all_vars]
            top_vars_clean = [str(x).strip() for x in top_vars] 
            # Ajout des top N manquantes dans all_vars

            for v in top_vars_clean:
                if v not in all_vars_clean:
                    all_vars_clean.append(v)

            st.session_state.selected_vars_plot = top_vars_clean.copy()
            
            # Vérification des variables problématiques
            invalid_vars = [v for v in st.session_state.selected_vars_plot if v not in all_vars_clean]
            if invalid_vars:
                st.warning(f"Ces variables ne sont pas présentes dans les options et seront ignorées : {invalid_vars}")

            # Ajout d'une section pour ajouter ou supprimer des variables
            st.markdown("**Sélection des variables à afficher :**")
            selected_vars = st.multiselect(
                    "Cochez les variables à afficher",
                    options=all_vars_clean,
                    default=st.session_state.selected_vars_plot,
                    key="var_multiselect"
                )

            # Si la sélection a changé, mise à jour de la session
            if selected_vars != st.session_state.selected_vars_plot:
                st.session_state.selected_vars_plot = selected_vars

            # Définition des paramètres pour l'API à l'API
            payload_filtered = {
                    "client_id": int(st.session_state.client_id),
                    "variables": st.session_state.selected_vars_plot
                }

            # Ajout des données modifiées si elles existent
            if st.session_state.data_modified:
                payload_filtered["modified_data"] = st.session_state.modified_data

            # Appel de l'API
            dist_data_filtered = call_api(f"{API_URL}/Data_plot_dist", payload=payload_filtered)

            # Affichage du graphique après mise à jour
            if dist_data_filtered and dist_data_filtered.get("plots_data"):
                plot_client_distributions(dist_data_filtered["plots_data"], colors)


#---------------------------------------------------------------------
# Description des variables
#---------------------------------------------------------------------

elif section == "Description des variables":

    # Ajout d'un logo
    st.image(str(LOGO_PATH), use_column_width=True)

    # Ajout d'un titre
    st.header("Description des variables")

    # Ajout d'une description
    st.markdown("""
        Pour obtenir davantage d’informations sur la signification des variables, 
        la section ci-dessous permet de sélectionner celles qui vous intéressent et 
        d’accéder à leur description. Vous pourrez également savoir si chaque variable
        provient directement de la table d’application (Application) ou si elle a été 
        extraite ou bien construite à partir d’autres sources de données (Engineered). 
    """, unsafe_allow_html=True)

    # Obtention de la liste de toutes les variables
    all_variables = get_all_variables() or []

    # Tri de la liste des variables
    all_variables.sort()

    # Ajout d'un champ pour selectionner les variables 
    selected_vars = st.multiselect(
        "Sélectionnez les variables pour voir leur description et type",
        options=all_variables,
        placeholder="Selectionnez une ou plusieurs variables..."
    )
  
    # Si des variables ont été selectionnées
    if selected_vars:

        # Définition des paramètres à envoyer à l'API
        payload = {
            "variable_names": selected_vars,
            "goal": "Description"
        }

        # Appel de l'API pour récupérer les description des variables
        data = call_api(f"{API_URL}/variables_info", payload=payload)

        # Si data existe
        if data:

            # Conversion en DataFrame
            df = pd.DataFrame(data)

            # Affichage légende
            st.subheader("Légende des couleurs")

            # Affichage d'une légende expliquant le code couleur utilisé
            st.markdown("""
                    <div style="display:flex; align-items:center; line-height:1.2; margin-bottom:2px;">
                        <span style="display:inline-block;width:15px;height:15px;background-color:#189696;margin-right:10px;"></span>
                        Application: Ces variables proviennent de la table Application
                    </div>
                    <div style="display:flex; align-items:center; line-height:1.2; margin-bottom:2px;">
                        <span style="display:inline-block;width:15px;height:15px;background-color:#1f0678;margin-right:10px;"></span>
                        Engineered: Ces variables ont été extraites ou construites à partir d'autres sources de données
                    </div>
                """, unsafe_allow_html=True)

            # Filtre de la table sur les variables utilisées par le modèle
            used_df = df[df['Utilisé'] == True]

            # Affichage des descriptions des variables selectionnées utilisées par le modèle
            if not used_df.empty:

                # Ajout d'un titre
                st.subheader("Description des variables utilisées par le modèle")
                
                # Affichage des descriptions
                for _, row in used_df.iterrows():
                    
                    # Extraction des informations
                    var_name = row.get("Row", "Nom inconnu")
                    description = row.get("Description", "Pas de description disponible")
                    source = row.get("Source", "Application")
                    color = "#189696" if source.lower() == "application" else "#1f0678"

                    # Affichage des descriptions
                    st.markdown(
                        f"""
                        <div style="margin-bottom:2px; line-height:1.2;">
                            <span style="display:inline-block;width:15px;height:15px;background-color:{color};margin-right:10px;"></span>
                            <strong>{var_name}</strong> : {description}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )




        
