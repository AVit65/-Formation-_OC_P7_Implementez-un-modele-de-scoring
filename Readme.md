📊 **Projet**

La société Prêt à Dépenser est une société financière  qui propose des crédits à la consommation. 
Dans une logique de gestion du risque, l’entreprise souhaite mettre en place un outil de scoring de crédits capable d’estimer 
la probabilité qu’un client rembourse son emprunt. Cet outil permettra de classer automatiquement les demandes en deux catégories : 
les demandes de prêts peu risquées qui seront acceptées ou les demandes de prêts risquées qui seront refusées. Pour développer ce modèle 
de classification, la société a fourni un large panel de données provenant de différentes sources et inclus des informations sociodémographiques et des données issues d'autres institutions financières.

🎓 **Compétences évaluées**
- Définir et mettre en œuvre un pipeline d’entraînement des modèles
- Définir la stratégie d’élaboration d’un modèle d’apprentissage supervisé
- Évaluer les performances des modèles d’apprentissage supervisé
- Mettre en œuvre un logiciel de version de code
- Suivre la performance d’un modèle en production et en assurer la maintenance
- Concevoir un déploiement continu d'un moteur d’inférence sur une plateforme Cloud


📂 **Architecture du repository**

```
*Note*: Pour alléger le dépôt GitHub, les objets contenant l'explainer et les valeurs shap inclus sont issus d'échantillons représentatifs extraits des tables complètes utilisées dans les notebooks. Ces échantillons permettent de tester efficacement l’API et le dashboard sans nécessiter l’intégralité des données volumineuses.

OC_P7_Implementer_un_outil_de_scoring/
│
├── .github/workflows/                    # Workflow de test et déploiement automatique de l’API et du dashboard                    │
├── API/                                  # Script principal de l’API et dépendances nécéssaires                  
├── Config/                               # Fichier de configuration                            
├── Data/                                 # Données à télécharger sur Kaggle
├── mlflow/                               # Artifacts et métadonnées MLflow
├── notebooks/                            # Notebooks d’exploration, d’analyse et de modélisation 
├── Output/                                    
│   ├── Analyses_bibariées/               # Résultats des analyses bivariées (visualisations)
│   ├── Analyses_univariées/              # Résultats des analyses univariées (visualisations)
│   ├── Comparaison_modèles/              # Résultats de comparaison des modèles
│   ├── Data_clients/                     # Jeu de données client test pour l'API (échantillon)
│   ├── Data_Drift/                       # Résultat de l'nalyses de dérive des données
│   ├── Evolution_seuil_classification/   # Résultat de l'étude de l'impact du seuil de classification
│   ├── Explicativité/                    # Résultats de l'analyse dexplicabilité du modèle retenu
│   ├── input/                            # Données d'entrée du modèle formatées
│   ├── Optimisation/                     # Résultats des optimisations d’hyperparamètres
│   ├── Performances/                     # Résultats de performance des modèles
│   └── Pipelines/                        # Pipeline de machine learning pré entraîné
│   └── Shap_value/                       # Valeur shap calculées pour l'explicativité
│   └── Variables/                        # Tables avec informations descriptives des variables
├── Streamlit/                            # Script principal du dashboard et dépendances nécéssaires                                   
└── Test/                                 # Script des tests unitaires de l’API 
└── Util/                                 # Code du ConditionalImputer                              
├── README.md                             # Documentation générale du projet
├── .python-version                       # Version de Python utilisée 

```
🗄️ **Données**

Les tables de données brutes listées ci-dessous et utilisées dans les notebook d'exploration, de modélisation et d'analyse de dérive peuvent être téléchargées sur [Kaggle]( https://www.kaggle.com/c/home-credit-default-risk/data)  

- application_{train|test}.csv
- bureau.csv
- bureau_balance.csv
- POS_CASH_balance.csv
- credit_card_balance.csv
- previous_application.csv
- installments_payments.csv
- HomeCredit_columns_description.csv

**Liens vers le dashboard et vers l'API** 

- API: https://api-oc-p7.onrender.com/docs#/  
- Dashboard : https://oc-p7-cu77.onrender.com/

