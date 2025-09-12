**CreditRiskScore**

La société Prêt à Dépenser est une société financière  qui propose des crédits à la consommation. 
Dans une logique de gestion du risque, l’entreprise souhaite mettre en place un outil de scoring de crédits capable d’estimer 
la probabilité qu’un client rembourse son emprunt. Cet outil permettra de classer automatiquement les demandes en deux catégories : 
les demandes de prêts peu risquées qui seront acceptées ou les demandes de prêts risquées qui seront refusées. Pour développer ce modèle 
de classification, la société a fourni un large panel de données provenant de différentes sources et inclus des informations sociodémographiques et des données issues d'autres institutions financières.

**Architecture du repository**

```
OC_P7_Implementer_un_outil_de_scoring/
│
├── .github/workflows/                         
│   ├── test_and_deployment.yml                # Workflow de test et déploiement automatique de l’API et du dashboard
│
├── API/                                       
│   ├── __init__.py                            # Fichier d’initialisation 
│   └── api.py                                 # Script principal de l’API
│   └── requirements.txt                       # Liste des dépendances Python nécessaires
│
├── Config/                                    
│   ├── __init__.py                            # Fichier d’initialisation 
│   └── config.py                              # Fichier de configuration
│ 
├── Data/                                      # Données à télécharger sur Kaggle
│ 
├── notebooks/                                 # Notebooks d’exploration, d’analyse et de modélisation
│  └── Notebook_EDA_FeatEng.ipynb              # Notebook d'analyse du data drift
│  └── Notebook_datadrift.ipynb                # Notebook d'exploration et de nettoyage des données
│  └── Notebook_modelisation.ipynb             # Notebook de développement de l'outil de scoring
│  └── requirements.txt                        # Liste des dépendances Python nécessaires
│  └── requirements_DataDrift.txt              # Liste des dépendances Python nécessaires pour le Data Drift
│ 
├── Output/                                    
│   ├── Data/clients/App_test_final.csv        # Jeu de données client test pour l'API
│   └── Pipelines/pipeline_to_deployed.joblib  # Pipeline de machine learning pré entraîné
│
├── Streamlit/                                 
│   └── streamlit.py                           # Script principal du dashboard
│   └── requirements.txt                       # Liste des dépendances Python nécessaires
│ 
└── Test/                                      
│  ├── __init__.py                             # Fichier d’initialisation 
│  └── test_api.py                             # Script des tests unitaires de l’API
│  └── requirements.txt                        # Liste des dépendances Python nécessaires
│ 
└── Util/                                      
│  ├── __init__.py                             # Fichier d’initialisation 
│  └── Conditional_imputer.py                  # Code du ConditionalImputer
│
├── README.md                                  # Documentation générale du projet
├── .gitignore                                 # Liste des fichiers et dossiers à ignorer par Git
├── .python-version                            # Version de Python utilisée 

```
**Données**

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

