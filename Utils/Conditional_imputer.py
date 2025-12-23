from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class ConditionalImputer(BaseEstimator, TransformerMixin):
    def __init__(self, ref_col, variables):
        """
        Paramètres
            ref_col : colonne de référence pour la condition
            variables : liste des colonnes à imputer
        """
        
        # Initialisation des paramètres
        self.ref_col = ref_col
        self.variables = variables
        self.medians_ = {}
    
    def fit(self, X, y=None):
        
        # calcul de la médiane des variables pour les cas où ref_col > 0
        for var in self.variables:
            self.medians_[var] = X.loc[X[self.ref_col] > 0, var].median()
        
        return self
    
    def transform(self, X):
        
        # Copie de la table de données
        X = X.copy()
        
        # Pour chaque variable
        for var in self.variables:
            mask = X[self.ref_col] > 0
            # Si ref_col > 0, les NA sont remplacé par la médiane
            X.loc[mask, var] = X.loc[mask, var].fillna(self.medians_[var])
            # Sinon (ref_col <= 0), les NA sont remplacées par -1
            X.loc[~mask, var] = X.loc[~mask, var].fillna(-1)
    
        return X
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array(self.variables)
        return np.array(input_features)
