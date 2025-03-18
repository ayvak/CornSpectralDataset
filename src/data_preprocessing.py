import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import GradientBoostingRegressor

class OutlierRemover(BaseEstimator, TransformerMixin):
    """
    Example outlier remover that removes samples
    whose spectral data is beyond a specified z-score threshold.
    """
    def __init__(self, z_thresh=3.0):
        self.z_thresh = z_thresh
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X, y=None):
        z_scores = np.abs((X - np.mean(X, axis=0)) / np.std(X, axis=0))
        mask = (z_scores < self.z_thresh).all(axis=1)
        return X[mask], (y[mask] if y is not None else None), mask

def load_and_preprocess_data(csv_path="../data/MLE-Assignment.csv"):
    # Load data
    df = pd.read_csv(csv_path)
    # Separate features and target
    X = df.drop(["hsi_id","vomitoxin_ppb"], axis=1).values
    y = df["vomitoxin_ppb"].values.reshape(-1, 1)
    
    # 2) Remove outliers (example)
    outlier_remover = OutlierRemover(z_thresh=3.0)
    X_no_outliers, y_no_outliers, mask = outlier_remover.transform(X, y)
    
    # 3) Scale the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y.ravel(), scaler

