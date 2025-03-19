import pytest
import numpy as np
from src.processing_draft import process_data
from src.model_training import build_regression_model
from src.evaluation import evaluate_regression
from sklearn.preprocessing import MinMaxScaler


def test_process_data():
    df_processed, transform_params = process_data("data/MLE-Assignment.csv")
    assert df_processed is not None, "Processed DataFrame should not be None"
    assert len(df_processed) > 0, "Processed DataFrame should not be empty"
    assert 'vomitoxin_ppb' in df_processed.columns, "Target column should be present in the processed DataFrame"

def test_build_model():
    model = build_regression_model(input_dim=100, hidden_units=[64, 32], dropout_rate=0.2, l2_reg=0.01)
    assert model is not None, "Model should be built successfully"
    # Check layer count
    assert len(model.layers) > 0, "Model should have layers"

def test_evaluate_regression():
    # Create a dummy model and data for testing
    class DummyModel:
        def predict(self, X):
            return np.zeros((X.shape[0], 1))

    model = DummyModel()
    
    # Generate random data
    X_test = np.random.rand(10, 100)
    y_test = np.random.rand(10, 1)
    
    # Normalize X_test using MinMaxScaler (similar to your preprocessing)
    scaler = MinMaxScaler()
    X_test = scaler.fit_transform(X_test)
    
    # Simulate log transformation and shifting for y_test
    y_test = np.log1p(y_test)  # Apply log transformation
    
    transform_params = {
        'shift_values': {'vomitoxin_ppb': 0},
        'log_applied': {'vomitoxin_ppb': True}  # Simulate that log transformation was applied
    }
    target_col = 'vomitoxin_ppb'
    
    mae, rmse, r2 = evaluate_regression(model, X_test, y_test, transform_params, target_col, log_transform=True)
    assert mae >= 0, "MAE should be non-negative"
    assert rmse >= 0, "RMSE should be non-negative"
    assert -1 <= r2 <= 1, "R2 should be between -1 and 1"