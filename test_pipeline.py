import pytest
import numpy as np
from src.data_preprocessing import load_and_preprocess_data
from src.model_training import build_regression_model

def test_load_and_preprocess_data():
    X, y, scaler = load_and_preprocess_data("../data/corn_data.csv")
    assert X.shape[0] == len(y), "X and y must have the same number of samples"
    assert X.shape[1] > 0, "Features must be present"

def test_build_model():
    model = build_regression_model(input_dim=100)
    assert model is not None, "Model should be built successfully"
    # Check layer count
    assert len(model.layers) > 0, "Model should have layers"