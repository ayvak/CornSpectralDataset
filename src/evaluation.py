import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pandas as pd

def inverse_transform_data(series, transform_params, col):
    """
    Inverse the transformations applied to a series.

    Parameters:
    - series: pandas Series with processed data
    - transform_params: dictionary containing transformation parameters from training
    - col: string, name of the column to inverse transform

    Returns:
    - original_series: Series with original data
    """
    # Step 1: Inverse log transform if applied during training
    if transform_params['log_applied'][col]:
        transformed_series = np.exp(series)
        shift_value = transform_params['shift_values'][col]
        shifted_series = transformed_series - shift_value
    else:
        shifted_series = series
    
    # Step 2: Inverse clipping (not needed as clipping is not a reversible operation)
    # Just assign the shifted series to the original series
    
    original_series = shifted_series
    
    return original_series

def evaluate_regression(model, X_test, y_test, transform_params, log_transform=False):


    y_pred = model.predict(X_test).flatten()

    # if log_transform:
    #     # Inverse transform the predictions and target values
    #     y_pred = np.expm1(y_pred)
    #     y_test = np.expm1(y_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print("MAE :", mae)
    print("RMSE:", rmse)
    print("R2  :", r2)
    
    # Plot actual vs predicted
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.set_aspect('equal')
    plt.grid(True)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel("Actual DON")
    plt.ylabel("Predicted DON")
    plt.title("Actual vs. Predicted DON")
    plt.show()
    
    # Residual plot
    residuals = y_test - y_pred.reshape(-1, 1)
    plt.figure(figsize=(6,4))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.gca().set_aspect('equal')
    plt.xlabel("Predicted DON")
    plt.ylabel("Residuals")
    plt.title("Residual Plot")
    plt.show()
    
    return mae, rmse, r2