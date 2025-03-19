import numpy as np
import logging
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def inverse_transform(series, transform_params, col):
    """
    Inverse the log and shift transformations applied during preprocessing.
    """
    if transform_params['log_applied'][col]:
        shift_value = transform_params['shift_values'][col]
        series = np.exp(series) - shift_value
    return series

def evaluate_regression(model, X_test, y_test, transform_params, target_col):
    logging.info("Starting evaluation")
    y_pred = model.predict(X_test).flatten()
    
    # Inverse transform the predictions and target values if log_transform is True
    if log_transform:
        y_pred = inverse_transform(y_pred, transform_params, target_col)
        y_test = inverse_transform(y_test, transform_params, target_col)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    logging.info(f"Evaluation completed - MAE: {mae}, RMSE: {rmse}, R2: {r2}")
    
    # Plot actual vs predicted
    plt.figure(figsize=(6,6))
    plt.scatter(y_test, y_pred, alpha=0.7)
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
    plt.grid(True)
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.xlabel("Actual DON")
    plt.ylabel("Predicted DON")
    plt.title("Actual vs. Predicted DON")
    plt.show()
    
    return mae, rmse, r2