import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data(csv_path, skewness_threshold=1.0):
    logging.info("Starting data processing")
    """
    Process the training DataFrame by:
    - Removing target outliers using IQR,
    - Handling negative values, applying log transform if needed, and
    - Normalizing features.
    
    Returns the processed DataFrame and transformation parameters.
    """
    # Load data
    df = pd.read_csv(csv_path)
    logging.info("Data loaded")

    # Separate features and target
    features = df.drop(["hsi_id","vomitoxin_ppb"], axis=1)
    target = df["vomitoxin_ppb"]
    target_col = "vomitoxin_ppb"  

    # --- Remove outliers in the target variable using IQR ---
    Q1_target = target.quantile(0.25)
    Q3_target = target.quantile(0.75)
    IQR_target = Q3_target - Q1_target
    lower_bound_target = Q1_target - 1.5 * IQR_target
    upper_bound_target = Q3_target + 1.5 * IQR_target
    keep_mask = (target >= lower_bound_target) & (target <= upper_bound_target)
    target = target[keep_mask]
    features = features.loc[keep_mask]
    logging.info("Outliers removed from target variable")

    # Recompute the index list of rows we now process
    all_cols = list(features.columns) + [target_col]
    
    # Initialize dictionaries to store transformation parameters
    shift_values = {}
    log_applied = {}
    
    processed_features = pd.DataFrame(index=features.index)
    processed_target = pd.Series(index=target.index, name=target_col)
    
    # Process each column (features and target)
    for col in all_cols:
        if col == target_col:
            series = target
        else:
            series = features[col]
        
        # Check skewness to decide on log transform
        skewness = series.skew()
        if abs(skewness) > skewness_threshold:
            log_applied[col] = True
            # Handle negative or zero values by shifting
            min_val = series.min()
            if min_val <= 0:
                shift_value = 1 - min_val  # Shift so minimum becomes 1
                shifted_series = series + shift_value
                shift_values[col] = shift_value
            else:
                shifted_series = series
                shift_values[col] = 0
            # Apply log transform
            transformed_series = np.log(shifted_series)
        else:
            log_applied[col] = False
            transformed_series = series
            shift_values[col] = 0
        
        # Assign transformed series to features or target
        if col == target_col:
            processed_target = transformed_series
        else:
            processed_features[col] = transformed_series
    
    # Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    features_normalized = pd.DataFrame(
        scaler.fit_transform(processed_features),
        columns=processed_features.columns,
        index=processed_features.index
    )
    logging.info("Features normalized")

    # Combine processed features and target
    df_processed = pd.concat([features_normalized, processed_target], axis=1)
    
    # Store transformation parameters
    transform_params = {
        'shift_values': shift_values,
        'log_applied': log_applied,
        'scaler': scaler,
        'target_log_applied': log_applied[target_col]
    }
    
    logging.info("Data processing completed")
    return df_processed, transform_params