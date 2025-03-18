import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def process_data(csv_path, skewness_threshold=1.0):
    """
    Process the training DataFrame by clipping outliers, handling negative values,
    applying log transform if needed, and normalizing features. Returns the processed
    DataFrame and transformation parameters.

    Parameters:
    - df: pandas DataFrame with features and target
    - target_col: string, name of the target column
    - skewness_threshold: float, threshold for skewness to apply log transform

    Returns:
    - df_processed: processed pandas DataFrame
    - transform_params: dictionary containing transformation parameters
    """
    # Separate features and target
    df = pd.read_csv(csv_path)

    # Separate features and target
    features = df.drop(["hsi_id","vomitoxin_ppb"], axis=1)
    target = df["vomitoxin_ppb"]
    target_col = "vomitoxin_ppb"  
    all_cols = list(features.columns) + ["vomitoxin_ppb"]
    
    # Initialize dictionaries to store transformation parameters
    clip_bounds = {}
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
        
        # Step 1: Clip outliers using IQR
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        clipped_series = series.clip(lower_bound, upper_bound)
        clip_bounds[col] = (lower_bound, upper_bound)
        
        # Step 2: Check skewness to decide on log transform
        skewness = clipped_series.skew()
        
        if abs(skewness) > skewness_threshold:
            log_applied[col] = True
            # Step 3: Handle negative or zero values by shifting
            min_val = clipped_series.min()
            if min_val <= 0:
                shift_value = 1 - min_val  # Shift so minimum becomes 1
                shifted_series = clipped_series + shift_value
                shift_values[col] = shift_value
            else:
                shifted_series = clipped_series
                shift_values[col] = 0
            # Step 4: Apply log transform
            transformed_series = np.log(shifted_series)
        else:
            log_applied[col] = False
            transformed_series = clipped_series
            shift_values[col] = 0
        
        # Assign transformed series to features or target
        if col == target_col:
            processed_target = transformed_series
        else:
            processed_features[col] = transformed_series
    
    # Step 5: Normalize features using MinMaxScaler
    scaler = MinMaxScaler()
    features_normalized = pd.DataFrame(
        scaler.fit_transform(processed_features),
        columns=processed_features.columns,
        index=processed_features.index
    )
    
    # Step 6: Combine processed features and target
    df_processed = pd.concat([features_normalized, processed_target], axis=1)
    
    # Step 7: Store transformation parameters
    transform_params = {
        'clip_bounds': clip_bounds,        # Clipping bounds for each column
        'shift_values': shift_values,      # Shift values for columns needing log transform
        'log_applied': log_applied,        # Boolean flags for log transform per column
        'scaler': scaler,                  # Fitted scaler for features
        'target_log_applied': log_applied[target_col]  # Specific flag for target
    }
    
    return df_processed, transform_params

def transform_test_data(test_df, target_col, transform_params):
    """
    Apply the same transformations to the test DataFrame using parameters from training.

    Parameters:
    - test_df: pandas DataFrame with features and target (unscaled)
    - target_col: string, name of the target column
    - transform_params: dictionary containing transformation parameters from training

    Returns:
    - test_df_processed: processed test pandas DataFrame
    """
    # Separate features and target
    features = test_df.drop(target_col, axis=1)
    target = test_df[target_col]
    
    all_cols = list(features.columns) + [target_col]
    
    processed_features = pd.DataFrame(index=features.index)
    processed_target = pd.Series(index=target.index, name=target_col)
    
    # Apply transformations to each column
    for col in all_cols:
        if col == target_col:
            series = target
        else:
            series = features[col]
        
        # Step 1: Clip using training bounds
        lower, upper = transform_params['clip_bounds'][col]
        clipped_series = series.clip(lower, upper)
        
        # Step 2: Apply shift and log transform if applied during training
        if transform_params['log_applied'][col]:
            shift_value = transform_params['shift_values'][col]
            shifted_series = clipped_series + shift_value
            transformed_series = np.log(shifted_series)
        else:
            transformed_series = clipped_series
        
        # Assign transformed series to features or target
        if col == target_col:
            processed_target = transformed_series
        else:
            processed_features[col] = transformed_series
    
    # Step 3: Normalize features using training scaler
    scaler = transform_params['scaler']
    features_normalized = pd.DataFrame(
        scaler.transform(processed_features),
        columns=processed_features.columns,
        index=processed_features.index
    )
    
    # Step 4: Combine processed features and target
    test_df_processed = pd.concat([features_normalized, processed_target], axis=1)
    
    return test_df_processed