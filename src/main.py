import numpy as np
import optuna
from model_training import train_neural_network, plot_training_history
from evaluation import evaluate_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from interpretability import compute_shap_values, plot_shap_summary
from sklearn.linear_model import Lasso
from processing_draft import process_data

def objective(trial):
    # 1) Load & preprocess
    df, transform = process_data("./data/MLE-Assignment.csv")
    X = df.drop(["vomitoxin_ppb"], axis=1).values
    y = df["vomitoxin_ppb"].values.reshape(-1, 1)

    # 2) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Hyperparameters to optimize
    hidden_units = trial.suggest_categorical('hidden_units', [[64, 32], [128, 64], [256, 128]])
    dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
    l2_reg = trial.suggest_float('l2_reg', 1e-5, 1e-2)
    batch_size = trial.suggest_int('batch_size', 16, 64)
    epochs = trial.suggest_int('epochs', 50, 200)
    
    # 3) Train model
    model, history = train_neural_network(X_train, y_train, epochs=epochs, batch_size=batch_size, hidden_units=hidden_units, dropout_rate=dropout_rate, l2_reg=l2_reg)
    
    # 4) Evaluate
    mae, rmse, r2 = evaluate_regression(model, X_test, y_test)
    
    return mae

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    
    print("Best hyperparameters: ", study.best_params)
    
    # Train final model with best hyperparameters
    best_params = study.best_params
    
    df, transformed = process_data("./data/MLE-Assignment.csv")
    X = df.drop(["vomitoxin_ppb"], axis=1).values
    y = df["vomitoxin_ppb"].values.reshape(-1, 1)

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply Lasso regression for feature selection
    lasso = Lasso(alpha=0.07)
    lasso.fit(X_scaled, y)
    
    # Select the features with non-zero coefficients
    selected_features = np.where(lasso.coef_ != 0)[0]
    print("Selected features: ", selected_features)
    X_selected = X_scaled[:, selected_features]
    
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
    model, history = train_neural_network(X_train, y_train,)
                                        #   , epochs=best_params['epochs'], batch_size=best_params['batch_size'], hidden_units=best_params['hidden_units'], dropout_rate=best_params['dropout_rate'], l2_reg=best_params['l2_reg'])
    
    plot_training_history(history)
    mae, rmse, r2 = evaluate_regression(model, X_train[100:], y_train[100:],transformed,"vomitoxin_ppb")
    mae, rmse, r2 = evaluate_regression(model, X_test, y_test,transformed,"vomitoxin_ppb")
    print("Final model performance - MAE: {}, RMSE: {}, R2: {}".format(mae, rmse, r2))

    shap_values = compute_shap_values(model, X_test)
    plot_shap_summary(shap_values, X_test)
