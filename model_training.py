import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import matplotlib.pyplot as plt

def build_regression_model(input_dim, hidden_units, dropout_rate, l2_reg):
    model = keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(input_dim,)))
    
    for units in hidden_units:
        model.add(keras.layers.Dense(units, activation="relu", kernel_regularizer=regularizers.l2(l2_reg)))
        model.add(keras.layers.Dropout(dropout_rate))
    model.add(keras.layers.Dense(1, activation="linear"))  # Output layer for regression
    
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model

def train_gradient_boosting(X, y, n_estimators=100, max_depth=3, learning_rate=0.1, min_samples_split=2, min_samples_leaf=1):
    model = GradientBoostingRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    model.fit(X, y)
    return model

def train_neural_network(X, y, val_split=0.2, epochs=150, batch_size=16, hidden_units=[64, 32], dropout_rate=0.2, l2_reg=0.01):
    input_dim = X.shape[1]
    model = build_regression_model(input_dim=input_dim, hidden_units=hidden_units, dropout_rate=dropout_rate, l2_reg=l2_reg)
    
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    
    history = model.fit(
        X, y,
        validation_split=val_split,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0 
    )
    
    return model, history

def train_random_forest(X, y, n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1):
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        random_state=42
    )
    
    model.fit(X, y)
    return model

def plot_training_history(history):
    # Plot training & validation loss values
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    # Plot training & validation MAE values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'])
    plt.plot(history.history['val_mae'])
    plt.title('Model MAE')
    plt.ylabel('MAE')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.show()