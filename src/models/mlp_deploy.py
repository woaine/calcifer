import os
import random
import numpy as np
import tensorflow as tf
import pandas as pd
import joblib
import argparse

from keras.api.models import Sequential
from keras.api.layers import InputLayer, Dense, PReLU, BatchNormalization, Activation, Dropout
from keras.api.regularizers import L2
from keras.api.metrics import R2Score, RootMeanSquaredError
from keras.api.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.api.backend import clear_session

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error

from math import sqrt

import sys

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if src_root not in sys.path:
    sys.path.append(src_root)
    
from features import create_features
from sklearn.model_selection import train_test_split

# Set seed for reproducibility
SEED_VALUE = 42
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

def create_model(n_features, layers, activation, l2_alpha, batch_normalization, dropout, optimizer):
    model = Sequential([InputLayer(shape=(n_features,))])
    for neurons in layers:
        model.add(Dense(neurons, kernel_initializer='he_uniform', kernel_regularizer=L2(l2_alpha) if l2_alpha else None))
        if batch_normalization:
            model.add(BatchNormalization())
        model.add(PReLU() if activation == 'prelu' else Activation(activation))
        if dropout:
            model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae", RootMeanSquaredError(), R2Score()])
    
    return model
        
def train(X_train, y_train, X_test, y_test, hyperparameters: tuple, epochs:int, scaler: str, l2_alpha: float, batch_normalization: bool, dropout: float, optimizer: str, y_scaler, model_path: str, training_history_path: str):
    layers, activation, batch_size = hyperparameters
    
    model = create_model(X_train.shape[1], layers, activation, l2_alpha, batch_normalization, dropout, optimizer)
    callbacks = [
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=25, min_lr=1e-6),
        EarlyStopping(monitor='loss', patience=epochs, restore_best_weights=True)
    ]

    result = model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, callbacks=callbacks, validation_data=(X_test, y_test))
    y_pred = model.predict(X_test)

    y_pred = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test = y_scaler.inverse_transform(y_test.reshape(-1, 1))

    mae = calculate_confidence_intervals(y_test, y_pred, 'mae')
    rmse = calculate_confidence_intervals(y_test, y_pred, 'rmse')

    history, conf = process_results(result, mae, rmse, hyperparameters, epochs, scaler, y_scaler)
    save_results(history, conf, training_history_path)
    model.save(os.path.join(model_path, 'best_model.keras'))

    clear_session()

def process_results(result: dict, mae: tuple, rmse: tuple, hyperparameters: tuple, epochs: int, scaler: str, y_scaler: object):
    scale_factor = y_scaler.scale_[0] if scaler == 'standard' else y_scaler.data_max_[0] - y_scaler.data_min_[0]

    history = {
        'layers': [hyperparameters[0]]*epochs,
        'activation': [hyperparameters[1]]*epochs,
        'batch_size': [hyperparameters[2]]*epochs,
    }
    metrics = ['loss', 'val_loss', 'mae', 'val_mae', 'root_mean_squared_error', 'val_root_mean_squared_error']
    for k, v in result.history.items():
        if k in metrics:
            scale_adjustment = scale_factor**2 if 'loss' in k and scaler == 'minmax' else scale_factor
            history[k] = [val * scale_adjustment for val in v]
        else:
            history[k] = v

    conf = {
        'layers': hyperparameters[0],
        'activation': hyperparameters[1],
        'batch_size': hyperparameters[2],
        'mae': mae[0],
        'mae_lower': mae[1],
        'mae_upper': mae[2],
        'rmse': rmse[0],
        'rmse_lower': rmse[1],
        'rmse_upper': rmse[2],
    }

    return history, conf

def save_results(history, conf, training_history_path):
    history = pd.DataFrame(history)
    file_path = f"{training_history_path}/best_model_training.csv"
    history.to_csv(file_path, index=False)

    conf = pd.DataFrame(conf)
    file_path = f"{training_history_path}/conf.csv"
    conf.to_csv(file_path, index=False)

def calculate_confidence_intervals(y_true, y_pred, metric='rmse', alpha=0.05, n_bootstraps=1000):
    """
    Calculate confidence intervals for model performance metrics using bootstrap.
    
    Parameters:
    -----------
    y_true : array-like
        True target values
    y_pred : array-like
        Predicted target values
    metric : str, optional (default='rmse')
        Performance metric to calculate ('rmse', 'mse', or 'mae')
    alpha : float, optional (default=0.05)
        Significance level for confidence intervals (95% CI = 0.05)
    n_bootstraps : int, optional (default=1000)
        Number of bootstrap samples
        
    Returns:
    --------
    tuple
        (metric_value, lower_bound, upper_bound)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    
    # Calculate the actual metric value
    if metric == 'rmse':
        actual_metric = sqrt(mean_squared_error(y_true, y_pred))
    elif metric == 'mse':
        actual_metric = mean_squared_error(y_true, y_pred)
    elif metric == 'mae':
        actual_metric = mean_absolute_error(y_true, y_pred)
    else:
        raise ValueError("Metric must be 'rmse', 'mse', or 'mae'")
    
    # Initialize array to store bootstrap results
    bootstrap_metrics = np.zeros(n_bootstraps)
    
    # Generate bootstrap samples and calculate metrics
    for i in range(n_bootstraps):
        # Sample with replacement
        indices = np.random.choice(len(y_true), len(y_true), replace=True)
        
        # Calculate metric on bootstrap sample
        if metric == 'rmse':
            bootstrap_metrics[i] = sqrt(mean_squared_error(y_true[indices], y_pred[indices]))
        elif metric == 'mse':
            bootstrap_metrics[i] = mean_squared_error(y_true[indices], y_pred[indices])
        else:  # mae
            bootstrap_metrics[i] = mean_absolute_error(y_true[indices], y_pred[indices])
    
    # Calculate confidence interval bounds
    lower_bound = np.percentile(bootstrap_metrics, alpha/2 * 100)
    upper_bound = np.percentile(bootstrap_metrics, (1 - alpha/2) * 100)
    
    return actual_metric, lower_bound, upper_bound

def load_and_scale_data(augmented:bool, preprocessed: bool, feature_engineered: bool, scaler: str, model_path: str):
    data_file = 'preprocessed_augmented_data.csv' if augmented and preprocessed else \
                'augmented_data.csv' if augmented else \
                'preprocessed_external_data.csv' if preprocessed else \
                'external_data.csv'
    
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed', data_file))
    data = pd.read_csv(data_path)

    if feature_engineered:
        data = create_features(data)

    data['Tg'], data['Tc'] = round(data['Tg'], 2), round(data['Tc'], 1)

    X, y = data.drop(columns=['Tc']).to_numpy(), data['Tc'].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED_VALUE)

    if scaler == 'standard':
        x_scaler = StandardScaler().fit(X_train)
        y_scaler = StandardScaler().fit(y_train.reshape(-1, 1))
    else:
        x_scaler = MinMaxScaler().fit(X_train)
        y_scaler = MinMaxScaler().fit(y_train.reshape(-1, 1))
        
    X_train = x_scaler.transform(X_train)
    X_test = x_scaler.transform(X_test)
    y_train = y_scaler.transform(y_train.reshape(-1, 1)).flatten()
    y_test = y_scaler.transform(y_test.reshape(-1, 1)).flatten()

    joblib.dump(x_scaler, os.path.join(model_path, 'fs.pkl'))
    joblib.dump(y_scaler, os.path.join(model_path, 'ts.pkl'))

    return (X_train, y_train, X_test, y_test), y_scaler
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape',  type=str, required=True, help='Number of neurons per layer')
    parser.add_argument('--activation', type=str, required=True, help='Activation function')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size to process')
    parser.add_argument('--augmented', type=int, default=0, help='Augmented or non-augmented')
    parser.add_argument('--preprocessed', type=int, default=0, help='Preprocessed or non-processed')
    parser.add_argument('--feature_engineered', type=int, default=0, help='Feature-engineered or not')
    parser.add_argument('--scaler', type=str, default=None, help='Scaler to use')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train a model')
    parser.add_argument('--l2_alpha', type=float, default=0.001, help='Apply L2 regularizer')
    parser.add_argument('--batch_normalization', type=int, default=1, help='Apply batch optimization to architecture')
    parser.add_argument('--dropout', type=float, default=0.2, help='Apply dropout rate to architecture')
    parser.add_argument('--optimizer', type=str, default='adamw', help='Optimizer to use')
    opt = parser.parse_args()

    scale = 'standardized' if opt.scaler == 'standard' else 'normalized'
    feature = 'engineered' if opt.feature_engineered else 'non_engineered'
    preprocessing = 'preprocessed' if opt.preprocessed else 'non_processed'
    data_type = 'augmented' if opt.augmented else 'external'
    model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/mlp/', scale, feature, preprocessing, data_type))
    training_history_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../reports/results/training', scale, feature, preprocessing, data_type))

    os.makedirs(model_path, exist_ok=True)
    os.makedirs(training_history_path, exist_ok=True)

    hyperparameters = ([int(x.strip()) for x in opt.shape.split(',') if x.strip()], opt.activation, opt.batch_size)
    data, y_scaler = load_and_scale_data(opt.augmented, opt.preprocessed, opt.feature_engineered, opt.scaler, model_path)

    print(f"\nTraining {scale}, {feature}, {preprocessing}, {data_type} model with config {hyperparameters}...\n")
    train(*data, hyperparameters, opt.epochs, opt.scaler, opt.l2_alpha, opt.batch_normalization, opt.dropout, opt.optimizer, y_scaler, model_path, training_history_path)
