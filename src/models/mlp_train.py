import os
import time
import json
import yaml
import random
import argparse
import re
import numpy as np
import pandas as pd
import tensorflow as tf

from itertools import product

from keras.api.models import Sequential
from keras.api.layers import InputLayer, Dense, PReLU, Activation, BatchNormalization, Dropout
from keras.api.regularizers import L2
from keras.api.metrics import RootMeanSquaredError, R2Score
from keras.api.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.api.backend import clear_session

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler

import sys

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if src_root not in sys.path:
    sys.path.append(src_root)
    
from features import create_features

SEED_VALUE = 42
random.seed(SEED_VALUE)
np.random.seed(SEED_VALUE)
tf.random.set_seed(SEED_VALUE)

def create_model(n_features, layers, activation, l2_alpha, batch_normalization, dropout, optimizer):
    model = Sequential([InputLayer(input_shape=(n_features,))])
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

def scale_data(X_train, X_test, y_train, y_test, scaler_type):
    scaler_cls = StandardScaler if scaler_type == 'standard' else MinMaxScaler
    X_scaler, y_scaler = scaler_cls(), scaler_cls()
    X_train, X_test = X_scaler.fit_transform(X_train), X_scaler.transform(X_test)
    y_train, y_test = y_scaler.fit_transform(y_train.reshape(-1, 1)).flatten(), y_scaler.transform(y_test.reshape(-1, 1)).flatten()
   
    return X_train, X_test, y_train, y_test, y_scaler

def train(X_train, X_test, y_train, y_test, model, epochs):
    callbacks = [
        ReduceLROnPlateau(factor=0.5, patience=25, min_lr=1e-6),
        EarlyStopping(patience=epochs, restore_best_weights=True)
    ]
   
    return model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, verbose=2, callbacks=callbacks)
    
def grid_search(X, y, hyperparameters_grid: dict, dir_path: str, epochs: int, scaler: str, l2_alpha: float, batch_normalization: bool, dropout: float, optimizer: str, resume_training: bool):
    os.makedirs(dir_path, exist_ok=True)
    hyperparameters_file = f"{dir_path}/hyperparameters.txt"

    if resume_training:
        hyperparameters_list = load_remaining_params(hyperparameters_file)
    else:
        hyperparameters_list = product(*hyperparameters_grid.values())
        save_perm_params(hyperparameters_list, hyperparameters_file)

    for hyperparameters in hyperparameters_list:  
        kf = KFold(n_splits=5, shuffle=True, random_state=SEED_VALUE)
        fold_histories = {}
        start_time = time.time()

        for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), start=1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            if scaler:
                X_train, X_test, y_train, y_test, y_scaler = scale_data(X_train, X_test, y_train, y_test, scaler)
            model = create_model(X.shape[1], *hyperparameters, l2_alpha, batch_normalization, dropout, optimizer)
            result = train(X_train, X_test, y_train, y_test, model, epochs)
            fold_histories = process_results(result, fold_histories, hyperparameters, epochs, fold, scaler, y_scaler)
        duration = time.time() - start_time
        save_results(hyperparameters, duration, fold_histories, dir_path)
        remove_trained_params(hyperparameters, hyperparameters_file)

        clear_session()
    
    print("\nGrid Search Complete!")

def process_results(result: dict, fold_histories: dict, hyperparameters: tuple, epochs: int, fold: int, scaler: str, y_scaler: object):
    scale_factor = y_scaler.scale_[0] if scaler == 'standard' else y_scaler.data_max_[0] - y_scaler.data_min_[0]
    fold_history = {
        'layers': [hyperparameters[0]] * epochs,
        'activation': [hyperparameters[1]] * epochs,
        'batch_size': [hyperparameters[2]] * epochs,
        'fold': [fold] * epochs,
        'epoch': list(range(1, epochs + 1))
    }

    metrics = ['loss', 'val_loss', 'mae', 'val_mae', 'root_mean_squared_error', 'val_root_mean_squared_error']
    for k, v in fold_history.items():
        fold_histories.setdefault(k, []).extend(
            [val * (scale_factor**2 if 'loss' in k and scaler == 'minmax' else scale_factor) if scaler and k in metrics else v for val in v]
        )

    return fold_histories

def save_results(hyperparameters: tuple, duration: float, fold_histories: dict, dir_path: str):
    def create_summary_df(layers: list, activation: str, batch_size: int, duration: float, fold_histories: dict, metrics: list):
        training_duration_df = pd.DataFrame({'layers': [layers], 'activation': [activation], 'batch_size': [batch_size], 'duration': [duration]})
        all_fold_df = pd.DataFrame(fold_histories)
        cv_df = all_fold_df.groupby(['layers', 'activation', 'batch_size', 'epoch']).mean().reset_index().drop(columns=['fold'])
        mean_results_df = pd.DataFrame({
            "layers": [layers],
            "activation": [activation], 
            "batch_size": [batch_size],
            **{metric: [cv_df[metric].mean().round(6)] for metric in metrics}
        })
        peak_results_df = cv_df.loc[[cv_df['val_loss'].idxmin()]][['layers', 'activation', 'batch_size', 'epoch'] + metrics].round(6)
    
        return training_duration_df, all_fold_df, cv_df, mean_results_df, peak_results_df

    file_names = [
        "training_duration_history.csv",
        "all_fold_history.csv",
        "cv_history.csv",
        "mean_results.csv",
        "peak_results.csv"
    ]
    metrics = [
        'loss', 'val_loss', 'mae', 'val_mae', 'root_mean_squared_error', 
        'val_root_mean_squared_error', 'r2_score', 'val_r2_score'
    ]
    layers, activation, batch_size = hyperparameters
    batch_size = int(batch_size)

    dfs = create_summary_df(layers, activation, batch_size, duration, fold_histories, metrics)
    for df, file_name in zip(dfs, file_names):
        file_path = f"{dir_path}/{file_name}"
        df.to_csv(file_path, mode='a', header=not os.path.exists(file_path), index=False)
    
def save_perm_params(hyperparameters_list, hyperparameters_file):
    with open(hyperparameters_file, "w") as f:
        for hyperparameters in hyperparameters_list:
            f.write(",".join([json.dumps(hyperparameters[0])] + list(map(str, hyperparameters[1:]))) + "\n")

def load_remaining_params(param_file: str) -> list:
    params = []
    with open(param_file, "r") as f:
        for line in f:
            match = re.match(r"\[(.*?)\],(.*)", line)
            list_part, rest_part = list(map(int, match.group(1).split(","))), match.group(2).split(",")
            params.append((list_part, *rest_part))

    return params

def remove_trained_params(trained_params, param_file):
    trained_params_str = ",".join([json.dumps(trained_params[0])] + list(map(str, trained_params[1:])))

    with open(param_file, "w") as f:
        remaining_params = [line.strip() for line in f if line.strip() != trained_params_str]
        f.write("\n".join(remaining_params))        
        
def load_hyperparameters(config_path: str):
    with open(f"../../config/{config_path}", 'r') as file:
        config = yaml.safe_load(file)
    
        param_grid = config.get('param_grid', {})
        layers = param_grid.get('layers', [])
        activations = param_grid.get('activations', [])
        batch_sizes = param_grid.get('batch_sizes', [])
        
        return {
            'layers': layers,
            'activations': activations,
            'batch_sizes': batch_sizes
        }

def load_data(augmented:bool, preprocessed: bool, feature_engineered: bool):
    data_file = 'augmented_preprocessed.csv' if augmented and preprocessed else \
                'data_augmented.csv' if augmented else \
                'external_preprocessed.csv' if preprocessed else \
                'dataset_external.csv'
    
    data = pd.read_csv(f"../../data/processed/{data_file}")

    if feature_engineered:
        data = create_features(data)

    data['Tg'], data['Tc'] = round(data['Tg'], 2), round(data['Tc'], 1)
    
    return data.drop(columns=['Tc']).values, data['Tc'].values 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='model_config.yaml', help='Hyperparameters filename')
    parser.add_argument('--augmented', type=bool, default=False, help='Augmented or non-augmented')
    parser.add_argument('--preprocessed', type=bool, default=False, help='Preprocessed or non-processed')
    parser.add_argument('--feature_engineered', type=bool, default=False, help='Feature-engineered or not')
    parser.add_argument('--scaler', type=str, default=None, help='Scaler to use')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train a model')
    parser.add_argument('--l2_alpha', type=float, default=0.01, help='Apply L2 regularizer')
    parser.add_argument('--batch_normalization', type=bool, default=False, help='Apply batch optimization to architecture')
    parser.add_argument('--dropout', type=float, default=0, help='Apply dropout rate to architecture')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use')
    parser.add_argument('--resume', type=int, default=0, help='Resume most recent training')
    opt = parser.parse_args()

    hyperparameters = load_hyperparameters(opt.config)
    data = load_data(opt.augmented, opt.preprocessed, opt.feature_engineered)

    scale = 'standardized' if opt.scaler == 'standard' else 'normalized'
    feature = 'engineered' if opt.feature_engineered else 'non_engineered'
    preprocessing = 'preprocessed' if opt.preprocessed else 'non_processed'
    data_type = 'augmented' if opt.augmented else 'external'
    dir_path = f"../../reports/results/training/{scale}/{feature}/{preprocessing}/{data_type}"

    grid_search(*data, hyperparameters, dir_path, opt.epochs, opt.scaler, opt.l2_alpha, opt.batch_normalization, opt.dropout, opt.optimizer, opt.resume)