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

import sys

src_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
if src_root not in sys.path:
    sys.path.append(src_root)
    
from features import create_features

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
        
def train(X, y, hyperparameters: tuple, l2_alpha: float, batch_normalization: bool, dropout: float, optimizer: str, model_save_path: str):
    layers, activation, batch_size = hyperparameters
    
    model = create_model(X.shape[1], layers, activation, l2_alpha, batch_normalization, dropout, optimizer)
    callbacks = [
        ReduceLROnPlateau(monitor='loss', factor=0.5, patience=25, min_lr=1e-6),
        EarlyStopping(monitor='loss', patience=300, restore_best_weights=True)
    ]

    model.fit(X, y, batch_size=batch_size, epochs=300, verbose=2, callbacks=callbacks)
    model.save(model_save_path)

    clear_session()

def load_and_scale_data(augmented:bool, preprocessed: bool, feature_engineered: bool, scaler: str):
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

    if scaler == 'standard':
        x_scaler = StandardScaler().fit(X)
        y_scaler = StandardScaler().fit(y.reshape(-1, 1))
    else:
        x_scaler = MinMaxScaler().fit(X)
        y_scaler = MinMaxScaler().fit(y.reshape(-1, 1))
        
    X_train = x_scaler.transform(X)
    y_train = y_scaler.transform(y.reshape(-1, 1)).flatten()

    joblib.dump(x_scaler, x_scaler_path)
    joblib.dump(y_scaler, y_scaler_path)

    return X_train, y_train
     
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--shape',  type=str, required=True, help='Number of neurons per layer')
    parser.add_argument('--activation', type=str, required=True, help='Activation function')
    parser.add_argument('--batch_size', type=int, required=True, help='Batch size to process')
    parser.add_argument('--augmented', type=bool, default=False, help='Augmented or non-augmented')
    parser.add_argument('--preprocessed', type=bool, default=False, help='Preprocessed or non-processed')
    parser.add_argument('--feature_engineered', type=bool, default=False, help='Feature-engineered or not')
    parser.add_argument('--scaler', type=str, default=None, help='Scaler to use')
    parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train a model')
    parser.add_argument('--l2_alpha', type=float, default=0.01, help='Apply L2 regularizer')
    parser.add_argument('--batch_normalization', type=bool, default=False, help='Apply batch optimization to architecture')
    parser.add_argument('--dropout', type=float, default=0, help='Apply dropout rate to architecture')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer to use')
    opt = parser.parse_args()

    scale = 'standardized' if opt.scaler == 'standard' else 'normalized'
    feature = 'engineered' if opt.feature_engineered else 'non_engineered'
    preprocessing = 'preprocessed' if opt.preprocessed else 'non_processed'
    data_type = 'augmented' if opt.augmented else 'external'
    dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/mlp/', scale, feature, preprocessing, data_type))

    os.makedirs(dir_path, exist_ok=True)
    x_scaler_path = os.path.join(dir_path, 'fs.pkl')
    y_scaler_path = os.path.join(dir_path, 'ts.pkl')
    model_save_path = os.path.join(dir_path, 'best_model.keras')

    hyperparameters = ([int(x.strip()) for x in opt.shape.split(',') if x.strip()], opt.activation, opt.batch_size)
    data = load_and_scale_data(opt.augmented, opt.preprocessed, opt.feature_engineered, opt.scaler)

    print(f"\nTraining {scale}, {feature}, {preprocessing}, {data_type} model with config {hyperparameters}...\n")
    train(*data, hyperparameters, opt.l2_alpha, opt.batch_normalization, opt.dropout, opt.optimizer, model_save_path)
