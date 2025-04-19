import joblib
import os
import pandas as pd

from keras.api.models import load_model

class CalciferNet:
    def __init__(self, name: str, scale, feature, preprocessing, data_type):
        self.name = name
        
        dir_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/mlp/', scale, feature, preprocessing, data_type))
        
        model_path = os.path.join(dir_path, 'best_model.keras')
        self.model = self._load_model(model_path)

        x_scaler_path = os.path.join(dir_path, 'fs.pkl')
        self.feature_scaler = joblib.load(x_scaler_path)

        y_scaler_path =  os.path.join(dir_path, 'ts.pkl')
        self.target_scaler = joblib.load(y_scaler_path)

    def _load_model(self, model_path: str):
        try:
            model = load_model(model_path)
            
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
        
    def predict(self, input_data: pd.DataFrame):
        input_data = self.feature_scaler.transform(input_data.to_numpy())
        return self.target_scaler.inverse_transform(self.model.predict(input_data, verbose=False))