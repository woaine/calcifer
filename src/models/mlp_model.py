import joblib

from keras.api.models import load_model

class CalciferNet:
    def __init__(self, name: str, type: str, model_file: str, X_scaler_file: str, y_scaler_file: str):
        self.name = name
        
        model_path = f"../../models/mlp/{type}/{model_file}"
        self.model = self._load_model(model_path)

        x_scaler_path = f"../../models/mlp/{type}/{X_scaler_file}"
        self.feature_scaler = joblib.load(x_scaler_path)

        y_scaler_path =  f"../../models/mlp/{type}/{y_scaler_file}"
        self.target_scaler = joblib.load(y_scaler_path)

    def _load_model(self, model_path: str):
        try:
            model = load_model(model_path)
            
            return model
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")