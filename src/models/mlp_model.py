import joblib
import os

from keras.api.models import load_model
from utilities import Logger

class CalciferNet:
    def __init__(self, name: str, scale, feature, preprocessing, data_type):
        self.name = name
        
        model_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../models/mlp'))
        self.model = self._load_model(f"{model_path}/{scale}/{feature}/{preprocessing}/{data_type}/best_model.keras")
        self._feature_scaler = joblib.load(f"{model_path}/{scale}/{feature}/{preprocessing}/{data_type}/fs.pkl")
        self._target_scaler = joblib.load(f"{model_path}/{scale}/{feature}/{preprocessing}/{data_type}/ts.pkl")

    def _load_model(self, model_path: str):
        try:
            model = load_model(model_path)
        except Exception as e:
            raise RuntimeError(f"Error loading model: {e}")
        
        Logger().log(f"Loaded {self.name} model successfully.")
        return model
        
    def predict(self, input):
        input_data = self._feature_scaler.transform(input)
        return self._target_scaler.inverse_transform(self.model.predict(input_data, verbose=False))