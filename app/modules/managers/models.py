from utilities import singleton
from src.models import CalciferNet
from enum import Enum

@singleton
class Model:
    def __init__(self):
        self.load_model("SED", ModelType["SED"].value)
    
    def load_model(self, name, model_type):
        self.model = CalciferNet(name, *model_type)
            
class ModelType(Enum):
    SED = ('standardized', 'non_engineered', 'non_processed', 'external')
    SAD = ('standardized', 'non_engineered', 'non_processed', 'augmented')
    SPED = ('standardized', 'non_engineered', 'preprocessed', 'external')
    SPAD = ('standardized', 'non_engineered', 'preprocessed', 'augmented')
    NED = ('normalized', 'non_engineered', 'non_processed', 'external')
    NAD = ('normalized', 'non_engineered', 'non_processed', 'augmented')
    NPED = ('normalized', 'non_engineered', 'preprocessed', 'external')
    NPAD = ('normalized', 'non_engineered', 'preprocessed', 'augmented')

    