from utilities import ObservableVariable, singleton

@singleton
class State:
    def __init__(self):
        self._variables = {
            'state': 0,
            'predicting': 0,
            'saving': 0,
            'recording': ObservableVariable(value=0),
            'config_updated': ObservableVariable(value=0)
        }

    def __getattr__(self, name):
        if name in self._variables:
            return self._variables[name]
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")