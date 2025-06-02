import numpy as np

class ObservableVariable:
    def __init__(self, value):
        self._value = value
        self._listeners = []

    @property
    def value(self):
        return self._value

    @value.setter
    def value(self, new_value):
        if isinstance(self._value, np.ndarray) and isinstance(new_value, np.ndarray):
            if not np.array_equal(self._value, new_value):
                self._value = new_value
                self._notify_listeners()
        elif isinstance(self._value, np.ndarray) ^ isinstance(new_value, np.ndarray):
            self._value = new_value
            self._notify_listeners()
        elif self._value != new_value:
            self._value = new_value
            self._notify_listeners()

    def add_listener(self, listener):
        if callable(listener):
            self._listeners.append(listener)

    def remove_listener(self, index):
        self._listeners.pop(index)

    def _notify_listeners(self):
        for listener in self._listeners:
            listener(self._value)