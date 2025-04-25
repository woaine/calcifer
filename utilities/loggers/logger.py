import logging
import threading
import os
import queue
from datetime import datetime

from utilities import ObservableVariable, lock_singleton, SystemLogHandler

@lock_singleton
class Logger:
    def __init__(self):
        self._current_log = ObservableVariable(value="")
        self._log_queue = queue.Queue()
        self._logger = self._create_logger()
        self._log_methods = {
            'debug': self._logger.debug,
            'info': self._logger.info,
            'warning': self._logger.warning,
            'error': self._logger.error
        }

    @property
    def current_log(self):
        return self._current_log
    
    @property
    def logger(self):
        return self._logger
    
    def _create_logger(self):
        logger = logging.getLogger("AppLogger")
        logger.setLevel(logging.INFO)

        logs = self.get_logs()

        file_handler = logging.FileHandler(logs)
        console_handler = logging.StreamHandler()
        custom_handler = SystemLogHandler(self)

        for handler in [file_handler, console_handler, custom_handler]:
            handler.setLevel(logging.DEBUG if handler == file_handler else logging.INFO)
            handler.setFormatter(logging.Formatter("%(asctime)s :: %(levelname)s :: %(message)s"))
            
            logger.addHandler(handler)

        return logger

    def get_logs(self):
        logs_path = os.path.join(os.path.dirname(__file__), '../../app/logs')
        os.makedirs(logs_path, exist_ok=True)

        return os.path.join(logs_path, f"app_{datetime.now().strftime('%Y-%m-%d')}.txt")
    
    def log(self, message, level='info'):
        self._log_methods.get(level, self._logger.info)(message)