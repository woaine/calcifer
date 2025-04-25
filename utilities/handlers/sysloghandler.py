import logging

class SystemLogHandler(logging.Handler):
    def __init__(self, logger):
        super().__init__()
        self.logger = logger

    def emit(self, record):
        log_entry = self.format(record)

        self.logger.current_log.value = log_entry
        self.logger.logger.debug(log_entry)