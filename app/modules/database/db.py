import MySQLdb
import threading
import queue
import os

from configparser import ConfigParser

from utilities import Logger, lock_singleton

@lock_singleton
class Database:
    def __init__(self):
        self._connected = False
        self._connection = None
        
        self._listeners = []

        config_path = os.path.join(os.path.dirname(__file__), '../../../config')
        host, user, password, database, port = self._parse_config(f"{config_path}/db_config.ini")
        self._connect(host, user, password, database, port)

        self._queue = queue.Queue()

        threading.Thread(target=self._process_queue, daemon=True).start()
    
    def _parse_config(self, config):
        try:
            parser = ConfigParser()
            parser.read(config)
        except Exception as e:
            raise e

        if not parser.has_section('database'):
            raise Exception("Missing 'database' section.")
        
        required_attributes = ['DB_HOST', 'DB_USER', 'DB_PASS', 'DB_NAME', 'DB_PORT']
        for attribute in required_attributes:
            if not parser.has_option('database', attribute):
                raise Exception(f"Missing required attribute '{attribute}'.")
        
        return parser.get('database', 'DB_HOST'), parser.get('database', 'DB_USER'), \
            parser.get('database', 'DB_PASS'), parser.get('database', 'DB_NAME'), \
            parser.getint('database', 'DB_PORT')
    
    def _connect(self, host, user, password, database, port):
        Logger().log("Connecting to the database server...")
        
        try:
            if not self._connected:
                self._connection = MySQLdb.connect(
                    host=host,
                    user=user,
                    password=password,
                    database=database,
                    port=port
                )
                self._connected = True
        except Exception as e:
            raise e
        
        Logger().log("Connected successfully to the database server")
    
    def disconnect(self):
        if self._connected:
            self._connection.close()
            self._connected = False
        
        Logger().log("Disconnected successfully from the database server")
    
    def add_listener(self, listener):
        """Add a listener to be notified when new data is inserted."""
        self._listeners.append(listener)

    def notify_listeners(self, data):
        """Notify all listeners about the new data."""
        for listener in self._listeners:
            listener(data)

    def insert_data(self, model, Tg, Ta, Tp):
        self._queue.put(("insert", (model, Tg, Ta, Tp)))

    def get_data(self, callback):
        self._queue.put(("retrieve", callback))

    def _process_queue(self):
        while True:
            task_type, task_data = self._queue.get()

            if task_type == "insert":
                model, Tg, Ta, Tp = task_data
                query = """
                INSERT INTO temperatures (model, Tg, Ta, Tp, timestamp)
                VALUES (%s, %s, %s, %s, NOW())
                """
                try:
                    with self._connection.cursor() as cursor:
                        cursor.execute(query, task_data)
                        self._connection.commit()
                        
                        self.notify_listeners((model, Tg, Tp))
                except Exception as e:
                    Logger().log(f"Failed to insert temperature data: {e}", "error")

            elif task_type == "retrieve":
                callback = task_data
                query = "SELECT model, Tg, Tp FROM temperatures WHERE DATE(timestamp) = CURDATE()"
                try:
                    with self._connection.cursor() as cursor:
                        cursor.execute(query)
                        result = cursor.fetchall()
                        Logger().log(f"Data from result: {result}", "debug")
                        if callback:
                            callback(result)
                except Exception as e:
                    Logger().log(f"Failed to retrieve data: {e}", "error")
                    if callback:
                        callback(None)

            self._queue.task_done()
