import simplepyble
import cv2
import os

from configparser import ConfigParser
from flirpy.camera.lepton import Lepton
from utilities import Logger, singleton

class Device():
    def __init__(self, name, device_type):
        self.DEVICE_NAME = name
        self.TYPE = device_type
    
    def get_data(self):
        pass

    def disconnect(self):
        pass

@singleton
class ThermalCamera(Lepton, Device):
    def __init__(self, name="Lepton 3.1R"):
        Lepton.__init__(self)
        Device.__init__(self, name, "Thermal Camera")

        self._setup_camera()

    def _setup_camera(self):
        try:
            self.setup_video()
        except Exception as e:
            raise e

    def get_data(self):
        return self.grab()
    
    def disconnect(self):
        Logger().log("Disconnected from thermal camera successfully.")
        return super().disconnect()

@singleton
class Sensor(Device):
    def __init__(self, config='sensor_config.ini', name='LYWSD03MMC'):
        Device.__init__(self, name, "Sensor")

        config_path = os.path.join(os.path.dirname(__file__), '../../../config')
        self.DEVICE_NAME = name
        self._ADDRESS, self._SERVICE_UUID, self._TEMPERATURE_HUMIDITY_CHAR_UUID = self._parse_config(f"{config_path}/{config}")
        self._adapter, self._peripheral = self._scan()
        
        self._connect()

    def _parse_config(self, config):
        try:
            parser = ConfigParser()
            parser.read(config)
        except Exception as e:
            raise e
        
        if not parser.has_section("sensor"):
            raise Exception("Missing the 'sensor' section.")
        
        required_attributes = ['SERVICE_UUID', 'TEMPERATURE_HUMIDITY_CHAR_UUID']
        for attribute in required_attributes:
            if not parser.has_option('sensor', attribute):
                raise Exception(f"Missing required attribute '{attribute}'.")
            
        return tuple([parser.get("sensor", option) for option in parser.options("sensor")])   
        
    def _scan(self):
        adapters = simplepyble.Adapter.get_adapters()

        if not adapters:
            raise Exception("No Bluetooth adapters found.")
        
        adapter = adapters[0]
        adapter.set_callback_on_scan_start(lambda: Logger().log(f"Scanning for {self.DEVICE_NAME} device..."))
        adapter.set_callback_on_scan_stop(lambda: Logger().log("Scan complete."))
        adapter.set_callback_on_scan_found(lambda peripheral: self.on_device_found(adapter, peripheral))

        adapter.scan_for(50000)
        peripherals = adapter.scan_get_results()

        if not peripherals:
            raise Exception("No device found")
        
        for peripheral in peripherals:
            if peripheral.address() == self._ADDRESS:
                return adapter, peripheral
            
        raise Exception(f"{self.DEVICE_NAME} not found")
    
    def on_device_found(self, adapter, peripheral):
        Logger().log(f"Scanned {peripheral.identifier()} [{peripheral.address()}]")

        if peripheral.address() == self._ADDRESS:
            Logger().log(f"Found {peripheral.identifier()} [{peripheral.address()}]")
            adapter.scan_stop()

    def _connect(self):
        Logger().log("Connecting to LYWSD003MMC...")

        try:
            self._peripheral.connect()
        except Exception as e:
            raise e
        
        Logger().log("Connected successfully!")

    def get_data(self):
        return int.from_bytes(self._peripheral.read(
            self._SERVICE_UUID, 
            self._TEMPERATURE_HUMIDITY_CHAR_UUID
        )[:2], byteorder='little') / 100

    def disconnect(self):
        if self._peripheral:
            self._peripheral.disconnect()

            Logger().log("Disconnected from LYWSD003MMC successfully.")