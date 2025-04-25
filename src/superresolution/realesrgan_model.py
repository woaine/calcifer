from realesrgan_ncnn_py import Realesrgan

from utilities import Logger, singleton
@singleton
class SuperResolution:
    def __init__(self):
        self.model = self._load_model()

    def _load_model(self):
        try:
            model = Realesrgan(0, model=2)
        except Exception as e:
            raise e

        Logger().log("Loaded superresolution model successfully.")
        return model
    
    def resolve(self, frame):
        return self.model.process_cv2(frame)