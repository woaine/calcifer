from realesrgan_ncnn_py import Realesrgan

class SuperResolution:
    def __init__(self):
        self.model = Realesrgan(0, model=2)
    
    def resolve(self, frame):
        return self.model.process_cv2(frame)